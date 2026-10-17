"""Expected value, break-even save rate and sensitivity tables for the flagged
group. Everything is computed from the arguments; no configuration is read.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from retention_platform.inference.outputs import SNAPSHOT_NOTICE, document_header
from retention_platform.inference.ranking import top_k_count

SAVE_RATE_SWEEP = (0.05, 0.10, 0.20, 0.30, 0.40)
MONTHS_KEPT_SWEEP = (1, 3, 6, 12)
OFFER_COST_SWEEP = (10, 20, 40, 60)
ARPU_MULTIPLES = (0.8, 1.0, 1.2)

ARPU_DERIVED = "derived"
ARPU_OVERRIDE = "override"

TABLE_COLUMNS = ["customer_id", "churn_probability", "rank"]
GUARDED_TABLE_COLUMNS = [*TABLE_COLUMNS, "in_top_k"]


class EconomicsError(Exception):
    """Raised when the data cannot be used for the economics calculation."""


def _is_real_number(value: object) -> bool:
    """True for a finite int or float (not bool) that is representable as a float."""
    if isinstance(value, (bool, np.bool_)):
        return False
    if isinstance(value, (int, np.integer)):
        try:
            float(value)
        except OverflowError:
            return False
        return True
    return isinstance(value, (float, np.floating)) and math.isfinite(value)


@dataclass(frozen=True)
class EconomicsAssumptions:
    """The base assumptions. arpu_proxy None means derive it from the group."""

    save_rate: float
    months_kept: int
    offer_cost: float
    arpu_proxy: float | None = None

    def __post_init__(self) -> None:
        if not (_is_real_number(self.save_rate) and 0 < self.save_rate <= 1):
            raise ValueError(
                f"save_rate must be a number with 0 < save_rate <= 1, got {self.save_rate!r}"
            )
        if not (
            isinstance(self.months_kept, int)
            and _is_real_number(self.months_kept)
            and self.months_kept >= 1
        ):
            raise ValueError(f"months_kept must be a whole number >= 1, got {self.months_kept!r}")
        if not (_is_real_number(self.offer_cost) and self.offer_cost > 0):
            raise ValueError(f"offer_cost must be a finite number > 0, got {self.offer_cost!r}")
        if self.arpu_proxy is not None and not (
            _is_real_number(self.arpu_proxy) and self.arpu_proxy > 0
        ):
            raise ValueError(
                f"arpu_proxy must be null or a finite number > 0, got {self.arpu_proxy!r}"
            )


@dataclass(frozen=True)
class BaseCase:
    """The group figures at the base assumptions."""

    customers_contacted: int
    total_offer_cost: float
    expected_would_be_churners: float
    expected_customers_saved: float
    gross_revenue_kept: float
    net_expected_value: float
    ev_per_contacted_customer: float
    break_even_save_rate: float
    arpu_proxy: float
    arpu_source: str
    arpu_customers_used: int | None


@dataclass(frozen=True, eq=False)
class EconomicsResults:
    """The base case and the four sensitivity tables."""

    base_case: BaseCase
    one_at_a_time: pd.DataFrame
    save_rate_by_months: pd.DataFrame
    break_even_by_cost_and_months: pd.DataFrame
    k_sweep: pd.DataFrame


@dataclass(frozen=True)
class _Data:
    """The checked table and monthly charges, all in the table's row order."""

    ids: list
    probabilities: np.ndarray
    ranks: np.ndarray
    charges: np.ndarray


def _check_fraction(value: object, name: str) -> None:
    if not (_is_real_number(value) and 0 < value <= 1):
        raise ValueError(f"{name} must be a number with 0 < {name} <= 1, got {value!r}")


def _as_floats(values: pd.Series, what: str, ids: list) -> np.ndarray:
    """The values as float64, refusing anything that is missing, not a number
    or not finite; the message names the first offending customer."""
    result = np.empty(len(values), dtype="float64")
    for position, value in enumerate(values.tolist()):
        customer = ids[position]
        if value is None or value is pd.NA or (isinstance(value, (float, np.floating)) and value != value):
            raise EconomicsError(f"{what} is missing for customer {customer!r}")
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.integer, np.floating)):
            raise EconomicsError(f"{what} is not a number for customer {customer!r}: {value!r}")
        try:
            number = float(value)
        except OverflowError:
            raise EconomicsError(f"{what} is not finite for customer {customer!r}") from None
        if not math.isfinite(number):
            raise EconomicsError(f"{what} is not finite for customer {customer!r}")
        result[position] = number
    return result


def _checked_data(table: pd.DataFrame, monthly_charge: pd.Series, required: list[str]) -> _Data:
    if not isinstance(table, pd.DataFrame):
        raise ValueError(f"table must be a DataFrame, got {type(table).__name__}")
    missing = [column for column in required if column not in table.columns]
    if missing:
        raise ValueError(f"table is missing columns {missing}")
    if not isinstance(monthly_charge, pd.Series):
        raise ValueError(f"monthly_charge must be a Series, got {type(monthly_charge).__name__}")

    n_rows = len(table)
    if n_rows == 0:
        raise EconomicsError("the table has no rows")

    ids = table["customer_id"].tolist()
    if len(set(ids)) != n_rows:
        duplicate = next(value for value in ids if ids.count(value) > 1)
        raise EconomicsError(f"duplicate customer_id in the table: {duplicate!r}")

    rank = table["rank"]
    if rank.dtype == np.bool_ or not pd.api.types.is_integer_dtype(rank.dtype) or rank.isna().any():
        raise EconomicsError(f"rank must be a complete integer column, got dtype {rank.dtype}")
    ranks = rank.to_numpy(dtype="int64")
    if not np.array_equal(np.sort(ranks), np.arange(1, n_rows + 1)):
        raise EconomicsError(f"rank is not a complete 1..{n_rows} sequence")

    probabilities = _as_floats(table["churn_probability"], "churn_probability", ids)
    outside = np.flatnonzero((probabilities < 0) | (probabilities > 1))
    if outside.size:
        position = int(outside[0])
        raise EconomicsError(
            f"churn_probability is outside [0, 1] for customer {ids[position]!r}: "
            f"{probabilities[position]!r}"
        )

    series_ids = monthly_charge.index.tolist()
    if len(set(series_ids)) != len(series_ids):
        duplicate = next(value for value in series_ids if series_ids.count(value) > 1)
        raise EconomicsError(f"duplicate customer_id in the monthly_charge index: {duplicate!r}")
    known_ids, known_series_ids = set(ids), set(series_ids)
    absent = [value for value in ids if value not in known_series_ids]
    if absent:
        raise EconomicsError(f"monthly_charge has no value for customer {absent[0]!r}")
    extra = [value for value in series_ids if value not in known_ids]
    if extra:
        raise EconomicsError(f"monthly_charge has a value for unknown customer {extra[0]!r}")

    charges = _as_floats(monthly_charge.loc[ids], "monthly_charge", ids)
    negative = np.flatnonzero(charges < 0)
    if negative.size:
        position = int(negative[0])
        raise EconomicsError(
            f"monthly_charge is negative for customer {ids[position]!r}: {charges[position]!r}"
        )
    return _Data(ids=ids, probabilities=probabilities, ranks=ranks, charges=charges)


def _group_positions(data: _Data, k_frac: float) -> np.ndarray:
    """Row positions of the first top_k_count(n, k_frac) rows by rank, in rank order."""
    size = top_k_count(len(data.ids), k_frac)
    return np.argsort(data.ranks, kind="stable")[:size]


def _check_flags(table: pd.DataFrame, data: _Data, k_frac: float) -> None:
    flags = table["in_top_k"]
    if flags.dtype != np.bool_:
        raise EconomicsError(f"in_top_k must be a boolean column, got dtype {flags.dtype}")

    expected = top_k_count(len(data.ids), k_frac)
    flagged = int(flags.sum())
    if flagged != expected:
        raise EconomicsError(
            f"in_top_k marks {flagged} rows but top_k_count for k={k_frac!r} is {expected}"
        )
    first_by_rank = {data.ids[position] for position in _group_positions(data, k_frac)}
    flagged_ids = {data.ids[position] for position in np.flatnonzero(flags.to_numpy())}
    if flagged_ids != first_by_rank:
        raise EconomicsError(f"the in_top_k rows are not the first {expected} rows by rank")


def flagged_group(table: pd.DataFrame, monthly_charge: pd.Series, k_frac: float) -> pd.DataFrame:
    """The first top_k_count(n, k_frac) rows of the ranked table, in rank
    order, with customer_id, churn_probability and monthly_charge.

    monthly_charge is a Series indexed by customer_id and is matched to the
    table by id.
    """
    _check_fraction(k_frac, "k_frac")
    data = _checked_data(table, monthly_charge, TABLE_COLUMNS)
    positions = _group_positions(data, k_frac)
    return pd.DataFrame(
        {
            "customer_id": pd.Series([data.ids[p] for p in positions], dtype=object),
            "churn_probability": data.probabilities[positions],
            "monthly_charge": data.charges[positions],
        }
    )


def _arpu(charges: np.ndarray, assumptions: EconomicsAssumptions) -> tuple[float, str, int | None]:
    if assumptions.arpu_proxy is not None:
        return float(assumptions.arpu_proxy), ARPU_OVERRIDE, None
    mean = math.fsum(charges.tolist()) / len(charges)
    if not mean > 0:
        raise EconomicsError(f"the derived ARPU proxy is not above 0: {mean!r}")
    return mean, ARPU_DERIVED, len(charges)


def _figures(
    probabilities: np.ndarray, arpu: float, save_rate: float, months_kept: int, offer_cost: float
) -> dict:
    """The group figures for one set of inputs."""
    contacted = len(probabilities)
    would_be_churners = math.fsum(probabilities.tolist())
    if would_be_churners == 0:
        raise EconomicsError("the sum of the churn probabilities of the group is 0")

    total_cost = contacted * offer_cost
    saved = would_be_churners * save_rate
    gross = saved * arpu * months_kept
    net = gross - total_cost
    return {
        "customers_contacted": contacted,
        "total_offer_cost": float(total_cost),
        "expected_would_be_churners": would_be_churners,
        "expected_customers_saved": saved,
        "gross_revenue_kept": gross,
        "net_expected_value": net,
        "ev_per_contacted_customer": net / contacted,
        "break_even_save_rate": total_cost / (arpu * months_kept * would_be_churners),
    }


def analyze_economics(
    table: pd.DataFrame,
    monthly_charge: pd.Series,
    k_frac: float,
    k_values: list[float],
    assumptions: EconomicsAssumptions,
) -> EconomicsResults:
    """The base case and the sensitivity tables for the flagged group.

    table is the ranked scoring table (customer_id, churn_probability, rank,
    in_top_k); monthly_charge is a Series indexed by customer_id. The table
    is checked first and EconomicsError names the condition that fails. The
    inputs are not changed.
    """
    _check_fraction(k_frac, "k_frac")
    k_values = list(k_values)
    if not k_values:
        raise ValueError("k_values is empty")
    for k in k_values:
        _check_fraction(k, "k_values entry")
    if not isinstance(assumptions, EconomicsAssumptions):
        raise ValueError(f"assumptions must be EconomicsAssumptions, got {type(assumptions).__name__}")

    data = _checked_data(table, monthly_charge, GUARDED_TABLE_COLUMNS)
    _check_flags(table, data, k_frac)

    def group_for(k: float) -> tuple[np.ndarray, np.ndarray]:
        positions = _group_positions(data, k)
        return data.probabilities[positions], data.charges[positions]

    save_rate, months_kept, offer_cost = (
        assumptions.save_rate,
        assumptions.months_kept,
        assumptions.offer_cost,
    )
    probabilities, charges = group_for(k_frac)
    arpu, source, used = _arpu(charges, assumptions)

    base_figures = _figures(probabilities, arpu, save_rate, months_kept, offer_cost)
    base_case = BaseCase(**base_figures, arpu_proxy=arpu, arpu_source=source, arpu_customers_used=used)

    def at(save=save_rate, months=months_kept, cost=offer_cost, arpu_value=arpu) -> dict:
        return _figures(probabilities, arpu_value, save, months, cost)

    rows = []
    variants = [
        *[("save_rate", v, at(save=v)) for v in SAVE_RATE_SWEEP],
        *[("months_kept", v, at(months=v)) for v in MONTHS_KEPT_SWEEP],
        *[("offer_cost", v, at(cost=v)) for v in OFFER_COST_SWEEP],
        *[("arpu_multiple", v, at(arpu_value=arpu * v)) for v in ARPU_MULTIPLES],
    ]
    for name, value, figures in variants:
        rows.append(
            {
                "input": name,
                "value": value,
                "net_expected_value": figures["net_expected_value"],
                "ev_per_contacted_customer": figures["ev_per_contacted_customer"],
                "break_even_save_rate": figures["break_even_save_rate"],
            }
        )
    one_at_a_time = pd.DataFrame(rows)

    save_rate_by_months = pd.DataFrame(
        [[at(save=s, months=m)["net_expected_value"] for m in MONTHS_KEPT_SWEEP] for s in SAVE_RATE_SWEEP],
        index=pd.Index(SAVE_RATE_SWEEP, name="save_rate"),
        columns=pd.Index(MONTHS_KEPT_SWEEP, name="months_kept"),
    )
    break_even_by_cost_and_months = pd.DataFrame(
        [[at(months=m, cost=c)["break_even_save_rate"] for c in OFFER_COST_SWEEP] for m in MONTHS_KEPT_SWEEP],
        index=pd.Index(MONTHS_KEPT_SWEEP, name="months_kept"),
        columns=pd.Index(OFFER_COST_SWEEP, name="offer_cost"),
    )

    sweep_rows = []
    for k in k_values:
        k_probabilities, k_charges = group_for(k)
        k_arpu, _, _ = _arpu(k_charges, assumptions)
        figures = _figures(k_probabilities, k_arpu, save_rate, months_kept, offer_cost)
        sweep_rows.append(
            {
                "k": k,
                "customers_contacted": figures["customers_contacted"],
                "arpu_proxy": k_arpu,
                "expected_would_be_churners": figures["expected_would_be_churners"],
                "net_expected_value": figures["net_expected_value"],
                "ev_per_contacted_customer": figures["ev_per_contacted_customer"],
                "break_even_save_rate": figures["break_even_save_rate"],
            }
        )

    return EconomicsResults(
        base_case=base_case,
        one_at_a_time=one_at_a_time,
        save_rate_by_months=save_rate_by_months,
        break_even_by_cost_and_months=break_even_by_cost_and_months,
        k_sweep=pd.DataFrame(sweep_rows),
    )


DOCUMENT_TITLE = "Economics Summary"
SIMULATED_PHRASE = "Simulated result under stated assumptions"
BREAK_EVEN_ABOVE_ONE = "Above 100%: cannot pay off at any save rate"


def _money(value: float) -> str:
    text = f"{value:,.2f}"
    return "0.00" if text == "-0.00" else text


def _whole_or_money(value: float) -> str:
    return str(int(value)) if float(value) == int(value) else _money(value)


def _percent(fraction: float) -> str:
    """A fraction as a percentage, whole when it is whole, so that
    0.30000000000000004 prints as 30%."""
    text = f"{round(fraction * 100, 6):.2f}".rstrip("0").rstrip(".")
    return f"{text}%"


def _break_even(value: float) -> str:
    if not math.isfinite(value):
        raise ValueError(f"the break-even save rate is not finite: {value!r}")
    if value > 1:
        return BREAK_EVEN_ABOVE_ONE
    return f"{100 * value:.1f}%"


def _table(headers: list[str], rows: list[list[str]]) -> str:
    lines = [f"| {' | '.join(headers)} |", f"|{'|'.join('---' for _ in headers)}|"]
    lines.extend(f"| {' | '.join(row)} |" for row in rows)
    return "\n".join(lines)


def _one_at_a_time_cells(name: str, value: float) -> tuple[str, str]:
    if name == "save_rate":
        return "Save rate", _percent(value)
    if name == "months_kept":
        return "Months kept", str(int(value))
    if name == "offer_cost":
        return "Offer cost", str(int(value))
    return "ARPU multiple", f"{value:.1f}x"


def _check_document_arguments(
    results: EconomicsResults,
    assumptions: EconomicsAssumptions,
    k_frac: float,
    customers_scored: int,
) -> None:
    if not isinstance(results, EconomicsResults):
        raise ValueError(f"results must be EconomicsResults, got {type(results).__name__}")
    if not isinstance(assumptions, EconomicsAssumptions):
        raise ValueError(f"assumptions must be EconomicsAssumptions, got {type(assumptions).__name__}")
    _check_fraction(k_frac, "k_frac")
    contacted = results.base_case.customers_contacted
    if isinstance(customers_scored, bool) or not isinstance(customers_scored, int):
        raise ValueError(f"customers_scored must be an int, got {customers_scored!r}")
    if customers_scored < contacted:
        raise ValueError(
            f"customers_scored ({customers_scored}) is below customers_contacted ({contacted})"
        )
    expected_contacted = top_k_count(customers_scored, k_frac)
    if expected_contacted != contacted:
        raise ValueError(
            f"top_k_count({customers_scored}, {k_frac!r}) is {expected_contacted} "
            f"but the results contacted {contacted} customers"
        )
    is_override = results.base_case.arpu_source == ARPU_OVERRIDE
    if is_override != (assumptions.arpu_proxy is not None):
        raise ValueError("the assumptions and the results disagree on the ARPU proxy source")


def render_economics_document(
    results: EconomicsResults,
    assumptions: EconomicsAssumptions,
    k_frac: float,
    customers_scored: int,
    scoring_run_id: str,
    model_run_id: str,
) -> str:
    """The economics document as Markdown text: the fixed run-ID header, then
    eight sections. Every figure comes from results and assumptions, so the
    same inputs give the same text. Nothing is written to disk."""
    _check_document_arguments(results, assumptions, k_frac, customers_scored)
    header = document_header(DOCUMENT_TITLE, scoring_run_id, model_run_id)

    base = results.base_case
    arpu = _money(base.arpu_proxy)
    cost = _whole_or_money(assumptions.offer_cost)
    save = _percent(assumptions.save_rate)
    months = str(assumptions.months_kept)
    base_inputs = (
        f"save rate {save}, months kept {months}, offer cost {cost} currency units, "
        f"ARPU proxy {arpu} currency units"
    )

    if base.arpu_source == ARPU_DERIVED:
        arpu_source = (
            f"derived: mean monthly_charge of the {base.arpu_customers_used:,} flagged customers "
            "(observed statistic)"
        )
    else:
        arpu_source = "user-supplied what-if value, not an observed value"

    sections: list[tuple[str, list[str]]] = []

    sections.append(
        (
            "What this document is",
            [
                "This document is generated by the scoring run, from the same execution as the "
                "prioritized outreach list. It asks what contacting the flagged group might be worth. "
                f"The flagged group is the top {_percent(k_frac)} of the scored customers: "
                f"{base.customers_contacted:,} of {customers_scored:,} customers.",
                "The data holds no information about what an offer costs, how many contacted customers "
                "an offer would keep, or for how long. Every result below is therefore a simulated "
                "result under stated assumptions, not a measured campaign outcome.",
                "Three kinds of figure are kept apart. Observed: the ARPU proxy, when it is derived "
                "from the monthly charges of the flagged group. Assumed: save rate, months kept and "
                "offer cost. Simulated: the expected value, the break-even save rate and every table "
                "below.",
                SNAPSHOT_NOTICE,
            ],
        )
    )

    sections.append(
        (
            "Assumptions",
            [
                "The assumptions behind every result, with the sensitivity list used for each.",
                _table(
                    ["Input", "Base value", "Source and sensitivity list"],
                    [
                        [
                            "Save rate",
                            save,
                            "project assumption, not estimated from the data; list: "
                            + ", ".join(_percent(v) for v in SAVE_RATE_SWEEP),
                        ],
                        [
                            "Months kept",
                            months,
                            "project assumption, not estimated from the data; list: "
                            + ", ".join(str(v) for v in MONTHS_KEPT_SWEEP),
                        ],
                        [
                            "Offer cost (currency units)",
                            cost,
                            "project assumption, not estimated from the data; list: "
                            + ", ".join(str(v) for v in OFFER_COST_SWEEP),
                        ],
                        [
                            "ARPU proxy (currency units per month)",
                            arpu,
                            f"{arpu_source}; list: "
                            + ", ".join(f"{v:.1f}x" for v in ARPU_MULTIPLES)
                            + " the base ARPU proxy",
                        ],
                    ],
                ),
                "Money amounts are in currency units; nothing in the data establishes a currency. "
                "monthly_charge is a revenue proxy, not profit and not necessarily realized revenue.",
            ],
        )
    )

    base_rows = [
        ["Customers contacted", f"{base.customers_contacted:,}"],
        ["Total offer cost (currency units)", _money(base.total_offer_cost)],
        ["Expected would-be churners", _money(base.expected_would_be_churners)],
        ["Expected customers saved", _money(base.expected_customers_saved)],
        ["Gross revenue kept (currency units)", _money(base.gross_revenue_kept)],
        ["Net expected value (currency units)", _money(base.net_expected_value)],
        ["Expected value per contacted customer (currency units)", _money(base.ev_per_contacted_customer)],
        ["Break-even save rate", _break_even(base.break_even_save_rate)],
    ]
    sections.append(
        (
            "Base case",
            [
                f"{SIMULATED_PHRASE}. Fixed inputs: {base_inputs}.",
                _table(["Figure", "Value"], base_rows),
                "No ROI is calculated, because the figures are simulated and not measured campaign "
                "outcomes.",
            ],
        )
    )

    sweep_rows = []
    for row in results.one_at_a_time.itertuples(index=False):
        label, shown = _one_at_a_time_cells(row.input, row.value)
        sweep_rows.append(
            [
                label,
                shown,
                _money(row.net_expected_value),
                _money(row.ev_per_contacted_customer),
                _break_even(row.break_even_save_rate),
            ]
        )
    sections.append(
        (
            "One input at a time",
            [
                f"{SIMULATED_PHRASE}. Each row changes one input; the other inputs stay at their "
                f"base values ({base_inputs}).",
                _table(
                    [
                        "Input",
                        "Value",
                        "Net expected value (currency units)",
                        "Expected value per contacted customer (currency units)",
                        "Break-even save rate",
                    ],
                    sweep_rows,
                ),
            ],
        )
    )

    by_months = results.save_rate_by_months
    sections.append(
        (
            "Save rate and months kept",
            [
                f"{SIMULATED_PHRASE}. Net expected value in currency units. Fixed inputs: offer cost "
                f"{cost} currency units, ARPU proxy {arpu} currency units. Rows are save rates, "
                "columns are months kept.",
                _table(
                    ["Save rate", *[f"{m} {'month' if m == 1 else 'months'}" for m in by_months.columns]],
                    [
                        [_percent(rate), *[_money(v) for v in by_months.loc[rate]]]
                        for rate in by_months.index
                    ],
                ),
            ],
        )
    )

    by_cost = results.break_even_by_cost_and_months
    sections.append(
        (
            "Break-even save rate",
            [
                f"{SIMULATED_PHRASE}. Fixed inputs: ARPU proxy {arpu} currency units. Rows are "
                "months kept, columns are offer costs in currency units.",
                _table(
                    ["Months kept", *[f"Offer cost {c}" for c in by_cost.columns]],
                    [
                        [str(m), *[_break_even(v) for v in by_cost.loc[m]]]
                        for m in by_cost.index
                    ],
                ),
            ],
        )
    )

    if base.arpu_source == ARPU_DERIVED:
        arpu_rule = "the ARPU proxy is derived separately from each K's flagged group"
    else:
        arpu_rule = f"the same ARPU override of {arpu} currency units is used for every K"
    sections.append(
        (
            "Outreach group size",
            [
                "Each K uses its own flagged group, the first rows of the ranked list. Without an ARPU "
                "override the ARPU proxy is derived separately for each K; with an override the same "
                "override is used for every K. The other inputs stay at their base values. This sweep "
                "is a sensitivity analysis and is not a recommendation to change K.",
                f"{SIMULATED_PHRASE}. Fixed inputs: save rate {save}, months kept {months}, offer cost "
                f"{cost} currency units; {arpu_rule}.",
                _table(
                    [
                        "K",
                        "Customers contacted",
                        "ARPU proxy (currency units)",
                        "Expected would-be churners",
                        "Net expected value (currency units)",
                        "Expected value per contacted customer (currency units)",
                        "Break-even save rate",
                    ],
                    [
                        [
                            _percent(row.k),
                            f"{int(row.customers_contacted):,}",
                            _money(row.arpu_proxy),
                            _money(row.expected_would_be_churners),
                            _money(row.net_expected_value),
                            _money(row.ev_per_contacted_customer),
                            _break_even(row.break_even_save_rate),
                        ]
                        for row in results.k_sweep.itertuples(index=False)
                    ],
                ),
            ],
        )
    )

    sections.append(
        (
            "Limitations",
            [
                "- ARPU is revenue, not profit.",
                "- Future revenue is not discounted.",
                "- The offer cost is paid for every contacted customer.",
                "- The offer is assumed to help only would-be churners; this is not an uplift estimate.",
                "- Save rate and months kept are not estimated from the data.",
                "- The expected values inherit the calibration quality of the churn probabilities.",
            ],
        )
    )

    blocks = []
    for heading, paragraphs in sections:
        blocks.append(f"## {heading}")
        if all(paragraph.startswith("- ") for paragraph in paragraphs):
            blocks.append("\n".join(paragraphs))
        else:
            blocks.extend(paragraphs)
    return header + "\n" + "\n\n".join(blocks) + "\n"
