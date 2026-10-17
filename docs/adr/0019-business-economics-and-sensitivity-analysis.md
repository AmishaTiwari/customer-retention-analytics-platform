# ADR 0019: Business Economics and Sensitivity Analysis

**Status:** Accepted

---

## Context

Commit 11 turns the ranked list into decision support: what is the outreach group worth, and how sure can we be? Platform Output Contract v1 listed business economics (expected value, ARPU, offer cost, ROI) as non-deliverables and said that Commit 11 extends the same customer list under a reviewed v2 amendment.

The data cannot answer the economic questions on its own. The dataset has no information about what a retention offer costs, how many contacted customers an offer would keep, or for how long. The field selected for the ARPU proxy is `monthly_charge`. The only monetary statistic this economics analysis uses as an observed input is the ARPU proxy (the mean `monthly_charge` of the flagged group). The expected value and the break-even cannot be observed: they are simulated under stated assumptions. The design has to keep three things apart: what was observed, what was assumed, and what was simulated.

This ADR fixes the expected-value model, where its inputs come from, the assumptions and the sensitivity ranges, how the sensitivity is shown, and why none of it is written into the customer list or the run summary. The exact layout of the economics document is a Commit 11 documentation matter. The output file names and the publication rules are in Platform Output Contract v2.

---

## Decision

### 1. Expected value with every assumption explicit

For each contacted customer:

```
EV_i = churn_probability_i × save_rate × ARPU_proxy × months_kept − offer_cost
```

The group expected value is the sum of `EV_i` over the flagged top-K group (`in_top_k = True`; 141 customers in the reference run). The churn probability is the calibrated `churn_probability` of the scoring run and differs per customer. Save rate, months kept, ARPU proxy and offer cost are shared assumptions. No customer label and no held-out precision enters the calculation.

The same analysis reports the **group break-even save rate**, the save rate at which the group expected value is zero: `(N × offer_cost) / (ARPU_proxy × months_kept × Σ churn_probability_i)`. It is exact because the formula is linear in the save rate. A break-even above 100% means the offer cannot pay off at any save rate, and is shown as such (section 4).

### 2. The ARPU proxy comes from the flagged group

The ARPU proxy is the mean `monthly_charge` of the flagged top-K customers of the scoring run (model inputs only, no labels). It is derived on every run, and the economics document reports the value and the number of customers used. `business.economics.arpu_proxy` stays in the configuration with `null` as the default (derive from the flagged group). A positive number is allowed as an explicit what-if override. The document states the source, derived or user-supplied, and never presents an override as an observed value.

Evidence for the reference population (a one-time, read-only, label-free inspection): the training split (5,629 customers) has a mean `monthly_charge` of 64.9165, the held-out reference population (1,408) has 64.1270, and the flagged group (141) has 78.1330, which is 20.36% above the training mean. A fixed training mean would understate the flagged group's average monthly charge by about 16.9%, so the group's own mean is the better basis for the group's economics. This is a judgment, not a measurement. `monthly_charge` is a revenue proxy, not profit and not necessarily realized revenue.

### 3. Assumptions are explicit project assumptions

The base values and sensitivity lists are chosen assumptions. They are not derived from the dataset and are not verified ConnectTel figures:

| Input | Base | Sensitivity list |
|---|---|---|
| `save_rate` | 0.20 | 0.05, 0.10, 0.20, 0.30, 0.40 |
| `months_kept` | 3 | 1, 3, 6, 12 |
| `offer_cost` | 20 currency units | 10, 20, 40, 60 |
| ARPU proxy | derived (section 2) | 0.8×, 1.0×, 1.2× the applicable base ARPU |

The save-rate list is chosen to cover the group's break-even region. Three months is an illustrative horizon informed by the Business Design's 90-day example, not evidence that saved customers stay three months, and twelve months is a scenario, not a claim of a one-year contract. The economics document says "currency units" because nothing establishes a currency. It states next to the assumptions that save rate and months kept are not estimated from the data, and lists the limitations: ARPU is revenue, not profit; no discounting; the offer cost is paid for every contacted customer; the offer is assumed to help only would-be churners, and this is not an uplift estimate; the group expected value inherits the calibration quality of the probabilities.

The configuration gains `business.economics.save_rate` and `business.economics.months_kept` next to the existing `arpu_proxy` and `offer_cost`; these hold the base values. The **sensitivity lists** (and the 0.8×, 1.0×, 1.2× ARPU multiples) are fixed project constants of the economics code, not configuration. Making them configurable would be a separate decision. The validation bounds of the configuration keys are fixed at implementation.

### 4. How the sensitivity is shown

The economics document contains, as plain Markdown tables with no charts and no new dependency:

1. a base-case summary (customers contacted, total offer cost, expected would-be churners, expected customers saved, gross revenue kept, net expected value, expected value per contacted customer, break-even save rate);
2. one-at-a-time sweeps (16 rows: 5 save rates, 4 months kept, 4 offer costs, 3 ARPU multiples);
3. a save rate × months kept table (5 × 4);
4. a break-even table, months kept × offer cost (4 × 4);
5. a K sweep (5 rows).

The full scenario grid is not generated. In the K sweep, with `arpu_proxy` null the ARPU is derived separately for each K from that K's flagged group; with a numeric override the same override is used for every K. K = 10% remains the primary outreach policy, and the sweep is sensitivity analysis, not a recommendation to change K. A break-even above 100% is shown as "Above 100%: cannot pay off at any save rate", not as `N/A`. **No ROI is calculated**, because the figures are simulated results and not measured campaign outcomes. Every table states its fixed inputs and is labelled "simulated result under stated assumptions".

### 5. No economics in the customer list or the run summary

`prioritized_outreach_list.csv` gets no per-customer economics column, and `run_summary.json` carries no economics field. The economics analysis is a separate generated document. Reasons:

- The list stays focused on risk prioritization and the agreed tier and driver fields.
- Under shared assumptions the per-customer expected value is a linear function of the churn probability (at the base values, 46.88·p − 20), so it never changes the ranking, and its monetary meaning depends on assumptions.
- A column for all customers would need an ARPU for customers outside the flagged group and would couple the list to economics configuration and overrides.
- A column for the flagged customers only would leave empty cells, which the contract forbids.
- The document can show assumptions, scenarios and break-even next to the simulated figures, which a column cannot.

---

## Alternatives considered

- **`EV = p × A − c`, with the value of a saved customer as one number.** Rejected: it hides a 100% save rate and a one-month horizon, and allows neither a sensitivity analysis nor a break-even.
- **Break-even only, with no expected-value figure.** Rejected as the whole answer: the break-even is kept as one line of the analysis, but the decision-maker also needs the size of the simulated gain or loss.
- **A fixed ARPU from the training mean.** Rejected after the evidence in section 2: it understates the flagged group's monthly charge.
- **ARPU from the churners' own mean.** Rejected: it uses labels, which scoring does not have and the economics must not use.
- **A per-customer expected-value column for all customers, or for the flagged customers only.** Rejected for the reasons in section 5.
- **Show a return on investment.** Rejected: with simulated inputs it would look like a measured result.

---

## Consequences

- The derived ARPU proxy is an **observed statistic**, calculated from the `monthly_charge` of the flagged group. The expected value, the break-even and every other result built on the assumptions are **simulated results under stated assumptions**. The documents keep the two apart, and no document may present an assumption or an override as an observation.
- The economics depend on the calibration quality of the churn probabilities. ADR-0014 records the known limits of the calibration.
- The derived ARPU changes with the scoring run, because it follows the flagged group. A run that uses an override says so.
- The configuration grows by two keys, `save_rate` and `months_kept`. Their validation, and that of the existing `offer_cost` and the `arpu_proxy` override rules (positive values, sensible ranges), is part of the implementation and is covered by tests. The sensitivity lists add no configuration.
- The economics calculation needs `monthly_charge` for the scored customers. The saved model schema is checked at implementation to confirm the field is available in the scoring population.
- A later decision to act on a probability or expected-value threshold, or to add run-specific retention of earlier documents, is outside Commit 11.

---

**Decision Date:** 17 October 2026 (decisions approved 15 and 16 October 2026; recorded here before implementation)
