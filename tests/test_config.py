"""Validation tests for config.load_config's business.k,
business.k_sensitivity, business.risk_tiers and business.economics rules.

Each failing case writes a modified copy of the real config/config.yaml
under tmp_path and loads it through load_config's config_path argument,
so the real config file is never touched. No database or model fitting is
involved.
"""

from __future__ import annotations

import yaml
import pytest

from retention_platform.config import (
    DEFAULT_CONFIG_PATH,
    ConfigValidationError,
    load_config,
)


def _load_with(tmp_path, mutate):
    """Load the real config, apply mutate to its raw dict, and run
    load_config on the modified copy written under tmp_path."""
    raw = yaml.safe_load(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
    mutate(raw)

    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    return load_config(config_path)


def test_real_config_loads_with_expected_k_values():
    config = load_config()

    assert config["business"]["k"] == 0.10
    assert config["business"]["k_sensitivity"] == [0.05, 0.10, 0.15, 0.20, 0.25]


def test_missing_k_raises(tmp_path):
    with pytest.raises(ConfigValidationError, match="Missing required key in 'business': 'k'"):
        _load_with(tmp_path, lambda raw: raw["business"].pop("k"))


def test_null_k_raises(tmp_path):
    def mutate(raw):
        raw["business"]["k"] = None

    with pytest.raises(ConfigValidationError, match=r"business\.k must be"):
        _load_with(tmp_path, mutate)


@pytest.mark.parametrize("bad_k", [0, -0.1, 1.5], ids=["zero", "negative", "above_one"])
def test_out_of_range_k_raises(tmp_path, bad_k):
    def mutate(raw):
        raw["business"]["k"] = bad_k

    with pytest.raises(ConfigValidationError, match=r"business\.k must be"):
        _load_with(tmp_path, mutate)


def test_bool_k_raises(tmp_path):
    def mutate(raw):
        raw["business"]["k"] = True

    with pytest.raises(ConfigValidationError, match=r"business\.k must be"):
        _load_with(tmp_path, mutate)


def test_missing_sweep_raises(tmp_path):
    with pytest.raises(
        ConfigValidationError, match="Missing required key in 'business': 'k_sensitivity'"
    ):
        _load_with(tmp_path, lambda raw: raw["business"].pop("k_sensitivity"))


def test_empty_sweep_raises(tmp_path):
    def mutate(raw):
        raw["business"]["k_sensitivity"] = []

    with pytest.raises(ConfigValidationError, match=r"business\.k_sensitivity must be"):
        _load_with(tmp_path, mutate)


def test_non_numeric_sweep_entry_raises(tmp_path):
    def mutate(raw):
        raw["business"]["k_sensitivity"] = [0.05, "10%", 0.15]

    with pytest.raises(ConfigValidationError, match=r"business\.k_sensitivity\[1\]"):
        _load_with(tmp_path, mutate)


def test_business_not_a_mapping_raises(tmp_path):
    def mutate(raw):
        raw["business"] = [0.10]

    with pytest.raises(ConfigValidationError, match="business must be a mapping"):
        _load_with(tmp_path, mutate)


def test_two_problems_reported_in_one_error(tmp_path):
    def mutate(raw):
        raw["business"]["k"] = 0
        raw["business"]["k_sensitivity"] = []

    with pytest.raises(ConfigValidationError) as exc_info:
        _load_with(tmp_path, mutate)

    message = str(exc_info.value)
    assert "2 issue(s)" in message
    assert "business.k must be" in message
    assert "business.k_sensitivity must be" in message


def test_real_config_loads_with_expected_business_settings():
    business = load_config()["business"]

    assert business["risk_tiers"] == {"medium_edge": 0.25}
    assert business["economics"]["arpu_proxy"] is None
    assert business["economics"]["offer_cost"] == 20
    assert business["economics"]["save_rate"] == 0.20
    assert business["economics"]["months_kept"] == 3


@pytest.mark.parametrize("bad_k", [0.25, 0.30], ids=["equal_to_edge", "above_edge"])
def test_k_not_below_medium_edge_raises(tmp_path, bad_k):
    def mutate(raw):
        raw["business"]["k"] = bad_k

    with pytest.raises(ConfigValidationError) as exc_info:
        _load_with(tmp_path, mutate)

    message = str(exc_info.value)
    assert f"business.k ({bad_k!r}) must be below" in message
    assert "business.risk_tiers.medium_edge (0.25)" in message


def test_valid_pair_loads(tmp_path):
    def mutate(raw):
        raw["business"]["k"] = 0.10
        raw["business"]["risk_tiers"]["medium_edge"] = 0.25

    config = _load_with(tmp_path, mutate)

    assert config["business"]["k"] == 0.10
    assert config["business"]["risk_tiers"]["medium_edge"] == 0.25


def test_sweep_is_not_constrained_by_medium_edge(tmp_path):
    def mutate(raw):
        raw["business"]["k_sensitivity"] = [0.05, 0.25, 0.40]

    config = _load_with(tmp_path, mutate)

    assert config["business"]["k_sensitivity"] == [0.05, 0.25, 0.40]


def test_missing_risk_tiers_raises(tmp_path):
    with pytest.raises(
        ConfigValidationError, match="Missing required key in 'business': 'risk_tiers'"
    ):
        _load_with(tmp_path, lambda raw: raw["business"].pop("risk_tiers"))


def test_risk_tiers_not_a_mapping_raises(tmp_path):
    def mutate(raw):
        raw["business"]["risk_tiers"] = [0.25]

    with pytest.raises(ConfigValidationError, match=r"business\.risk_tiers must be a mapping"):
        _load_with(tmp_path, mutate)


def test_missing_medium_edge_raises(tmp_path):
    with pytest.raises(
        ConfigValidationError, match="Missing required key in 'business.risk_tiers': 'medium_edge'"
    ):
        _load_with(tmp_path, lambda raw: raw["business"]["risk_tiers"].pop("medium_edge"))


@pytest.mark.parametrize(
    "bad_edge",
    [0, -0.1, 1.5, True, "0.25", None],
    ids=["zero", "negative", "above_one", "bool", "string", "null"],
)
def test_invalid_medium_edge_raises(tmp_path, bad_edge):
    def mutate(raw):
        raw["business"]["risk_tiers"]["medium_edge"] = bad_edge

    with pytest.raises(ConfigValidationError) as exc_info:
        _load_with(tmp_path, mutate)

    message = str(exc_info.value)
    assert "business.risk_tiers.medium_edge must be" in message
    assert "1 issue(s)" in message


def test_missing_economics_raises(tmp_path):
    with pytest.raises(
        ConfigValidationError, match="Missing required key in 'business': 'economics'"
    ):
        _load_with(tmp_path, lambda raw: raw["business"].pop("economics"))


def test_economics_not_a_mapping_raises(tmp_path):
    def mutate(raw):
        raw["business"]["economics"] = [20]

    with pytest.raises(ConfigValidationError, match=r"business\.economics must be a mapping"):
        _load_with(tmp_path, mutate)


@pytest.mark.parametrize("key", ["arpu_proxy", "offer_cost", "save_rate", "months_kept"])
def test_missing_economics_key_raises(tmp_path, key):
    with pytest.raises(
        ConfigValidationError, match=f"Missing required key in 'business.economics': '{key}'"
    ):
        _load_with(tmp_path, lambda raw: raw["business"]["economics"].pop(key))


NAN = float("nan")
INF = float("inf")


@pytest.mark.parametrize(
    ("key", "bad_value"),
    [
        *[("offer_cost", v) for v in (0, -5, True, "20", NAN, INF, None, [20])],
        *[("save_rate", v) for v in (0, -0.2, 1.5, True, "0.2", NAN, INF, None, [0.2])],
        *[("months_kept", v) for v in (0, -3, 2.5, 3.0, True, False, "3", NAN, INF, None, [3])],
        *[("arpu_proxy", v) for v in (0, -80, True, False, "80", NAN, INF, [80])],
    ],
)
def test_invalid_economics_value_raises(tmp_path, key, bad_value):
    def mutate(raw):
        raw["business"]["economics"][key] = bad_value

    with pytest.raises(ConfigValidationError) as exc_info:
        _load_with(tmp_path, mutate)

    message = str(exc_info.value)
    assert f"business.economics.{key} must be" in message
    assert "1 issue(s)" in message


@pytest.mark.parametrize("value", [None, 80.0, 80], ids=["null", "float", "int"])
def test_arpu_proxy_accepts_null_and_positive_number(tmp_path, value):
    def mutate(raw):
        raw["business"]["economics"]["arpu_proxy"] = value

    config = _load_with(tmp_path, mutate)

    assert config["business"]["economics"]["arpu_proxy"] == value


def test_invalid_pair_and_invalid_save_rate_reported_together(tmp_path):
    def mutate(raw):
        raw["business"]["k"] = 0.30
        raw["business"]["economics"]["save_rate"] = 1.5

    with pytest.raises(ConfigValidationError) as exc_info:
        _load_with(tmp_path, mutate)

    message = str(exc_info.value)
    assert "2 issue(s)" in message
    assert "business.k (0.3) must be below business.risk_tiers.medium_edge (0.25)" in message
    assert "business.economics.save_rate must be" in message


@pytest.mark.parametrize(
    ("section", "key"),
    [
        ("economics", "offer_cost"),
        ("economics", "arpu_proxy"),
        ("economics", "save_rate"),
        ("economics", "months_kept"),
        ("risk_tiers", "medium_edge"),
    ],
)
def test_huge_integer_is_rejected_with_validation_error(tmp_path, section, key):
    def mutate(raw):
        raw["business"][section][key] = 10**400

    with pytest.raises(ConfigValidationError, match=f"business.{section}.{key} must be"):
        _load_with(tmp_path, mutate)


def test_large_float_representable_integer_offer_cost_is_accepted(tmp_path):
    def mutate(raw):
        raw["business"]["economics"]["offer_cost"] = 10**15

    config = _load_with(tmp_path, mutate)

    assert config["business"]["economics"]["offer_cost"] == 10**15
