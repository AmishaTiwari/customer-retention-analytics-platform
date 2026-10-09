"""Validation tests for config.load_config's business.k and
business.k_sensitivity rules.

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
