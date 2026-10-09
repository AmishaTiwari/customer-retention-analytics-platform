"""Configuration loading and validation.

Single loader for config/config.yaml, used by every pipeline stage.
Validates presence and types of required keys at startup and fails loudly
on missing or malformed configuration.
"""

from __future__ import annotations

from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = REPO_ROOT / "config" / "config.yaml"

REQUIRED_SECTIONS = [
    "paths",
    "reproducibility",
    "data",
    "business",
    "tuning",
    "inference",
]

REQUIRED_PATH_KEYS = [
    "raw_data",
    "interim_db",
    "processed_dataset",
    "models_dir",
    "evaluation_dir",
    "runs_dir",
    "outputs_dir",
]

_OPTIONAL_PATH_KEYS = {"processed_dataset"}


class ConfigValidationError(Exception):
    """Raised when config/config.yaml is missing required sections or values."""


def _is_fraction(value: object) -> bool:
    """True for an int or float (not bool) with 0 < value <= 1."""
    return isinstance(value, (int, float)) and not isinstance(value, bool) and 0 < value <= 1


def _validate_business(business: object) -> list[str]:
    """Return problems with business, business.k and business.k_sensitivity.

    Called only when the business key is present; a missing section is
    already reported as a missing top-level section.
    """
    if not isinstance(business, dict):
        return [f"business must be a mapping, got {business!r}"]

    errors: list[str] = []

    if "k" not in business:
        errors.append("Missing required key in 'business': 'k'")
    elif not _is_fraction(business["k"]):
        errors.append(f"business.k must be a number with 0 < k <= 1, got {business['k']!r}")

    if "k_sensitivity" not in business:
        errors.append("Missing required key in 'business': 'k_sensitivity'")
    else:
        sweep = business["k_sensitivity"]
        if not isinstance(sweep, list) or not sweep:
            errors.append(f"business.k_sensitivity must be a non-empty list, got {sweep!r}")
        else:
            for index, value in enumerate(sweep):
                if not _is_fraction(value):
                    errors.append(
                        f"business.k_sensitivity[{index}] must be a number with "
                        f"0 < value <= 1, got {value!r}"
                    )

    return errors


def load_config(config_path: Path | None = None) -> dict:
    """Load, validate, and resolve config/config.yaml.

    Raises ConfigValidationError listing every problem found if the
    configuration is missing required sections/keys or has invalid values.
    """
    config_path = config_path or DEFAULT_CONFIG_PATH

    with config_path.open("r", encoding="utf-8") as fh:
        raw_config = yaml.safe_load(fh)

    if not isinstance(raw_config, dict):
        raise ConfigValidationError(f"{config_path} is empty or not a valid YAML mapping")

    errors: list[str] = []

    for section in REQUIRED_SECTIONS:
        if section not in raw_config:
            errors.append(f"Missing required top-level section: '{section}'")

    paths = raw_config.get("paths", {})
    for key in REQUIRED_PATH_KEYS:
        if key not in paths:
            errors.append(f"Missing required key in 'paths': '{key}'")

    for key in REQUIRED_PATH_KEYS:
        if key not in paths:
            continue
        value = paths[key]
        if key in _OPTIONAL_PATH_KEYS:
            if value is not None and not (isinstance(value, str) and value):
                errors.append(
                    f"paths.{key} must be null or a non-empty string, got {value!r}"
                )
        else:
            if not (isinstance(value, str) and value):
                errors.append(f"paths.{key} must be a non-empty string, got {value!r}")

    if "business" in raw_config:
        errors.extend(_validate_business(raw_config["business"]))

    if errors:
        raise ConfigValidationError(
            f"Configuration validation failed with {len(errors)} issue(s):\n"
            + "\n".join(f"- {error}" for error in errors)
        )

    resolved_paths = {
        key: (None if paths[key] is None else REPO_ROOT / paths[key])
        for key in REQUIRED_PATH_KEYS
    }

    return {
        "paths": resolved_paths,
        "reproducibility": raw_config["reproducibility"],
        "data": raw_config["data"],
        "business": raw_config["business"],
        "tuning": raw_config["tuning"],
        "inference": raw_config["inference"],
    }
