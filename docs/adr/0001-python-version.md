# ADR 0001: Standardize on Python 3.12

**Status:** Accepted

---

## Context

The planning documents define the overall project architecture but intentionally do not specify a Python version.

Before implementing the repository, we needed to decide which Python version would be used so that the project remains reproducible and all dependencies are compatible.

---

## Decision

This repository will use **Python 3.12**.

The `pyproject.toml` file will specify:

```toml
requires-python = ">=3.12,<3.13"
```

---

## Rationale

This decision was made because:

- The project uses **XGBoost** as one of its planned candidate models, and the latest versions require Python 3.12 or newer.
- Python 3.12 has broad and mature support across the Machine Learning ecosystem, including libraries such as pandas, NumPy, scikit-learn, and XGBoost.
- Using Python 3.12 provides a stable environment while avoiding potential compatibility issues that may arise with newer Python releases.
- Restricting the project to Python 3.12 helps ensure that anyone cloning the repository can reproduce the same environment.

---

## Consequences

- The repository will target **Python 3.12** throughout development.
- All project dependencies will be selected and tested against Python 3.12.
- If a future change requires a different Python version, a new ADR will be created instead of silently changing this decision.

---

**Decision Date:** 16 July 2026