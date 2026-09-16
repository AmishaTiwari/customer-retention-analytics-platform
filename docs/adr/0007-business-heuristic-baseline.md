# ADR 0007: Business Heuristic Baseline Definition

**Status:** Accepted

---

## Context

The locked ML System Design's Candidate Model Strategy names a business heuristic baseline as a required non-ML reference point, evaluated alongside Logistic Regression, Random Forest, and XGBoost, but leaves the heuristic's actual definition open. A defensible baseline must be genuinely simple (a rule a retention team could state and apply without computation) and grounded in real data, not invented arbitrarily.

---

## Decision

The business heuristic flags every customer on a month-to-month contract (`is_month_to_month = True`) as at-risk. On the full dataset this yields 45.75% precision and 88.51% recall, flagging 51.2% of the customer base — a strong, clean, whole-population binary churn signal identified during the EDA/investigation (45.75% churn rate vs. 6.23% for fixed-term contracts).

---

## Alternatives considered

- **`is_month_to_month OR internet_without_security`** (compound rule): higher recall (94.85%) but lower precision (37.82%), flagging 66.4% of the base — too broad to usefully prioritize outreach, and as a compound rule it blurs the line between "simple heuristic" and "hand-built model," a role Logistic Regression already fills properly.
- **`no_addons_despite_internet`** alone: comparable rate gap (46.91% vs. 26.24%) but covers only 81 customers — too small a population for a baseline meant to score everyone.

---

## Consequences

The heuristic reads directly from the unencoded `feat_churn`-shaped input (only `is_month_to_month`), not the preprocessed `X` used by the ML models — a deliberate asymmetry, since the two only need to agree at the prediction level, not the input-representation level. Implemented in `src/retention_platform/models/candidates.py::predict_business_heuristic`, verified against real data in `tests/test_heuristic.py`.

---

**Decision Date:** 16 September 2026
