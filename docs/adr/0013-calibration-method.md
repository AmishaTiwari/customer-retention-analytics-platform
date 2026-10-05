# ADR 0013: Calibration Method

**Status:** Accepted

---

## Context

The locked ML System Design requires probability calibration to be "fitted using data independent of final evaluation," and lists calibration assessment as one of the Primary Metrics that feed ADR-0008's model-selection rule, alongside Precision@K, Lift@K, and PR-AUC. What it doesn't specify is *which* calibration method to use — that was left open, the same way ADR-0009 later had to fix the tuning objective and search strategy the design had also left open.

This ADR is being written slightly out of order: `calibrate_candidate()` in `evaluation/compare.py` already implements `CalibratedClassifierCV(method="sigmoid")`, added while correcting Commit 9's model-selection protocol (see ADR-0014). This document simply makes that choice official and puts the reasoning on record, so it exists in the project's history rather than only in code.

scikit-learn's `CalibratedClassifierCV` supports two methods: Platt scaling (`sigmoid`) and isotonic regression (`isotonic`).

---

## Decision

**Use sigmoid (Platt scaling)** for all calibration in this project — both the per-fold calibration used during candidate comparison, and the final refit on the full training set before test-set evaluation.

The deciding factor is sample size. Each CV fold's training portion has roughly a few hundred churn examples to calibrate against (5 folds over ~5,600 training rows at a ~26.5% churn rate). Isotonic regression is a flexible, non-parametric fit, but that flexibility becomes a liability at small sample sizes — it tends to overfit, especially on the minority class, which is a well-known limitation of isotonic calibration. Sigmoid's two-parameter logistic curve needs far less data to fit reliably. The trade-off is that sigmoid assumes a roughly S-shaped miscalibration pattern, which is a reasonable assumption for all three candidates here (Logistic Regression, Random Forest, XGBoost) — none of them are expected to produce a wildly irregular calibration curve.

---

## Alternatives considered

- **Isotonic regression** — more flexible in theory, but that flexibility works against it at this sample size; rejected.
- **No calibration** — not a real option, since the locked Validation Strategy explicitly requires it.
- **A different method per candidate** (e.g. sigmoid for LR, isotonic for the tree ensembles) — rejected for the same reason ADR-0009 picked one shared tuning approach for all three candidates: a single method keeps the comparison fair and avoids adding yet another per-candidate knob to a process that's already being tightened up for rigor.

---

## Consequences

- `CALIBRATION_METHOD` in `evaluation/compare.py` is fixed to `"sigmoid"`, used both during comparison and for the final model's calibration.
- If future evidence shows a candidate's calibration curve doesn't fit sigmoid's assumption well, that's a reason to open a new ADR revisiting this choice — not to change it quietly.
- This ADR only settles the calibration *method*. It's separate from the Commit 9 test-set-usage fix recorded in ADR-0014, which settles *where in the pipeline* calibration happens.

---

**Decision Date:** 30 September 2026