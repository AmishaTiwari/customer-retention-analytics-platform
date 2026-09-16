# ADR 0008: Model Selection Rule

**Status:** Accepted

---

## Context

The locked ML System Design's Model Comparison Strategy names eight comparison considerations (business-aligned metrics, ML success metrics, calibration, prediction stability, model complexity, interpretability, training efficiency, and performance relative to the business heuristic baseline) as a flat list, without specifying how they combine into a single decision. Separately, the design locks two structural facts this ADR builds on rather than restates: (1) business-aligned metrics drive final model selection, while ML success/threshold-independent metrics guide Commit 8's tuning process rather than final selection; (2) Precision@K, lift/ranking analysis, PR-AUC, calibration assessment, and business-oriented threshold analysis are named "Primary Metrics," with ROC-AUC, Precision, Recall, F1, and the confusion matrix as "Supporting Metrics" — a locked category split, though the design does not rank items within either category against each other. This ADR fixes the selection rule the design deferred, before Commit 8 (Hyperparameter Tuning) begins, so the rule is locked before any tuned-performance results exist that could bias it.

---

## Decision

Final model selection follows this rule:

1. **Primary basis: the Primary Metrics as a set** (Precision@K, lift/ranking analysis, PR-AUC, calibration assessment, business-oriented threshold analysis), interpreted together rather than any single one taken as dominant over the others — the design does not establish an internal ranking among them, and this ADR does not invent one. Where they agree on which candidate is best, that candidate is selected.
2. **Performance relative to the business heuristic baseline** is reported alongside every candidate's Primary Metrics as required context, per the design's listing of this comparison as a named consideration. A candidate performing worse than the heuristic on Precision@K is a significant negative signal, but this ADR does not establish it as an automatic disqualification; the decision remains a holistic reading of the Primary Metrics together with this comparison, not a hard gate.
3. **When the Primary Metrics do not agree** on a single best candidate (i.e., different metrics favor different candidates, or results are materially indistinguishable — see Consequences for this term's undefined status), the remaining named considerations — calibration quality, prediction stability, model complexity, interpretability, and training efficiency — are each examined for the candidates still in contention. The Candidate Model Strategy names interpretability, alongside business value and engineering robustness, as a stated model-selection priority — this ADR treats interpretability as carrying real weight among these tie-breakers on that basis. No locked design text ranks calibration, prediction stability, model complexity, or training efficiency against each other or against interpretability; how these four combine when they conflict with one another is not resolved by this ADR and is left as an explicit decision to make at the point it's actually needed, informed by the real values observed at that time rather than a rule fixed now without that information.
4. **Supporting Metrics and ML success/threshold-independent metrics** (ROC-AUC, Precision, Recall, F1, confusion matrix) are diagnostic — used during Commit 8's tuning process and for interpretation context — but do not independently drive final selection, per the design's explicit statement that tuning metrics are separate from the metrics driving final selection.

---

## Consequences

Three dependencies are explicitly left open by this ADR, not resolved prematurely:

- **K's value is not yet locked** (a separate future ADR, per the design's own ADR queue). This rule is written in terms of "the locked K" without assuming a number.
- **"Materially indistinguishable" (step 3) is left undefined here.** Setting a specific numeric threshold now, with no repeated-fold variance data yet available, would be inventing a number the design doesn't support. This is named as an open gap; if reached in practice during Commit 9, it should be resolved with real evaluation data in hand, or via an amendment to this ADR if a general rule is needed sooner.
- **The relative weighting among calibration quality, prediction stability, model complexity, and training efficiency is not resolved by this ADR.** Only interpretability has explicit textual support as a named priority among these five; the other four are named as considerations by the design but not ranked. If a tie-breaking decision among these four specifically is ever needed, it should be made with real observed values in hand, not speculatively now.

This rule is fixed before any tuned model results exist. If tuning outcomes create pressure to revisit this weighting, that requires a new ADR superseding this one, not a silent change during Commit 9's evaluation.

---

**Decision Date:** 16 September 2026
