# ADR 0011: Commit 9 Candidate-Set Definition

**Status:** Accepted

---

## Context

Commit 7 produced untuned baseline models (Logistic Regression, Random Forest, XGBoost) plus the business heuristic. Commit 8 produced tuned versions of the three ML models (ADR-0009). ADR-0008 locks *how* final model selection is performed once a candidate set exists, but neither ADR-0008, ADR-0009, nor the ML System Design ever states *which* artifact — tuned or untuned — represents each model family when that selection rule is applied in Commit 9. This gap was surfaced before Commit 9 Step 1 (evaluating all candidates) began, specifically because starting implementation would have silently frozen an unexamined default (using the tuned models) without it ever having been a documented decision.

Commit 7's own notes describe its untuned comparison as "a snapshot, not the final model decision," and ADR-0009 records a tuned-vs-untuned check against the tuning objective (PR-AUC) as part of Commit 8's diagnostics — but ADR-0009 explicitly scopes that check to tuning-stage evidence, separate from the Primary Metrics that govern Commit 9's actual selection. Neither document says what should happen to the candidate set as a result of that check.

To close this gap with real evidence rather than assumption, a one-time scratch check (not committed as project code) measured Test PR-AUC for the untuned and tuned version of each ML family, using the same train/test split, preprocessing fit, and seed for both arms of each family:

| Model | Untuned Test PR-AUC | Tuned Test PR-AUC | Change |
|---|---:|---:|---:|
| Logistic Regression | 0.7888 | 0.7895 | +0.0007 |
| Random Forest | 0.7560 | 0.7696 | +0.0136 |
| XGBoost | 0.7645 | 0.8047 | +0.0402 |

No family regressed under tuning. The size of the improvement varies substantially by family (negligible for LR, modest for RF, substantial for XGBoost), which is itself relevant: it rules out a uniform "tuning obviously helps" assumption while also ruling out "tuning made things worse," for all three.

---

## Decision

**Commit 9's candidate set is: the business heuristic, tuned Logistic Regression, tuned Random Forest, and tuned XGBoost.** The untuned Commit 7 models are retained as documented baseline/reference context but do not enter ADR-0008's final-selection comparison.

This is a candidate-*definition* decision, not a model-selection decision: it fixes what "the LR/RF/XGBoost candidate" refers to before ADR-0008's already-locked Primary Metrics rule is applied to it. It does not evaluate, rank, or select among the four candidates — that remains ADR-0008's exclusive basis, exercised in Commit 9 Step 1 onward.

**Rationale:**
1. No family showed a tuning regression on the tuning objective (PR-AUC), so there is no case here where the untuned version would be the stronger representative.
2. The Repository Architecture's downstream design (`artifacts/models/`, one persisted bundle per family) already assumes a single representative artifact per model family survives past Commit 9 — this decision is consistent with, though not derived from, that shape.
3. Treating tuned and untuned as separate parallel candidates was considered and rejected: ADR-0008's tie-breakers (interpretability, complexity, calibration, stability, training efficiency) are written for comparing distinct model families and have no defined behavior for intra-family tuned-vs-untuned comparisons. Introducing that would mean either stretching ADR-0008 into an untested context or inventing new tie-break logic — a larger scope expansion than this candidate-definition question calls for.

---

## Consequences

- Commit 9 Step 1 (`evaluate_all_candidates` in `evaluation/compare.py`) evaluates exactly four candidates: heuristic, tuned LR, tuned RF, tuned XGBoost.
- Untuned Commit 7 results remain in the project's history and notes as baseline context but are not part of the Commit 9 comparison table or ADR-0008's selection.
- The PR-AUC tuned-vs-untuned check above is diagnostic evidence for *this* decision only; it is not used to select the final model. Final selection uses only the Primary Metrics named in ADR-0008 (Precision@K, lift/ranking, PR-AUC as one input among them, calibration, business-oriented threshold analysis), applied to the four candidates named here.
- If a future model family or a retuning ever shows a genuine regression under tuning, this ADR's rule should be revisited explicitly (amendment or superseding ADR), not silently overridden.

---

**Decision Date:** 28 September 2026
