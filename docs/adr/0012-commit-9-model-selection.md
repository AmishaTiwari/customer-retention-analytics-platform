# ADR 0012: Commit 9 Model Selection — Logistic Regression

**Status:** Accepted

---

## Context

ADR-0011 fixed Commit 9's candidate set: the business heuristic, tuned Logistic
Regression, tuned Random Forest, and tuned XGBoost. ADR-0008 fixed the rule for
selecting among them: read the Primary Metrics (Precision@K, lift/ranking
analysis, PR-AUC, calibration assessment, business-oriented threshold analysis)
together; if they agree on one candidate, select it; if they disagree, fall
through to the named tie-breakers (calibration, prediction stability, model
complexity, interpretability, training efficiency), with interpretability
named as carrying explicit additional weight.

Commit 9 Steps 1–2 (`evaluation/compare.py`) produced the following Primary
Metrics for all four candidates, across the locked K sensitivity range
(ADR-0010: K = {5%, 10%, 15%, 20%, 25%}, primary K = 10%):

| Candidate | K=5% | K=10% | K=15% | K=20% | K=25% | PR-AUC | Brier |
|---|---:|---:|---:|---:|---:|---:|---:|
| business_heuristic | 0.4000 | 0.4043 | 0.4313 | 0.4291 | 0.4432 | 0.4330 | 0.3153 |
| logistic_regression | 0.9286 | 0.9078 | 0.8483 | 0.8014 | 0.7500 | 0.7895 | 0.1049 |
| random_forest | 0.9286 | 0.8511 | 0.8152 | 0.7837 | 0.7330 | 0.7696 | 0.1101 |
| xgboost | 0.9286 | 0.8936 | 0.8626 | 0.8121 | 0.7500 | 0.8047 | 0.1003 |

(Precision@K shown; Lift@K moves in lockstep, same relative ordering.)

**Random Forest is eliminated from contention.** It is never the best
candidate on any Primary Metric at any K, and both Logistic Regression and
XGBoost dominate it throughout. The remaining comparison is between Logistic
Regression and XGBoost only.

**The Primary Metrics disagree between LR and XGBoost:**
- Logistic Regression wins Precision@K/Lift@K at the locked primary K = 10%
  (0.9078 vs. 0.8936), and at K = 5% (tied) and K = 25% (tied).
- XGBoost wins Precision@K/Lift@K at K = 15% and K = 20%.
- XGBoost wins PR-AUC (0.8047 vs. 0.7895) and calibration/Brier score (0.1003
  vs. 0.1049).

No ranking is established among Primary Metrics by ADR-0008, so this
disagreement — different metrics favoring different candidates — is exactly
the condition ADR-0008 defines as triggering Step 3 (the tie-breakers).

---

## Decision

**Select tuned Logistic Regression as the final Commit 9 model.**

### Tie-breaker evidence

All figures below come from the fitted `RandomizedSearchCV` objects
(`cv_results_`, `best_params_`) produced in Commit 8, and from
`evaluation/compare.py`'s `TieBreakerEvidence` extraction — no new tuning or
experiments were run to produce them.

| Tie-breaker | Logistic Regression | XGBoost | Favors |
|---|---|---|---|
| Calibration (Brier, lower is better) | 0.1049 | 0.1003 | XGBoost (gap: 0.0046) |
| Prediction stability (CV std, lower is better) | 0.0232 | 0.0177 | XGBoost (gap: 0.0055) |
| Model complexity | 1 linear model, 1 active tuned hyperparameter (`C = 4.57`) | 215 boosted trees, `max_depth = 8` | Logistic Regression |
| Interpretability | Signed, per-feature coefficients; already used to diagnose leakage in Commit 7 | Unsigned, coarser `feature_importances_`; never exercised in this project | Logistic Regression |
| Training efficiency (best-config fit time) | 0.3141s | 0.3129s | Essentially a wash (0.0012s difference) |

### Interpretability caveat (applies to all three model families equally)

A correlation-matrix diagnostic was run on the actual post-preprocessing
feature matrix used by all three models (`X_train` after `ColumnTransformer`
encoding, 55 columns via `preprocessor.get_feature_names_out()`) — not the 43
pre-encoding columns referenced in `commit_06.ipynb`; the difference is
expected, since one-hot expansion of 5 categorical columns turns each into
multiple dummy columns, and the post-encoding representation is the correct
one to check, since it is what every candidate model actually trains on.

The check found three exact ±1.0 redundant-encoding pairs and several
non-trivial correlated pairs (0.6–0.97) among the 55 features. This is a
caveat on interpreting *any* of the three models' native feature-level output,
not an LR-specific concern: correlated predictors destabilize LR's coefficient
signs/magnitudes, and equally dilute RF/XGBoost's impurity- or gain-based
`feature_importances_` among correlated features. It does not, however, remove
LR's *relative* interpretability advantage over the ensembles (signed,
per-feature coefficients vs. unsigned aggregate importances) — it is a
caveat to attach to whichever model is selected, not a differentiator between
them. Redundant-encoding columns should be flagged before coefficients are
presented as business explanation.

### Applying ADR-0008's weighting

ADR-0008 establishes only that interpretability "carries real weight" among
the tie-breakers; it explicitly leaves the relative weighting of calibration,
prediction stability, model complexity, and training efficiency undefined,
to be resolved with real observed values when needed. The following weights
are this project's own reasoned judgment for Commit 9's specific business
context — they are **not** derived mechanically from ADR-0008, which supplies
only the "interpretability matters more" principle, not a percentage:

| Tie-breaker | Weight | Rationale |
|---|---:|---|
| Interpretability | 30% | Named priority in ADR-0008; `01_business_design.md` requires outputs interpretable to non-technical stakeholders. |
| Calibration | 25% | This is a churn/retention system; the predicted probability itself carries business meaning, not just the ranking. |
| Prediction stability | 20% | Consistent performance across training splits is valuable for a model retrained periodically in production. |
| Model complexity | 15% | Simpler models are easier to maintain, audit, and debug — secondary to predictive/business behavior, but not negligible. |
| Training efficiency | 10% | A real engineering consideration, but fit times here are sub-second for both candidates, making this low-impact. |

**Reasoning, not a weighted score:** the two highest-weighted criteria
(interpretability 30%, calibration 25%) point in opposite directions, so the
decision is not a matter of mechanically multiplying weights by outcomes.
What breaks the tie is the *strength* of each signal, not just its assigned
weight:

- Logistic Regression's interpretability advantage is strong and structural —
  a demonstrated, already-used capability (signed per-feature coefficients)
  versus a capability XGBoost does not have at all (no directional
  explanation). This maps directly to a named business-design constraint, not
  a stylistic preference.
- Logistic Regression's complexity advantage is similarly strong and
  unambiguous — one linear model that can be fully audited by inspecting a
  coefficient table, versus 215 sequentially-built trees that cannot be
  manually reasoned about.
- XGBoost's calibration advantage (0.0046 Brier) and stability advantage
  (0.0055 std) are real but small in absolute terms, and no repeated-fold
  variance estimate exists to establish that either gap is not noise —
  ADR-0008 itself flags "materially indistinguishable" as an open, undefined
  threshold, and these gaps sit close enough to it to warrant caution before
  treating them as decisive.
- Training efficiency contributes no signal either way at the selected
  configurations.

On balance, the two criteria carrying the strongest, most structurally
grounded signals (interpretability and model complexity, together 45% of the
assigned weight) both favor Logistic Regression, while the criteria favoring
XGBoost (calibration and stability, together 45% of the assigned weight) are
real but comparatively thin. This is the basis for selecting Logistic
Regression.

**This is explicitly a reasoned, project-specific judgment, not a
mathematically forced or objectively unique outcome.** A different, equally
defensible weighting is possible — for example, an analyst who weighted
calibration more heavily (if downstream decisions depended on the precise
probability value rather than the ranking) could reasonably reach XGBoost
instead. Nothing in the evidence makes that reading illegitimate, only
different. The judgment that LR's advantages are "strong" while XGBoost's are
"weak" is itself a qualitative call made by the project team, stated here
plainly as judgment rather than presented as an objective conclusion the
numbers alone forced.

---

## Consequences

- Tuned Logistic Regression is the model carried forward into Commit 10
  (Inference Pipeline) as the persisted, production model.
- Random Forest and tuned/untuned XGBoost remain in the project's history and
  evaluation artifacts as documented comparison context but are not carried
  forward past Commit 9.
- The correlation-analysis caveat above should be carried into any future
  presentation of LR's coefficients as business explanation (e.g., in Commit
  11's business-facing outputs): redundant-encoding columns should be flagged
  or consolidated before coefficients are presented as standalone drivers of
  churn risk.
- If Commit 10 or later work surfaces evidence that changes this picture
  (e.g., production data showing the calibration gap is larger or more
  consequential than observed here), that should be handled as an amendment
  to or supersession of this ADR, not a silent change.

---

**Decision Date:** 29 September 2026
