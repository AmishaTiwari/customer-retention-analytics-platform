# ADR 0009: Hyperparameter Tuning Strategy

**Status:** Accepted

---

## Context

The locked ML System Design's Hyperparameter Tuning Strategy requires reproducible search strategies, evaluation against baseline models, avoidance of unnecessary complexity, and states that "specific search strategies, tuning budgets, and optimization objectives will be finalized through Architecture Decision Records (ADRs) before experimentation begins." It further locks that tuning optimizes "threshold-independent evaluation metrics" — without naming which one — while final model selection remains driven by the Business-aligned Machine Learning Metrics per ADR-0008, a separation this ADR does not revisit. `config.yaml`'s `tuning` block (`budget_per_model`, `objective_metric`) and `data` block (`cv_folds`, `cv_repeats`) have stood as `null` since Commit 1, marked "TODO (ADR before Commit 8)." This ADR resolves those four values so Commit 8 can begin.

---

## Decision

1. **Search strategy: `RandomizedSearchCV`** over `GridSearchCV`, for all three tuned candidates (Logistic Regression, Random Forest, XGBoost). A single shared search mechanism, with budget as the one tunable knob, keeps the comparison consistent across model families rather than requiring a separately hand-designed grid per family.
2. **Objective metric: PR-AUC.** The design names "threshold-independent evaluation metrics" without specifying which one; both ROC-AUC and PR-AUC qualify, and this ADR selects PR-AUC as the new decision. PR-AUC is chosen over ROC-AUC because it is sensitive to performance on the minority (churn) class, which ROC-AUC is not, and because it is one of the design's own named Primary Metrics — making it a closer proxy for what Commit 9's selection will ultimately weigh, even though ADR-0008's selection rule itself is not being revisited here.
3. **Cross-validation: 5 folds, 1 repeat**, stratified. This resolves `config.yaml`'s `data.cv_folds`/`cv_repeats`, left open since Commit 1. Repeats beyond 1 are not used here; if fold-to-fold variance later proves large enough to bear on ADR-0008's undefined "materially indistinguishable," that is Commit 9's finding to raise, not something to pre-empt in this ADR.
4. **Budget per model family: `n_iter=30`**, identical across Logistic Regression, Random Forest, and XGBoost. A single shared number keeps the comparison fair — no candidate family gets a larger search purely by design choice.
5. All searches use `random_state` from config for reproducibility, consistent with every prior commit's convention.

---

## Alternatives considered

- **`GridSearchCV`**: rejected as the primary method because it requires a hand-designed grid per model family, reintroducing the asymmetric per-model tuning effort a single shared budget is meant to avoid.
- **ROC-AUC as the objective metric**: rejected in favor of PR-AUC — see Decision, item 2.
- **Bayesian optimization (e.g. `Optuna`)**: rejected as introducing a tuning-process dependency and complexity beyond what the locked scope calls for; `RandomizedSearchCV` is already part of the approved stack via scikit-learn.
- **10-fold CV**: rejected in favor of 5-fold as the simpler, more common default, with no stated project reason to require the additional cost of 10.

---

## Consequences

- `config.yaml`'s four `null` tuning-related values (`tuning.budget_per_model`, `tuning.objective_metric`, `data.cv_folds`, `data.cv_repeats`) are resolved by this ADR and populated in Commit 8.
- `n_iter=30` and `cv_folds=5` are fixed as project conventions, not derived from an analysis of each model's actual parameter space; if a specific model's tuning results suggest this budget is inadequate, that is a Commit 8 finding to raise, not a defect in this ADR.
- Tuned models are compared against their own Commit 7 untuned baselines and against the business heuristic, using this ADR's PR-AUC objective — not the final Primary Metrics set, which remains ADR-0008's exclusive basis for the eventual Commit 9 selection.

---

**Decision Date:** 17 September 2026
