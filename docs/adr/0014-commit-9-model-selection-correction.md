# ADR 0014: Commit 9 Model Selection Correction — XGBoost

**Status:** Accepted

**Supersedes:** ADR-0012

---

## Context

ADR-0012 (29 September 2026) selected tuned Logistic Regression as the Commit 9 model. The evidence behind it, the Primary Metrics table and the Brier score used as a tie-breaker, was computed by scoring the candidates on the held-out test set. The locked Validation Strategy says candidates are compared with stratified cross-validation inside the training set, that calibration is fitted on data independent of final evaluation, and that the untouched holdout is used once, for final evaluation. Using the test set to compare candidates and pick a model breaks that rule: the test set can no longer serve as an unbiased check on the model that was chosen with it. This was found while reviewing Commit 9, before Commit 10 started, and corrected on the branch `fix/09-model-selection-correction`.

The correction changed the evaluation protocol only. Candidate comparison now uses out-of-fold predictions from the same stratified 5-fold CV that Commit 8 used for tuning (`evaluate_candidates_cv` in `evaluation/compare.py`), with each ML candidate wrapped in sigmoid calibration (`calibrate_candidate`, ADR-0013) and the business heuristic scored on the same training data. The test set is now reached by exactly one function, `evaluate_final_model_on_test`, once, for the selected model and the fixed heuristic, and a test that parses the source fails if any comparison or selection function refers to test-set identifiers. Left unchanged: ADR-0008's selection rule, ADR-0011's candidate set, ADR-0013's calibration method, the Commit 8 tuning results (reused, not re-run), preprocessing, and the tie-breaker evidence, which was checked and does not depend on the test set (it comes from the tuning searches' `cv_results_`).

---

## Decision

**Select tuned XGBoost, with sigmoid calibration (ADR-0013), as the final Commit 9 model.** ADR-0012's selection is superseded.

### Corrected comparison (training-set CV, calibrated, out-of-fold)

K = 10% is about 560 customers in this table.

| Candidate | P@5% | P@10% | P@15% | P@20% | P@25% | PR-AUC | Brier |
|---|---:|---:|---:|---:|---:|---:|---:|
| business_heuristic | 0.4875 | 0.4583 | 0.4751 | 0.4760 | 0.4748 | 0.4360 | 0.3064 |
| logistic_regression | 0.9146 | 0.8845 | 0.8235 | 0.7682 | 0.7178 | 0.7692 | 0.1102 |
| random_forest | 0.9288 | 0.8828 | 0.8282 | 0.7735 | 0.7228 | 0.7768 | 0.1090 |
| xgboost | 0.9502 | 0.9059 | 0.8483 | 0.7877 | 0.7264 | 0.8004 | 0.1046 |

Under ADR-0008 step 1, the Primary Metrics agree: XGBoost is best on Precision@K at all five K values, on PR-AUC and on Brier score. Because they agree, the tie-breakers are not reached, and the weighted judgment that ADR-0012 applied does not apply. The margins are small at some K (for example 0.7264 against 0.7228 for Random Forest at K = 25%), and ADR-0008 leaves "materially indistinguishable" undefined. That threshold was deliberately not invoked: the rule as written only asks whether the Primary Metrics agree, and they do.

Checks made before accepting the result. Tuning and comparison use identical CV folds (the tuning searches' `cv` object and `_build_cv` produce the same splits, verified for all three searches). Because hyperparameters were chosen on the folds the comparison reuses, the estimates are somewhat optimistic, and that optimism could favour the candidate with the largest search space, XGBoost. Two checks bound this: the top five tuning configurations are tightly clustered (spread in mean CV score: Logistic Regression 0.0037, Random Forest 0.0006, XGBoost 0.0015), and a pessimistic adjustment (the best configuration's score minus the search's median score) still leaves XGBoost ahead on PR-AUC.

### Final held-out test evaluation

After selection, the calibrated XGBoost was fitted on the full training set and scored once on the test set, together with the fixed heuristic. The result is recorded as found and was not used to select or reconsider the model.

| Metric | Training CV | Held-out test |
|---|---:|---:|
| XGBoost Precision@10% | 0.9059 | 0.8936 |
| XGBoost PR-AUC | 0.8004 | 0.8032 |
| XGBoost Brier | 0.1046 | 0.1017 |
| Heuristic Precision@10% | 0.4583 | 0.4043 |

At K = 10% the test set flags 141 customers, so one customer moves Precision@K by about 0.007; the 0.0123 gap between the CV and test Precision@10% is under two customers. The held-out result is consistent with the CV estimate. This is one consistency check, not proof that the selection optimism described above is zero. For transparency: the test-set numbers from the original protocol (the table in ADR-0012) had already been seen before the correction; the corrected selection was made on the training-CV table alone.

### Calibration verification (post-hoc)

After the corrected selection, the project checked whether calibration actually helps. This was not part of the original selection protocol. It is a post-hoc check on training data only; it does not touch the test set. The rules were fixed before the script was run and before any raw out-of-fold numbers had been seen, and were recorded in the project's implementation tracker (kept outside this repository) on 5 October 2026. One prior exposure is disclosed: the final test-set Brier of the calibrated model (0.1017) had been seen next to the earlier raw model's test-set Brier (0.1003). The rules use training data only and treat that test-set gap as non-evidence.

What was compared: raw against calibrated out-of-fold predictions for all three tuned ML candidates, on the same five folds, with pooled Brier, log loss, PR-AUC and Precision@K, plus per-fold Brier and log loss.

Rule 1 (calibration trigger, XGBoost only): "worse" means calibrated is strictly higher than raw, unrounded, with Brier and log loss counted separately. The rule fires if calibrated is worse on Brier in at least 4 of 5 folds and worse on log loss in at least 4 of 5 folds. Firing means stop and discuss; it is not an automatic reversal. If it did not fire, calibration and the selection stand unchanged.

Rule 2 (sensitivity, reported only): whether XGBoost is also best on the Primary Metrics under raw probabilities. It cannot change the selection by itself while Rule 1 has not fired.

| XGBoost, pooled out-of-fold | Raw | Calibrated |
|---|---:|---:|
| Brier | 0.1031 | 0.1046 |
| Log loss | 0.3249 | 0.3367 |
| PR-AUC | 0.7988 | 0.8004 |
| Precision@10% | 0.9023 | 0.9059 |

Rule 1 fired: calibrated was worse than raw on Brier in 4 of 5 folds and on log loss in 5 of 5 folds, by small amounts (about 0.002 Brier in each affected fold). Rule 2: XGBoost is best on all eight columns (Brier, log loss, PR-AUC and Precision@K at the five K values) under raw probabilities as well as calibrated ones, so the choice of model does not depend on the calibration choice. For context only, Logistic Regression was worse calibrated on Brier in 4 of 5 folds but by about 0.0001 to 0.0003 (log loss: 3 of 5), and Random Forest was mixed (Brier 2 of 5, log loss 3 of 5).

The binned predicted-against-observed table (ten quantile bins of the out-of-fold predictions) shows why. Raw XGBoost predictions sit close to the diagonal (for example mean predicted 0.677 against observed 0.677 in the ninth bin). The sigmoid step compresses the low end: its four lowest bins predict between 0.045 and 0.056 where observed churn is between 0.002 and 0.028, and it over-predicts in the 0.58 to 0.84 bin (0.724 predicted, 0.670 observed). In the top bin, the group an outreach list would target, the calibrated version is closer (0.894 predicted against 0.906 observed, versus 0.867 raw), but that is one bin of 563 rows and within noise.

**Outcome: calibration is kept for this project.** Rule 1 fired and was discussed. The reasons for keeping the locked decision: the effect is small; the selected model is the same with or without calibration (Rule 2); the outreach list is built from rankings, which calibration barely changes; ADR-0013 records that the locked Validation Strategy requires calibration, so dropping it is a decision about a locked document and not a quiet tweak; and switching late would mean changes to `compare.py`, its tests and the final evaluation. ADR-0013 names evidence that a candidate's curve does not fit sigmoid's assumption as a reason to revisit the method. This diagnostic is partial evidence of that for XGBoost, whose raw output is already close to the diagonal. It is recorded here, and revisiting the method, or the calibrate-every-candidate policy, is the main open question for a future version.

One limitation of the diagnostic itself. The two arms differ in how they are fitted: raw is one XGBoost model fitted on each fold's full training portion, while the calibrated arm is `CalibratedClassifierCV`, which fits several XGBoost models each on four fifths of that portion, calibrates them, and averages them. The diagnostic therefore compares the two options that could actually be shipped, but it cannot separate the effect of the sigmoid step from the effect of the different fitting procedure. The reliability-curve plot named in the Business Design is deferred to the Commit 12 README; the binned table above is the supporting calibration evidence until then.

---

## Alternatives considered

- **Keep Logistic Regression (ADR-0012)** — rejected: it was selected on test-set evidence, and under the corrected comparison the Primary Metrics agree on XGBoost, so ADR-0008 step 1 applies and the tie-breaker judgment is not reached.
- **Nested cross-validation for the comparison** — would remove the shared-fold optimism, but it is a larger change than this correction warrants. Rejected for scope and recorded as a carried-forward limitation below.
- **Ship raw XGBoost instead of calibrated** — considered after Rule 1 fired, since raw scored slightly better on Brier and log loss. Not adopted, for the reasons in the outcome paragraph above. One more: the raw model's test-set Brier had already been seen, so a switch now would be harder to separate from test-set influence, even though Rule 1 fired on training data alone.
- **A different calibration method for XGBoost (for example isotonic)** — not tested. ADR-0013's sample-size reasoning still stands, and trying methods after seeing results would be a new decision, not part of this correction.
- **Re-tuning the candidates** — not needed; the Commit 8 tuning is unchanged by this correction.

---

## Consequences

- The final model carried forward is calibrated XGBoost: the tuned XGBoost from Commit 8 wrapped by `calibrate_candidate` (sigmoid, ADR-0013) and fitted on the full training set. The production fit and its persistence belong to Commit 10 and will reuse `calibrate_candidate`.
- ADR-0012's guidance to use Logistic Regression coefficients for business explanation no longer applies. Its correlation-matrix finding (three exact redundant-encoding pairs and several 0.6 to 0.97 correlated pairs among the 55 post-preprocessing features) remains accurate and applies equally to feature-importance output from XGBoost. How the model's drivers are explained is decided in Commit 11.
- ADR-0012 is kept as history with the status "Superseded by ADR-0014". Its prediction-stability, fit-time and parameter evidence is unchanged, but it is no longer used for selection.
- The test set is reached only by `evaluate_final_model_on_test`, once, for the selected model and the fixed heuristic. Any later change to this protocol needs a new ADR.
- Known limitations carried forward:
  1. Preprocessing (the `ColumnTransformer`) is fit once on the full training set before tuning and CV, so each CV fold's preprocessing has seen its own validation rows. The expected effect is small, and the issue pre-dates this correction. It was not fixed because doing so means changing the locked `preprocess.py` that Commit 10 depends on.
  2. Tuning and candidate comparison share the same CV folds (non-nested CV), so hyperparameters were selected on folds the comparison reuses. This is mitigated by the tight top-five plateau, the pessimistic adjustment, and the consistent held-out test result, which is a consistency check and not proof.
  3. The calibration diagnostic cannot isolate the sigmoid effect from the different fitting procedure inside `CalibratedClassifierCV`, and the calibrate-every-candidate policy was kept despite weak evidence that it helps XGBoost.

---

**Decision Date:** 6 October 2026
