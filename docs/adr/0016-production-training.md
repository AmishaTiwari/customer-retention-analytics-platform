# ADR 0016: Production Training Procedure

**Status:** Accepted

---

## Context

Commit 9 selected tuned XGBoost with sigmoid calibration (ADR-0014). The tuned hyperparameters exist today only inside a cached, git-ignored scratch file, and the final fit was a one-time scratch run. Commit 10 needs a **production** training step (`make train`) that anyone can run from the repository alone and that produces the bundle described in ADR-0015.

Three questions have to be answered before any code is written:

1. **Where do the production hyperparameters come from?** Re-running the search, copying values into code, and loading the scratch file are all possible, and they behave very differently when something changes (a library version, the data, the configuration).
2. **How do we know the production model is the model that was evaluated?** ADR-0014 recorded the training-CV table and the held-out result for a specific configuration. A production model with different hyperparameters would make those numbers describe a different model.
3. **Which data does training touch?** The locked Validation Strategy reserves the held-out test split for final evaluation, and the reference scoring run (contract section 2) scores that same split as a historical demonstration. Training must not have seen it.

Reproducibility of the tuned parameters was checked on 10 October 2026: all seven XGBoost parameters are identical across the values recorded in the Commit 9 correction notes, the cached search object, and a fresh run of the seeded search (xgboost 3.3.0, scikit-learn 1.9.0).

---

## Decision

### 1. Production training re-runs the seeded search

`models/train.py` runs `tune_xgboost` on the **5,629-row training split**, with exactly the settings of ADR-0009 (PR-AUC objective, 30 iterations, five-fold stratified cross-validation with one repeat, seed from configuration). Nothing about the search is changed by this ADR. The fixed (non-searched) settings come from the same code path: `eval_metric="logloss"` and `random_state` from the configuration seed.

### 2. The result is checked against a reference constant

`train.py` holds a **module-level reference constant** with the seven XGBoost parameters that ADR-0014 selected. After the search, the tuned parameters are compared with the constant. The reference values are recorded here:

| Parameter | Reference value |
|---|---|
| `n_estimators` | `215` |
| `max_depth` | `8` |
| `learning_rate` | `0.02171140223187999` |
| `subsample` | `0.9442922333025374` |
| `colsample_bytree` | `0.6440207698110707` |
| `min_child_weight` | `9` |
| `scale_pos_weight` | `1` |

The comparison is **exact equality**. The recorded values were reproduced to every digit on 10 October 2026, so no tolerance is introduced.

### 3. A mismatch stops training

- If any of the seven parameters differs from the reference, training **stops before any model bundle or manifest is written**. Nothing is published under `artifacts/models/`.
- The training run's tracking folder and its `run.log` may already exist at that point, because they are created at the start of the run (ADR-0018). The failure is therefore recorded there. `metrics.json`, the completion marker, is absent.
- There is **no override flag**. Changing the constant needs a new ADR that supersedes this one, because it means the production model is no longer the model that ADR-0014 evaluated.

### 4. What is fitted, and on which data

After the check passes:

- `calibrate_candidate(...)` (ADR-0013) returns an **unfitted** `CalibratedClassifierCV` wrapped around the tuned XGBoost estimator, with `method="sigmoid"` and `cv=build_cv(config)` (five stratified folds, one repeat, seed from configuration). It passes no `ensemble` argument. The scikit-learn default, `ensemble="auto"`, resolves to `True` for an ordinary (not frozen) estimator, so it is the ensemble behaviour that applies.
- `train.py` calls `.fit` on this wrapper with the **5,629-row training split**. For each of the five folds, scikit-learn fits a **clone** of the tuned estimator on that fold's training portion (about four fifths of the training split) and fits a sigmoid calibrator on that fold's held-out portion. The production model is the **average of these five calibrated pairs**. The tuned estimator passed in is only a template: **no base estimator is refitted on all 5,629 training rows**. This is the same construction as the one-time Commit 9 final fit (`calibrate_candidate`, then `.fit` on the training split).
- The preprocessor is the one fitted on the training split, as in Commit 9.
- **Training, tuning and calibration use only the 5,629-row training split.** The 1,408-row test split is never used for training. It is used only as the scoring input of the reference run, without labels (contract section 2). Its row count is recorded in the manifest as a count only (ADR-0015).

### 5. Headline metrics recorded in the manifest

The manifest carries the training-split cross-validation estimates only (Precision@10%, Lift@10%, PR-AUC, Brier score), labelled as in the contract and ADR-0015, with a pointer to ADR-0014 for the held-out result (D4). They are computed from out-of-fold predictions on the training split: the unfitted calibrated model is cross-validated over the `build_cv` folds (each outer fold fits and calibrates it on that fold's training portion and predicts the held-out portion), and the scores come from the functions in `metrics.py`. This is the method of the Commit 9 comparison. Training does not import from `evaluation/` (ADR-0017). The metrics are **recorded, not gated**: there is no second enforced check that a metric must reach a value.

The held-out test result is **not recomputed** in Commit 10. It stays recorded in ADR-0014.

---

## Alternatives considered

- **Load the tuned model from the cached scratch file.** Rejected: the file is git-ignored scratch output, so a stranger running `make reproduce` would not have it, and a pickle from another environment is not a reproducible input.
- **Copy the seven values into code and skip the search.** Rejected: the values would have no visible link to the procedure that produced them, and training could never notice that the data, the libraries or the configuration had drifted from what ADR-0014 evaluated.
- **Re-run the search and accept whatever it returns.** Rejected: a different version of a library could silently produce a different model while the README and ADR-0014 still describe the old one.
- **Re-run the search, warn on a mismatch and continue, or allow an override flag.** Rejected: a warning in a log is easy to miss and an override makes the guarantee optional. The check exists to stop `make reproduce` when the model would differ.
- **Train the final model on the training and test splits together.** Rejected: it breaks the Validation Strategy, and the reference run would then score customers the model was trained on, which makes the demonstration meaningless.
- **Skip calibration in production.** Rejected: calibration is locked (ADR-0013) and the ADR-0014 numbers describe the calibrated model.

---

## Consequences

- `make train` includes the full tuning search, so training takes as long as that search does. This is the cost of a model that is reproducible from the repository alone.
- `make reproduce` stops if the tuned parameters differ from the reference. This is intended. On a machine with different library versions, a mismatch is a finding to investigate, not something to bypass.
- The hyperparameters in the manifest (ADR-0015) are those of the reference constant, so the manifest, this ADR and ADR-0014 describe one model.
- The ADR-0014 held-out numbers describe this model configuration fitted on the training split. They are not regenerated in Commit 10, and the README presents them as recorded there (Commit 12).
- The limitations recorded in ADR-0014 carry over unchanged to the production model: preprocessing is fitted once on the full training split, cross-validation is not nested and shares folds with tuning, and the calibration diagnostic cannot isolate the effect of the sigmoid step. The training-CV headline metrics are therefore potentially mildly optimistic, as the contract states.
- `models/train.py` (Commit 10, step 7) implements this ADR, using the tracking helpers of step 6 and the persistence rules of ADR-0015.

---

**Decision Date:** 11 October 2026 (decision D3 approved 10 October 2026 and its wording revised 11 October 2026; recorded here before implementation)
