# ADR 0017: Module Layout and Import Direction

**Status:** Accepted

---

## Context

The locked Repository Architecture promises that dependencies run in one direction and that "nothing imports upward" (section 2), and that boundaries are "enforced by structure, not by discipline" through "import direction rules, and tests" (section 1). Until Commit 10 no rule was written down and nothing checked it.

Commit 10 adds two entry points that need code which currently sits in the wrong place:

- **`models/train.py`** needs the metric functions (for the training-CV headline metrics, ADR-0016) and `calibrate_candidate`. The metric functions live in `evaluation/metrics.py` and the calibration code lives in `evaluation/compare.py`. Both are in a *later* stage than `models/`, so training would import upward.
- **`inference/score.py`** needs the same Top-K rule as the metrics. Importing it from `evaluation/` would make inference depend on the comparison stage.
- **Both entry points** need the same run-tracking operations (ADR-0018). Whichever of `models/` or `inference/` owned them, the other would have to import across stages.

So three pieces of code have to move or be created before the entry points are written, and the rule they must follow has to be recorded and tested. This ADR does that. It changes no behaviour: every move is a relocation with unchanged logic.

The locked Repository Architecture lists the package layout in section 4. That listing names `evaluation/metrics.py` and has no `models/calibrate.py` or `tracking.py`, and it names `config.py` and `logging_setup.py` as the only root modules. This ADR is the recorded deviation from that listing. Section 4 is left unchanged as history (see Consequences).

---

## Decision

### 1. Stage order and the rule

The package stages, in dependency order, are:

```
data  ->  features  ->  pipeline  ->  models  ->  evaluation  ->  inference
```

Root modules (`config.py`, `logging_setup.py`, and the two added below) sit outside the stages.

**The rule: nothing imports upward.** A module may import only what its row in the table in section 4 allows. The table is an allow-list, not "anything earlier". It is deliberately narrower than the stage order, so a new dependency between stages is a visible, reviewed change to the table and not a side effect of an `import` line.

### 2. The three layout changes

| Change | Where it ends up | Imports from the package |
|---|---|---|
| Metric functions (`precision_at_k`, `lift_at_k`, `pr_auc`, `brier_score`, `confusion_matrix_at_k`, `sensitivity_report`) | `retention_platform/metrics.py` (moved from `evaluation/metrics.py`) | nothing |
| `CALIBRATION_METHOD`, `build_cv` (renamed from `_build_cv`) and `calibrate_candidate` | `retention_platform/models/calibrate.py` (moved out of `evaluation/compare.py`) | nothing (scikit-learn only) |
| Shared experiment-tracking operations: create a run folder, write `config.json` and `metrics.json`, wire the per-run log file (ADR-0018) | `retention_platform/tracking.py` (new) | `config`, `logging_setup` |

- **`metrics.py` at the root.** Training, comparison and scoring all use it, so it belongs to none of the three stages. It stays pure: it imports nothing from the package.
- **`models/calibrate.py`.** Calibration belongs to the model stage, and `evaluation/compare.py` imports it from there (earlier stage, allowed). Behaviour is unchanged: `calibrate_candidate` still returns an unfitted `CalibratedClassifierCV` with `method="sigmoid"` and `cv=build_cv(config)` (ADR-0013). `build_cv` loses its underscore because it is now imported by other modules.
- **`tracking.py` at the root**, next to `config.py` and `logging_setup.py`, for the same reason those two are there: it is cross-cutting, and no stage owns it. It contains no training, scoring or metric-calculation logic.
- **Comparison-specific code stays in `evaluation/compare.py`.** Nothing else moves.

### 3. What stays where

`data/`, `features/`, `pipeline/`, `models/candidates.py`, `models/tune.py`, `evaluation/compare.py`, `evaluation/reports.py` and the `inference/` modules keep the places given in Repository Architecture section 4. The new entry-point modules of Commit 10 (`models/train.py`, `models/persist.py`, `inference/validate.py`, `inference/score.py`, `inference/outputs.py`) are already named there.

### 4. The import table

`set()` / "nothing" means the file may import nothing from `retention_platform` (standard library and third-party imports are not restricted). A dotted entry such as `models.persist` allows that module only. A bare entry such as `models` allows the whole subpackage.

| File or selector | May import from the package |
|---|---|
| `__init__.py`, `config.py`, `logging_setup.py`, `metrics.py`, `models/calibrate.py` | nothing |
| `tracking.py` (†) | `config`, `logging_setup` |
| `data/*` | `config`, `logging_setup`, `data` |
| `features/*` | `config`, `logging_setup`, `data`, `features` |
| `pipeline/*` | `config`, `logging_setup`, `data`, `features`, `pipeline` |
| `models/*` | `config`, `logging_setup`, `pipeline`, `metrics`, `tracking` (†), `models` |
| `evaluation/*` | `config`, `metrics`, `models`, `pipeline`, `evaluation` |
| `inference/*` | `config`, `logging_setup` (‡), `metrics`, `tracking` (†), `models.persist`, `pipeline.split` (‡), `inference` |

Reading the table:

- **A module may import its own stage** (`models/*` lists `models`, `evaluation/*` lists `evaluation`, and so on). For example, `inference/score.py` imports `inference/validate.py`.
- **An exact row wins over a wildcard row.** `models/calibrate.py` has its own row (nothing), which overrides `models/*`.
- **The table is narrower than the stage order on purpose.** `models/*` does not list `data` or `features`; `evaluation/*` and `inference/*` do not list `data` or `features`; `inference/*` cannot import `models.train`, `models.tune` or anything in `evaluation/`. `inference/*` reaches the model stage only through `models.persist`.
- **(†) Rows added in Commit 10, step 6**, when `tracking.py` is created. The import test has no `tracking` entries until then, because the module does not exist.
- **(‡) Entries added to the `inference/*` row in step 8**, when `inference/score.py` is written:
  - **`pipeline.split`** is an explicit, module-level exception, and the only place `inference/*` reaches into `pipeline/`. The reference run selects the held-out customer population with the existing `split_feat_churn` (the closed "scoring input" decision). It does **not** allow `inference/*` to import anything else from `pipeline/`, in the same way `models.persist` does not open the rest of `models/`.
  - **`logging_setup`** lets the scoring entry point call `setup_logging()` itself (see "Logging responsibility" below).

**Logging responsibility.** Entry points configure console logging themselves by calling `setup_logging()` at startup, as `data/prepare.py` already does. `models/train.py` and `inference/score.py` follow the same pattern, which is why `inference/*` lists `logging_setup`. `tracking.py` handles per-run tracking only: the run folder, `config.json`, `metrics.json` and wiring the per-run `run.log` through the file handler that step 6 adds to `logging_setup.py` (ADR-0018). That is why `tracking.py` imports `logging_setup`. It does not initialise console logging. `evaluation/*` has no `logging_setup` entry because Commit 10 adds no evaluation entry point (`make evaluate` is a placeholder). The tracking API itself is designed in ADR-0018.

### 5. Enforcement

`tests/test_import_direction.py` encodes the table as `IMPORT_RULES` and checks it:

- It reads each source file's `import` statements with Python's `ast` module. No package code is executed.
- Every file under the package must have a row, and every row must match at least one file, so a new module without a row fails the test, and a stale row fails too.
- Known limit: imports made at run time through `importlib.import_module` are not seen by this test.

Changing a row is a reviewed edit to both the test and this table.

### 6. Top-K count rule

The number of customers flagged for outreach is

```
k = max(1, round(n * K))
```

where `n` is the number of scored rows and `K` is the capacity fraction from configuration (0.10). This is the rule `metrics.py` already uses for Precision@K, Lift@K and the confusion matrix, and the scoring output uses the same formula.

- Python's `round()` sends an exact half to the nearest **even** integer. For 25 rows at K = 0.10, `25 * 0.10` is exactly `2.5`, and `round(2.5)` is `2`, so 2 rows are flagged.
- The 1,408-row reference run: `1408 * 0.10 = 140.8`, which rounds to **141** flagged customers. The 5,629-row training CV: `562.9`, which rounds to 563.
- `max(1, ...)` guarantees at least one flagged row.
- A public `top_k_count(n_rows, k_frac)` helper is added in step 8, when the output code first needs it. It carries the same formula. No change is made to `metrics.py` logic in this commit.

### 7. Ranking ties: documented difference

Two places rank by predicted probability, and they break ties differently:

- **Evaluation metrics** (`metrics.py`) rank with `np.argsort(y_score)[::-1]`. Among rows with equal scores, the order is not defined by any project rule; it follows NumPy's default sort order, reversed.
- **Production scoring** (the output contract) ranks by probability descending and breaks ties by `customer_id` ascending, compared as text.

The two can differ only where scores tie, and only in which tied rows land inside the top K. The difference is documented here and is **not refactored** in Commit 10: the metrics are the Commit 9 numbers recorded in ADR-0014, and the production order is a deliberate, deterministic rule the output depends on.

---

## Alternatives considered

- **Leave `metrics.py` and the calibration code in `evaluation/` and let `models/train.py` and `inference/` import them.** Rejected: it creates upward imports, which is exactly what the locked architecture rules out, and it makes training and scoring depend on the comparison stage.
- **Copy the metric functions or the Top-K formula into `train.py` and the scorer.** Rejected: two definitions of the project's single "top K by predicted risk" operating point can drift apart. The `metrics.py` docstring already states that this logic must exist in one place.
- **Put `calibrate_candidate` in `models/train.py`.** Rejected: `evaluation/compare.py` would then import the training entry point. A small module with no package imports can be used by both without either depending on the other's code.
- **Put the tracking operations in `models/` or in `inference/`.** Rejected: the other entry point would then import across stages. The root is where the locked architecture already puts cross-cutting modules.
- **A `utils.py` for shared code.** Rejected by the locked architecture (section 4): a catch-all module is where structure goes to die.
- **Enforce the direction by code review only.** Rejected: the locked architecture says boundaries are enforced by structure and tests, not by the author remembering them.
- **Edit the locked Repository Architecture section 4 to match the new layout.** Rejected: the document is locked. This ADR records the deviation, and the old listing stays as history, as the older ADRs that name `compare.py` do.
- **An import-checking library.** Rejected: a new dependency for a rule that a short `ast`-based test enforces, and the project's scope does not add tooling for this.

---

## Consequences

- Training and scoring import only what the table allows, and the test fails on any other import, including in files added later.
- The three moves were behaviour-preserving and verified as such: the metric tests pass after the move, and after the calibration move the Commit 9 comparison script reproduced the ADR-0014 training-CV table exactly for all four candidates.
- `models/*` stays independent of `data` and `features`. Code in the model stage that needs SQL or raw-data files, such as the fingerprints of ADR-0015, reads those files directly using configured paths. It does not import the loader modules (`data/prepare.py`, `features/build.py`).
- Repository Architecture section 4, ADR-0013 and ADR-0014 still name `evaluation/metrics.py` and `compare.py` as the home of this code. They are left as history. The stale mentions are on the Commit 12 cleanup list.
- The table gains the `tracking` entries and the `tracking.py` row in step 6, and `pipeline.split` and `logging_setup` for `inference/*` in step 8. Those are the only planned changes.
- The Top-K rounding behaviour at exact halves (ties to even) is now written down. It affects only row counts where `n * K` is exactly a half, which does not happen for the 1,408-row reference run.
- The tie-break difference between evaluation and production is a known, documented asymmetry.

---

**Decision Date:** 11 October 2026 (the layout, the import rule and the Top-K and tie-break notes were decided 10-11 October 2026; recorded here before implementation of the entry points)
