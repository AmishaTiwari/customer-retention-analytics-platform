# ADR 0018: Experiment Tracking and Per-Run Logging

**Status:** Accepted

---

## Context

The locked Repository Architecture asks for "basic experiment tracking" with zero infrastructure. Section 7 describes `artifacts/runs/` as one directory per run holding the resolved configuration, a metrics JSON file and a log file. It notes that a local MLflow *file store* would also be permissible under the locked scope, and that plain JSON is preferred for dependency-freeness, "recorded as an ADR". Section 5 requires every training and scoring run to snapshot its resolved configuration, and section 6 requires "a per-run log file alongside the run's artifacts". The lock checklist still has one open item: *Experiment-tracking format decided*.

Commit 10 is the first commit with training and scoring runs, so the decision is needed now. Four other decisions already point at it:

- ADR-0015 names the training run's tracking folder and says the per-run `config.json` holds exactly the content hashed into `config_hash`.
- ADR-0016 says a hyperparameter mismatch leaves a recorded failure in the training run's folder.
- The Platform Output Contract names the scoring run's folder, its id, and the "tracking failed" failure class (class 3).
- ADR-0017 puts the shared tracking code in `tracking.py` at the package root and lets entry points call `setup_logging()`.

This ADR fixes the format, the folder layout, what each file holds, when a run counts as complete, what happens when tracking fails, and what `tracking.py` is responsible for. Section 7 describes the run folder only in general terms. This ADR adds a run-type level under `artifacts/runs/` and does not contradict it.

---

## Decision

### 1. Format and scope

- **Plain JSON files and a plain text log. No MLflow file store, and no new dependency.**
- **Only training runs and scoring runs are tracked**, as the locked architecture says. `ingest`, `prepare` and `features` keep console logging only.
- Tracking is write-only for the platform: nothing in training, scoring or model loading reads a run folder back. A run's tracking folder never decides whether a model or an output is valid.

### 2. Folder layout and run ids

```
artifacts/runs/
├── train/<run_id>/
│   ├── config.json
│   ├── run.log
│   └── metrics.json     written last; absent if the run did not complete
└── score/<run_id>/
    ├── config.json
    ├── run.log
    └── metrics.json     written last; absent if the run did not complete
```

- The location is the configured `runs_dir`. `artifacts/runs/*` is already git-ignored.
- The run type is clear from the path, and a training run and a scoring run that start in the same second cannot collide, because they live under different folders.
- **A training run's id is the same id as its model directory** under `artifacts/models/` (ADR-0015). **A scoring run has its own id** in the same format, `YYYYMMDDTHHMMSSZ` (contract section 1).
- **An existing run folder is never reused or overwritten.** If the folder for a new run already exists, the run stops with a clear error before any run-specific work (see section 5).
- The scoring run's folder holds that run's configuration, metrics and log. It does **not** hold a copy of the customer list or the summary (contract section 3.2).

### 3. What each file holds

| File | Training run | Scoring run |
|---|---|---|
| `config.json` | The resolved configuration **without the `paths` section**. This is exactly the content hashed into `config_hash` (ADR-0015), so the hash can be re-checked by loading the file and serialising it canonically | The same: the resolved configuration without `paths` |
| `run.log` | The run's log lines, UTF-8 | The run's log lines, UTF-8 |
| `metrics.json` | The training-CV headline metrics, **the same object as the manifest's `training_cv_metrics`** (`k`, `training_rows`, `estimate_type`, `precision_at_k`, `lift_at_k`, `pr_auc`, `brier_score`), with the same labels (ADR-0015, contract section 5.3) | Operational details only, with exactly these keys: `k` (the K used in this run), `customers_scored`, `customers_flagged` and `model_run_id` (the model run used). **No model-quality metric**, because scoring has no labels |

`config.json` and `metrics.json` are written as readable, indented UTF-8 JSON.

### 4. Per-run logging

- `logging_setup.py` gains a **file handler** that writes `run.log` into the run folder. **Console logging is unchanged and preserved.**
- Entry points configure console logging themselves with `setup_logging()` (ADR-0017). Attaching the run's file handler is a separate step, done through `tracking.py`.
- **The handler is removed and its file closed when the run ends**, on success and on failure. Repeated runs in one process do not duplicate log lines or leak handlers, and the run folder can be deleted afterwards on Windows.
- The file is opened with an explicit UTF-8 encoding.

### 5. Completion marker

- **Run-specific work begins only after tracking is initialized.** An entry point first resolves the configuration and sets up console logging (`setup_logging()`, ADR-0017). It then initializes tracking: it creates the run folder, writes `config.json` and opens `run.log`. Only after all three are in place does run-specific work begin, so that the work can be recorded in the log: for a scoring run, model selection, validation, scoring and output publication; for a training run, the model-path existence check of ADR-0015, the tuning search, the hyperparameter check (ADR-0016), the fit and model publication. Because of this, `config.json` and `run.log` exist **from the start of the run-specific work**.
- **`metrics.json` is written last**, only after the run's required work and its other tracking writes have succeeded. For a training run that means after the model directory is published. For a scoring run it means after both output files are published.
- **`metrics.json` is never left as a partial file.** It is published from a temporary file: the content is written to a temporary file in the same run folder (a name that is not `metrics.json`, for example a `.tmp-` prefix), flushed and closed, and only then renamed to `metrics.json` in one step. If writing or the rename fails, the temporary file is removed where possible, the failure is a tracking failure (section 6), and **no `metrics.json` exists**. If the cleanup itself also fails, only the temporary file remains, and it is not a completion marker. This keeps the guarantee of the Platform Output Contract (section 9) that an unsuccessful run leaves no `metrics.json`.
- **A run folder with `config.json` and `run.log` but no `metrics.json` is a failed or incomplete run** for tracking purposes. There is no separate status field.
- The ADR-0016 hyperparameter mismatch is an example: the folder and its `run.log` exist and record the failure, `metrics.json` does not exist, and no bundle, manifest or model directory was written.

### 6. Tracking failure rule (both run types)

The **required tracking operations** are: creating the run folder, writing `config.json`, opening `run.log`, and writing `metrics.json`. A failure of any of these is a tracking failure under the rules below. Individual log lines are **not** required tracking operations: a failure to write one log line does not fail the run.

- **Training:** if the model is published but a required tracking write fails, the command **exits non-zero** and reports the **published model run id** and the tracking error. The published model stays intact and is not rolled back, because model validity never depends on the tracking folder. The non-zero exit stops `make reproduce`.
- **Scoring:** if both outputs are published but a required tracking write fails, the **outputs are kept**, the command **exits non-zero**, and the message says that the outputs were generated but tracking failed (contract class 3).
- **At the start of a run:** if the run folder, `config.json` or `run.log` cannot be created, the command exits non-zero before any model or output is written or replaced. For a scoring run this is **contract class 2a** (a failure before any output file is replaced that is not an input validation failure; Platform Output Contract revision 5, section 3.3). The existing output files stay unchanged. The contract does not classify training runs, so for a training run this bullet is the rule.
- The run ids written in the logs, in `outputs/run_summary.json` (`model_run_id`, `scoring_run_id`) and in the folder names agree.

### 7. What `tracking.py` is responsible for

`tracking.py` holds only the operations that training and scoring share. It contains **no training, scoring or metric-calculation logic** and imports only `config` and `logging_setup` (ADR-0017). It provides, in step 6:

| Operation | Behaviour |
|---|---|
| Run id | Formats a UTC instant as `YYYYMMDDTHHMMSSZ`, in one place. The training entry point passes the same instant to the manifest's `trained_at_utc` (ADR-0015). `models/persist.py` uses the same format definition for its strict run-id matching |
| Config snapshot | Returns the resolved configuration without `paths`. `config_hash` (ADR-0015) is computed from this same content |
| Start a run | Creates `artifacts/runs/<type>/<run_id>/` (failing if it already exists), writes `config.json`, and attaches the `run.log` file handler |
| Write metrics | Writes `metrics.json` through a temporary file in the run folder, then renames it, so a failed write leaves no `metrics.json` (section 5). Called last |
| End a run | Removes the file handler and closes the file, on success and on failure. Offered as a context manager so that it cannot be forgotten |

Exact function names and signatures are fixed when step 6 is implemented and are covered by its tests.

---

## Alternatives considered

- **A local MLflow file store.** Permissible under the locked scope, but it adds a dependency and a storage format for a project that records two kinds of run and never reads them back. Section 7 already prefers plain JSON for dependency-freeness. The per-run layout maps onto MLflow's model, so moving to it later is a migration and not a rewrite (Repository Architecture section 12).
- **One flat folder, `artifacts/runs/<run_id>/`.** Rejected: the run type would not be visible from the path, and a training run and a scoring run with the same timestamp would collide.
- **Track every command (`ingest`, `prepare`, `features`).** Rejected: the locked architecture asks for training and scoring runs, and those commands have nothing to record beyond their console output.
- **An explicit `status` field instead of the missing `metrics.json`.** Rejected: a status written at the start and updated at the end is left as "running" by a crash, and the marker approach needs no update in place.
- **Capture the console with shell redirection in the Makefile.** Rejected: the run id exists only inside the Python process, so the Makefile cannot name the folder, and the redirect would not keep console output.
- **Keep a copy of the customer list or summary in the scoring run folder.** Rejected by the contract: earlier lists are not retained.
- **Roll back the published model or outputs when tracking fails, or only warn.** Rejected: model validity never depends on tracking, and a warning is easy to miss. The command fails loudly and keeps what was published.

---

## Consequences

- Run folders accumulate. Nothing prunes them, and `artifacts/runs/*` is git-ignored.
- A failed or aborted run leaves a folder without `metrics.json`. This is intended, because the folder is the record of the failure.
- A published model whose tracking write failed has a run folder without `metrics.json`. It remains a valid model, since validity is decided by the bundle files only (ADR-0015).
- `logging_setup.py` is a shared module, so the file handler is added in its own step (step 6), with its own tests.
- The step 6 and step 7/8 tests cover this ADR: the file handler and its cleanup, the config snapshot and its hash, the metrics files, the run-id relationships, the failure rules for both run types, and the import-direction check for `tracking.py`.
- The lock checklist item *Experiment-tracking format decided* in the Repository Architecture is ticked with a pointer to this ADR. No other change is made to that locked document.

---

**Decision Date:** 11 October 2026 (decisions approved 10 October 2026, open points closed 11 October 2026; recorded here before implementation)
