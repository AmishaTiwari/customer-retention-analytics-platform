# ADR 0015: Model Persistence, Manifest and Input Schema

**Status:** Accepted

---

## Context

Commit 10 turns the selected model (tuned XGBoost with sigmoid calibration, ADR-0014) into something that can be trained once, saved, and used later to score customers. Four things must be settled before any implementation code is written, because the Platform Output Contract (`docs/platform_output_contract.md`) and the tests depend on them:

1. **What is saved, where, and under what name**, so that a model is never overwritten and the newest valid one can be found without guessing.
2. **What the saved model says about itself** (the manifest), including where it came from.
3. **What input the saved model accepts** (the input schema), and the exact rules a table must pass before it is scored. The contract (section 6) states the rules in business terms and points here for the full rule set.
4. **What is recorded as provenance but never enforced** (fingerprints), and how it is hashed so that the values are the same on every machine.

The locked Repository Architecture (section 7) already fixes the principle: a persisted model is a **bundle** (preprocessing, model and calibrator together) with a `manifest.json`, artifacts carry a run identifier, and "latest" is a pointer decision, never an ambiguous filename. This ADR makes that concrete. It does not change any locked document.

---

## Decision

### 1. Bundle layout

Each training run is stored as one directory:

```
artifacts/models/<run_id>/
├── bundle.joblib      fitted preprocessor + calibrated model + schema hash
└── manifest.json      what the bundle is, where it came from, how it was measured
```

- The bundle is serialized with `joblib`. **Loading a joblib file executes code, so a bundle must only be loaded if it was produced by this repository on a trusted machine.** This is documented, not mitigated.
- The preprocessor and the calibrated model are saved **together**, so the train/serve consistency guarantee cannot be broken by loading one without the other.
- `artifacts/models/*` is already git-ignored.

### 2. Run identifier

- `run_id` is a UTC timestamp in the form `YYYYMMDDTHHMMSSZ`, generated **once**, at the start of training.
- The manifest records the same `run_id`, and the directory name is the `run_id`.
- The same `run_id` names the training run's tracking folder (`artifacts/runs/train/<run_id>/`, ADR-0018).
- A scoring run has its own, separate id in the same format (contract section 1).

### 3. Publication

- **The existence check comes after the run is initialized and before any training or model work.** Training first initializes its tracking folder, `config.json` and `run.log` (ADR-0018). It then checks the final path `artifacts/models/<run_id>/`, and **stops if that path already exists in any form** (empty, incomplete or complete). The check runs before the tuning search, before any fitting and before anything is written for publication, so a failed check writes nothing under `artifacts/models/`. The stop is recorded in the training run's `run.log`, and the run has no `metrics.json`.
- Both files are written into a temporary directory created **inside `artifacts/models/`** (so it is on the same filesystem), named with a `.tmp-` prefix. It is removed after failures where possible.
- The run directory is published with a **single `os.rename`**, and only after both files are complete. `os.rename` is used instead of `os.replace` because the intent is never to replace anything.
- If the rename fails because the target now exists, the error says the run id collided and that nothing was replaced. Any other failure is re-raised unchanged.

### 4. No-overwrite guarantee and its accepted limitation

**Guarantee.** Training never replaces a directory that already contains files. Renaming onto a non-empty directory fails on Linux, macOS and Windows, so a published run (which always holds `bundle.joblib` and `manifest.json`) is never overwritten, including when two runs race with the same run id.

**What the guarantee does not cover.** An empty destination on Linux and macOS, where `os.rename` succeeds. Training stops at the existence check if the final path exists in any form, so an empty directory can only be replaced if it appears between that check and the rename. The loader treats an empty or incomplete directory as not a run.

**Accepted limitation (the mechanism is deliberately not redesigned):**

- On Linux and macOS, an empty final directory created between the initial existence check and the rename may be replaced.
- If publication fails, the new model may remain in the temporary directory and will not be selected.
- Safeguards that are kept: temporary directories are cleaned up after failures where possible; temporary, empty and incomplete directories are ignored when a model is selected automatically; every invalid run directory that is skipped produces a log warning.
- There is no `fsync`. Publication is atomic against other processes, **not against a power cut**.
- "Newest" trusts the system clock.

### 5. Loading and run selection

- Only folder names that **exactly match** the run-id format are considered.
- A run is **valid** only if both files exist and the manifest's `run_id` equals the folder name. A run's validity never depends on its tracking folder.
- Automatic selection takes the **highest valid name**, never the file modification time, and names each skipped folder in a log warning.
- An optional `--run-id` selects a specific run. It must match the strict format **before any path is built**. A missing or invalid requested run fails clearly, with **no fallback** to another model.
- With no valid run, scoring stops with a "run `make train` first" message.
- The selected run id is written to the logs and to `run_summary.json` as `model_run_id`.
- The loader's enforced checks are the **schema hash** and the **compatibility of the schema with the fitted preprocessor** (section 6). Nothing else is enforced; in particular the fingerprints (section 8) are not.

### 6. Input schema

The schema is the list of columns a scored table must contain, with the kind of value each accepts. It is **generated at training time**, not typed by hand.

| SQL-declared type (`feat_churn`) | Schema kind | Meaning |
|---|---|---|
| `BOOLEAN` | boolean-like | `True`, `False`, `0` or `1` |
| `INTEGER` | integer-like | a whole number (`12` and `12.0` are accepted, `12.5` is not) |
| `DOUBLE` | decimal | any finite number |
| `VARCHAR` | categorical | text that is one of the categories the fitted encoder knows |

Each schema entry records the column name, its kind, whether it is nullable, and for categorical columns the allowed values taken from the fitted encoder. `customer_id` is part of the scored table but is not a model input.

**It is checked three ways**, so that a wrong schema cannot be saved quietly:

1. the kinds come from the SQL-declared types through the explicit mapping above;
2. the schema is cross-checked against the fitted preprocessor's column set and categorical encoder values;
3. a pinned test covers every schema column by name and kind.

**The number of columns is whatever the generated schema contains.** It is pinned by a test and is deliberately not written in this ADR.

**Schema hash.** The canonical schema is stored in the manifest, and its sha256 hash is stored in the bundle. At load time the loader checks the two against each other and checks the schema against the fitted preprocessor. A mismatch stops scoring.

### 7. Input validation rules

These are the rules the validator enforces before anything is scored (contract section 6 states them in business terms). A failure is a class 1 failure in the contract: nothing is published, and existing output files stay unchanged.

| Rule | Detail |
|---|---|
| Columns | Reject missing columns and extra columns. The target column and the excluded columns (for example `city`, `state`, `country`, `zip_code`) are extra columns. Accept reordered columns |
| Table | Reject an empty table |
| `customer_id` | Reject duplicates, and missing, empty-string or whitespace-only values |
| Missing values | Reject NaN anywhere, and infinity. Python `None` is accepted **only** in the two nullable columns `internet_type` and `offer`. `pd.NA` is rejected everywhere, including those two columns |
| Sentinels | Reject the literal strings `No Internet` and `No Offer` as raw input. They are internal sentinels created by feature engineering |
| Categories | Reject unseen categories, and categorical values that are not text |
| Numbers | Reject numeric strings. Reject non-whole numbers in integer-like columns |
| Booleans | Reject anything other than `True`, `False`, `0` or `1` in boolean-like columns |
| Dtypes | Accept compatible dtypes: `int64` or `float64` for integer-like columns, `float32` for decimal columns, 0/1 for boolean-like columns, string or category for text columns |
| Messages | Name the failed condition, and the column and row where that is meaningful. A missing column or an empty table needs no row |

### 8. Manifest

`manifest.json` records:

| Content | Notes |
|---|---|
| `run_id` | the same value as the folder name |
| input schema and its hash | section 6 |
| tuned hyperparameters | the values the model was trained with |
| library versions | the packages that matter for loading the bundle |
| `trained_at_utc` | ISO 8601 in UTC with a `Z` suffix, for example `2026-10-11T12:00:00Z`. Taken from the **same instant** as `run_id`, so the two cannot disagree (a test checks this). Never typed by hand |
| `package_version` | read from the installed package metadata, which comes from the version in `pyproject.toml`. Never hardcoded |
| `training_cv_metrics` | `k`, `training_rows`, `estimate_type`, `precision_at_k`, `lift_at_k`, `pr_auc`, `brier_score`. The field names and the exact `estimate_type` sentence are the ones in contract section 5.3, because the run summary copies this block from the manifest |
| `held_out_results_reference` | `"ADR-0014"`. The held-out test numbers are **not** copied into the manifest |
| `fingerprints` | section 9 |

The metrics are the training-split cross-validation estimates only (Precision@10%, Lift@10%, PR-AUC, Brier score), labelled as five-fold out-of-fold estimates that use the same folds as tuning and are potentially mildly optimistic. There is no second, enforced metrics check. Exact key names for fields not named in this table are fixed when `persist.py` is written and covered by tests.

### 9. Fingerprints: provenance only

**Fingerprints are traceability metadata and nothing more.** They record which configuration, SQL and data produced a model. The loader never compares them with anything, and a mismatch can never block scoring. The loader's enforced checks are only those in section 5.

The `fingerprints` block holds:

- **`config_hash`**: sha256 of the canonical JSON (sorted keys, separators `(",", ":")`) of every configuration section **except `paths`**, which differs per machine. It is a whole-configuration identity stamp: an edit to any section changes it, including sections that do not affect training. The per-run `config.json` snapshot (ADR-0018) holds exactly the hashed content, so the hash can be re-checked and a mismatch can be diffed.
- **`sql_hash`**: sha256 over the `.sql` files only (not `.gitkeep`), in the **actual execution order** across both SQL loaders (`data/prepare.py` for `sql/01` to `sql/04`, and the feature build for `sql/05`). The order is read from the loaders, not inferred from file names. Each script contributes its file name, a NUL byte, then its normalized contents.
- **`data_fingerprint`**: sha256 of each of the five raw CSV files, keyed by file name (`raw_files_sha256`), plus the training and test row counts (`train_rows`, `test_rows`). The counts are counts only; no test labels are read. No hash of training customer ids is added, because the split is deterministic given the data, the SQL and the configuration seed.
- **`hash_normalization`**: the text `"CRLF to LF before hashing"`.

**Hashing rules.** Files are streamed in chunks. Every hash (digest) value, including the schema hash, is a lowercase, 64-character SHA-256 hexadecimal string with no prefix. SHA-256 is the algorithm for every digest in the manifest; it is stated here and is not repeated in each field name. Row counts (`train_rows`, `test_rows`) and `hash_normalization` are not hash values. Before hashing, `\r\n` is converted to `\n` **in memory**; the files on disk are never modified. The implementation must handle a CRLF pair that is split across two chunks, and tests cover that case.

**Why line endings are normalized.** Checked on 10 October 2026: `git ls-files --eol` shows every raw CSV and every SQL script as `i/lf w/crlf`, `core.autocrlf=true` is set on the owner's machine, and there is no `.gitattributes`. Without normalization the same files would hash differently on this Windows checkout and on a Linux or Mac clone.

---

## Alternatives considered

- **Separate files for the preprocessor, the model and the calibrator.** Rejected: it allows loading one without the others, which breaks the bundle-as-unit rule of the Repository Architecture.
- **A single "latest" file, a symlink or a pointer file.** Rejected: the Repository Architecture forbids ambiguous names. "Latest" is resolved at load time from the run ids, and the chosen id is logged.
- **Selecting the newest model by file modification time.** Rejected: copying or restoring a file changes its modification time. The highest valid run-id name is used instead.
- **Overwriting an existing run on a collision.** Rejected: a published model is never replaced. A collision fails clearly.
- **Redesigning publication to close the empty-destination gap** (for example with locking or `fsync`). Rejected for scope: the gap is narrow, the safeguards above remain, and the limitation is recorded rather than hidden.
- **Enforcing fingerprints when loading.** Rejected: `config_hash` changes with any configuration edit, including edits that do not affect training, so enforcement would block scoring for harmless reasons.
- **Hashing the raw bytes without normalizing line endings.** Rejected: the hashes would differ between operating systems for identical content.
- **Adding a `.gitattributes` file to fix line endings.** Rejected: it changes tracked files in an inference commit and could renormalize the raw data committed under ADR-0005.
- **Asking contributors to change `core.autocrlf`.** Rejected: it is a per-machine setting the repository cannot enforce.

---

## Consequences

- `models/persist.py` and `models/train.py` (Commit 10, step 7) implement sections 1 to 5, 8 and 9. The validator (`inference/validate.py`, step 5) implements sections 6 and 7. Tests assert each rule here and conformance to the Platform Output Contract.
- Because the contract copies `model_health` from the manifest, a change to the manifest fields named in section 8 is a contract amendment, not just an implementation change.
- A stranger can load a bundle only by trusting the file (section 1). The README states this when it describes model loading.
- Two machines can legitimately produce different `config_hash` values for the same training result. That is by design (section 9).
- The accepted limitations in section 4 are part of the documented behaviour of this project. Changing any rule in this ADR needs a new ADR that supersedes this one.

---

**Decision Date:** 11 October 2026 (decisions approved 10 to 11 October 2026, recorded here before implementation)
