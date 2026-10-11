# Platform Output Contract — v1

> **Status:** v1, revision 5. Approved by the owner on 11 Oct 2026 (section 10).
> **Scope:** the outputs of the Commit 10 inference pipeline only.
> **Implemented by:** Commit 10 steps 5 to 8. `tests/test_inference.py` asserts conformance to this document (section 9).
> **Changes after approval:** only by a reviewed amendment (section 8).

Every statement in this document restates an approved decision. Section 10 records the owner's review decisions.

---

## 1. Purpose and terminology

This contract is the business-facing specification of what the platform emits when it scores customers: the files, their exact structure, and the guarantees a reader can rely on. It does not prescribe internal code structure.

| Term | Meaning |
|---|---|
| scored table | the customer table given to the scoring function |
| n | the number of customers in the scored table |
| K | the fraction of scored customers to flag, from configuration (`business.k`, currently 0.10) |
| flag count | the number of customers flagged, `max(1, round(n × K))` |
| model run | one training run and its model, identified by a model run ID |
| scoring run | one execution of scoring, identified by its own scoring run ID |

Run IDs are UTC timestamps in the form `YYYYMMDDTHHMMSSZ`.

## 2. Snapshot notice

The reference run (`make score`) scores the **1,408-customer held-out test split**. That is a historical snapshot used to demonstrate the pipeline. It is **not** a live list of customers to contact.

The scoring function itself accepts any table that passes validation (section 6) and never receives or reads target labels. Only the reference run is restricted to the test split.

## 3. Outputs and publication

### 3.1 Files

Both files are written to `outputs/`. The paths are fixed constants of this contract and are not configurable.

| File | Purpose |
|---|---|
| `outputs/prioritized_outreach_list.csv` | One row per scored customer, ranked by churn risk (section 4) |
| `outputs/run_summary.json` | Description of the scoring run that produced the list (section 5) |

### 3.2 Replacement and retention

- When publication succeeds, the new output pair replaces the previous one. **Publication succeeding and the command succeeding are different things.** Publication succeeds when both files have been replaced. The command succeeds (exit code 0) only if publication succeeded **and** the run's required tracking also completed. If tracking fails after publication, the new outputs stay in place and the command still fails (class 3, section 3.3).
- Earlier lists are **not retained** once a new pair has been published. The tracking folder of a scoring run (`artifacts/runs/score/<scoring_run_id>/`) keeps that run's configuration, metrics and log, **not a copy of the customer list or the summary**. Anyone who needs an earlier list must copy it before the next run.
- Input validation happens **before** any output is published.
- Each file is replaced as a whole: at its final path a reader sees either the previous complete file or the new complete file, never a partly written one. The CSV is published first and the summary second.
- The pair is **not** replaced atomically. Two separate files cannot be swapped in one step, and this contract does not require a transaction mechanism.
- **No backup and no rollback.** Once a file's own destination has been replaced, its previous version is not kept anywhere, even if publication of the pair then fails (class 2b). A failed run does not restore it.

### 3.3 Failure behaviour

Every failure below ends the command with a **non-zero exit code** and a clear message. There are three classes: input validation failure (1), output generation or publication failure (2, split into 2a and 2b), and post-publication tracking failure (3). They are distinct.

| Class | When it happens | Effect on the output files | Message reports |
|---|---|---|---|
| **1. Input validation failure** | The scored table breaks a rule in section 6, before any output is generated | **Neither new file is written. Existing output files stay unchanged.** | The failed condition, naming the column and row where meaningful |
| **2a. Output generation failure** | Before any output file is replaced, and not an input validation failure (for example no valid model found, the scoring run's tracking folder, configuration snapshot or log file cannot be created at the start of the run, or an error while scoring or building the outputs) | Neither new file is written. Existing output files stay unchanged. | What failed |
| **2b. Output publication failure** | While the files are being replaced (for example the CSV is locked by another program) | Each file is old or new as a whole. Because the pair is not atomic, the CSV may be new while the summary is still the previous one. The previous CSV is not restored. | Which file was replaced and which was not, and that the pair may be inconsistent until a successful run |
| **3. Post-publication tracking failure** | Both output files were published (publication succeeded), then a required tracking write failed, so the command as a whole fails | **Both new files are kept.** | That the outputs were generated but tracking failed, and the tracking error |

A scoring run whose required work or tracking did not complete leaves a tracking folder without `metrics.json`. (If the folder itself could not be created, there is no folder.)

## 4. Prioritized Outreach List (`prioritized_outreach_list.csv`)

### 4.1 Format

UTF-8, comma-separated, one header row, no index column. `in_top_k` is written as `True` or `False`. `churn_probability` is written without deliberate rounding by the output layer, so the available prediction precision is preserved. No cell is empty.

### 4.2 Columns

Exactly these four columns, in this order. No other column is present in v1.

| Column | Type | Meaning |
|---|---|---|
| `customer_id` | text | Customer identifier, as it appears in the scored table |
| `churn_probability` | number, 0 to 1 | Calibrated probability of voluntary churn from the selected model |
| `rank` | integer, 1 to n | Position by risk. 1 is the highest churn probability. Unique within the file |
| `in_top_k` | boolean | `True` for the customers flagged for outreach (`rank` ≤ flag count) |

### 4.3 Row guarantees

1. **One row per scored customer.** Every customer in the scored table appears exactly once. None is dropped or added.
2. **Unique, non-empty `customer_id`.** A table with duplicate or empty customer IDs fails validation (class 1). Together with the other column rules this means no CSV cell is ever empty.
3. **Rank order.** Rows are written in ascending `rank`, which is non-increasing `churn_probability`.
4. **Tie-breaking.** Customers with equal `churn_probability` are ordered by `customer_id` ascending, as text. As a result `rank` has no ties.
5. **Flag count.** Exactly `k = max(1, round(n × K))` rows have `in_top_k = True`, namely the rows with `rank` ≤ k. `round` is Python's `round()`: exact halves go to the nearest even integer (25 × 0.10 = 2.5 flags 2). Flagging is by `rank`, so a probability tie across the boundary cannot flag more or fewer than k customers.
6. **Reference run.** n = 1,408 and K = 0.10 give k = 141 flagged customers.

### 4.4 Determinism

The same scored table, model, configuration and implementation should produce the same ranking and flags, assuming the same runtime behaviour (library versions and numerical behaviour on the machine). The contract does not claim bit-identical probabilities across different environments. It does guarantee that, whatever the probabilities are, equal probabilities are ordered by `customer_id` and flags follow `rank`.

## 5. Run summary (`run_summary.json`)

A single JSON object. In v1 it contains exactly the fields below, at each level: no field may be omitted, null or added. A new field requires a reviewed amendment (section 8). No library versions, schema hash or other detailed model metadata appear here; they stay in the model's `manifest.json`.

### 5.1 Top-level fields

| Field | Type | Required | Meaning |
|---|---|---|---|
| `contract_version` | string | yes | `"1"` |
| `k` | number | yes | The K used in this scoring run, a fraction with 0 < K ≤ 1, from configuration at scoring time |
| `customers_scored` | integer | yes | n, at least 1. Equals the number of data rows in the CSV |
| `customers_flagged` | integer | yes | The flag count. Equals the number of rows with `in_top_k = True` |
| `model_run_id` | string | yes | Model run ID of the model used for scoring |
| `scoring_run_id` | string | yes | Run ID of this scoring run. Equals the name of its tracking folder |
| `snapshot_notice` | string | yes | Exactly: `Historical snapshot: this list scores the held-out test split for demonstration. It is not a live outreach list.` |
| `model_health` | object | yes | Information about the model, copied from its manifest (section 5.2) |

### 5.2 `model_health`

`model_health` describes the **model**, not the scored customers. Scoring has no labels, so no accuracy or other quality figure is measured on the scored customers, and nothing in this object may be read as one.

| Field | Type | Required | Meaning |
|---|---|---|---|
| `trained_at_utc` | string | yes | When the model was trained, ISO 8601 in UTC with a `Z` suffix |
| `package_version` | string | yes | Version of the platform package that trained the model |
| `training_cv_metrics` | object | yes | Training-split headline metrics (section 5.3) |
| `held_out_results_reference` | string | yes | `"ADR-0014"`: where the held-out test results are recorded. The numbers themselves are **not** copied here |

### 5.3 `training_cv_metrics`

| Field | Type | Required | Meaning |
|---|---|---|---|
| `k` | number | yes | The K at which the metrics were computed (training time) |
| `training_rows` | integer | yes | Number of training rows the estimates are based on |
| `estimate_type` | string | yes | Exactly: `Five-fold out-of-fold estimates on the training split, using the same folds as tuning; potentially mildly optimistic.` |
| `precision_at_k` | number | yes | Precision@K (Precision@10% when K = 0.10) |
| `lift_at_k` | number | yes | Lift@K |
| `pr_auc` | number | yes | PR-AUC |
| `brier_score` | number | yes | Brier score |

The two `k` values are independent. The top-level `k` is the K applied to this scoring run. `training_cv_metrics.k` is the K the model's metrics were computed at when it was trained. They are equal as long as `business.k` has not changed since training.

### 5.4 Illustration

The values are illustrative, not the output of a real run. Real values come from the scoring run and the model manifest.

```json
{
  "contract_version": "1",
  "k": 0.1,
  "customers_scored": 1408,
  "customers_flagged": 141,
  "model_run_id": "20261011T120000Z",
  "scoring_run_id": "20261011T123500Z",
  "snapshot_notice": "Historical snapshot: this list scores the held-out test split for demonstration. It is not a live outreach list.",
  "model_health": {
    "trained_at_utc": "2026-10-11T12:00:00Z",
    "package_version": "<version>",
    "training_cv_metrics": {
      "k": 0.1,
      "training_rows": 5629,
      "estimate_type": "Five-fold out-of-fold estimates on the training split, using the same folds as tuning; potentially mildly optimistic.",
      "precision_at_k": 0.9059,
      "lift_at_k": 3.4222,
      "pr_auc": 0.8004,
      "brier_score": 0.1046
    },
    "held_out_results_reference": "ADR-0014"
  }
}
```

## 6. Input accepted by scoring

The scoring function accepts a scored table with `customer_id` plus the model's input columns. It never receives or reads target labels. The validator fails loudly and runs **before any output is generated**; a validation failure is a class 1 failure (section 3.3).

**Rejected**

- missing columns and extra columns, including the target column
- duplicate `customer_id`, empty `customer_id` values (missing, an empty string, or text containing only whitespace) and empty tables
- NaN and infinite values
- unseen categorical values, and categorical values that are not text
- numeric strings
- the literal internal sentinels `No Internet` and `No Offer` as raw input
- Python `None` anywhere except the two nullable columns `internet_type` and `offer`
- `pd.NA`, including in those two columns (they accept only Python `None`)
- non-whole numbers in integer-like columns (`12.0` is accepted, `12.5` is not)
- anything other than `True`, `False`, `0` or `1` in boolean-like columns

**Accepted**

- reordered columns
- compatible dtypes (for example int64 or float64 for an integer-like column, float32 for a decimal column, 0 and 1 for a boolean-like column, string or category for a text column)

**Reporting.** The validator reports the failed condition clearly and names the column and the row where that is meaningful. Table-level failures (a missing column, an empty table) do not require a row.

The full rule set and the schema it is derived from are recorded in ADR-0015.

## 7. Non-deliverables of v1

v1 deliberately does **not** provide:

- risk tiers
- customer-level explanations (drivers)
- business economics (expected value, ARPU, offer cost, ROI)
- a threshold or economics summary
- any evaluation of model quality on the scored customers

Commit 11 extends the same CSV under a reviewed v2 amendment (section 8).

## 8. Amendment policy and the v2 rule

- Any change after approval is a reviewed amendment, made in this document before it is implemented.
- Commit 11 extends the **same** `prioritized_outreach_list.csv` under v2. No second customer-list file is created.
- **Compatibility rule for v2 (approved):** the four v1 columns keep the same names, types, order and meanings, and any new columns are appended after them.
- v1 forbids unapproved additions to the run summary (section 5). A v2 amendment may change the run summary only through the same review.

## 9. Conformance

`tests/test_inference.py` asserts, on the reference run:

**Outreach list**

- the exact column names and order; one row per scored customer; unique `customer_id`
- `rank` is 1 to n without gaps or ties; rows are in ascending `rank`; `churn_probability` is within 0 to 1 and non-increasing down the file
- ties are ordered by `customer_id` ascending as text
- exactly `max(1, round(n × K))` rows have `in_top_k = True`, and they are the first rows
- `in_top_k` is written as `True` and `False`; `churn_probability` is not rounded by the output layer
- no cell in the CSV is empty

**Run summary**

- the key set at every level is exactly the one in section 5, with the stated types and no extra or missing fields
- `customers_scored` and `customers_flagged` agree with the CSV
- `model_run_id` names an existing model run and `scoring_run_id` names the scoring run's tracking folder
- the held-out test numbers do not appear in the summary, and `held_out_results_reference` points to ADR-0014

**Label isolation**

- a table containing the target column is rejected
- the table handed to the scoring function has exactly `customer_id` plus the model's input columns
- scoring never reads a label column

**Validation and failure behaviour**

- each rejection rule in section 6 fails with a message that names the failed condition, and names the column and row where meaningful; this includes a missing `customer_id`, an empty-string `customer_id` and a whitespace-only `customer_id`
- after a validation failure, neither output file is written, and pre-existing output files are unchanged byte for byte
- after a failure before publication starts (class 2a), existing output files are unchanged
- if the scoring run's tracking folder, configuration snapshot or log file cannot be created at the start of the run, the command ends with a non-zero exit before any output file is written or replaced, and pre-existing output files are unchanged byte for byte (class 2a)
- if outputs are published and the tracking write then fails, both outputs are kept, the exit is non-zero, and the message says the outputs were generated but tracking failed
- an unsuccessful run leaves no `metrics.json` in its tracking folder

**Per-file replacement and publication order**

- the CSV is published before the summary
- a simulated failure while writing the CSV leaves both existing output files unchanged, with no partly written file at either final path
- a simulated failure after the CSV is published and before the summary is published (class 2b) leaves the new complete CSV and the previous complete summary, ends with a non-zero exit, and the message names which file was replaced and which was not, and says the pair may be inconsistent; the previous CSV is not restored and no copy of it is kept
- a simulated failure while replacing the summary never leaves a partly written summary at its final path
- publication succeeding while the command fails (a tracking failure after both files are published) keeps both new complete files and ends with a non-zero exit

**Retention**

- a successful run replaces the previous output pair, and its tracking folder holds no copy of the CSV or the summary

---

## 10. Review record

Owner decisions taken in the review of revision 2 (11 Oct 2026), now part of this contract:

1. **Per-file whole replacement and publish order** (sections 3.2, 3.3): approved. Each file is replaced as a whole, the CSV first and the summary second. The pair is not atomic, and that limitation stays explicit.
2. **`snapshot_notice` wording** (section 5.1): approved as written.
3. **Run-summary names and formats** (sections 5.2, 5.3): approved, including the ISO 8601 UTC timestamp with `Z`, the field names, the metric keys and the `estimate_type` wording.

Revision 3 also resolves two consistency points raised in that review: publication success versus command success (section 3.2), and empty `customer_id` values (sections 4.3 and 6).

Owner approval of revision 3 (11 Oct 2026), including the rejection of missing, empty-string and whitespace-only `customer_id` values. The owner added one clarification: a previous output file is not retained when its own destination is replaced, even if publication of the pair then fails. Revision 4 changes only the status line and this clarification (sections 3.2, 3.3 and 9).

Owner approval of revision 5 (11 Oct 2026): a failure to create the scoring run's tracking folder, configuration snapshot or log file at the start of the run is class 2a. Class 2a is defined by the failure happening before any output file is replaced, excluding input validation failures; this also settles that "no valid model found" belongs to 2a. Revision 5 changes only the status line, the class 2a trigger (section 3.3), one sentence after the failure table (section 3.3), one conformance bullet (section 9) and this record.
