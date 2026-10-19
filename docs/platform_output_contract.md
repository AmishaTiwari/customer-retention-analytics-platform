# Platform Output Contract — v2

> **Status:** v2, revision 2. Approved by the owner on 17 Oct 2026 (section 10).
> **Scope:** the outputs of the scoring pipeline as extended by Commit 11 (business decision support).
> **Replaces:** v1, revision 5 (approved 11 Oct 2026). v2 is the reviewed amendment that v1 required for Commit 11. It replaces v1 in this file; v1 stays in the Git history, and its review record is kept in section 10.1.
> **Implemented by:** Commit 11. `tests/test_inference.py` asserts conformance to this document (section 9).
> **Changes after approval:** only by a reviewed amendment (section 8).

Every statement in this document restates an approved decision. Section 10 records the owner's review decisions.

---

## 1. Purpose and terminology

This contract is the business-facing specification of what the platform emits when it scores customers: the files, their exact structure, and the guarantees a reader can rely on. It does not prescribe internal code structure.

| Term | Meaning |
|---|---|
| scored table | the customer table given to the scoring function |
| n | the number of customers in the scored table |
| K | the fraction of scored customers to flag, from configuration (`business.k`, currently 0.10). It must be above 0 and below the Medium edge (section 4.4) |
| flag count | the number of customers flagged, `max(1, round(n × K))` |
| Medium edge | the fraction of scored customers up to which the Medium tier reaches, from configuration (`business.risk_tiers`, currently 0.25). It is above K and at most 1 (section 4.4) |
| Medium count | the number of customers in the High and Medium tiers together, `max(1, round(n × Medium edge))` |
| risk tier | `High`, `Medium` or `Low`, assigned from `rank` (section 4.4) |
| driver | one of the input features that moves a customer's score most, in either direction (section 4.5) |
| label map | the fixed plain-language label of every input feature (section 4.5) |
| output files | the four files in section 3.1 |
| model run | one training run and its model, identified by a model run ID |
| scoring run | one execution of scoring, identified by its own scoring run ID |

Run IDs are UTC timestamps in the form `YYYYMMDDTHHMMSSZ`.

## 2. Snapshot notice

The reference run (`make score`) scores the **1,408-customer held-out test split**. That is a historical snapshot used to demonstrate the pipeline. It is **not** a live list of customers to contact.

The scoring function itself accepts any table that passes validation (section 6) and never receives or reads target labels. Only the reference run is restricted to the test split.

The two Markdown documents (section 3.1) describe the same snapshot. What they say is specified in ADR-0021 and the Commit 11 documentation, not in this contract.

## 3. Outputs and publication

### 3.1 Files

All four files are written to `outputs/`. The paths are fixed constants of this contract and are not configurable. They are Git-ignored like the rest of `outputs/`.

| File | Purpose |
|---|---|
| `outputs/prioritized_outreach_list.csv` | One row per scored customer, ranked by churn risk, with risk tier and drivers (section 4) |
| `outputs/economics_summary.md` | The economics document of the scoring run: expected value, break-even and sensitivity under stated assumptions. Plain Markdown |
| `outputs/business_recommendations.md` | The business recommendations document of the scoring run. Plain Markdown |
| `outputs/run_summary.json` | Description of the scoring run that produced the other files, and an index to them (section 5) |

`outputs/` is the **latest published output set**, not an archive.

### 3.2 Replacement and retention

- When publication succeeds, the new output files replace the previous ones. **Publication succeeding and the command succeeding are different things.** Publication succeeds when all four files have been replaced. The command succeeds (exit code 0) only if publication succeeded **and** the run's required tracking also completed. If tracking fails after publication, the new outputs stay in place and the command still fails (class 3, section 3.5).
- Earlier outputs are **not retained** once new ones have been published. The tracking folder of a scoring run (`artifacts/runs/score/<scoring_run_id>/`) keeps that run's configuration, metrics and log, **not a copy of any output file**. Anyone who needs earlier outputs must copy them before the next run. Run-specific output retention is outside Commit 11.
- Input validation happens **before** any output is published.
- Each file is replaced as a whole: at its final path a reader sees either the previous complete file or the new complete file, never a partly written one.
- The four files are **not** replaced atomically. Separate files cannot be swapped in one step, and this contract does not require a transaction mechanism.
- **No backup and no rollback.** Once a file's own destination has been replaced, its previous version is not kept anywhere, even if publication then fails (class 2b). A failed run does not restore it, and the run does not retry.

### 3.3 Publication sequence and checks

All four outputs are built in memory before any file is written or replaced. Then:

1. **Temporary files.** Each output is written to a temporary file next to its final path. If any of these writes fails, nothing has been replaced (class 2a).
2. **Replacement in a fixed order:** (1) the outreach CSV, (2) the economics document, (3) the recommendations document, and (4) `run_summary.json` **last**. The result of each replacement (replaced or not) is recorded.
3. **Stop at the first failure.** If a replacement fails, the later files are not tried and nothing is rolled back (class 2b).
4. **Checks before the summary is replaced.** After the three deliverables have been replaced and before `run_summary.json` is replaced:
   - each of the three deliverable paths exists and points to a file, in the folder this run published to (in tests, a temporary folder);
   - the two Markdown documents are **read back from disk**, and each carries the scoring run ID and the model run ID in the fixed places of section 3.4, equal to the IDs of the summary about to be written. An ID found anywhere else in the text does not count.

   If a check fails, the failure is class 2b, no new summary is published and the previous summary stays unchanged.
5. **Tracking.** The run's `metrics.json` is written last, after all four outputs are published.

**What these checks prove, and what they do not.** A recorded successful replacement shows that this run replaced that file. The existence check alone shows only that something is at the path: an old file passes it. The run-ID read-back shows that the two documents agree with the summary of this run. **The CSV has no run-ID field**, so its provenance rests on the recorded replacement result and not on its contents. If the CSV replacement fails, the failure report says it was not replaced, even if an old CSV exists. The contract does not claim that the output folder is consistent after a failed run.

### 3.4 Run IDs in the Markdown documents

Both Markdown documents begin with the same fixed header, so that the run-ID check reads exactly these places:

```
# <document title>

Scoring run ID: <scoring_run_id>
Model run ID: <model_run_id>
```

Line 1 is the title, line 2 is empty, line 3 is the scoring run ID line and line 4 is the model run ID line, each exactly in the form shown, with a run ID in the form of section 1. The check reads lines 3 and 4 only.

### 3.5 Failure behaviour

Every failure below ends the command with a **non-zero exit code** and a clear message. There are three classes: input validation failure (1), output generation or publication failure (2, split into 2a and 2b), and post-publication tracking failure (3). They are distinct.

| Class | When it happens | Effect on the output files | Message reports |
|---|---|---|---|
| **1. Input validation failure** | The scored table breaks a rule in section 6, before any output is generated | **No new file is written. Existing output files stay unchanged.** | The failed condition, naming the column and row where meaningful |
| **2a. Output generation failure** | Before any output file is replaced, and not an input validation failure (for example no valid model found; the scoring run's tracking folder, configuration snapshot or log file cannot be created at the start of the run; an error while scoring or building any of the four outputs; a driver compatibility error, section 4.5; an invalid K or Medium edge, section 4.4; a failure to write a temporary file) | No new file is written. Existing output files stay unchanged. | What failed |
| **2b. Output publication failure** | While the files are being replaced (for example a file is locked by another program), or a check before the summary (section 3.3) fails | Each file is old or new as a whole. Because publication is not atomic, some files can be new while others are still the previous ones. Nothing is restored. If the failure happens before the summary is replaced, the previous summary is unchanged | **For each of the four outputs, whether this run replaced it** (from the recorded result of each replacement, never inferred from a file being present), and that the output set may be inconsistent until a successful run. The user investigates and reruns `make score` if appropriate |
| **3. Post-publication tracking failure** | All four output files were published (publication succeeded), then a required tracking write failed, so the command as a whole fails | **All four new files are kept.** | That the outputs were generated but tracking failed, and the tracking error |

A scoring run whose required work or tracking did not complete leaves a tracking folder without `metrics.json`. (If the folder itself could not be created, there is no folder.)

## 4. Prioritized Outreach List (`prioritized_outreach_list.csv`)

### 4.1 Format

UTF-8, comma-separated, one header row, no index column. `in_top_k` is written as `True` or `False`. `churn_probability` is written without deliberate rounding by the output layer, so the available prediction precision is preserved. No cell is empty. The CSV writer quotes every field as CSV syntax requires, so a field that contains a comma, a quotation mark or a line break is written correctly and reads back unchanged. The labels (section 4.5) and the other fixed text values contain none of these characters, so in practice only `customer_id`, which this contract does not restrict beyond section 6, can need quoting.

### 4.2 Columns

Exactly these eleven columns, in this order. The first four are the v1 columns and keep their names, types, order and meanings. The other seven are appended after them. No other column is present in v2.

| Column | Type | Meaning |
|---|---|---|
| `customer_id` | text | Customer identifier, as it appears in the scored table |
| `churn_probability` | number, 0 to 1 | Calibrated probability of voluntary churn from the selected model |
| `rank` | integer, 1 to n | Position by risk. 1 is the highest churn probability. Unique within the file |
| `in_top_k` | boolean | `True` for the customers flagged for outreach (`rank` ≤ flag count) |
| `risk_tier` | text | `High`, `Medium` or `Low` (section 4.4) |
| `driver_1` | text | The strongest driver of this customer: a label from the label map |
| `driver_1_direction` | text | `raises risk` or `lowers risk` |
| `driver_2` | text | The second driver: a label from the label map |
| `driver_2_direction` | text | `raises risk` or `lowers risk` |
| `driver_3` | text | The third driver: a label from the label map |
| `driver_3_direction` | text | `raises risk` or `lowers risk` |

### 4.3 Row guarantees

1. **One row per scored customer.** Every customer in the scored table appears exactly once. None is dropped or added.
2. **Unique, non-empty `customer_id`.** A table with duplicate or empty customer IDs fails validation (class 1). Together with the other column rules this means no CSV cell is ever empty.
3. **Rank order.** Rows are written in ascending `rank`, which is non-increasing `churn_probability`.
4. **Tie-breaking.** Customers with equal `churn_probability` are ordered by `customer_id` ascending, as text. As a result `rank` has no ties.
5. **Flag count.** Exactly `k = max(1, round(n × K))` rows have `in_top_k = True`, namely the rows with `rank` ≤ k. `round` is Python's `round()`: exact halves go to the nearest even integer (25 × 0.10 = 2.5 flags 2). Flagging is by `rank`, so a probability tie across the boundary cannot flag more or fewer than k customers.
6. **Tiers follow rank.** Every row's `risk_tier` follows from its `rank` by the rule in section 4.4. `risk_tier` is `High` exactly when `in_top_k` is `True`.
7. **Three different drivers.** Every row has exactly three drivers, all different, in the order of section 4.5. Every direction cell is one of the two allowed values.
8. **Reference run.** n = 1,408 and K = 0.10 give k = 141 flagged customers, and 141 High, 211 Medium and 1,056 Low customers.

### 4.4 Risk tiers

Tiers are rank-based and relative to the scored table. They are not probability thresholds, and the contract defines no probability cut-off that changes an action.

| Tier | Rows |
|---|---|
| `High` | `rank` ≤ k (the flagged customers, `in_top_k = True`) |
| `Medium` | k < `rank` ≤ m, where m is the Medium count |
| `Low` | `rank` > m |

The three tiers always add up to n. For a very small n, the Medium tier can be empty. The Medium edge of 0.25 is an explicit project assumption (the largest K in ADR-0010's sensitivity range), not a previously locked business rule.

**Configuration rule.** The two settings must satisfy

```
0 < K < Medium edge <= 1
```

With the current Medium edge of 0.25, K must be below 0.25. The current K = 0.10 and Medium edge = 0.25 satisfy the rule, and nothing about their behaviour changes. The rule is strict on purpose: if K were equal to or above the Medium edge, the High tier would reach or pass the point where the Medium tier ends, and the tiers would no longer describe a High tier followed by a Medium tier. Because rounding never reverses the order of two fractions, K below the Medium edge guarantees that the Medium count m is never below the flag count k, for every n.

A configuration that breaks the rule stops the scoring run before any output is generated (class 2a, section 3.5). No tier is assigned from an invalid pair, and the existing output files stay unchanged. The rule covers `business.k` and the Medium edge only. The evaluation sweep `business.k_sensitivity` (which includes 0.25) is a separate setting and is not constrained by it.

### 4.5 Drivers

A driver describes which input features moved the model's score for this customer, and in which direction. The rules:

1. **Source.** For each of the five fitted XGBoost models inside the calibrated model, the model's own per-feature contributions are computed. The contributions of the encoded columns are added up to the 42 original input features, and the five models' results are averaged. No churn label is used.
2. **Selection.** The three features with the largest **absolute** averaged contribution are the customer's drivers, in either direction. `driver_1` has the largest, `driver_3` the third largest. All 42 input features are eligible: a feature is not left out because the business can or cannot act on it.
3. **Equal sizes.** If two features have the same absolute averaged contribution, the one that comes first in the model schema's column order ranks first.
4. **Direction.** `raises risk` if the averaged contribution is above zero, `lowers risk` if it is below zero. Direction is measured against the model's average prediction.
5. **Zero contribution.** A selected driver whose averaged contribution is exactly zero has no direction. If this occurs for any customer, the run stops before anything is published (class 2a). No filler value is written.
6. **Labels.** Every driver is shown as a fixed plain-language label from the label map, with the direction in its own column. The CSV carries no contribution numbers and no customer feature values. The label map has exactly one label for every input feature in the model schema. Labels are non-empty, are different from each other, and contain no comma, quotation mark or line break. Labels of related features stay clearly different (for example `contract` and `is_month_to_month`). A test fails if any input feature lacks a label.
7. **Compatibility check.** The driver calculation relies on the structure of the loaded model: five fitted XGBoost estimators; a fitted column transformer whose number of encoded columns equals the feature count of each estimator; and an assignment of the encoded columns to input features in which every encoded column has an owner, every owner is an input feature, and every input feature owns at least one encoded column. The current model has 55 encoded columns and 42 input features. The scoring code checks this structure when it runs. If it does not hold, the run stops with a clear message before anything is published (class 2a). The contract does not require that a future model has this structure; it requires that an incompatible one is not used silently.
8. **Interpretation caveat.** Drivers are **model associations, not proven causes**, and they do not show that a retention offer will work. The contributions are in the model's raw-score (log-odds) units, averaged over five fold models, and are not an exact breakdown of the calibrated probability. Related features share credit. They are measured against the model's average prediction. This caveat is stated here and in the recommendations document, and is not repeated in every CSV row.

### 4.6 Determinism

The same scored table, model, configuration and implementation should produce the same ranking, flags and tiers, assuming the same runtime behaviour (library versions and numerical behaviour on the machine). The contract does not claim bit-identical probabilities or contributions across different environments. It does guarantee that, whatever the probabilities are, equal probabilities are ordered by `customer_id`, flags follow `rank`, tiers follow `rank`, and equal driver sizes are ordered by the schema column order.

## 5. Run summary (`run_summary.json`)

A single JSON object. In v2 it contains exactly the fields below, at each level: no field may be omitted, null or added. A new field requires a reviewed amendment (section 8). No library versions, schema hash or other detailed model metadata appear here; they stay in the model's `manifest.json`. The summary is the last output file published (section 3.3).

### 5.1 Top-level fields

| Field | Type | Required | Meaning |
|---|---|---|---|
| `contract_version` | string | yes | `"2"` |
| `k` | number | yes | The K used in this scoring run, from configuration at scoring time. A fraction with 0 < K < Medium edge (section 4.4); with the current Medium edge of 0.25, K is below 0.25 |
| `customers_scored` | integer | yes | n, at least 1. Equals the number of data rows in the CSV |
| `customers_flagged` | integer | yes | The flag count. Equals the number of rows with `in_top_k = True` |
| `model_run_id` | string | yes | Model run ID of the model used for scoring |
| `scoring_run_id` | string | yes | Run ID of this scoring run. Equals the name of its tracking folder |
| `snapshot_notice` | string | yes | Exactly: `Historical snapshot: this list scores the held-out test split for demonstration. It is not a live outreach list.` |
| `model_health` | object | yes | Information about the model, copied from its manifest (section 5.2) |
| `output_files` | object | yes | The repository-relative paths of the three deliverable files (section 5.4) |

The fields `contract_version` through `model_health` keep their v1 meanings. `output_files` is the one new field, appended after the others. The summary has **no** `risk_tiers` object and no tier counts: they can be derived from the CSV.

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

### 5.4 `output_files`

| Field | Type | Required | Value |
|---|---|---|---|
| `outreach_list` | string | yes | `outputs/prioritized_outreach_list.csv` |
| `economics_summary` | string | yes | `outputs/economics_summary.md` |
| `business_recommendations` | string | yes | `outputs/business_recommendations.md` |

The paths are repository-relative, with the `outputs/` prefix, and are resolved against the repository root. They are the same in every run. The summary does not list itself, and it does not list model bundles, tracking files or other intermediate artifacts. Its value is discoverability: it says which scoring run and which model run the files in the folder come from, subject to the publication rules in section 3.3.

### 5.5 Illustration

The values are illustrative, not the output of a real run. Real values come from the scoring run and the model manifest.

```json
{
  "contract_version": "2",
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
  },
  "output_files": {
    "outreach_list": "outputs/prioritized_outreach_list.csv",
    "economics_summary": "outputs/economics_summary.md",
    "business_recommendations": "outputs/business_recommendations.md"
  }
}
```

The first rows of the CSV look like this. The IDs, probabilities and labels are illustrative; real labels come from the label map.

```
customer_id,churn_probability,rank,in_top_k,risk_tier,driver_1,driver_1_direction,driver_2,driver_2_direction,driver_3,driver_3_direction
AAAA-00001,0.94,1,True,High,<label A>,raises risk,<label B>,raises risk,<label C>,raises risk
AAAA-00002,0.93,2,True,High,<label A>,raises risk,<label C>,raises risk,<label D>,lowers risk
```

## 6. Input accepted by scoring

The scoring function accepts a scored table with `customer_id` plus the model's input columns. It never receives or reads target labels. The validator fails loudly and runs **before any output is generated**; a validation failure is a class 1 failure (section 3.5). **This part is unchanged from v1.**

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

## 7. Non-deliverables of v2

v2 now provides risk tiers, customer-level drivers, and the economics and business recommendations documents. v2 deliberately does **not** provide:

- ROI or any measured campaign result. The economics document reports simulated results under stated assumptions, not measured outcomes
- per-customer economics columns in the CSV, or economics fields in the run summary
- any evaluation of model quality on the scored customers
- probability-based tier columns, or probability or expected-value cut-offs that change an action
- any claim that a driver is a cause, or that a retention offer will work
- an archive of earlier outputs (run-specific output retention)
- records of actual customer contact or outcomes

## 8. Amendment policy and the v2 rule

- Any change after approval is a reviewed amendment, made in this document before it is implemented.
- Commit 11 extends the **same** `prioritized_outreach_list.csv` under v2. No second customer-list file is created.
- **Compatibility rule for v2 (approved, and met by this document):** the four v1 columns keep the same names, types, order and meanings, and the new columns are appended after them.
- v1 forbids unapproved additions to the run summary. v2 changes the run summary only through this review: the version string and the one new field, `output_files`.
- The extension of publication from two files to four (section 3.3) and the new failure report (section 3.5) are part of this same review.

## 9. Conformance

`tests/test_inference.py` asserts, on the reference run:

**Outreach list**

- the exact eleven column names and order; one row per scored customer; unique `customer_id`
- `rank` is 1 to n without gaps or ties; rows are in ascending `rank`; `churn_probability` is within 0 to 1 and non-increasing down the file
- ties are ordered by `customer_id` ascending as text
- exactly `max(1, round(n × K))` rows have `in_top_k = True`, and they are the first rows
- `in_top_k` is written as `True` and `False`; `churn_probability` is not rounded by the output layer
- no cell in the CSV is empty
- `risk_tier` is `High` exactly when `in_top_k` is `True`; the tier counts follow section 4.4 (141 High, 211 Medium, 1,056 Low on the reference run) and add up to n
- every customer has three different non-empty drivers; every driver is a label of the label map; every direction cell is `raises risk` or `lowers risk`
- the drivers of a customer are the three largest absolute averaged contributions, in order, with ties by schema column order (checked against an independent selection)
- the label map has a label for every input feature of the model schema, and its labels are unique and free of commas, quotation marks and line breaks

**Configuration rule**

- the configuration `0 < K < Medium edge <= 1` is accepted for the current values (K = 0.10, Medium edge = 0.25)
- a configuration with K equal to or above the Medium edge, with K at or below 0, or with a Medium edge above 1 is rejected with a message that names the failed condition; the run stops before any output is generated (class 2a), and pre-existing output files are unchanged byte for byte

**Run summary**

- the key set at every level is exactly the one in section 5, with the stated types and no extra or missing fields; `contract_version` is `"2"`
- `customers_scored` and `customers_flagged` agree with the CSV
- `model_run_id` names an existing model run and `scoring_run_id` names the scoring run's tracking folder
- the held-out test numbers do not appear in the summary, and `held_out_results_reference` points to ADR-0014
- `output_files` has exactly the three keys and values of section 5.4, and the three paths exist and point to files after publication
- the summary has no `risk_tiers` object

**Markdown documents**

- both documents begin with the header of section 3.4, with the scoring run ID and the model run ID of the run

**Label isolation**

- a table containing the target column is rejected
- the table handed to the scoring function has exactly `customer_id` plus the model's input columns
- scoring and the driver calculation never read a label column

**Driver compatibility**

- the driver calculation stops with a clear message, before anything is published, if the loaded model does not have five fitted XGBoost estimators, if the number of encoded columns produced by the preprocessor differs from the feature count of any estimator, or if the encoded columns cannot be fully assigned to input features (an encoded column without an owner, an owner that is not an input feature, or an input feature that owns no encoded column)
- the contribution additivity, the top-three selection and the batch invariance are tested in the test suite, not at run time

**Validation and failure behaviour**

- each rejection rule in section 6 fails with a message that names the failed condition, and names the column and row where meaningful; this includes a missing `customer_id`, an empty-string `customer_id` and a whitespace-only `customer_id`
- after a validation failure, none of the four output files is written, and pre-existing output files are unchanged byte for byte
- after a failure before publication starts (class 2a), existing output files are unchanged
- if the scoring run's tracking folder, configuration snapshot or log file cannot be created at the start of the run, the command ends with a non-zero exit before any output file is written or replaced, and pre-existing output files are unchanged byte for byte (class 2a)
- a failure to write any temporary file ends the run as class 2a, with no output file replaced
- if outputs are published and the tracking write then fails, all four outputs are kept, the exit is non-zero, and the message says the outputs were generated but tracking failed
- an unsuccessful run leaves no `metrics.json` in its tracking folder

**Per-file replacement and publication order**

- the files are replaced in the order CSV, economics document, recommendations document, summary
- a simulated failure at **each** of the four replacements (class 2b) stops publication at that file, leaves the files before it new and complete and the files from it onward as they were, never leaves a partly written file at a final path, ends with a non-zero exit, and the message states for each of the four outputs whether this run replaced it, and says the output set may be inconsistent; nothing is restored
- a simulated failure of the checks before the summary (an expected file missing; a document whose run IDs disagree with the summary, or are missing) is reported as class 2b and leaves the previous summary unchanged
- the success path publishes all four files, with the summary written last
- publication succeeding while the command fails (a tracking failure after all four files are published) keeps all four new complete files and ends with a non-zero exit

**Retention**

- a successful run replaces the previous output files, and its tracking folder holds no copy of any of them

---

## 10. Review record

### 10.1 Review record of v1 (kept)

Owner decisions taken in the review of v1 revision 2 (11 Oct 2026), now part of this contract:

1. **Per-file whole replacement and publish order** (sections 3.2, 3.5): approved. Each file is replaced as a whole. The limitation that the files are not replaced atomically stays explicit.
2. **`snapshot_notice` wording** (section 5.1): approved as written.
3. **Run-summary names and formats** (sections 5.2, 5.3): approved, including the ISO 8601 UTC timestamp with `Z`, the field names, the metric keys and the `estimate_type` wording.

v1 revision 3 also resolved two consistency points raised in that review: publication success versus command success, and empty `customer_id` values.

Owner approval of v1 revision 3 (11 Oct 2026), including the rejection of missing, empty-string and whitespace-only `customer_id` values. The owner added one clarification: a previous output file is not retained when its own destination is replaced, even if publication then fails. Revision 4 changed only the status line and this clarification.

Owner approval of v1 revision 5 (11 Oct 2026): a failure to create the scoring run's tracking folder, configuration snapshot or log file at the start of the run is class 2a. Class 2a is defined by the failure happening before any output file is replaced, excluding input validation failures; this also settles that "no valid model found" belongs to 2a.

### 10.2 Review record of v2

Owner decisions of the Commit 11 planning (15 to 17 Oct 2026) that this contract restates:

| Subject | What it fixes in this contract |
|---|---|
| Deliverables | The extended CSV, one economics document, one recommendations document (sections 3.1, 4) |
| Risk tiers | The tier rule, the Medium edge and the configuration rule `0 < K < Medium edge <= 1` (section 4.4) |
| Drivers | How drivers are calculated, selected and shown (section 4.5) |
| Economics | No economics in the CSV or the run summary (section 7) |
| CSV columns | The eleven columns, their order and allowed values (section 4.2) |
| Run summary | Version `"2"`, the `output_files` object, its keys and paths, no `risk_tiers`, written last (section 5) |
| File names and Git policy | The fixed paths; `outputs/` is the latest set, not an archive (sections 3.1, 3.2) |
| Publication | Order, per-file replacement, failure report, checks before the summary, run IDs in the documents (sections 3.3 to 3.5) |
| Driver compatibility | The run-time check of the model structure the driver calculation relies on (section 4.5, point 7) |

The module layout of the new code has no effect on this contract.

Points settled in the drafting of v2 and approved by the owner on 17 Oct 2026:

1. `output_files` sits after `model_health` in the run summary (section 5.1).
2. The run IDs in the two documents are on fixed lines: title on line 1, an empty line 2, the scoring run ID on line 3, the model run ID on line 4 (section 3.4).
3. The contract states the rules of the label map. The 42 labels and their test live in the code and are reviewed at implementation (section 4.5, point 6).
4. Equal absolute driver contributions are ordered by the model schema's column order (section 4.5, point 3).
5. An exactly zero averaged contribution for a selected driver stops the run before publication, class 2a (section 4.5, point 5).
6. A driver compatibility failure is class 2a (sections 3.5 and 4.5).
7. All four outputs are built in memory and all four temporary files are written before the first replacement. A temporary-file write failure is class 2a. Publication stops at the first replacement failure, with no rollback (section 3.3).
8. Determinism covers tiers and drivers, with no promise of bit-identical numbers across environments (section 4.6).
9. The Medium tier can be empty for a very small n (section 4.4).
10. Only the header of the two Markdown documents is specified here (section 3.4). Their sections, wording rules and evidence rules are in ADR-0021 and the Commit 11 documentation. The interpretation caveat is stated here once (section 4.5, point 8) and recorded in ADR-0020.

**Correction to the first draft (owner, 17 Oct 2026).** The first draft allowed K up to 1, while the Medium edge is 0.25 and must be above K. The contract now states `0 < K < Medium edge <= 1` (terms, sections 3.5, 4.4, 5.1 and 9). With the current Medium edge of 0.25, K must be below 0.25. This is a consistency correction. The approved tier design, the 25% Medium edge and the current K = 0.10 behaviour are unchanged. The owner confirmed three consequences:

- an invalid K or Medium edge is a class 2a failure: the run stops before any output is generated;
- where the check runs and the message wording are implementation details, and implementing it replaces the v1 bound `0 < K <= 1` in the configuration check;
- `business.k_sensitivity` (which includes 0.25) is an evaluation and economics sweep, not the setting that defines the tier boundaries, and is outside this rule. The approved sweep is unchanged.

**Approval:** owner approval of v2, revision 1 (17 Oct 2026).

**Revision 2 (17 Oct 2026), a reviewed amendment of the approved revision 1.** Section 4.1 said that no cell needs quoting because text values cannot contain a comma, a quotation mark or a line break. That is true of the labels and the other fixed text values, but section 6 does not restrict the characters of `customer_id`. The owner chose not to add a customer-ID restriction. Section 4.1 now says that the CSV writer quotes fields as CSV syntax requires. Revision 2 changes only the status line, section 4.1 and this record. No other rule, column or guarantee changes.

**Approval:** owner approval of v2, revision 2 (17 Oct 2026). The v1 review record is kept in section 10.1, and v1 stays in the Git history.

**Wording correction (19 Oct 2026), before the merge to `main`.** The driver compatibility wording in section 4.5, point 7, and in section 9 was corrected to describe the check the scoring code performs: the encoded-column count is compared with the feature count of each estimator, and the earlier wording that required the count to be 55 was removed because the code never required that number. No output field, file or published behaviour changed, and the contract version stays "2".
