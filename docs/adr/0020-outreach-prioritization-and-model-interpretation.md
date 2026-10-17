# ADR 0020: Outreach Prioritization and Model Interpretation

**Status:** Accepted

---

## Context

The Commit 10 outreach list gives each customer a calibrated churn probability, a rank and a flag (`in_top_k`). Commit 11 adds two things on top of it: a coarser **risk tier** so the list can be read in groups, and a short **explanation** of what moved each customer's score.

The selected model is a tuned XGBoost with sigmoid calibration. Calibration builds five fitted model-and-calibrator pairs (5 folds × 1 repeat), the tuned estimator is only a template, and the final probability is the mean of five calibrated probabilities (ADR-0013, ADR-0016). Any explanation has to be derived from those five fitted models. It also has to avoid a new dependency (the locked scope), use no customer labels, and not claim more than a model can support.

The Commit 5 notes left one interpretation question open: how to read the Fiber Optic internet type in a model that also uses charges and other related features. This ADR records the diagnostic evidence for that question.

Platform Output Contract v2 holds the exact CSV schema, the allowed values and the failure behaviour. This ADR records the decisions and the reasons behind them.

---

## Decision

### 1. Risk tiers are rank-based

- **High** = the flagged top-K customers, exactly `in_top_k` (K is read from `business.k` and not stored as a second number).
- **Medium** = the customers after the top K, through the top 25% of the scored population.
- **Low** = everyone outside the top 25%.

Boundary counts use `max(1, round(n × fraction))`, the same rule as the flag count (ADR-0017). For the 1,408-customer reference run this gives 141 High, 211 Medium and 1,056 Low customers. The 25% edge is an explicit project assumption: it is the largest K in ADR-0010's sensitivity range, and it is not a previously locked business rule. Only the Medium edge is added to the configuration (`business.risk_tiers`), and the configuration must satisfy `0 < K < Medium edge <= 1`, so a configuration in which the Medium tier would end before the High tier does is rejected before scoring.

`churn_probability` stays a numeric column. There is no probability-based tier column and no probability cut-off that changes an action in Commit 11; such thresholds may be revisited if the economics (ADR-0019) justify them. The economics document reports the mean calibrated probability per tier.

### 2. Customer-level drivers come from XGBoost's own contributions

For each of the five fitted XGBoost models inside the calibrated model, the built-in `pred_contribs` (a TreeSHAP-style calculation) gives each encoded column's signed contribution to the raw score. The contributions of the 55 encoded columns are summed back to the 42 original input features, and the five models' results are averaged. The `shap` library is not added, and no label is used.

### 3. How drivers are shown

- Each customer gets **exactly three drivers**: the three input features with the largest **absolute** averaged contribution, in either direction, strongest first.
- Only a **plain-language label** and a **direction** (`raises risk` or `lowers risk`) are shown. There are no contribution numbers and no customer feature values.
- **All 42 features are eligible.** A feature is not left out because the business can or cannot act on it. Actionability is discussed in the recommendations document (ADR-0021).
- A **complete label map** gives every input feature one fixed label, with a test that fails if any feature of the model schema has none. Labels of related features must stay clearly different (for example `contract` and `is_month_to_month`).
- Equal absolute sizes are ordered by the model schema's column order. A selected driver whose averaged contribution is exactly zero has no direction, so the run stops before publication instead of writing a filler value.
- Direction is measured against the model's average prediction.

### 4. Global importance is a separate measure

For the recommendations document, global importance uses XGBoost's built-in `total_gain`. In each of the five models, `total_gain` is summed over the encoded columns of the same input feature and turned into a share of that model's total gain. The five shares per feature are then averaged. Global gain importance and customer-level contributions are different measures and are not compared as if on one scale.

### 5. The CSV carries seven flat text columns

The seven columns are appended after the four v1 columns: `risk_tier`, then `driver_n` and `driver_n_direction` for n = 1 to 3. Each fact is an explicit field with its own column (the tier, each driver and each direction), so filtering and validation work column by column and need no text splitting, delimiters or parsing. The extra width is an accepted trade-off. Because every customer has exactly three drivers, no cell is empty.

### 6. Fiber Optic: the diagnostic evidence

A one-time, read-only, label-free diagnostic on the reference model run (`20261014T164739Z`) measured how much the model relies on the Fiber Optic indicator:

- `internet_type` has an average gain share of 1.152% (0.831% to 1.447% over the five models) and ranks 20th of 42 input features (19th to 23rd per model).
- Its Fiber Optic encoded column averages 0.822% (0.568% to 1.101%), which is 71.3% of the feature's share, and ranks 26th of 55 encoded columns by the mean (22nd to 29th per model).
- `monthly_charge` has a share of 5.712% (4th of 42) and a correlation of 0.772 with the Fiber Optic column on the training features. `internet_service` has 0.004%.

Two different things must be kept apart. **Observed in the data:** Fiber Optic customers churn more (40.65%, against 25.66% for Cable and 18.48% for DSL, Commit 4); this is a difference between groups of customers. **Measured by the diagnostic:** how much the model relies on the Fiber Optic indicator when it predicts. The diagnostic does not explain the observed difference, and it does not establish an effect or a cause of Fiber Optic on churn. The spread across the five models is context, not proof of instability. The Commit 5 follow-up is therefore resolved at feature level: the recommendations document reports `internet_type` as one original feature, with the current run's share and rank and the correlation caveat, and no separate Fiber-specific percentage. These figures are evidence in this ADR only; the document never hard-codes them (ADR-0021).

### 7. Drivers are computed at scoring time

Customer drivers and global importance are computed at scoring time from the model returned by `load_run`. The existing model bundle format is unchanged, so no retraining and no new model run are needed. The driver module checks the structure it relies on at run time (five fitted XGBoost estimators, the 55 encoded columns, and a complete assignment of each encoded column to one input feature) and stops with a clear error if it does not hold. The detailed verification results are kept in the implementation tracker.

### 8. The interpretation caveat

This is the one place where the caveat is stated in full. Drivers and importance shares are **model associations, not proven causes**, and they do not show that a retention offer will work. Contributions are in the model's raw-score (log-odds) units, averaged over five fold models, and are not an exact breakdown of the calibrated probability. Correlated features share credit. The contract and the recommendations document carry the wording that their readers need, and the caveat is not repeated in every CSV row. ADR-0019 and ADR-0021 refer to this section.

---

## Alternatives considered

- **Probability-based tiers (fixed cut-offs).** Not adopted in Commit 11: no probability cut-off that changes an action is defined, and the economics may justify one later. Rank-based tiers stay relative to the scored table.
- **The `shap` library.** Not adopted: a new dependency for a calculation XGBoost already provides, and the locked scope adds none.
- **Top three risk-raising drivers only.** Rejected: a customer with very low risk can have fewer than three positive contributions, which forces empty or filler entries and conflicts with the no-empty-cell rule.
- **All 42 contributions as columns.** Rejected: too wide, and the log-odds numbers carry false precision.
- **Four columns, with the direction inside the cell (`Contract (raises risk)`), or one `top_drivers` text.** Rejected: filtering by direction needs text parsing, and the checks need more parsing rules. The single text is the most compact but the hardest to filter and test.
- **Filtering out features the business cannot act on.** Rejected: it would hide what the model uses. Actionability is discussed in the recommendations document.
- **Storing the encoded-to-input column map or the importance shares in the model bundle.** Rejected: it would change the bundle format, `persist.py`, the check that the bundle has exactly three keys, training and the manifest, and need a new model run, for a calculation that is cheap and already verified at scoring time.

---

## Consequences

- The tier counts and the driver rules are tested against the contract, including: `High` exactly when `in_top_k` is `True`; three non-empty, different drivers per customer; every direction cell one of the two values; and the label-map test.
- The 42 labels are written and reviewed at implementation. Labels must keep related features apart; `number of referrals` and `referred a friend` are an example that must stay distinct.
- Strong features such as `latitude`, `longitude`, `population` and `age` are not directly actionable for a retention team, but they may still appear as drivers. This is intended and is handled in the recommendations document.
- A future model with another structure fails the driver compatibility check until the new structure is reviewed. It is never explained silently.
- The CSV is wider (eleven columns), and the output validation and the contract tests change accordingly.
- The interpretation caveat has one home (section 8). A change to its meaning needs a new ADR, because ADRs are append-only.

---

**Decision Date:** 17 October 2026 (decisions approved 15 and 16 October 2026, the structure of this ADR and section 7 confirmed 17 October 2026; recorded here before implementation)
