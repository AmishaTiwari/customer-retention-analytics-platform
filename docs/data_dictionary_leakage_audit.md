# Data Dictionary & Leakage Audit

> **Status:** Living document during Commit 3; locks once the modeling dataset (`mv_churn`) is frozen.

This document records the disposition of every raw column across the five source tables, and the engineering rationale behind each classification. It formalizes decisions already made during Dataset Understanding (Stage 2), ADR-0006, and the SQL data preparation layer — it does not introduce new classifications here.

**Dispositions used:**
- **Feature** — available at scoring time, included in the modeling dataset (`mv_churn`).
- **Identifier** — a key used for joins or row identity, not a predictive feature.
- **Target Construction Only** — used to build or directly represent the modeling target; excluded from the feature set.
- **Excluded (Leakage)** — only known after churn has occurred; not realistically available at scoring time.
- **Excluded (Other)** — excluded for reasons unrelated to leakage (redundancy, constant value, non-customer identifier).

---

## Demographics

| Column | Type | Disposition | Rationale |
|---|---|---|---|
| Customer ID | VARCHAR | Identifier | Join key across all customer-level tables. |
| Count | INTEGER | Excluded (Other) | Constant reporting/dashboarding artifact from the source BI tool; carries no customer information. |
| Gender | VARCHAR | Feature | Static customer attribute, available at scoring time. |
| Age | INTEGER | Feature | Static customer attribute. |
| Under 30 | BOOLEAN | Feature | Static customer attribute; converted from raw Yes/No text to BOOLEAN in `sql/02_cleaning/`. |
| Senior Citizen | BOOLEAN | Feature | Same. |
| Married | BOOLEAN | Feature | Same. |
| Dependents | BOOLEAN | Feature | Same. |
| Number of Dependents | INTEGER | Feature | Static customer attribute. |

---

## Location

| Column | Type | Disposition | Rationale |
|---|---|---|---|
| Customer ID | VARCHAR | Identifier | Join key. |
| Count | INTEGER | Excluded (Other) | Constant reporting artifact. |
| Country | VARCHAR | Feature | Static attribute (single value in this dataset, retained for structural completeness). |
| State | VARCHAR | Feature | Static attribute (single value in this dataset). |
| City | VARCHAR | Feature | Static customer attribute. |
| Zip Code | VARCHAR | Feature | Static customer attribute; also the join key into Population. |
| Lat Long | VARCHAR | Excluded (Other) | Redundant once Latitude/Longitude are available as separate numeric columns. |
| Latitude | DOUBLE | Feature | Static customer attribute. |
| Longitude | DOUBLE | Feature | Static customer attribute. |

---

## Population

| Column | Type | Disposition | Rationale |
|---|---|---|---|
| ID | INTEGER | Excluded (Other) | This table's own row identifier; not meaningful at the customer level. |
| Zip Code | VARCHAR | Identifier | Join key into Location; not a predictive feature itself. |
| Population | INTEGER | Feature | Zip-code-level population figure, joined onto each customer via Location. |

---

## Services

| Column | Type | Disposition | Rationale |
|---|---|---|---|
| Customer ID | VARCHAR | Identifier | Join key. |
| Count | INTEGER | Excluded (Other) | Constant reporting artifact. |
| Quarter | VARCHAR | Excluded (Other) | Constant value ("Q3") across all rows in this snapshot; zero variance, no predictive value. |
| Referred a Friend | BOOLEAN | Feature | Service/behavioral attribute, available at scoring time; converted from raw Yes/No text in `sql/02_cleaning/`. |
| Number of Referrals | INTEGER | Feature | Same availability reasoning. |
| Tenure in Months | INTEGER | Feature | Same. |
| Offer | VARCHAR | Feature | Same. |
| Phone Service | BOOLEAN | Feature | Same; boolean conversion as above. |
| Avg Monthly Long Distance Charges | DOUBLE | Feature | Same. |
| Multiple Lines | BOOLEAN | Feature | Same; boolean conversion as above. |
| Internet Service | BOOLEAN | Feature | Same; boolean conversion as above. |
| Internet Type | VARCHAR | Feature | Same. |
| Avg Monthly GB Download | INTEGER | Feature | Same. |
| Online Security | BOOLEAN | Feature | Same; boolean conversion as above. |
| Online Backup | BOOLEAN | Feature | Same; boolean conversion as above. |
| Device Protection Plan | BOOLEAN | Feature | Same; boolean conversion as above. |
| Premium Tech Support | BOOLEAN | Feature | Same; boolean conversion as above. |
| Streaming TV | BOOLEAN | Feature | Same; boolean conversion as above. |
| Streaming Movies | BOOLEAN | Feature | Same; boolean conversion as above. |
| Streaming Music | BOOLEAN | Feature | Same; boolean conversion as above. |
| Unlimited Data | BOOLEAN | Feature | Same; boolean conversion as above. |
| Contract | VARCHAR | Feature | Same availability reasoning. |
| Paperless Billing | BOOLEAN | Feature | Same; boolean conversion as above. |
| Payment Method | VARCHAR | Feature | Same availability reasoning. |
| Monthly Charge | DOUBLE | Feature | Same. |
| Total Charges | DOUBLE | Feature | Same. |
| Total Refunds | DOUBLE | Feature | Same. |
| Total Extra Data Charges | INTEGER | Feature | Same. |
| Total Long Distance Charges | DOUBLE | Feature | Same. |
| Total Revenue | DOUBLE | Feature | Same. |

---

## Status

| Column | Type | Disposition | Rationale |
|---|---|---|---|
| Customer ID | VARCHAR | Identifier | Join key. |
| Count | INTEGER | Excluded (Other) | Constant reporting artifact. |
| Quarter | VARCHAR | Excluded (Other) | Constant value ("Q3"). |
| Satisfaction Score | INTEGER | Feature | Empirically verified as fully populated across all `Customer Status` values (Stayed, Joined, Churned alike) during Dataset Understanding, Stage 2 — reflects ongoing customer sentiment, not an exit-only survey artifact. |
| Customer Status | VARCHAR | Target Construction Only | Directly represents the outcome the target is built from; including it as a feature would trivially expose the target to the model. |
| Churn Label | BOOLEAN | Target Construction Only | A direct restatement of Customer Status (Churned vs. not); converted to BOOLEAN in `sql/02_cleaning/`; same exposure risk as Customer Status. |
| Churn Value | INTEGER | Target Construction Only | Numeric restatement of Churn Label; same exposure risk. |
| Churn Score | INTEGER | Excluded (Leakage) | An internal churn-risk score. As with CLTV, this has not been empirically tested for availability the way Satisfaction Score was; it is excluded based on its business meaning — a "churn score" of this kind is presumed to be computed as part of, or closely tied to, the churn/exit process — rather than a confirmed finding that it is unavailable for active customers in this specific dataset. If this assumption is ever revisited, it should be verified empirically (e.g., checking whether Churn Score is populated for Stayed/Joined customers the same way Satisfaction Score was) before being reclassified. |
| CLTV | INTEGER | Excluded (Leakage) | Customer Lifetime Value. This has not been empirically tested for availability the way Satisfaction Score was; it is excluded based on its business meaning — CLTV figures of this kind are commonly computed or finalized in conjunction with a customer's outcome — rather than a confirmed finding that it is retrospective in this specific dataset. If this assumption is ever revisited, it should be verified empirically (e.g., checking whether CLTV values are stable over time, or whether they correlate suspiciously with churn outcome) before being reclassified. |
| Churn Category | VARCHAR | Target Construction Only | Used, together with Churn Reason, to construct `is_voluntary_churn` per the mapping decided in ADR-0006. Only populated for churned customers; excluded from the feature set once the target is built. Not itself a leakage column in the temporal sense — it is a target-construction input. |
| Churn Reason | VARCHAR | Target Construction Only | Same as Churn Category — the granular input to the ADR-0006 mapping, excluded from features once the target is constructed. Not itself a leakage column in the temporal sense — it is a target-construction input. |

---

## Summary

- **34 raw columns** across 5 source tables.
- **44 columns** carried into the final modeling dataset (`mv_churn`): 42 features, plus the `customer_id` identifier and the `is_voluntary_churn` target. See `sql/04_modeling_view/01_mv_churn.sql` for the exact join and selection logic.
- **7 columns excluded from the modeling feature set** — but these fall into three distinct categories, not one:
  - **2 post-outcome / temporal leakage columns** (only knowable after churn has occurred): `Churn Score`, `CLTV`
  - **3 target-derived columns** (would trivially expose the target if included as features): `Customer Status`, `Churn Label`, `Churn Value`
  - **2 target-construction inputs** (used to build the target per ADR-0006, then excluded once the target exists — not leakage in the temporal sense): `Churn Category`, `Churn Reason`
- **Remaining exclusions** (`Count`, `Quarter`, `Lat Long`, Population's `ID`) are unrelated to leakage — constant values or redundant/non-customer identifiers.
- All 7 exclusions in the leakage/target-derived/target-construction categories are enforced structurally in `mv_churn` by never being selected (not filtered after the fact), and verified permanently absent via `tests/test_leakage.py`.