# ADR 0005: Commit Raw Dataset Tables to the Repository

**Status:** Accepted

---

## Context

Repository Architecture §7 states `data/raw/` is git-ignored by default, with an explicit documented exception: if raw tables are small and license-compatible, committing them is acceptable, and the ingestion script becomes a verifier rather than a downloader. The Lock Checklist left this as an open decision ("Raw-data commit policy decided (ADR)").

The selected dataset (IBM Telco Customer Churn, IBM Cognos Analytics 11.1.3 multi-table sample) is small (~7,043 rows across a handful of tables) and is distributed as public sample/demonstration data suitable for portfolio use.

The dataset does not have one single stable canonical download URL — it circulates via IBM Cognos sample packages and secondary mirrors — which creates reproducibility risk if the pipeline depends on a live download at ingestion time.

---

## Decision

The raw IBM Telco multi-table sample dataset will be committed to the repository under `data/raw/`, exercising the documented exception in Repository Architecture §7.

`src/retention_platform/data/ingest.py` will act as a verification step: it checks that the expected raw files are present, with expected filenames and expected row counts, and fails loudly if the raw data does not match expectations. It does not download data.

---

## Rationale

This decision was made because:

- Reproducibility should be an executable claim: `make reproduce` must work for a stranger who clones the repo, with no dependency on an external mirror staying available.
- The repository size cost is negligible for a dataset this size.
- Licensing is compatible with public portfolio use.
- An ingestion script is still valuable even with committed data — verifying file identity (names, row counts) is a legitimate defensive-engineering practice, not just plumbing.

---

## Consequences

- `data/raw/` will **not** be git-ignored; `data/interim/` and `data/processed/` remain git-ignored as originally designed.
- `.gitignore` will be updated in a later step (the ingestion script commit) to reflect this — not in this step.
- If the dataset is later replaced or grows significantly, this decision should be revisited via a new ADR rather than silently reversed.

---

**Decision Date:** 20 July 2026