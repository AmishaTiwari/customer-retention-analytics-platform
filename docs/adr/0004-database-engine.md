# ADR 0004: Use DuckDB as the Local Analytical Database Engine

**Status:** Accepted

---

## Context

The Repository Architecture and ML System Design planning documents left the local analytical database engine as an open decision ("SQLite or DuckDB"), explicitly deferred to be resolved via ADR before SQL data preparation begins.

The SQL layer's job, per the locked SQL Strategy, is to load raw relational tables, join them, filter invalid records, construct the churn target, and produce a clean modeling view — an analytical/ELT-style workload, not a transactional one.

---

## Decision

This project will use **DuckDB** as the embedded local analytical database engine for all SQL-based data preparation (`sql/01_staging` through `sql/04_modeling_view`).

---

## Rationale

This decision was made because:

- DuckDB reads CSV/tabular raw files directly with native SQL, with no manual `CREATE TABLE` plus row-by-row insert step required, which matches the staging layer's actual job.
- Its warehouse-style SQL dialect (window functions, aggregations) is suited to the transformation-heavy workload described in the SQL Strategy.
- It is zero-infrastructure and single embedded file, consistent with the locked lightweight engineering scope — no server, no Docker.
- It has clean interop with pandas for handoff to the Python package (`features/build.py` and downstream pipeline stages).
- SQLite would have been a slightly more literal simulation of an OLTP source system, but the actual workload here is transformation/ELT, not transactional, so DuckDB better matches the real work.

---

## Consequences

- `pyproject.toml` will add `duckdb` as a dependency.
- `data/prepare.py` will connect to a local DuckDB file (e.g., `data/interim/retention_platform.duckdb`).
- All SQL scripts in `sql/` will target DuckDB's SQL dialect.
- If a future need requires a different engine, a new ADR will supersede this one.

---

**Decision Date:** 20 July 2026