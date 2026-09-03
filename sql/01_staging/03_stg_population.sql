-- Loads raw Population CSV into stg_population with normalized column names
-- and types. Zip Code is cast to VARCHAR since it is an identifier, not a
-- quantity, and DuckDB would otherwise infer it as numeric.

CREATE OR REPLACE TABLE stg_population AS
SELECT
    CAST("ID" AS INTEGER) AS id,
    CAST("Zip Code" AS VARCHAR) AS zip_code,
    CAST("Population" AS INTEGER) AS population
FROM read_csv('{raw_dir}/telco_customer_churn_population.csv', header = true);
