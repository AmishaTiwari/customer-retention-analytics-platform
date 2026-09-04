-- Loads raw Population CSV into stg_population with normalized column names
-- and types. Column types are declared explicitly via read_csv's columns
-- parameter so DuckDB never infers (and silently converts) a column's type
-- on its own -- notably Zip Code, which must stay VARCHAR since it is an
-- identifier, not a quantity.

CREATE OR REPLACE TABLE stg_population AS
SELECT
    "ID" AS id,
    "Zip Code" AS zip_code,
    "Population" AS population
FROM read_csv(
    '{raw_dir}/telco_customer_churn_population.csv',
    header = true,
    columns = {
        'ID': 'INTEGER',
        'Zip Code': 'VARCHAR',
        'Population': 'INTEGER'
    }
);
