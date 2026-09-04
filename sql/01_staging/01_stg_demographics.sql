-- Loads raw Demographics CSV into stg_demographics with normalized column
-- names and types. Column types are declared explicitly via read_csv's
-- columns parameter so DuckDB never infers (and silently converts) a
-- column's type on its own. "Count" is a constant reporting artifact, not
-- a customer attribute, and is dropped here.

CREATE OR REPLACE TABLE stg_demographics AS
SELECT
    "Customer ID" AS customer_id,
    "Gender" AS gender,
    "Age" AS age,
    "Under 30" AS under_30,
    "Senior Citizen" AS senior_citizen,
    "Married" AS married,
    "Dependents" AS dependents,
    "Number of Dependents" AS number_of_dependents
FROM read_csv(
    '{raw_dir}/telco_customer_churn_demographics.csv',
    header = true,
    columns = {
        'Customer ID': 'VARCHAR',
        'Count': 'BIGINT',
        'Gender': 'VARCHAR',
        'Age': 'INTEGER',
        'Under 30': 'VARCHAR',
        'Senior Citizen': 'VARCHAR',
        'Married': 'VARCHAR',
        'Dependents': 'VARCHAR',
        'Number of Dependents': 'INTEGER'
    }
);
