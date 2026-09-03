-- Loads raw Demographics CSV into stg_demographics with normalized column
-- names and types. "Count" is a constant reporting artifact, not a customer
-- attribute, and is dropped here.

CREATE OR REPLACE TABLE stg_demographics AS
SELECT
    "Customer ID" AS customer_id,
    "Gender" AS gender,
    CAST("Age" AS INTEGER) AS age,
    CAST("Under 30" AS VARCHAR) AS under_30,
    CAST("Senior Citizen" AS VARCHAR) AS senior_citizen,
    CAST("Married" AS VARCHAR) AS married,
    CAST("Dependents" AS VARCHAR) AS dependents,
    CAST("Number of Dependents" AS INTEGER) AS number_of_dependents
FROM read_csv('{raw_dir}/telco_customer_churn_demographics.csv', header = true);