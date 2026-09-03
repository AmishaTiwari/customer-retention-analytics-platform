-- Loads raw Location CSV into stg_location with normalized column names and
-- types. Zip Code is cast to VARCHAR since it is an identifier, not a
-- quantity, and DuckDB would otherwise infer it as numeric. "Count" is a
-- constant reporting artifact, not a customer attribute, and is dropped here.

CREATE OR REPLACE TABLE stg_location AS
SELECT
    "Customer ID" AS customer_id,
    "Country" AS country,
    "State" AS state,
    "City" AS city,
    CAST("Zip Code" AS VARCHAR) AS zip_code,
    "Lat Long" AS lat_long,
    "Latitude" AS latitude,
    "Longitude" AS longitude
FROM read_csv('{raw_dir}/telco_customer_churn_location.csv', header = true);
