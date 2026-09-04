-- Loads raw Location CSV into stg_location with normalized column names and
-- types. Column types are declared explicitly via read_csv's columns
-- parameter so DuckDB never infers (and silently converts) a column's type
-- on its own -- notably Zip Code, which must stay VARCHAR since it is an
-- identifier, not a quantity. "Count" is a constant reporting artifact, not
-- a customer attribute, and is dropped here.

CREATE OR REPLACE TABLE stg_location AS
SELECT
    "Customer ID" AS customer_id,
    "Country" AS country,
    "State" AS state,
    "City" AS city,
    "Zip Code" AS zip_code,
    "Lat Long" AS lat_long,
    "Latitude" AS latitude,
    "Longitude" AS longitude
FROM read_csv(
    '{raw_dir}/telco_customer_churn_location.csv',
    header = true,
    columns = {
        'Customer ID': 'VARCHAR',
        'Count': 'BIGINT',
        'Country': 'VARCHAR',
        'State': 'VARCHAR',
        'City': 'VARCHAR',
        'Zip Code': 'VARCHAR',
        'Lat Long': 'VARCHAR',
        'Latitude': 'DOUBLE',
        'Longitude': 'DOUBLE'
    }
);
