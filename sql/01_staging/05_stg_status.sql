-- Loads raw Status CSV into stg_status with normalized column names and
-- types. "Count" is a constant reporting artifact, not a customer
-- attribute, and is dropped here.

CREATE OR REPLACE TABLE stg_status AS
SELECT
    "Customer ID" AS customer_id,
    "Quarter" AS quarter,
    CAST("Satisfaction Score" AS INTEGER) AS satisfaction_score,
    "Customer Status" AS customer_status,
    CAST("Churn Label" AS VARCHAR) AS churn_label,
    CAST("Churn Value" AS INTEGER) AS churn_value,
    CAST("Churn Score" AS INTEGER) AS churn_score,
    CAST("CLTV" AS INTEGER) AS cltv,
    "Churn Category" AS churn_category,
    "Churn Reason" AS churn_reason
FROM read_csv('{raw_dir}/telco_customer_churn_status.csv', header = true);
