-- Loads raw Status CSV into stg_status with normalized column names and
-- types. Column types are declared explicitly via read_csv's columns
-- parameter so DuckDB never infers (and silently converts) a column's type
-- on its own. "Count" is a constant reporting artifact, not a customer
-- attribute, and is dropped here.

CREATE OR REPLACE TABLE stg_status AS
SELECT
    "Customer ID" AS customer_id,
    "Quarter" AS quarter,
    "Satisfaction Score" AS satisfaction_score,
    "Customer Status" AS customer_status,
    "Churn Label" AS churn_label,
    "Churn Value" AS churn_value,
    "Churn Score" AS churn_score,
    "CLTV" AS cltv,
    "Churn Category" AS churn_category,
    "Churn Reason" AS churn_reason
FROM read_csv(
    '{raw_dir}/telco_customer_churn_status.csv',
    header = true,
    columns = {
        'Customer ID': 'VARCHAR',
        'Count': 'BIGINT',
        'Quarter': 'VARCHAR',
        'Satisfaction Score': 'INTEGER',
        'Customer Status': 'VARCHAR',
        'Churn Label': 'VARCHAR',
        'Churn Value': 'INTEGER',
        'Churn Score': 'INTEGER',
        'CLTV': 'INTEGER',
        'Churn Category': 'VARCHAR',
        'Churn Reason': 'VARCHAR'
    }
);
