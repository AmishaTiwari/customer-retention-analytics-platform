-- Converts Churn Label from Yes/No text to real BOOLEAN. Uses an explicit
-- CASE WHEN rather than CAST(col AS BOOLEAN), since DuckDB's implicit
-- boolean casting was already found to silently misbehave on this
-- project's data. All other columns pass through unchanged.

CREATE OR REPLACE TABLE cln_status AS
SELECT
    customer_id,
    quarter,
    satisfaction_score,
    customer_status,
    CASE
        WHEN churn_label = 'Yes' THEN true
        WHEN churn_label = 'No' THEN false
        ELSE NULL
    END AS churn_label,
    churn_value,
    churn_score,
    cltv,
    churn_category,
    churn_reason
FROM stg_status
;
