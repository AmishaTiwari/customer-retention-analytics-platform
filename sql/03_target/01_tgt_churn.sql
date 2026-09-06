-- Constructs the voluntary-churn target per the locked voluntary-churn
-- mapping decision, excluding Deceased customers entirely. Uses
-- IS DISTINCT FROM instead of != so the comparison is NULL-safe:
-- churn_reason is NULL for non-churned customers, and a plain != would
-- incorrectly exclude those rows too, not just the Deceased ones.

CREATE OR REPLACE TABLE tgt_churn AS
SELECT
    customer_id,
    customer_status,
    churn_category,
    churn_reason,
    (customer_status = 'Churned') AS is_voluntary_churn
FROM cln_status
WHERE churn_reason IS DISTINCT FROM 'Deceased'
;
