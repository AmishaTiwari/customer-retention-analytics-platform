-- Converts Under 30 / Senior Citizen / Married / Dependents from Yes/No text
-- to real BOOLEAN. Uses an explicit CASE WHEN rather than CAST(col AS
-- BOOLEAN), since DuckDB's implicit boolean casting was already found to
-- silently misbehave on this project's data. All other columns pass
-- through unchanged.

CREATE OR REPLACE TABLE cln_demographics AS
SELECT
    customer_id,
    gender,
    age,
    CASE
        WHEN under_30 = 'Yes' THEN true
        WHEN under_30 = 'No' THEN false
        ELSE NULL
    END AS under_30,
    CASE
        WHEN senior_citizen = 'Yes' THEN true
        WHEN senior_citizen = 'No' THEN false
        ELSE NULL
    END AS senior_citizen,
    CASE
        WHEN married = 'Yes' THEN true
        WHEN married = 'No' THEN false
        ELSE NULL
    END AS married,
    CASE
        WHEN dependents = 'Yes' THEN true
        WHEN dependents = 'No' THEN false
        ELSE NULL
    END AS dependents,
    number_of_dependents
FROM stg_demographics
;
