-- Converts 13 Yes/No text columns to real BOOLEAN. Uses an explicit CASE
-- WHEN rather than CAST(col AS BOOLEAN), since DuckDB's implicit boolean
-- casting was already found to silently misbehave on this project's data.
-- All other columns pass through unchanged.

CREATE OR REPLACE TABLE cln_services AS
SELECT
    customer_id,
    quarter,
    CASE
        WHEN referred_a_friend = 'Yes' THEN true
        WHEN referred_a_friend = 'No' THEN false
        ELSE NULL
    END AS referred_a_friend,
    number_of_referrals,
    tenure_in_months,
    offer,
    CASE
        WHEN phone_service = 'Yes' THEN true
        WHEN phone_service = 'No' THEN false
        ELSE NULL
    END AS phone_service,
    avg_monthly_long_distance_charges,
    CASE
        WHEN multiple_lines = 'Yes' THEN true
        WHEN multiple_lines = 'No' THEN false
        ELSE NULL
    END AS multiple_lines,
    CASE
        WHEN internet_service = 'Yes' THEN true
        WHEN internet_service = 'No' THEN false
        ELSE NULL
    END AS internet_service,
    internet_type,
    avg_monthly_gb_download,
    CASE
        WHEN online_security = 'Yes' THEN true
        WHEN online_security = 'No' THEN false
        ELSE NULL
    END AS online_security,
    CASE
        WHEN online_backup = 'Yes' THEN true
        WHEN online_backup = 'No' THEN false
        ELSE NULL
    END AS online_backup,
    CASE
        WHEN device_protection_plan = 'Yes' THEN true
        WHEN device_protection_plan = 'No' THEN false
        ELSE NULL
    END AS device_protection_plan,
    CASE
        WHEN premium_tech_support = 'Yes' THEN true
        WHEN premium_tech_support = 'No' THEN false
        ELSE NULL
    END AS premium_tech_support,
    CASE
        WHEN streaming_tv = 'Yes' THEN true
        WHEN streaming_tv = 'No' THEN false
        ELSE NULL
    END AS streaming_tv,
    CASE
        WHEN streaming_movies = 'Yes' THEN true
        WHEN streaming_movies = 'No' THEN false
        ELSE NULL
    END AS streaming_movies,
    CASE
        WHEN streaming_music = 'Yes' THEN true
        WHEN streaming_music = 'No' THEN false
        ELSE NULL
    END AS streaming_music,
    CASE
        WHEN unlimited_data = 'Yes' THEN true
        WHEN unlimited_data = 'No' THEN false
        ELSE NULL
    END AS unlimited_data,
    contract,
    CASE
        WHEN paperless_billing = 'Yes' THEN true
        WHEN paperless_billing = 'No' THEN false
        ELSE NULL
    END AS paperless_billing,
    payment_method,
    monthly_charge,
    total_charges,
    total_refunds,
    total_extra_data_charges,
    total_long_distance_charges,
    total_revenue
FROM stg_services
;
