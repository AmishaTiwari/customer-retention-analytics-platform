-- Adds three hypothesis-driven derived features on top of the modeling
-- view:
--   num_addon_services -- count of the 8 opted-in add-on services.
--   is_month_to_month -- flags customers on a month-to-month contract.
--   internet_without_security -- flags internet customers with no
--     online security add-on.

CREATE OR REPLACE TABLE feat_churn AS
SELECT
    customer_id,
    is_voluntary_churn,
    gender,
    age,
    under_30,
    senior_citizen,
    married,
    dependents,
    number_of_dependents,
    country,
    state,
    city,
    zip_code,
    latitude,
    longitude,
    population,
    referred_a_friend,
    number_of_referrals,
    tenure_in_months,
    offer,
    phone_service,
    avg_monthly_long_distance_charges,
    multiple_lines,
    internet_service,
    internet_type,
    avg_monthly_gb_download,
    online_security,
    online_backup,
    device_protection_plan,
    premium_tech_support,
    streaming_tv,
    streaming_movies,
    streaming_music,
    unlimited_data,
    contract,
    paperless_billing,
    payment_method,
    monthly_charge,
    total_charges,
    total_refunds,
    total_extra_data_charges,
    total_long_distance_charges,
    total_revenue,
    satisfaction_score,

    (online_security::INTEGER + online_backup::INTEGER +
     device_protection_plan::INTEGER + premium_tech_support::INTEGER +
     streaming_tv::INTEGER + streaming_movies::INTEGER +
     streaming_music::INTEGER + unlimited_data::INTEGER)
        AS num_addon_services,

    (contract = 'Month-to-Month') AS is_month_to_month,

    (internet_service = true AND online_security = false)
        AS internet_without_security

FROM mv_churn;
