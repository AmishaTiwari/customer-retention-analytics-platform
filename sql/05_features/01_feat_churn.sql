-- Adds four hypothesis-driven derived features on top of the modeling
-- view:
--   num_addon_services -- count of the 8 opted-in add-on services.
--     Hypothesis: customers with fewer bundled services have less
--     invested in the relationship and lower switching cost, making
--     them more likely to churn.
--   is_month_to_month -- flags customers on a month-to-month contract.
--     Hypothesis: the absence of a contract term removes friction
--     from leaving, so month-to-month customers churn more easily
--     than customers under a fixed-term contract.
--   internet_without_security -- flags internet customers with no
--     online security add-on.
--     Hypothesis: internet customers who decline security coverage
--     are more exposed to service dissatisfaction (e.g. malware,
--     unwanted charges) that can drive churn.
--   no_addons_despite_internet -- flags internet customers who opted
--     into zero add-on services.
--     Hypothesis: distinguishes customers who have zero add-ons
--     because they lack internet service (structural, the majority
--     of the num_addon_services = 0 group) from the small group who
--     have internet but declined every add-on, a distinct pattern
--     worth isolating even though it covers only 81 customers.

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
        AS internet_without_security,

    (internet_service = true AND num_addon_services = 0)
        AS no_addons_despite_internet

FROM mv_churn;
