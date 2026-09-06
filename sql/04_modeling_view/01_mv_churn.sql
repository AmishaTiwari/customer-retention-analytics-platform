-- Assembles the final modeling dataset. Joins the customer-level tables on
-- customer_id, and brings in Population via zip_code. Two categories of
-- columns are permanently excluded, enforced simply by never selecting
-- them (not by filtering them out after the fact):
--   1. Post-outcome/temporal leakage columns: churn_score, cltv,
--      churn_category, churn_reason.
--   2. Target-derived columns that would trivially expose the target:
--      customer_status, churn_label, churn_value.

CREATE OR REPLACE TABLE mv_churn AS
SELECT
    t.customer_id,
    t.is_voluntary_churn,

    d.gender,
    d.age,
    d.under_30,
    d.senior_citizen,
    d.married,
    d.dependents,
    d.number_of_dependents,

    l.country,
    l.state,
    l.city,
    l.zip_code,
    l.latitude,
    l.longitude,

    p.population,

    s.referred_a_friend,
    s.number_of_referrals,
    s.tenure_in_months,
    s.offer,
    s.phone_service,
    s.avg_monthly_long_distance_charges,
    s.multiple_lines,
    s.internet_service,
    s.internet_type,
    s.avg_monthly_gb_download,
    s.online_security,
    s.online_backup,
    s.device_protection_plan,
    s.premium_tech_support,
    s.streaming_tv,
    s.streaming_movies,
    s.streaming_music,
    s.unlimited_data,
    s.contract,
    s.paperless_billing,
    s.payment_method,
    s.monthly_charge,
    s.total_charges,
    s.total_refunds,
    s.total_extra_data_charges,
    s.total_long_distance_charges,
    s.total_revenue,

    st.satisfaction_score

FROM tgt_churn t
INNER JOIN cln_demographics d ON t.customer_id = d.customer_id
INNER JOIN stg_location l ON t.customer_id = l.customer_id
LEFT JOIN stg_population p ON l.zip_code = p.zip_code
INNER JOIN cln_services s ON t.customer_id = s.customer_id
INNER JOIN stg_status st ON t.customer_id = st.customer_id;
