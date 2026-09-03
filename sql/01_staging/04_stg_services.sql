-- Loads raw Services CSV into stg_services with normalized column names and
-- types. "Count" is a constant reporting artifact, not a customer
-- attribute, and is dropped here.

CREATE OR REPLACE TABLE stg_services AS
SELECT
    "Customer ID" AS customer_id,
    "Quarter" AS quarter,
    CAST("Referred a Friend" AS VARCHAR) AS referred_a_friend,
    CAST("Number of Referrals" AS INTEGER) AS number_of_referrals,
    CAST("Tenure in Months" AS INTEGER) AS tenure_in_months,
    "Offer" AS offer,
    CAST("Phone Service" AS VARCHAR) AS phone_service,
    "Avg Monthly Long Distance Charges" AS avg_monthly_long_distance_charges,
    CAST("Multiple Lines" AS VARCHAR) AS multiple_lines,
    CAST("Internet Service" AS VARCHAR) AS internet_service,
    "Internet Type" AS internet_type,
    CAST("Avg Monthly GB Download" AS INTEGER) AS avg_monthly_gb_download,
    CAST("Online Security" AS VARCHAR) AS online_security,
    CAST("Online Backup" AS VARCHAR) AS online_backup,
    CAST("Device Protection Plan" AS VARCHAR) AS device_protection_plan,
    CAST("Premium Tech Support" AS VARCHAR) AS premium_tech_support,
    CAST("Streaming TV" AS VARCHAR) AS streaming_tv,
    CAST("Streaming Movies" AS VARCHAR) AS streaming_movies,
    CAST("Streaming Music" AS VARCHAR) AS streaming_music,
    CAST("Unlimited Data" AS VARCHAR) AS unlimited_data,
    "Contract" AS contract,
    CAST("Paperless Billing" AS VARCHAR) AS paperless_billing,
    "Payment Method" AS payment_method,
    "Monthly Charge" AS monthly_charge,
    "Total Charges" AS total_charges,
    "Total Refunds" AS total_refunds,
    CAST("Total Extra Data Charges" AS INTEGER) AS total_extra_data_charges,
    "Total Long Distance Charges" AS total_long_distance_charges,
    "Total Revenue" AS total_revenue
FROM read_csv('{raw_dir}/telco_customer_churn_services.csv', header = true);
