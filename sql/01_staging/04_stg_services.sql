-- Loads raw Services CSV into stg_services with normalized column names and
-- types. Column types are declared explicitly via read_csv's columns
-- parameter so DuckDB never infers (and silently converts) a column's type
-- on its own. "Count" is a constant reporting artifact, not a customer
-- attribute, and is dropped here.

CREATE OR REPLACE TABLE stg_services AS
SELECT
    "Customer ID" AS customer_id,
    "Quarter" AS quarter,
    "Referred a Friend" AS referred_a_friend,
    "Number of Referrals" AS number_of_referrals,
    "Tenure in Months" AS tenure_in_months,
    "Offer" AS offer,
    "Phone Service" AS phone_service,
    "Avg Monthly Long Distance Charges" AS avg_monthly_long_distance_charges,
    "Multiple Lines" AS multiple_lines,
    "Internet Service" AS internet_service,
    "Internet Type" AS internet_type,
    "Avg Monthly GB Download" AS avg_monthly_gb_download,
    "Online Security" AS online_security,
    "Online Backup" AS online_backup,
    "Device Protection Plan" AS device_protection_plan,
    "Premium Tech Support" AS premium_tech_support,
    "Streaming TV" AS streaming_tv,
    "Streaming Movies" AS streaming_movies,
    "Streaming Music" AS streaming_music,
    "Unlimited Data" AS unlimited_data,
    "Contract" AS contract,
    "Paperless Billing" AS paperless_billing,
    "Payment Method" AS payment_method,
    "Monthly Charge" AS monthly_charge,
    "Total Charges" AS total_charges,
    "Total Refunds" AS total_refunds,
    "Total Extra Data Charges" AS total_extra_data_charges,
    "Total Long Distance Charges" AS total_long_distance_charges,
    "Total Revenue" AS total_revenue
FROM read_csv(
    '{raw_dir}/telco_customer_churn_services.csv',
    header = true,
    columns = {
        'Customer ID': 'VARCHAR',
        'Count': 'BIGINT',
        'Quarter': 'VARCHAR',
        'Referred a Friend': 'VARCHAR',
        'Number of Referrals': 'INTEGER',
        'Tenure in Months': 'INTEGER',
        'Offer': 'VARCHAR',
        'Phone Service': 'VARCHAR',
        'Avg Monthly Long Distance Charges': 'DOUBLE',
        'Multiple Lines': 'VARCHAR',
        'Internet Service': 'VARCHAR',
        'Internet Type': 'VARCHAR',
        'Avg Monthly GB Download': 'INTEGER',
        'Online Security': 'VARCHAR',
        'Online Backup': 'VARCHAR',
        'Device Protection Plan': 'VARCHAR',
        'Premium Tech Support': 'VARCHAR',
        'Streaming TV': 'VARCHAR',
        'Streaming Movies': 'VARCHAR',
        'Streaming Music': 'VARCHAR',
        'Unlimited Data': 'VARCHAR',
        'Contract': 'VARCHAR',
        'Paperless Billing': 'VARCHAR',
        'Payment Method': 'VARCHAR',
        'Monthly Charge': 'DOUBLE',
        'Total Charges': 'DOUBLE',
        'Total Refunds': 'DOUBLE',
        'Total Extra Data Charges': 'INTEGER',
        'Total Long Distance Charges': 'DOUBLE',
        'Total Revenue': 'DOUBLE'
    }
);
