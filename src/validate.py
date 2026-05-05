import pandas as pd


EXPECTED_COLUMNS = [
    'customerID', 'gender', 'SeniorCitizen', 'Partner', 'Dependents',
    'tenure', 'PhoneService', 'MultipleLines', 'InternetService',
    'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
    'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling',
    'PaymentMethod', 'MonthlyCharges', 'TotalCharges', 'Churn'
]


def validate_data(df):
    print("=" * 45)
    print("       RUNNING DATA VALIDATION CHECKS")
    print("=" * 45)

    all_passed = True

    # ── Check 1: All expected columns are present ──
    missing_cols = [col for col in EXPECTED_COLUMNS if col not in df.columns]
    if missing_cols:
        print(f"❌ Column check failed — missing: {missing_cols}")
        all_passed = False
    else:
        print("✅ Column check passed — all 21 columns present")

    # ── Check 2: Minimum row count ──
    if len(df) < 100:
        print(f"❌ Row count check failed — only {len(df)} rows found")
        all_passed = False
    else:
        print(f"✅ Row count check passed — {len(df)} rows found")

    # ── Check 3: Tenure range (0 to 72 months) ──
    invalid_tenure = df[(df['tenure'] < 0) | (df['tenure'] > 72)]
    if len(invalid_tenure) > 0:
        print(f"❌ Tenure range check failed — {len(invalid_tenure)} invalid rows")
        all_passed = False
    else:
        print("✅ Tenure range check passed — all values between 0 and 72")

    # ── Check 4: MonthlyCharges range (0 to 200) ──
    invalid_charges = df[(df['MonthlyCharges'] <= 0) | (df['MonthlyCharges'] > 200)]
    if len(invalid_charges) > 0:
        print(f"❌ MonthlyCharges check failed — {len(invalid_charges)} invalid rows")
        all_passed = False
    else:
        print("✅ MonthlyCharges check passed — all values between 0 and 200")

    # ── Check 5: Churn column only contains Yes or No ──
    valid_churn_values = {'Yes', 'No'}
    actual_churn_values = set(df['Churn'].unique())
    invalid_churn = actual_churn_values - valid_churn_values
    if invalid_churn:
        print(f"❌ Churn values check failed — unexpected values: {invalid_churn}")
        all_passed = False
    else:
        print("✅ Churn values check passed — only 'Yes' and 'No' found")

    # ── Final result ──
    print("=" * 45)
    if all_passed:
        print("✅ ALL CHECKS PASSED — data is safe to use")
    else:
        print("❌ SOME CHECKS FAILED — review issues above")
    print("=" * 45)

    return all_passed


if __name__ == "__main__":
    from ingest import load_data
    df = load_data()
    validate_data(df)