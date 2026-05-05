import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import joblib
import os


def build_features(df):
    print("=" * 45)
    print("       RUNNING FEATURE ENGINEERING")
    print("=" * 45)

    # ── Step 1: Fix TotalCharges ──
    # Convert to numeric — blanks become NaN, then fill with 0
    # New customers haven't been charged yet, so 0 makes sense
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'] = df['TotalCharges'].fillna(0)
    print("✅ TotalCharges fixed — blanks filled with 0")

    # ── Step 2: Drop useless columns ──
    # customerID is just an identifier — no predictive value
    # Contract_numeric was created for segmentation — Contract column still exists
    df = df.drop(columns=['customerID', 'Contract_numeric'])
    print("✅ Dropped customerID and Contract_numeric")

    # ── Step 3: Encode binary categorical columns ──
    # These columns only have Yes/No or Male/Female — simple 1/0 mapping
    binary_cols = [
        'gender', 'Partner', 'Dependents', 'PhoneService',
        'PaperlessBilling', 'Churn'
    ]
    binary_map = {'Yes': 1, 'No': 0, 'Male': 1, 'Female': 0}
    df[binary_cols] = df[binary_cols].replace(binary_map).infer_objects(copy=False)
    print("✅ Binary columns encoded")

    # ── Step 4: Encode multi-category columns ──
    # These have 3+ categories — we use get_dummies (one-hot encoding)
    # drop_first=True avoids multicollinearity (dummy variable trap)
    multi_cols = [
        'MultipleLines', 'InternetService', 'OnlineSecurity',
        'OnlineBackup', 'DeviceProtection', 'TechSupport',
        'StreamingTV', 'StreamingMovies', 'Contract', 'PaymentMethod'
    ]
    df = pd.get_dummies(df, columns=multi_cols, drop_first=True)
    print("✅ Multi-category columns one-hot encoded")

    # ── Step 5: Engineer new features ──
    # charges_per_tenure: how much a customer pays relative to loyalty
    # +1 avoids division by zero for new customers with tenure=0
    df['charges_per_tenure'] = df['MonthlyCharges'] / (df['tenure'] + 1)

    # is_high_risk: customers on month-to-month with less than 12 months tenure
    # These are the most likely to churn — we make it an explicit signal
    df['is_high_risk'] = (
    (df['tenure'] < 12) &
    (df['Contract_One year'] == False) &
    (df['Contract_Two year'] == False)
    ).astype(int)
    print("✅ New features engineered: charges_per_tenure, is_high_risk")

    # ── Step 6: Separate features and target ──
    X = df.drop(columns=['Churn'])
    y = df['Churn']

    print(f"\nFeatures shape: {X.shape}")
    print(f"Target distribution before SMOTE:\n{y.value_counts()}")

    # ── Step 7: Train/Test split BEFORE SMOTE ──
    # We split first so test data is never touched by SMOTE
    # This prevents data leakage
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\n✅ Train/Test split done — Train: {len(X_train)}, Test: {len(X_test)}")

    # ── Step 8: Apply SMOTE on training data only ──
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    print(f"\nTarget distribution after SMOTE (train only):")
    print(pd.Series(y_train_resampled).value_counts())
    print("✅ SMOTE applied — classes are now balanced")

    # ── Step 9: Scale features ──
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_resampled)
    X_test_scaled = scaler.transform(X_test)
    print("✅ Features scaled with StandardScaler")

    # ── Step 10: Save scaler for use in API later ──
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    scaler_path = os.path.join(BASE_DIR, "data", "processed", "scaler.pkl")
    joblib.dump(scaler, scaler_path)
    print(f"✅ Scaler saved to {scaler_path}")

    # ── Step 11: Save feature names for API later ──
    feature_names = list(X.columns)
    feature_path = os.path.join(BASE_DIR, "data", "processed", "feature_names.pkl")
    joblib.dump(feature_names, feature_path)
    print(f"✅ Feature names saved to {feature_path}")

    print("=" * 45)
    print("✅ FEATURE ENGINEERING COMPLETE")
    print("=" * 45)

    return X_train_scaled, X_test_scaled, y_train_resampled, y_test, feature_names


if __name__ == "__main__":
    from ingest import load_data
    from validate import validate_data
    from segment import segment_customers

    df = load_data()
    is_valid = validate_data(df)

    if is_valid:
        df = segment_customers(df)
        X_train, X_test, y_train, y_test, feature_names = build_features(df)
        print(f"\nFinal training shape: {X_train.shape}")
        print(f"Final test shape: {X_test.shape}")
    else:
        print("❌ Pipeline stopped — data validation failed")