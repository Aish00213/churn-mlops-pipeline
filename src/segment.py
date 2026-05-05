import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


def segment_customers(df):
    print("=" * 45)
    print("       RUNNING CUSTOMER SEGMENTATION")
    print("=" * 45)

    # ── Step 1: Convert Contract to numeric ──
    # KMeans only works with numbers, not text
    contract_map = {
        'Month-to-month': 0,
        'One year': 1,
        'Two year': 2
    }
    df['Contract_numeric'] = df['Contract'].map(contract_map)

    # ── Step 2: Select RFM features ──
    # R = tenure (how long they've been a customer)
    # F = Contract_numeric (commitment level)
    # M = MonthlyCharges (how much they pay)
    rfm = df[['tenure', 'Contract_numeric', 'MonthlyCharges']].copy()

    # ── Step 3: Scale the features ──
    # KMeans uses distance — if MonthlyCharges is 0-120 and tenure is 0-72,
    # MonthlyCharges will dominate. Scaling puts all features on equal footing.
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm)

    # ── Step 4: Run KMeans clustering ──
    # n_clusters=3 → we want 3 segments: High, Mid, Low value
    # random_state=42 → makes results reproducible every time you run it
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    df['segment'] = kmeans.fit_predict(rfm_scaled)

    # ── Step 5: Print segment distribution ──
    print("\nCustomers per segment:")
    print(df['segment'].value_counts().sort_index())

    # ── Step 6: Print segment profiles so we understand what each cluster means ──
    print("\nSegment profiles (averages):")
    profile = df.groupby('segment')[['tenure', 'MonthlyCharges', 'Contract_numeric']].mean().round(2)
    print(profile)

    print("=" * 45)
    print("✅ Segmentation complete — 'segment' column added")
    print("=" * 45)

    return df


if __name__ == "__main__":
    from ingest import load_data
    from validate import validate_data

    df = load_data()
    is_valid = validate_data(df)

    if is_valid:
        df = segment_customers(df)
    else:
        print("❌ Pipeline stopped — data validation failed")