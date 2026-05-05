import os
import joblib
import numpy as np
import pandas as pd


# ── Retention campaign costs by segment ──
# These are business decisions — how much are we willing to spend
# to retain a customer based on their value tier?
RETENTION_COSTS = {
    0: 100,   # High Value  → premium retention (personal call + big discount)
    1: 30,    # Low Value   → minimal retention (small voucher)
    2: 60,    # Mid Value   → moderate retention (discount offer)
}

# ── Churn probability threshold ──
# Only act on customers with >50% churn probability
CHURN_THRESHOLD = 0.5


def calculate_clv(monthly_charges, tenure, avg_lifespan_months=24):
    """
    Calculate Customer Lifetime Value (CLV).

    CLV = Monthly Charges × Expected Remaining Months

    We estimate remaining months as:
    - If tenure < avg_lifespan → they have (avg_lifespan - tenure) months left
    - If tenure >= avg_lifespan → we give them 6 more months (loyal customer)

    This is a simplified CLV — real businesses use more complex models,
    but this demonstrates the concept clearly.
    """
    if tenure < avg_lifespan_months:
        remaining_months = avg_lifespan_months - tenure
    else:
        remaining_months = 6  # loyal customer — estimate 6 more months

    clv = monthly_charges * remaining_months
    return round(clv, 2)


def get_action(churn_probability, clv, retention_cost, segment):
    """
    Decide what action to take for a customer.

    Logic:
    - If churn probability is low → no action needed
    - If churn probability is high AND CLV > retention cost → target them
    - If churn probability is high BUT CLV < retention cost → let them go
      (spending more to retain than they're worth is bad business)
    """
    segment_labels = {0: "High Value", 1: "Low Value", 2: "Mid Value"}

    if churn_probability < CHURN_THRESHOLD:
        return {
            "action": "✅ No Action",
            "reason": "Low churn risk — customer is stable",
            "segment": segment_labels.get(segment, "Unknown"),
            "roi": None
        }

    # Customer is at risk — should we spend to retain them?
    roi = clv - retention_cost

    if roi > 0:
        # Worth retaining
        if segment == 0:
            action = "📞 High Priority — Personal call + 20% discount"
        elif segment == 2:
            action = "📧 Medium Priority — Email offer + 10% discount"
        else:
            action = "💌 Low Priority — Send retention voucher"

        return {
            "action": action,
            "reason": f"CLV (${clv}) > Retention Cost (${retention_cost}) → ROI = ${round(roi, 2)}",
            "segment": segment_labels.get(segment, "Unknown"),
            "roi": round(roi, 2)
        }
    else:
        # Not worth retaining
        return {
            "action": "❌ Do Not Target",
            "reason": f"CLV (${clv}) < Retention Cost (${retention_cost}) → Not worth retaining",
            "segment": segment_labels.get(segment, "Unknown"),
            "roi": round(roi, 2)
        }


def generate_recommendations(df_original, X_test_scaled, y_test, model, feature_names):
    """
    Generate full recommendations for all test customers.
    Combines churn prediction + CLV + retention cost → action.
    """
    print("=" * 55)
    print("       GENERATING BUSINESS RECOMMENDATIONS")
    print("=" * 55)

    # ── Get churn probabilities ──
    # predict_proba returns [prob_no_churn, prob_churn]
    # We take [:, 1] which is the probability of churn
    churn_probs = model.predict_proba(X_test_scaled)[:, 1]
    churn_preds = (churn_probs >= CHURN_THRESHOLD).astype(int)

    # ── Build results dataframe ──
    # We use the test indices to match back to original data
    test_indices = y_test.index
    results = []

    for i, idx in enumerate(test_indices):
        monthly_charges = df_original.loc[idx, 'MonthlyCharges']
        tenure = df_original.loc[idx, 'tenure']
        segment = df_original.loc[idx, 'segment']

        # Calculate CLV
        clv = calculate_clv(monthly_charges, tenure)

        # Get retention cost for this segment
        retention_cost = RETENTION_COSTS.get(segment, 50)

        # Get churn probability
        churn_prob = round(churn_probs[i], 4)

        # Get recommended action
        recommendation = get_action(churn_prob, clv, retention_cost, segment)

        results.append({
            "customer_index": idx,
            "monthly_charges": monthly_charges,
            "tenure": tenure,
            "segment": recommendation["segment"],
            "churn_probability": churn_prob,
            "actual_churn": int(y_test.iloc[i]),
            "clv": clv,
            "retention_cost": retention_cost,
            "action": recommendation["action"],
            "reason": recommendation["reason"],
            "roi": recommendation["roi"]
        })

    results_df = pd.DataFrame(results)

    # ── Summary statistics ──
    at_risk = results_df[results_df['churn_probability'] >= CHURN_THRESHOLD]
    worth_retaining = at_risk[at_risk['roi'] > 0]
    not_worth = at_risk[at_risk['roi'] <= 0]

    print(f"\n📊 Summary:")
    print(f"   Total test customers:        {len(results_df)}")
    print(f"   Predicted to churn:          {len(at_risk)}")
    print(f"   Worth retaining (ROI > 0):   {len(worth_retaining)}")
    print(f"   Not worth retaining:         {len(not_worth)}")
    print(f"   Total potential ROI:         ${worth_retaining['roi'].sum():,.2f}")

    # ── Show sample recommendations ──
    print(f"\n🎯 Sample Recommendations (first 5 at-risk customers):")
    display_cols = ['monthly_charges', 'tenure', 'segment',
                    'churn_probability', 'clv', 'action']
    print(at_risk[display_cols].head().to_string(index=False))

    # ── Save recommendations to CSV ──
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_path = os.path.join(BASE_DIR, "data", "processed", "recommendations.csv")
    results_df.to_csv(output_path, index=False)

    print(f"\n✅ Full recommendations saved to {output_path}")
    print("=" * 55)

    return results_df


if __name__ == "__main__":
    from ingest import load_data
    from validate import validate_data
    from segment import segment_customers
    from features import build_features

    # ── Run pipeline ──
    df = load_data()
    is_valid = validate_data(df)

    if not is_valid:
        print("❌ Pipeline stopped — data validation failed")
        exit()

    df = segment_customers(df)
    X_train, X_test, y_train, y_test, feature_names = build_features(df)

    # ── Load saved model ──
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(BASE_DIR, "data", "processed", "best_model.pkl")
    model = joblib.load(model_path)

    # ── Generate recommendations ──
    results_df = generate_recommendations(df, X_test, y_test, model, feature_names)

    print("\n🎉 Recommendation engine complete!")