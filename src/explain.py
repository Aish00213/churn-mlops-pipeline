import os
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt


def load_model_and_data():
    """Load the saved model, scaler and feature names."""
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    processed_dir = os.path.join(BASE_DIR, "data", "processed")

    model = joblib.load(os.path.join(processed_dir, "best_model.pkl"))
    scaler = joblib.load(os.path.join(processed_dir, "scaler.pkl"))
    feature_names = joblib.load(os.path.join(processed_dir, "feature_names.pkl"))

    return model, scaler, feature_names


def explain_global(model, X_train_scaled, feature_names):
    """
    Global explanation — which features matter most across ALL customers.
    This tells us what the model generally uses to make decisions.
    """
    print("=" * 45)
    print("       GLOBAL SHAP EXPLANATION")
    print("=" * 45)

    # ── Create SHAP explainer ──
    # TreeExplainer is optimized for tree-based models like XGBoost
    explainer = shap.TreeExplainer(model)

    # ── Calculate SHAP values ──
    # SHAP values tell us: for each feature, how much did it push
    # the prediction higher or lower compared to the average prediction?
    shap_values = explainer.shap_values(X_train_scaled)

    # ── Plot feature importance (global) ──
    # mean(|SHAP value|) across all customers = overall feature importance
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    plots_dir = os.path.join(BASE_DIR, "data", "processed")

    plt.figure()
    shap.summary_plot(
        shap_values,
        X_train_scaled,
        feature_names=feature_names,
        plot_type="bar",
        show=False
    )
    plt.title("Global Feature Importance (SHAP)")
    plt.tight_layout()
    global_plot_path = os.path.join(plots_dir, "shap_global.png")
    plt.savefig(global_plot_path, bbox_inches='tight', dpi=150)
    plt.close()

    print(f"✅ Global SHAP plot saved to {global_plot_path}")

    # ── Print top 10 most important features ──
    mean_shap = np.abs(shap_values).mean(axis=0)
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': mean_shap
    }).sort_values('importance', ascending=False)

    print("\n🏆 Top 10 most important features:")
    print(feature_importance.head(10).to_string(index=False))

    return explainer, shap_values


def explain_local(explainer, customer_data_scaled, feature_names, customer_index=0):
    """
    Local explanation — why did the model make THIS prediction
    for THIS specific customer?
    """
    print("\n" + "=" * 45)
    print(f"       LOCAL SHAP EXPLANATION — Customer #{customer_index}")
    print("=" * 45)

    # ── Get SHAP values for one customer ──
    single_customer = customer_data_scaled[customer_index:customer_index+1]
    shap_values_single = explainer.shap_values(single_customer)

    # ── Build explanation dataframe ──
    explanation = pd.DataFrame({
        'feature': feature_names,
        'shap_value': shap_values_single[0]
    }).sort_values('shap_value', ascending=False)

    # Positive SHAP = pushes toward churn
    # Negative SHAP = pushes away from churn
    print("\n🔴 Top reasons pushing toward CHURN:")
    print(explanation[explanation['shap_value'] > 0].head(5).to_string(index=False))

    print("\n🟢 Top reasons pushing AGAINST churn:")
    print(explanation[explanation['shap_value'] < 0].tail(5).to_string(index=False))

    # ── Save local SHAP plot ──
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    plots_dir = os.path.join(BASE_DIR, "data", "processed")

    plt.figure()
    shap.waterfall_plot(
        shap.Explanation(
            values=shap_values_single[0],
            base_values=explainer.expected_value,
            feature_names=feature_names
        ),
        show=False
    )
    plt.title(f"Local SHAP Explanation — Customer #{customer_index}")
    plt.tight_layout()
    local_plot_path = os.path.join(plots_dir, f"shap_local_customer_{customer_index}.png")
    plt.savefig(local_plot_path, bbox_inches='tight', dpi=150)
    plt.close()

    print(f"\n✅ Local SHAP plot saved to {local_plot_path}")
    print("=" * 45)

    return explanation


if __name__ == "__main__":
    from ingest import load_data
    from validate import validate_data
    from segment import segment_customers
    from features import build_features

    # ── Run pipeline to get training data ──
    df = load_data()
    is_valid = validate_data(df)

    if not is_valid:
        print("❌ Pipeline stopped — data validation failed")
        exit()

    df = segment_customers(df)
    X_train, X_test, y_train, y_test, feature_names = build_features(df)

    # ── Load saved model ──
    model, scaler, feature_names = load_model_and_data()

    # ── Run global explanation ──
    explainer, shap_values = explain_global(model, X_train, feature_names)

    # ── Run local explanation for first 3 test customers ──
    for i in range(3):
        explain_local(explainer, X_test, feature_names, customer_index=i)

    print("\n🎉 Explainability complete!")
    print("Check data/processed/ for SHAP plots.")