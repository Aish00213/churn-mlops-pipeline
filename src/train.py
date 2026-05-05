import os
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)
from xgboost import XGBClassifier


def evaluate_model(model, X_test, y_test, model_name):
    """Run predictions and return a dictionary of metrics."""
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy":  round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred), 4),
        "recall":    round(recall_score(y_test, y_pred), 4),
        "f1_score":  round(f1_score(y_test, y_pred), 4),
        "roc_auc":   round(roc_auc_score(y_test, y_prob), 4),
    }

    print(f"\n📊 {model_name} Results:")
    for metric, value in metrics.items():
        print(f"   {metric}: {value}")

    print(f"\n   Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(f"   TN={cm[0][0]}  FP={cm[0][1]}")
    print(f"   FN={cm[1][0]}  TP={cm[1][1]}")

    return metrics


def train_baseline(X_train, X_test, y_train, y_test):
    """Train Logistic Regression as the baseline model."""
    print("=" * 45)
    print("       TRAINING BASELINE — LOGISTIC REGRESSION")
    print("=" * 45)

    # Set MLflow experiment — all runs go under this experiment name
    mlflow.set_experiment("churn-prediction")

    with mlflow.start_run(run_name="baseline_logistic_regression"):

        # ── Train model ──
        model = LogisticRegression(max_iter=1000, random_state=42)
        model.fit(X_train, y_train)

        # ── Evaluate ──
        metrics = evaluate_model(model, X_test, y_test, "Logistic Regression")

        # ── Log parameters to MLflow ──
        # Parameters = settings you chose before training
        mlflow.log_param("model_type", "LogisticRegression")
        mlflow.log_param("max_iter", 1000)
        mlflow.log_param("random_state", 42)

        # ── Log metrics to MLflow ──
        # Metrics = results after training
        for metric, value in metrics.items():
            mlflow.log_metric(metric, value)

        # ── Log model to MLflow ──
        mlflow.sklearn.log_model(model, "model")

        print("\n✅ Baseline logged to MLflow")

    return model, metrics


def train_champion(X_train, X_test, y_train, y_test):
    """Train XGBoost as the champion model."""
    print("\n" + "=" * 45)
    print("       TRAINING CHAMPION — XGBOOST")
    print("=" * 45)

    mlflow.set_experiment("churn-prediction")

    with mlflow.start_run(run_name="champion_xgboost"):

        # ── Train model ──
        # scale_pos_weight handles any remaining imbalance
        model = XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            scale_pos_weight=1,
            random_state=42,
            eval_metric='logloss',
            verbosity=0
        )
        model.fit(X_train, y_train)

        # ── Evaluate ──
        metrics = evaluate_model(model, X_test, y_test, "XGBoost")

        # ── Log parameters to MLflow ──
        mlflow.log_param("model_type", "XGBoost")
        mlflow.log_param("n_estimators", 200)
        mlflow.log_param("max_depth", 6)
        mlflow.log_param("learning_rate", 0.1)
        mlflow.log_param("random_state", 42)

        # ── Log metrics to MLflow ──
        for metric, value in metrics.items():
            mlflow.log_metric(metric, value)

        # ── Log model to MLflow ──
        mlflow.xgboost.log_model(model, "model")

        print("\n✅ Champion logged to MLflow")

    return model, metrics


def save_best_model(baseline_metrics, champion_metrics,
                    baseline_model, champion_model):
    """Compare models and save the best one."""
    print("\n" + "=" * 45)
    print("       COMPARING MODELS")
    print("=" * 45)

    # We use F1 score to compare — it balances precision and recall
    # Important for imbalanced problems like churn
    baseline_f1 = baseline_metrics['f1_score']
    champion_f1 = champion_metrics['f1_score']

    print(f"   Baseline F1:  {baseline_f1}")
    print(f"   Champion F1:  {champion_f1}")

    if champion_f1 > baseline_f1:
        best_model = champion_model
        best_name = "XGBoost"
    else:
        best_model = baseline_model
        best_name = "Logistic Regression"

    print(f"\n🏆 Best model: {best_name}")

    # ── Save best model as pickle ──
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(BASE_DIR, "data", "processed", "best_model.pkl")
    joblib.dump(best_model, model_path)

    print(f"✅ Best model saved to {model_path}")
    print("=" * 45)

    return best_model


if __name__ == "__main__":
    from ingest import load_data
    from validate import validate_data
    from segment import segment_customers
    from features import build_features

    # ── Run full pipeline up to training ──
    df = load_data()
    is_valid = validate_data(df)

    if not is_valid:
        print("❌ Pipeline stopped — data validation failed")
        exit()

    df = segment_customers(df)
    X_train, X_test, y_train, y_test, feature_names = build_features(df)

    # ── Train both models ──
    baseline_model, baseline_metrics = train_baseline(X_train, X_test, y_train, y_test)
    champion_model, champion_metrics = train_champion(X_train, X_test, y_train, y_test)

    # ── Save best model ──
    best_model = save_best_model(
        baseline_metrics, champion_metrics,
        baseline_model, champion_model
    )

    print("\n🎉 Training pipeline complete!")
    print("Run 'mlflow ui' in your terminal to view experiment results.")