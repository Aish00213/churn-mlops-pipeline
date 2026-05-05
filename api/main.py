import os
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional

# ── Import recommendation logic ──
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.recommend import calculate_clv, get_action, RETENTION_COSTS


# ── Load model, scaler, feature names on startup ──
# We load these ONCE when the API starts — not on every request
# This makes predictions fast (no disk reads per request)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
processed_dir = os.path.join(BASE_DIR, "data", "processed")

model = joblib.load(os.path.join(processed_dir, "best_model.pkl"))
scaler = joblib.load(os.path.join(processed_dir, "scaler.pkl"))
feature_names = joblib.load(os.path.join(processed_dir, "feature_names.pkl"))


# ── Initialize FastAPI app ──
app = FastAPI(
    title="Churn Prediction API",
    description="Predicts customer churn and recommends retention actions",
    version="1.0.0"
)


# ── Input schema ──
# Pydantic validates every incoming request automatically
# If a required field is missing or wrong type → API returns 422 error
class CustomerInput(BaseModel):
    # Demographics
    gender: int = Field(..., description="0=Female, 1=Male")
    SeniorCitizen: int = Field(..., description="0=No, 1=Yes")
    Partner: int = Field(..., description="0=No, 1=Yes")
    Dependents: int = Field(..., description="0=No, 1=Yes")

    # Service info
    tenure: int = Field(..., description="Months as customer (0-72)")
    PhoneService: int = Field(..., description="0=No, 1=Yes")
    MonthlyCharges: float = Field(..., description="Monthly bill amount")
    TotalCharges: float = Field(..., description="Total amount paid")

    # Contract & billing
    PaperlessBilling: int = Field(..., description="0=No, 1=Yes")

    # One-hot encoded fields
    MultipleLines_Yes: int = Field(0)
    InternetService_Fiber_optic: int = Field(0)
    InternetService_No: int = Field(0)
    OnlineSecurity_Yes: int = Field(0)
    OnlineBackup_Yes: int = Field(0)
    DeviceProtection_Yes: int = Field(0)
    TechSupport_Yes: int = Field(0)
    StreamingTV_Yes: int = Field(0)
    StreamingMovies_Yes: int = Field(0)
    Contract_One_year: int = Field(0)
    Contract_Two_year: int = Field(0)
    PaymentMethod_Credit_card_automatic: int = Field(0)
    PaymentMethod_Electronic_check: int = Field(0)
    PaymentMethod_Mailed_check: int = Field(0)

    # Engineered features
    charges_per_tenure: Optional[float] = Field(None, description="Auto-calculated if not provided")
    is_high_risk: Optional[int] = Field(None, description="Auto-calculated if not provided")
    segment: int = Field(1, description="0=High Value, 1=Low Value, 2=Mid Value")


# ── Output schema ──
class PredictionOutput(BaseModel):
    churn_probability: float
    churn_prediction: str
    clv: float
    retention_cost: int
    action: str
    reason: str
    roi: Optional[float]


# ── Health check endpoint ──
@app.get("/")
def root():
    return {
        "status": "running",
        "model": "XGBoost Churn Predictor",
        "version": "1.0.0"
    }


@app.get("/health")
def health():
    return {"status": "healthy"}


# ── Prediction endpoint ──
@app.post("/predict", response_model=PredictionOutput)
def predict(customer: CustomerInput):
    try:
        # ── Step 1: Convert input to dataframe ──
        data = customer.dict()

        # Auto-calculate engineered features if not provided
        if data['charges_per_tenure'] is None:
            data['charges_per_tenure'] = data['MonthlyCharges'] / (data['tenure'] + 1)

        if data['is_high_risk'] is None:
            data['is_high_risk'] = int(
                data['tenure'] < 12 and
                data['Contract_One_year'] == 0 and
                data['Contract_Two_year'] == 0
            )

        # ── Step 2: Align columns with training feature names ──
        # Map API field names to exact feature names used during training
        field_map = {
            'InternetService_Fiber_optic': 'InternetService_Fiber optic',
            'InternetService_No': 'InternetService_No',
            'OnlineSecurity_Yes': 'OnlineSecurity_Yes',
            'OnlineBackup_Yes': 'OnlineBackup_Yes',
            'DeviceProtection_Yes': 'DeviceProtection_Yes',
            'TechSupport_Yes': 'TechSupport_Yes',
            'StreamingTV_Yes': 'StreamingTV_Yes',
            'StreamingMovies_Yes': 'StreamingMovies_Yes',
            'Contract_One_year': 'Contract_One year',
            'Contract_Two_year': 'Contract_Two year',
            'PaymentMethod_Credit_card_automatic': 'PaymentMethod_Credit card (automatic)',
            'PaymentMethod_Electronic_check': 'PaymentMethod_Electronic check',
            'PaymentMethod_Mailed_check': 'PaymentMethod_Mailed check',
            'MultipleLines_Yes': 'MultipleLines_Yes',
        }

        # Rename fields to match training column names
        for api_name, train_name in field_map.items():
            if api_name in data:
                data[train_name] = data.pop(api_name)

        # Build input dataframe with exact feature order from training
        input_df = pd.DataFrame([data])
        input_df = input_df.reindex(columns=feature_names, fill_value=0)

        # ── Step 3: Scale input ──
        input_scaled = scaler.transform(input_df)

        # ── Step 4: Predict ──
        churn_prob = float(model.predict_proba(input_scaled)[0][1])
        churn_label = "Yes" if churn_prob >= 0.5 else "No"

        # ── Step 5: Calculate CLV and recommendation ──
        clv = calculate_clv(customer.MonthlyCharges, customer.tenure)
        retention_cost = RETENTION_COSTS.get(customer.segment, 50)
        recommendation = get_action(churn_prob, clv, retention_cost, customer.segment)

        return PredictionOutput(
            churn_probability=round(churn_prob, 4),
            churn_prediction=churn_label,
            clv=clv,
            retention_cost=retention_cost,
            action=recommendation['action'],
            reason=recommendation['reason'],
            roi=recommendation['roi']
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))