# 📊 End-to-End MLOps Pipeline — Customer Churn Prediction

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0-orange)
![MLflow](https://img.shields.io/badge/MLflow-Tracking-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-REST_API-green?logo=fastapi)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red?logo=streamlit)
![Docker](https://img.shields.io/badge/Docker-Containerized-blue?logo=docker)
![CI/CD](https://img.shields.io/badge/GitHub_Actions-CI%2FCD-black?logo=githubactions)

A production-grade MLOps pipeline that goes beyond prediction into **business decision-making**. Upload any customer dataset, and the system automatically trains a model, explains predictions with SHAP, calculates Customer Lifetime Value, and recommends ROI-driven retention actions.

---

## 🎯 What Makes This Different

Most ML projects stop at "this customer will churn." This pipeline answers four questions:

| Question | How |
|---|---|
| **Who will churn?** | XGBoost classifier with SMOTE for class imbalance |
| **Why will they churn?** | SHAP explainability — global and per-customer |
| **Is it worth retaining them?** | CLV vs retention cost ROI calculation |
| **What action should we take?** | Segment-based recommendation engine |

---

## 🏗️ Architecture

```
Raw CSV (any industry)
        ↓
Data Ingestion + Validation
        ↓
Customer Segmentation (RFM + KMeans)
        ↓
Feature Engineering + SMOTE
        ↓
Model Training — Baseline + XGBoost (MLflow Tracking)
        ↓
SHAP Explainability Layer
        ↓
CLV + Retention Cost → Action Recommender
        ↓
FastAPI (REST endpoint) + Streamlit (Dashboard)
        ↓
Docker + GitHub Actions CI/CD
```

---

## 📁 Project Structure

```
churn-mlops-pipeline/
│
├── src/
│   ├── ingest.py          # Load and inspect raw data
│   ├── validate.py        # 5 automated data quality checks
│   ├── segment.py         # RFM-based KMeans segmentation
│   ├── features.py        # Feature engineering + SMOTE
│   ├── train.py           # Train models + MLflow tracking
│   ├── explain.py         # SHAP global + local explanations
│   └── recommend.py       # CLV + ROI + action recommender
│
├── api/
│   └── main.py            # FastAPI REST endpoint
│
├── app/
│   └── streamlit_app.py   # Generic multi-industry dashboard
│
├── docker/
│   ├── Dockerfile         # Container definition
│   └── docker-compose.yml # Run API + Dashboard together
│
├── .github/workflows/
│   └── ci.yml             # GitHub Actions CI/CD
│
└── requirements.txt
```

---

## 🚀 Quick Start

### 1. Clone the repository
```bash
git clone https://github.com/Aish00213/churn-mlops-pipeline.git
cd churn-mlops-pipeline
```

### 2. Create virtual environment
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Mac/Linux
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Download the dataset
Download the [Telco Customer Churn dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) and place it at:
```
data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv
```

### 5. Run the full pipeline
```bash
cd src
python train.py       # Runs ingest → validate → segment → features → train
python explain.py     # Generate SHAP explanations
python recommend.py   # Generate business recommendations
```

### 6. Start the API
```bash
cd ..
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```
API docs available at: `http://localhost:8000/docs`

### 7. Start the Dashboard
```bash
streamlit run app/streamlit_app.py
```
Dashboard available at: `http://localhost:8501`

---

## 🐳 Docker

Run both API and Dashboard with one command:
```bash
cd docker
docker-compose up --build
```

| Service | URL |
|---|---|
| FastAPI | http://localhost:8000/docs |
| Streamlit | http://localhost:8501 |

---

## 📊 Model Performance

| Model | F1 Score | ROC AUC | Accuracy |
|---|---|---|---|
| Logistic Regression (Baseline) | 0.5854 | 0.8149 | 0.7587 |
| **XGBoost (Champion)** | **0.5856** | **0.8141** | **0.7679** |

### Top Churn Drivers (SHAP)
1. **MonthlyCharges** — Higher bills = more likely to churn
2. **InternetService_Fiber optic** — Fiber customers churn more
3. **PaymentMethod_Electronic check** — Less committed payment method
4. **charges_per_tenure** — Engineered feature — pay vs loyalty ratio
5. **MultipleLines** — Multiple line subscribers at risk

---

## 🔌 API Usage

### Predict churn for a customer

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "gender": 0,
    "SeniorCitizen": 0,
    "Partner": 0,
    "Dependents": 0,
    "tenure": 2,
    "PhoneService": 1,
    "MonthlyCharges": 70.70,
    "TotalCharges": 151.65,
    "PaperlessBilling": 1,
    "InternetService_Fiber_optic": 1,
    "PaymentMethod_Electronic_check": 1,
    "segment": 1
  }'
```

### Response
```json
{
  "churn_probability": 0.77,
  "churn_prediction": "Yes",
  "clv": 1555.4,
  "retention_cost": 30,
  "action": "💌 Low Priority — Send retention voucher",
  "reason": "CLV ($1555.4) > Retention Cost ($30) → ROI = $1525.4",
  "roi": 1525.4
}
```

---

## 🌍 Generic Dashboard

The Streamlit dashboard works with **any customer dataset** from any industry:

- 📱 Telecom
- 🏦 Banking & Financial Services
- 🎵 SaaS & Streaming
- 🛒 E-commerce & Retail
- 🏥 Insurance

**How it works:**
1. Upload any CSV file
2. Select which column represents churn
3. The app automatically engineers features, trains a model, and generates recommendations
4. Download predictions and retention targets

---

## ⚙️ CI/CD Pipeline

Every push to `main` triggers:

```
Push to main
     ↓
✅ Run Tests
   - Validate all Python files compile
   - Check project structure integrity
     ↓
✅ Build Docker Image
   - Verify container builds successfully
```

---

## 🧠 Key Technical Decisions

**Why SMOTE after train/test split?**
Applying SMOTE before splitting causes data leakage — synthetic rows based on test data would appear in training. Test data must simulate real unseen data.

**Why F1 score, not accuracy?**
74% of customers don't churn. A model predicting "No Churn" always achieves 74% accuracy but is completely useless. F1 balances precision and recall for imbalanced datasets.

**Why segment before modeling?**
A high-value customer churning needs a different response than a low-value one. Segmenting first means the model learns patterns from similar customers rather than averaging across everyone.

**Why save the scaler as a pickle?**
The API must use the exact same scaling learned during training. Refitting the scaler on new data at prediction time would give different results.

---

## 📦 Tech Stack

| Category | Tool |
|---|---|
| ML Model | XGBoost, Scikit-learn |
| Imbalance Handling | SMOTE (imbalanced-learn) |
| Explainability | SHAP |
| Experiment Tracking | MLflow |
| API | FastAPI + Pydantic |
| Dashboard | Streamlit |
| Containerization | Docker + docker-compose |
| CI/CD | GitHub Actions |
| Data Processing | Pandas, NumPy |

---

## 👤 Author

**Aish00213** — [GitHub](https://github.com/Aish00213)

---

*Built as part of an ML Engineer portfolio — demonstrating end-to-end MLOps from raw data to production deployment.*