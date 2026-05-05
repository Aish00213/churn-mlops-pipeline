import os
import sys
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import shap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import f1_score, roc_auc_score
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

st.set_page_config(page_title="Churn Prediction Platform", page_icon="📊", layout="wide")

# ── Session state ──
for key in ['model', 'scaler', 'feature_names', 'results_df', 'df_raw',
            'target_col', 'churn_value']:
    if key not in st.session_state:
        st.session_state[key] = None
if 'step' not in st.session_state:
    st.session_state.step = 1


# ══════════════════════════════════════════
#   HELPERS
# ══════════════════════════════════════════
def auto_engineer_features(df, target_col):
    df = df.copy()
    y = df[target_col].copy()
    df = df.drop(columns=[target_col])

    # Drop ID-like columns
    for col in list(df.columns):
        if df[col].dtype == object and df[col].nunique() / len(df) > 0.9:
            df = df.drop(columns=[col])

    # Fix numeric strings
    for col in list(df.columns):
        if df[col].dtype == object:
            try:
                df[col] = pd.to_numeric(df[col], errors='raise')
            except:
                pass

    # Fill missing
    for col in df.columns:
        if df[col].dtype in ['float64', 'int64']:
            df[col] = df[col].fillna(df[col].median())
        else:
            df[col] = df[col].fillna(df[col].mode()[0])

    # Encode categoricals
    for col in list(df.select_dtypes(include='object').columns):
        n = df[col].nunique()
        if n == 2:
            df[col] = LabelEncoder().fit_transform(df[col])
        elif n <= 10:
            dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
            df = pd.concat([df.drop(columns=[col]), dummies], axis=1)
        else:
            df = df.drop(columns=[col])

    # Encode target
    if y.dtype == object:
        y = pd.Series(LabelEncoder().fit_transform(y), name=target_col)

    feature_names = list(df.columns)

    X_train, X_test, y_train, y_test = train_test_split(
        df, y, test_size=0.2, random_state=42, stratify=y
    )

    if y_train.value_counts().min() >= 6:
        try:
            X_train, y_train = SMOTE(random_state=42).fit_resample(X_train, y_train)
        except:
            pass

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    return X_train_s, X_test_s, y_train, y_test, feature_names, scaler


def train_model(X_train, X_test, y_train, y_test):
    model = XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1,
                          random_state=42, eval_metric='logloss', verbosity=0)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    return model, {
        'f1_score': round(f1_score(y_test, y_pred), 4),
        'roc_auc': round(roc_auc_score(y_test, y_prob), 4)
    }


def generate_results(df_raw, model, scaler, feature_names, target_col, churn_value):
    df = df_raw.copy()
    df[target_col] = (df[target_col] == churn_value).astype(int)

    df_feat = df.drop(columns=[target_col])
    for col in list(df_feat.columns):
        if df_feat[col].dtype == object and df_feat[col].nunique() / len(df_feat) > 0.9:
            df_feat = df_feat.drop(columns=[col])
    for col in list(df_feat.columns):
        if df_feat[col].dtype == object:
            try:
                df_feat[col] = pd.to_numeric(df_feat[col], errors='raise')
            except:
                pass
    for col in df_feat.columns:
        if df_feat[col].dtype in ['float64', 'int64']:
            df_feat[col] = df_feat[col].fillna(df_feat[col].median())
        else:
            df_feat[col] = df_feat[col].fillna(df_feat[col].mode()[0])
    for col in list(df_feat.select_dtypes(include='object').columns):
        n = df_feat[col].nunique()
        if n == 2:
            df_feat[col] = LabelEncoder().fit_transform(df_feat[col])
        elif n <= 10:
            dummies = pd.get_dummies(df_feat[col], prefix=col, drop_first=True)
            df_feat = pd.concat([df_feat.drop(columns=[col]), dummies], axis=1)
        else:
            df_feat = df_feat.drop(columns=[col])

    df_feat = df_feat.reindex(columns=feature_names, fill_value=0)
    scaled = scaler.transform(df_feat)
    churn_probs = model.predict_proba(scaled)[:, 1]

    # Find charge/tenure columns
    num_cols = df_raw.select_dtypes(include=['float64', 'int64']).columns.tolist()
    charge_col = next((c for c in num_cols if any(
        w in c.lower() for w in ['charge', 'revenue', 'amount', 'fee', 'price', 'pay'])), None)
    tenure_col = next((c for c in num_cols if any(
        w in c.lower() for w in ['tenure', 'month', 'duration', 'age', 'length'])), None)

    results = []
    for i in range(len(df_raw)):
        prob = round(float(churn_probs[i]), 4)
        monthly = float(df_raw.iloc[i][charge_col]) if charge_col else 50.0
        ten = int(df_raw.iloc[i][tenure_col]) if tenure_col else 12
        clv = round(monthly * max(24 - ten, 6), 2)
        roi = round(clv - 50, 2)

        if prob < 0.5:
            action = "✅ No Action — Low churn risk"
        elif roi > 0:
            action = "🎯 Target for Retention — ROI positive"
        else:
            action = "❌ Do Not Target — Cost exceeds CLV"

        results.append({
            'customer_index': i,
            'churn_probability': prob,
            'churn_prediction': 'Yes' if prob >= 0.5 else 'No',
            'estimated_clv': clv,
            'retention_cost': 50,
            'roi': roi,
            'action': action
        })

    return pd.DataFrame(results)


# ══════════════════════════════════════════
#   HEADER
# ══════════════════════════════════════════
st.title("📊 Churn Prediction Platform")
st.markdown("*Upload any customer dataset — we train a model and predict churn automatically*")
st.divider()

steps = ["1. Upload Data", "2. Configure", "3. Train & Predict", "4. Results"]
cols = st.columns(4)
for i, (col, step) in enumerate(zip(cols, steps)):
    if st.session_state.step == i + 1:
        col.markdown(f"**🔵 {step}**")
    elif st.session_state.step > i + 1:
        col.markdown(f"✅ {step}")
    else:
        col.markdown(f"⚪ {step}")
st.divider()


# ══════════════════════════════════════════
#   STEP 1 — Upload
# ══════════════════════════════════════════
if st.session_state.step == 1:
    st.subheader("📂 Upload Your Customer Dataset")
    st.markdown("""
    This platform works with customer data from **any industry**:
    📱 Telecom &nbsp;|&nbsp; 🏦 Banking &nbsp;|&nbsp; 🎵 SaaS &nbsp;|&nbsp;
    🛒 E-commerce &nbsp;|&nbsp; 🏥 Insurance

    **Only requirement:** Your CSV must have a column indicating whether the customer churned.
    """)

    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.session_state.df_raw = df
        st.success(f"✅ Uploaded — {len(df):,} rows × {len(df.columns)} columns")

        st.subheader("Preview")
        st.dataframe(df.head(), use_container_width=True)

        st.subheader("Column Summary")
        st.dataframe(pd.DataFrame({
            'Column': df.columns,
            'Type': df.dtypes.values,
            'Missing': df.isnull().sum().values,
            'Unique Values': df.nunique().values
        }), use_container_width=True)

        if st.button("Next →", use_container_width=True):
            st.session_state.step = 2
            st.rerun()


# ══════════════════════════════════════════
#   STEP 2 — Configure
# ══════════════════════════════════════════
elif st.session_state.step == 2:
    df = st.session_state.df_raw
    st.subheader("⚙️ Configure Your Dataset")
    st.markdown("Tell us which column represents **customer churn** in your data.")

    churn_candidates = [c for c in df.columns if any(
        w in c.lower() for w in ['churn', 'cancel', 'leave', 'exit', 'attrition'])]
    default_idx = list(df.columns).index(churn_candidates[0]) if churn_candidates else 0

    target_col = st.selectbox("Which column is your churn/target column?",
                               df.columns.tolist(), index=default_idx)

    st.markdown(f"**Unique values in `{target_col}`:**")
    unique_vals = df[target_col].unique().tolist()
    st.write(unique_vals)

    churn_value = st.selectbox("Which value means the customer CHURNED?", unique_vals)

    st.divider()
    st.markdown(f"**{len(df.columns) - 1} feature columns detected** — all other columns will be used for prediction.")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("← Back", use_container_width=True):
            st.session_state.step = 1
            st.rerun()
    with col2:
        if st.button("Train Model →", use_container_width=True):
            st.session_state.target_col = target_col
            st.session_state.churn_value = churn_value
            st.session_state.step = 3
            st.rerun()


# ══════════════════════════════════════════
#   STEP 3 — Train
# ══════════════════════════════════════════
elif st.session_state.step == 3:
    df = st.session_state.df_raw.copy()
    target_col = st.session_state.target_col
    churn_value = st.session_state.churn_value

    st.subheader("🤖 Training Model on Your Data")
    df[target_col] = (df[target_col] == churn_value).astype(int)

    progress = st.progress(0)
    status = st.empty()

    status.text("🔄 Engineering features automatically...")
    progress.progress(25)
    X_train, X_test, y_train, y_test, feature_names, scaler = auto_engineer_features(df, target_col)

    status.text("🔄 Training XGBoost model...")
    progress.progress(60)
    model, metrics = train_model(X_train, X_test, y_train, y_test)

    status.text("🔄 Generating predictions and recommendations...")
    progress.progress(80)
    results_df = generate_results(
        st.session_state.df_raw, model, scaler,
        feature_names, target_col, churn_value
    )

    progress.progress(100)
    status.text("✅ Complete!")

    st.session_state.model = model
    st.session_state.scaler = scaler
    st.session_state.feature_names = feature_names
    st.session_state.results_df = results_df

    st.divider()
    st.subheader("📊 Model Performance on Your Data")
    col1, col2 = st.columns(2)
    col1.metric("F1 Score", metrics['f1_score'],
                delta="Good" if metrics['f1_score'] > 0.6 else "Moderate")
    col2.metric("ROC AUC", metrics['roc_auc'],
                delta="Good" if metrics['roc_auc'] > 0.75 else "Moderate")

    st.info("**F1 Score** balances precision and recall. **ROC AUC** measures separation between churners and non-churners. Both above 0.75 is excellent.")

    if st.button("View Results →", use_container_width=True):
        st.session_state.step = 4
        st.rerun()


# ══════════════════════════════════════════
#   STEP 4 — Results
# ══════════════════════════════════════════
elif st.session_state.step == 4:
    results_df = st.session_state.results_df
    model = st.session_state.model
    feature_names = st.session_state.feature_names

    st.subheader("🎯 Predictions & Recommendations")

    at_risk = results_df[results_df['churn_probability'] >= 0.5]
    worth_retaining = at_risk[at_risk['roi'] > 0]

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Customers", f"{len(results_df):,}")
    col2.metric("Predicted to Churn", f"{len(at_risk):,}",
                delta=f"{len(at_risk)/len(results_df):.1%} of total")
    col3.metric("Worth Retaining", f"{len(worth_retaining):,}")
    col4.metric("Total Potential ROI", f"${worth_retaining['roi'].sum():,.0f}")

    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📊 Churn Probability Distribution")
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.hist(results_df['churn_probability'], bins=20, color='#ff4b4b', alpha=0.8, edgecolor='white')
        ax.axvline(0.5, color='yellow', linestyle='--', linewidth=1.5, label='0.5 threshold')
        ax.set_xlabel("Churn Probability")
        ax.set_ylabel("Customers")
        ax.legend(fontsize=8)
        fig.patch.set_alpha(0)
        ax.set_facecolor('none')
        ax.tick_params(colors='white')
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')
        st.pyplot(fig)
        plt.close()

    with col2:
        st.subheader("🎯 Recommended Actions")
        action_counts = results_df['action'].value_counts()
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.barh(range(len(action_counts)), action_counts.values, color='#4b9eff')
        ax.set_yticks(range(len(action_counts)))
        ax.set_yticklabels([a[:35] for a in action_counts.index], fontsize=8)
        fig.patch.set_alpha(0)
        ax.set_facecolor('none')
        ax.tick_params(colors='white')
        st.pyplot(fig)
        plt.close()

    st.divider()

    # SHAP
    st.subheader("🧠 Top Churn Drivers in Your Data")
    try:
        explainer = shap.TreeExplainer(model)
        df_temp = st.session_state.df_raw.copy()
        target_col = st.session_state.target_col
        churn_value = st.session_state.churn_value
        df_temp[target_col] = (df_temp[target_col] == churn_value).astype(int)
        X_tr, _, _, _, _, _ = auto_engineer_features(df_temp, target_col)
        shap_values = explainer.shap_values(X_tr[:200])
        mean_shap = np.abs(shap_values).mean(axis=0)
        shap_df = pd.DataFrame({'Feature': feature_names, 'Importance': mean_shap})\
            .sort_values('Importance', ascending=False).head(10)

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.barh(shap_df['Feature'][::-1], shap_df['Importance'][::-1], color='#4b9eff')
        ax.set_xlabel("Mean |SHAP Value|")
        ax.set_title("Top 10 Features Driving Churn in Your Dataset")
        fig.patch.set_alpha(0)
        ax.set_facecolor('none')
        ax.tick_params(colors='white')
        ax.xaxis.label.set_color('white')
        ax.title.set_color('white')
        st.pyplot(fig)
        plt.close()
    except Exception:
        st.info("SHAP explanation unavailable for this dataset.")

    st.divider()

    st.subheader("📋 Full Predictions Table")
    st.dataframe(
        results_df.sort_values('churn_probability', ascending=False),
        use_container_width=True
    )

    col1, col2 = st.columns(2)
    with col1:
        st.download_button("⬇️ Download All Predictions",
                           data=results_df.to_csv(index=False).encode('utf-8'),
                           file_name="all_predictions.csv", mime="text/csv",
                           use_container_width=True)
    with col2:
        if len(worth_retaining) > 0:
            st.download_button("⬇️ Download Retention Targets",
                               data=worth_retaining.to_csv(index=False).encode('utf-8'),
                               file_name="retention_targets.csv", mime="text/csv",
                               use_container_width=True)

    st.divider()
    if st.button("🔄 Start Over — Upload New Dataset", use_container_width=True):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()