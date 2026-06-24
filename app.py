"""
app.py  —  Customer Churn Prediction Dashboard
Run with:  streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib, json, os
from utils.preprocess import load_and_clean, encode_features

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Churn Predictor",
    page_icon="📉",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Dark sidebar */
    [data-testid="stSidebar"] { background-color: #0f172a; }
    [data-testid="stSidebar"] * { color: #e2e8f0 !important; }

    /* Metric cards */
    [data-testid="metric-container"] {
        background: #1e293b;
        border-radius: 10px;
        padding: 16px;
        border-left: 4px solid #6366f1;
    }

    /* Section headings */
    h2 { color: #6366f1 !important; }
    h3 { color: #94a3b8 !important; font-size: 1rem !important; }

    /* Page background */
    .main { background-color: #0f172a; color: #e2e8f0; }
</style>
""", unsafe_allow_html=True)

MODELS_DIR = "models"


# ── Helpers ───────────────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    if not os.path.exists(f"{MODELS_DIR}/meta.json"):
        return None, None, None, None, None
    lr     = joblib.load(f"{MODELS_DIR}/logistic_regression.pkl")
    rf     = joblib.load(f"{MODELS_DIR}/random_forest.pkl")
    scaler = joblib.load(f"{MODELS_DIR}/scaler.pkl")
    encoders = joblib.load(f"{MODELS_DIR}/encoders.pkl")
    with open(f"{MODELS_DIR}/meta.json") as f:
        meta = json.load(f)
    return lr, rf, scaler, encoders, meta


@st.cache_data
def load_dataset(path):
    df_raw = load_and_clean(path)
    df_enc, _ = encode_features(df_raw.copy())
    return df_raw, df_enc


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/combo-chart.png", width=60)
    st.title("Churn Predictor")
    st.markdown("---")

    uploaded = st.file_uploader("📂 Upload Telco CSV", type=["csv"])
    st.markdown("---")
    page = st.radio(
        "Navigate",
        ["🏠 Overview", "🔍 EDA", "🤖 Model Comparison", "🎯 Predict Single Customer"],
    )
    st.markdown("---")
    st.caption("Dataset: [Kaggle Telco Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)")


# ── Load data / models ────────────────────────────────────────────────────────
lr, rf, scaler, encoders, meta = load_models()

df_raw, df_enc = None, None
if uploaded:
    import tempfile, os
    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
        tmp.write(uploaded.getvalue())
        tmp_path = tmp.name
    df_raw, df_enc = load_dataset(tmp_path)
    os.unlink(tmp_path)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: OVERVIEW
# ═══════════════════════════════════════════════════════════════════════════════
if page == "🏠 Overview":
    st.title("📉 Customer Churn Dashboard")
    st.markdown("Predict and understand customer churn using ML — built with Python & Streamlit.")

    if df_raw is None:
        st.info("👆 Upload your Telco CSV from the sidebar to get started.")
        st.markdown("""
        ### What this app does
        - 📊 **EDA** — Visual exploration of churn patterns
        - 🤖 **Model Comparison** — Logistic Regression vs Random Forest metrics
        - 🎯 **Single Prediction** — Enter a customer's details and get churn probability
        
        ### How to set up
        ```bash
        # 1. Install dependencies
        pip install -r requirements.txt

        # 2. Train models (run once)
        python train_models.py --data data/WA_Fn-UseC_-Telco-Customer-Churn.csv

        # 3. Launch dashboard
        streamlit run app.py
        ```
        """)
        st.stop()

    total     = len(df_raw)
    churned   = int(df_raw["Churn"].sum())
    retained  = total - churned
    churn_pct = round(churned / total * 100, 1)
    avg_tenure = round(df_raw["tenure"].mean(), 1)
    avg_monthly = round(df_raw["MonthlyCharges"].mean(), 2)

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Customers", f"{total:,}")
    c2.metric("Churned",         f"{churned:,}", f"{churn_pct}%")
    c3.metric("Retained",        f"{retained:,}")
    c4.metric("Avg Tenure",      f"{avg_tenure} mo")
    c5.metric("Avg Monthly $",   f"${avg_monthly}")

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        fig = px.pie(
            values=[churned, retained],
            names=["Churned", "Retained"],
            color_discrete_sequence=["#f43f5e", "#6366f1"],
            title="Churn Distribution",
            hole=0.5,
        )
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color="#e2e8f0")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        tenure_fig = px.histogram(
            df_raw, x="tenure", color=df_raw["Churn"].map({1: "Churned", 0: "Retained"}),
            barmode="overlay", nbins=30,
            color_discrete_map={"Churned": "#f43f5e", "Retained": "#6366f1"},
            title="Churn by Tenure",
            labels={"tenure": "Tenure (months)", "color": "Status"},
        )
        tenure_fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color="#e2e8f0")
        st.plotly_chart(tenure_fig, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: EDA
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🔍 EDA":
    st.title("🔍 Exploratory Data Analysis")

    if df_raw is None:
        st.warning("Please upload a CSV first.")
        st.stop()

    tab1, tab2, tab3 = st.tabs(["Categorical Features", "Numeric Features", "Correlation"])

    with tab1:
        cat_cols = ["Contract", "InternetService", "PaymentMethod", "TechSupport"]
        for col in cat_cols:
            if col not in df_raw.columns:
                continue
            grp = df_raw.groupby(col)["Churn"].mean().reset_index()
            grp["Churn %"] = (grp["Churn"] * 100).round(1)
            fig = px.bar(
                grp, x=col, y="Churn %",
                color="Churn %", color_continuous_scale="RdPu",
                title=f"Churn Rate by {col}",
            )
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color="#e2e8f0")
            st.plotly_chart(fig, use_container_width=True)

    with tab2:
        num_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
        for col in num_cols:
            fig = px.box(
                df_raw, x=df_raw["Churn"].map({1: "Churned", 0: "Retained"}),
                y=col, color=df_raw["Churn"].map({1: "Churned", 0: "Retained"}),
                color_discrete_map={"Churned": "#f43f5e", "Retained": "#6366f1"},
                title=f"{col} — Churned vs Retained",
            )
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color="#e2e8f0")
            st.plotly_chart(fig, use_container_width=True)

    with tab3:
        corr = df_enc.corr()[["Churn"]].sort_values("Churn", ascending=False)
        fig = px.bar(
            corr.reset_index(), x="index", y="Churn",
            color="Churn", color_continuous_scale="RdBu_r",
            title="Feature Correlation with Churn",
            labels={"index": "Feature", "Churn": "Correlation"},
        )
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color="#e2e8f0")
        st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: MODEL COMPARISON
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🤖 Model Comparison":
    st.title("🤖 Model Performance Comparison")

    if meta is None:
        st.warning("Models not trained yet. Run `python train_models.py --data <path>` first.")
        st.stop()

    lr_r = meta["lr_results"]
    rf_r = meta["rf_results"]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("LR Accuracy",  f"{lr_r['accuracy']*100:.1f}%")
    c2.metric("LR AUC",       f"{lr_r['roc_auc']:.3f}")
    c3.metric("RF Accuracy",  f"{rf_r['accuracy']*100:.1f}%",
              delta=f"{(rf_r['accuracy']-lr_r['accuracy'])*100:+.1f}%")
    c4.metric("RF AUC",       f"{rf_r['roc_auc']:.3f}",
              delta=f"{rf_r['roc_auc']-lr_r['roc_auc']:+.3f}")

    st.markdown("---")
    col1, col2 = st.columns(2)

    # Bar comparison
    with col1:
        compare_df = pd.DataFrame({
            "Metric": ["Accuracy", "ROC-AUC"],
            "Logistic Regression": [lr_r["accuracy"], lr_r["roc_auc"]],
            "Random Forest":       [rf_r["accuracy"], rf_r["roc_auc"]],
        })
        fig = px.bar(
            compare_df.melt("Metric"), x="Metric", y="value",
            color="variable", barmode="group",
            color_discrete_map={"Logistic Regression": "#6366f1", "Random Forest": "#f43f5e"},
            title="Accuracy & AUC Comparison",
        )
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color="#e2e8f0", yaxis_range=[0, 1])
        st.plotly_chart(fig, use_container_width=True)

    # Feature importance (RF)
    with col2:
        if rf is not None and meta.get("feature_names"):
            fi = pd.DataFrame({
                "Feature":   meta["feature_names"],
                "Importance": rf.feature_importances_,
            }).sort_values("Importance", ascending=False).head(12)

            fig2 = px.bar(
                fi, x="Importance", y="Feature", orientation="h",
                color="Importance", color_continuous_scale="Purples",
                title="Top 12 Feature Importances (Random Forest)",
            )
            fig2.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color="#e2e8f0",
                               yaxis=dict(autorange="reversed"))
            st.plotly_chart(fig2, use_container_width=True)

    # Confusion matrices
    st.markdown("### Confusion Matrices")
    cm1, cm2 = st.columns(2)
    for col, results, name in [(cm1, lr_r, "Logistic Regression"), (cm2, rf_r, "Random Forest")]:
        cm = np.array(results["confusion_matrix"])
        fig = px.imshow(
            cm, text_auto=True,
            x=["Predicted: No", "Predicted: Yes"],
            y=["Actual: No",    "Actual: Yes"],
            color_continuous_scale="Blues",
            title=f"{name} — Confusion Matrix",
        )
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color="#e2e8f0")
        col.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: SINGLE PREDICTION
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🎯 Predict Single Customer":
    st.title("🎯 Predict — Single Customer")

    if lr is None or rf is None:
        st.warning("Models not trained yet. Run `python train_models.py --data <path>` first.")
        st.stop()

    feature_names = meta["feature_names"]

    st.markdown("Fill in the customer details below:")

    col1, col2, col3 = st.columns(3)

    with col1:
        gender         = st.selectbox("Gender", ["Male", "Female"])
        senior         = st.selectbox("Senior Citizen", ["No", "Yes"])
        partner        = st.selectbox("Has Partner", ["Yes", "No"])
        dependents     = st.selectbox("Has Dependents", ["No", "Yes"])
        tenure         = st.slider("Tenure (months)", 0, 72, 12)
        phone_service  = st.selectbox("Phone Service", ["Yes", "No"])

    with col2:
        multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
        internet       = st.selectbox("Internet Service", ["Fiber optic", "DSL", "No"])
        online_sec     = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
        online_bkp     = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
        device_prot    = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
        tech_support   = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])

    with col3:
        streaming_tv   = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
        streaming_mv   = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])
        contract       = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        paperless      = st.selectbox("Paperless Billing", ["Yes", "No"])
        payment        = st.selectbox("Payment Method", [
            "Electronic check", "Mailed check",
            "Bank transfer (automatic)", "Credit card (automatic)"
        ])
        monthly        = st.number_input("Monthly Charges ($)", 18.0, 120.0, 65.0, step=0.5)
        total          = st.number_input("Total Charges ($)", 0.0, 9000.0, monthly * tenure, step=1.0)

    if st.button("🔮 Predict Churn", use_container_width=True):
        raw_row = {
            "gender": gender, "SeniorCitizen": 1 if senior == "Yes" else 0,
            "Partner": partner, "Dependents": dependents, "tenure": tenure,
            "PhoneService": phone_service, "MultipleLines": multiple_lines,
            "InternetService": internet, "OnlineSecurity": online_sec,
            "OnlineBackup": online_bkp, "DeviceProtection": device_prot,
            "TechSupport": tech_support, "StreamingTV": streaming_tv,
            "StreamingMovies": streaming_mv, "Contract": contract,
            "PaperlessBilling": paperless, "PaymentMethod": payment,
            "MonthlyCharges": monthly, "TotalCharges": total,
        }

        row_df = pd.DataFrame([raw_row])

        # Encode using saved encoders
        for col in row_df.select_dtypes(include="object").columns:
            if col in encoders:
                le = encoders[col]
                val = row_df[col].astype(str).iloc[0]
                if val in le.classes_:
                    row_df[col] = le.transform([val])
                else:
                    row_df[col] = 0
            else:
                row_df[col] = 0

        # Align columns
        for f in feature_names:
            if f not in row_df.columns:
                row_df[f] = 0
        row_df = row_df[feature_names]

        row_sc = scaler.transform(row_df)

        lr_prob = lr.predict_proba(row_sc)[0][1]
        rf_prob = rf.predict_proba(row_sc)[0][1]
        avg_prob = (lr_prob + rf_prob) / 2

        st.markdown("---")
        r1, r2, r3 = st.columns(3)
        r1.metric("Logistic Regression", f"{lr_prob*100:.1f}% churn risk")
        r2.metric("Random Forest",       f"{rf_prob*100:.1f}% churn risk")
        r3.metric("Ensemble (avg)",      f"{avg_prob*100:.1f}% churn risk")

        if avg_prob >= 0.7:
            st.error(f"🚨 HIGH risk of churn ({avg_prob*100:.1f}%). Immediate retention action recommended.")
        elif avg_prob >= 0.4:
            st.warning(f"⚠️ MODERATE churn risk ({avg_prob*100:.1f}%). Consider a loyalty offer.")
        else:
            st.success(f"✅ LOW churn risk ({avg_prob*100:.1f}%). Customer appears stable.")

        # Gauge chart
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=avg_prob * 100,
            title={"text": "Ensemble Churn Probability (%)"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#6366f1"},
                "steps": [
                    {"range": [0,  40], "color": "#22c55e"},
                    {"range": [40, 70], "color": "#f59e0b"},
                    {"range": [70, 100], "color": "#f43f5e"},
                ],
            },
        ))
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color="#e2e8f0", height=300)
        st.plotly_chart(fig, use_container_width=True)
