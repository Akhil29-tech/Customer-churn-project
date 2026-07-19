# 📉 Customer Churn Prediction Dashboard

A full end-to-end Data Science project: EDA → ML modelling → Streamlit dashboard.

Built with Python, Scikit-learn, and Streamlit.

Live Demo - https://customer-churn-projects.streamlit.app/

---

## 🗂️ Project Structure

```
churn_predictor/
├── app.py                  ← Streamlit dashboard (main entry point)
├── train_models.py         ← Train & save ML models
├── requirements.txt
├── data/
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv   ← Download from Kaggle
├── models/
│   ├── logistic_regression.pkl
│   ├── random_forest.pkl
│   ├── scaler.pkl
│   ├── encoders.pkl
│   └── meta.json
├── notebooks/
│   └── 01_EDA.ipynb        ← Exploratory analysis
└── utils/
    └── preprocess.py       ← Shared data cleaning & encoding
```

---

## 🚀 Quick Start

### 1. Get the Dataset
Download from Kaggle:
👉 https://www.kaggle.com/datasets/blastchar/telco-customer-churn

Place the CSV inside the `data/` folder.

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Train the Models (run once)
```bash
python train_models.py --data data/WA_Fn-UseC_-Telco-Customer-Churn.csv
```

### 4. Launch the Dashboard
```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

---

## 📊 Dashboard Pages

| Page | What it shows |
|------|--------------|
| 🏠 Overview | KPI cards, churn distribution, tenure histogram |
| 🔍 EDA | Churn rate by contract/internet/payment, box plots, correlation |
| 🤖 Model Comparison | Accuracy, AUC, confusion matrices, feature importances |
| 🎯 Predict Single Customer | Fill a form → get churn probability with gauge chart |

---

## 🤖 Models Used

| Model | Accuracy | ROC-AUC |
|-------|----------|---------|
| Logistic Regression | ~80% | ~0.84 |
| Random Forest | ~82% | ~0.86 |

*(Actual numbers will appear after training)*

---

## 🧠 Key Findings (from EDA)
- Customers on **Month-to-month contracts** churn at ~43% vs ~3% for two-year contracts
- **Fiber optic** internet users churn more than DSL users
- Customers with **low tenure (< 12 months)** are the highest risk group
- **Electronic check** payment method has the highest churn rate

---

## 🔗 Dataset Credit
[IBM Telco Customer Churn — Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
