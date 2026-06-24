"""
train_models.py
---------------
Run this once to train & save both models:
    python train_models.py --data data/WA_Fn-UseC_-Telco-Customer-Churn.csv
"""
import argparse, joblib, json, os
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, roc_auc_score,
    classification_report, confusion_matrix
)
from utils.preprocess import load_and_clean, encode_features, split_and_scale

MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)


def evaluate(model, X_test, y_test, name):
    preds = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]
    results = {
        "model": name,
        "accuracy": round(accuracy_score(y_test, preds), 4),
        "roc_auc":  round(roc_auc_score(y_test, proba), 4),
        "report":   classification_report(y_test, preds, output_dict=True),
        "confusion_matrix": confusion_matrix(y_test, preds).tolist(),
    }
    return results


def main(data_path):
    print(f"Loading data from {data_path} …")
    df = load_and_clean(data_path)
    df_enc, encoders = encode_features(df)
    X_train, X_test, y_train, y_test, scaler, feature_names = split_and_scale(df_enc)

    # --- Logistic Regression ---
    print("Training Logistic Regression …")
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_train, y_train)
    lr_results = evaluate(lr, X_test, y_test, "Logistic Regression")
    print(f"  Accuracy: {lr_results['accuracy']}  |  AUC: {lr_results['roc_auc']}")

    # --- Random Forest ---
    print("Training Random Forest …")
    rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    rf_results = evaluate(rf, X_test, y_test, "Random Forest")
    print(f"  Accuracy: {rf_results['accuracy']}  |  AUC: {rf_results['roc_auc']}")

    # --- Save artefacts ---
    joblib.dump(lr,     f"{MODELS_DIR}/logistic_regression.pkl")
    joblib.dump(rf,     f"{MODELS_DIR}/random_forest.pkl")
    joblib.dump(scaler, f"{MODELS_DIR}/scaler.pkl")
    joblib.dump(encoders, f"{MODELS_DIR}/encoders.pkl")

    meta = {
        "feature_names": feature_names,
        "lr_results": lr_results,
        "rf_results": rf_results,
    }
    with open(f"{MODELS_DIR}/meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print("\n✅ All models & metadata saved to /models/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to Telco CSV file")
    args = parser.parse_args()
    main(args.data)
