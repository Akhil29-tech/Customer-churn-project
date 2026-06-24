import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler

def load_and_clean(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df.dropna(subset=["TotalCharges"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    df.drop(columns=["customerID"], inplace=True, errors="ignore")
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})
    return df

def encode_features(df: pd.DataFrame):
    df = df.copy()
    encoders = {}
    for col in df.select_dtypes(include="object").columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le
    return df, encoders

def split_and_scale(df: pd.DataFrame, target: str = "Churn"):
    from sklearn.model_selection import train_test_split
    X = df.drop(columns=[target])
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)
    return X_train_sc, X_test_sc, y_train, y_test, scaler, list(X.columns)
