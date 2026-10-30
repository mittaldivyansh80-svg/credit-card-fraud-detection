import os
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

TARGET = "is_fraud"
ID_COLUMN = "transaction_id"
FEATURES = [
    "amount",
    "transaction_hour",
    "merchant_category",
    "foreign_transaction",
    "location_mismatch",
    "device_trust_score",
    "velocity_last_24h",
    "cardholder_age",
]
NUMERIC_FEATURES = [
    "amount", "transaction_hour", "foreign_transaction",
    "location_mismatch", "device_trust_score",
    "velocity_last_24h", "cardholder_age",
]
CATEGORICAL_FEATURES = ["merchant_category"]

def train_model(csv_path, model_path, metadata_path):
    df = pd.read_csv(csv_path)
    required = FEATURES + [TARGET]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError("Dataset is missing required columns: " + ", ".join(missing))
    df = df[required].dropna().copy()
    df[TARGET] = df[TARGET].astype(int)
    if not set(df[TARGET].unique()).issuperset({0, 1}):
        raise ValueError("is_fraud must contain both 0 (legitimate) and 1 (fraud).")

    X = df[FEATURES]
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    numeric_pipe = Pipeline([("imputer", SimpleImputer(strategy="median"))])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])
    prep = ColumnTransformer([
        ("numeric", numeric_pipe, NUMERIC_FEATURES),
        ("categorical", categorical_pipe, CATEGORICAL_FEATURES)
    ])
    model = Pipeline([
        ("preprocess", prep),
        ("classifier", RandomForestClassifier(
            n_estimators=180, class_weight="balanced_subsample",
            min_samples_leaf=2, random_state=42, n_jobs=-1
        ))
    ])
    model.fit(X_train, y_train)
    scores = model.predict_proba(X_test)[:, 1]
    predictions = (scores >= 0.5).astype(int)
    metadata = {
        "trained_on_supplied_dataset": True,
        "rows": int(len(df)),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "legitimate_count": int((y == 0).sum()),
        "fraud_count": int((y == 1).sum()),
        "precision": float(precision_score(y_test, predictions, zero_division=0)),
        "recall": float(recall_score(y_test, predictions, zero_division=0)),
        "f1": float(f1_score(y_test, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, scores)),
        "features": FEATURES,
        "merchant_categories": sorted(df["merchant_category"].astype(str).unique().tolist()),
    }
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)
    joblib.dump(metadata, metadata_path)
    return model, metadata, df
