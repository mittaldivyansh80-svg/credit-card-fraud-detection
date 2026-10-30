# Credit Card Fraud Detection Dashboard

This project is configured for the dataset you supplied (`creditcard.csv.csv`). The file is included in `data/creditcard.csv`.

## Features
- One Random Forest model only.
- The model is trained from the included dataset on first app launch.
- Single transaction form with all model input features.
- CSV batch prediction and downloadable results.
- No synthetic demo-data fallback.

## Run on Windows / VS Code
Open this folder in VS Code and run these commands in Terminal:

```powershell
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

The first launch trains and evaluates the model using a stratified 75/25 train-test split. Metrics are saved in `models/metadata.joblib`; the dashboard focuses on transaction input and the final prediction.

## Fields
- amount
- transaction_hour
- merchant_category
- foreign_transaction
- location_mismatch
- device_trust_score
- velocity_last_24h
- cardholder_age

`is_fraud` is the training target (0 = legitimate, 1 = fraud) and is not entered for a new transaction.
