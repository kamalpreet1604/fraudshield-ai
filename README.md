# FraudShield AI

Fraud detection and risk scoring on the IEEE-CIS Fraud Detection dataset.
Portfolio project, not connected to any real bank.

Work in progress. Plan: data and EDA → time-based split → Logistic Regression baseline → XGBoost → cost-based threshold → FastAPI → Streamlit → Docker / Cloud Run → Pub/Sub replay.

## Data

The data is not in this repo (Kaggle competition rules). To get it:

1. Join the competition and accept the rules: https://www.kaggle.com/competitions/ieee-fraud-detection/rules
2. Put your Kaggle API token in `~/.kaggle/` (kaggle.com → Settings → API).
3. Run:

```
uv run kaggle competitions download -c ieee-fraud-detection -p data/raw
```

Then unzip into `data/raw/`.
