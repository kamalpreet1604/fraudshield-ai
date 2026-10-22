"""Load the saved model and score one payment.

A payment is a dict like {"TransactionAmt": 120.5, "ProductCD": "C", ...}.
Any column that is not given counts as missing (NaN), which XGBoost handles.
"""
from functools import lru_cache
from pathlib import Path

import numpy as np
import joblib
import pandas as pd

MODEL_PATH = Path("models/fraud_model.joblib")


@lru_cache(maxsize=1)
def load_bundle():
    # loaded once, then kept in memory for every next request
    return joblib.load(MODEL_PATH)


def prepare(payment: dict) -> pd.DataFrame:
    """Turn one payment into a one-row table, prepared exactly like the training data."""
    bundle = load_bundle()
    row = pd.DataFrame([payment]).reindex(columns=bundle["columns"])   # missing columns -> NaN

    row[bundle["num_cols"]] = row[bundle["num_cols"]].apply(pd.to_numeric, errors="coerce").astype("float32")
    for col in bundle["cat_cols"]:
        cats = bundle["categories"][col]
        known = row[col].where(row[col].isin(cats))   # unseen value -> NaN
        row[col] = pd.Categorical(known, categories=cats)
    return row


def score(payment: dict) -> dict:
    bundle = load_bundle()
    proba = float(bundle["model"].predict_proba(prepare(payment))[:, 1][0])
    return {
        "fraud_score": round(proba, 4),
        "decision": "review" if proba >= bundle["cutoff"] else "allow",
        "cutoff": bundle["cutoff"],
    }
