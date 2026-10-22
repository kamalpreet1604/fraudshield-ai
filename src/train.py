"""Train the final XGBoost model and save everything needed to score new payments.

Same steps as notebooks/02_split_and_features.ipynb (Days 3, 6 and 7):
time split -> drop >90% empty columns -> text columns as categories -> XGBoost.

Run from the project root:
    uv run python -m src.train
"""
from pathlib import Path

import joblib
import pandas as pd
from xgboost import XGBClassifier

RAW = Path("data/raw")
MODELS = Path("models")

# chosen on the validation weeks in Day 7 (cheapest total cost)
CUTOFF = 0.45
REVIEW_COST = 10


def load_data():
    df = pd.read_csv(RAW / "train_transaction.csv")
    df1 = pd.read_csv(RAW / "train_identity.csv")
    df_merged = pd.merge(df, df1, on="TransactionID", how="left")
    df_merged["week"] = df_merged["TransactionDT"] // (86400 * 7)
    return df_merged


def main():
    df_merged = load_data()
    # same time split as the notebook: train weeks 0-17, validation 18-21
    train = df_merged[df_merged["week"] <= 17]
    val = df_merged[(df_merged["week"] >= 18) & (df_merged["week"] <= 21)]
    del df_merged

    drop = ["isFraud", "TransactionID", "TransactionDT", "week"]
    y_train, y_val = train["isFraud"], val["isFraud"]
    X_train, X_val = train.drop(columns=drop), val.drop(columns=drop)
    del train, val

    # drop columns that are more than 90% empty in train
    missing_pct = X_train.isnull().mean() * 100
    cols_to_drop = missing_pct[missing_pct > 90].index
    X_train = X_train.drop(columns=cols_to_drop)
    X_val = X_val.drop(columns=cols_to_drop)

    num_cols = list(X_train.select_dtypes(include="number").columns)
    cat_cols = list(X_train.select_dtypes(exclude="number").columns)
    X_train[num_cols] = X_train[num_cols].astype("float32")
    X_val[num_cols] = X_val[num_cols].astype("float32")

    # category lists learned from train; unseen values in validation become NaN
    categories = {}
    for col in cat_cols:
        X_train[col] = X_train[col].astype("category")
        categories[col] = list(X_train[col].cat.categories)
        known = X_val[col].where(X_val[col].isin(categories[col]))
        X_val[col] = pd.Categorical(known, categories=categories[col])

    pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

    model = XGBClassifier(
        n_estimators=1000,
        learning_rate=0.1,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=pos_weight,
        tree_method="hist",
        enable_categorical=True,
        eval_metric="aucpr",
        early_stopping_rounds=50,
        n_jobs=-1,
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=50)
    print("best tree:", model.best_iteration)

    # one file with the model and everything needed to prepare a new payment
    MODELS.mkdir(exist_ok=True)
    bundle = {
        "model": model,
        "columns": list(X_train.columns),   # exact column order the model expects
        "num_cols": num_cols,
        "cat_cols": cat_cols,
        "categories": categories,
        "cutoff": CUTOFF,
        "review_cost": REVIEW_COST,
    }
    joblib.dump(bundle, MODELS / "fraud_model.joblib")
    print("saved", MODELS / "fraud_model.joblib")


if __name__ == "__main__":
    main()
