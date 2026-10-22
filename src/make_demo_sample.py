"""Save a small set of real payments from the test weeks for the dashboard's "Try a real payment" button.

The full CSVs are too big to load in the dashboard, so this picks 200 payments once
(50 fraud, 150 genuine) from test weeks 22-26, which the model never trained on.

Run from the project root:
    uv run python -m src.make_demo_sample
"""
from pathlib import Path

import pandas as pd

from src.train import load_data

OUT = Path("data/processed/demo_payments.csv")


def main():
    df_merged = load_data()
    test = df_merged[df_merged["week"] >= 22]
    del df_merged

    fraud = test[test["isFraud"] == 1].sample(50, random_state=42)
    genuine = test[test["isFraud"] == 0].sample(150, random_state=42)
    sample = pd.concat([fraud, genuine]).sample(frac=1, random_state=42)   # shuffle

    OUT.parent.mkdir(parents=True, exist_ok=True)
    sample.drop(columns=["week"]).to_csv(OUT, index=False)
    print("saved", len(sample), "payments to", OUT)


if __name__ == "__main__":
    main()
