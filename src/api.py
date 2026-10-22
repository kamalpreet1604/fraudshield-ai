"""FastAPI service: send a payment, get back a fraud score and a decision.

Run from the project root:
    uv run uvicorn src.api:app --reload
Then open http://127.0.0.1:8000/docs to try it in the browser.
"""
from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict

from src.predict import load_bundle, score

app = FastAPI(title="FraudShield AI", description="Fraud score for one card payment.")


class Payment(BaseModel):
    # the most useful fields are listed so /docs shows them;
    # any other dataset column (C1, V258, id_31, ...) can be sent too
    model_config = ConfigDict(extra="allow")

    TransactionAmt: float
    ProductCD: str | None = None
    card4: str | None = None
    card6: str | None = None
    P_emaildomain: str | None = None
    R_emaildomain: str | None = None
    DeviceType: str | None = None
    DeviceInfo: str | None = None


class Prediction(BaseModel):
    fraud_score: float
    decision: str
    cutoff: float


@app.get("/health")
def health():
    load_bundle()   # fails here if the model file is missing
    return {"status": "ok"}


@app.post("/predict", response_model=Prediction)
def predict(payment: Payment):
    # exclude_none: fields left empty count as missing
    return score(payment.model_dump(exclude_none=True))
