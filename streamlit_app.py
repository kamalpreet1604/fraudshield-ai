"""FraudShield AI dashboard: a simple screen for a fraud analyst.

It talks to the FastAPI service, so start that first:
    uv run uvicorn src.api:app --reload
Then, in a second terminal:
    uv run streamlit run streamlit_app.py
"""
import json
import os
from pathlib import Path

import pandas as pd
import requests
import streamlit as st

API_URL = os.environ.get("FRAUD_API_URL", "http://127.0.0.1:8000")
DEMO_PATH = Path(__file__).parent / "data" / "processed" / "demo_payments.csv"   # made by: uv run python -m src.make_demo_sample

st.set_page_config(page_title="FraudShield AI", page_icon=":material/shield:", layout="wide")


def api_is_up() -> bool:
    try:
        # first call after the API starts loads the model, so allow a few seconds
        return requests.get(f"{API_URL}/health", timeout=15).ok
    except requests.RequestException:
        return False


@st.cache_data
def load_demo_payments() -> pd.DataFrame | None:
    """200 real payments from the test weeks (50 fraud, 150 genuine), with all their columns."""
    return pd.read_csv(DEMO_PATH) if DEMO_PATH.exists() else None


def send_payment(payment: dict, truth: str | None = None):
    r = requests.post(f"{API_URL}/predict", json=payment, timeout=10)
    if r.ok:
        st.session_state["last_result"] = r.json()
        st.session_state["last_truth"] = truth
    else:
        st.error(f"The API refused the payment: {r.text}")


st.title(":material/shield: FraudShield AI")
st.caption("Score a card payment and review the latest decisions. Model: XGBoost, cut-off 0.45.")

if not api_is_up():
    st.error(
        f"Can't reach the fraud API at {API_URL}. Start it first with "
        "`uv run uvicorn src.api:app --reload`, then refresh this page.",
        icon=":material/cloud_off:",
    )
    st.stop()

left, right = st.columns([1, 1], gap="large")

# ---- left: the payment form ----
with left:
    with st.form("payment"):
        st.subheader("Check a payment")
        amount = st.number_input("Amount ($)", min_value=0.01, value=120.0, step=10.0)
        product = st.selectbox("Product type", ["W", "C", "R", "H", "S"])
        card_type = st.segmented_control("Card type", ["credit", "debit"], default="debit")
        email = st.selectbox(
            "Buyer email domain",
            ["gmail.com", "yahoo.com", "hotmail.com", "outlook.com", "anonymous.com", "aol.com", "att.net", "(none)"],
        )
        device = st.segmented_control("Device", ["desktop", "mobile"], default="desktop")
        submitted = st.form_submit_button("Check payment", type="primary", icon=":material/search:")

    if submitted:
        payment = {"TransactionAmt": amount, "ProductCD": product}
        if card_type:
            payment["card6"] = card_type
        if email != "(none)":
            payment["P_emaildomain"] = email
        if device:
            payment["DeviceType"] = device
        send_payment(payment)

    # a real payment from the test weeks, with all 400+ columns and its true answer
    with st.container(border=True):
        st.subheader("Try a real payment")
        demo = load_demo_payments()
        if demo is None:
            st.caption("Run `uv run python -m src.make_demo_sample` once to enable this.")
        else:
            st.caption("Picks a payment from the test weeks, which the model never trained on.")
            kind = st.segmented_control("Pick from", ["Any", "Fraud", "Genuine"], default="Any")
            if st.button("Try a real payment", icon=":material/shuffle:"):
                pool = demo
                if kind == "Fraud":
                    pool = demo[demo["isFraud"] == 1]
                elif kind == "Genuine":
                    pool = demo[demo["isFraud"] == 0]
                row = pool.sample(1).iloc[0]
                truth = "fraud" if row["isFraud"] == 1 else "genuine"
                # empty values are left out (they count as missing); to_json turns numpy numbers into plain ones
                payment = json.loads(row.drop(["isFraud", "TransactionID", "TransactionDT"]).dropna().to_json())
                send_payment(payment, truth)

# ---- right: the answer for the last payment ----
with right:
    with st.container(border=True):
        st.subheader("Result")
        result = st.session_state.get("last_result")
        if result is None:
            st.info("Fill in a payment and press **Check payment**.", icon=":material/arrow_back:")
        else:
            st.metric("Fraud score", f"{result['fraud_score']:.1%}")
            if result["decision"] == "review":
                st.badge("Review: send to a fraud analyst", icon=":material/warning:", color="red")
            else:
                st.badge("Allow: payment goes through", icon=":material/check_circle:", color="green")
            truth = st.session_state.get("last_truth")
            if truth is not None:
                caught = (truth == "fraud") == (result["decision"] == "review")
                st.markdown(
                    f"**Real answer:** this payment was **{truth}**. "
                    + (":green[The model got it right.]" if caught else ":red[The model got it wrong.]")
                )
                st.caption(f"Payment #{result['id']}, a real payment with all its columns. "
                           f"Anything at or above {result['cutoff']:.0%} goes to review.")
            else:
                st.caption(
                    f"Payment #{result['id']}. Anything at or above {result['cutoff']:.0%} goes to review. "
                    "The form sends only a few fields, so scores run low; try a real payment for the full picture."
                )

st.divider()

# ---- bottom: history from the SQLite database ----
history = pd.DataFrame(requests.get(f"{API_URL}/history", params={"limit": 500}, timeout=10).json())

st.subheader("Recent decisions")
if history.empty:
    st.info("No payments scored yet.")
else:
    reviewed = history[history["decision"] == "review"]
    with st.container(horizontal=True):
        st.metric("Payments scored", len(history), border=True)
        st.metric("Sent to review", len(reviewed), border=True)
        st.metric("Review rate", f"{len(reviewed) / len(history):.0%}", border=True)
        st.metric("Money under review", f"${reviewed['amount'].sum():,.0f}", border=True)

    chart_col, table_col = st.columns([1, 2], gap="large")
    with chart_col:
        with st.container(border=True):
            st.markdown("**Allowed vs review**")
            counts = history["decision"].value_counts().rename_axis("decision").reset_index(name="payments")
            st.bar_chart(counts, x="decision", y="payments", color="decision")
    with table_col:
        st.dataframe(
            history[["id", "created_at", "amount", "product", "fraud_score", "decision"]],
            hide_index=True,
            column_config={
                "id": st.column_config.NumberColumn("#", width="small"),
                "created_at": st.column_config.DatetimeColumn("Time (UTC)", format="D MMM, HH:mm:ss"),
                "amount": st.column_config.NumberColumn("Amount", format="dollar"),
                "product": "Product",
                "fraud_score": st.column_config.ProgressColumn("Fraud score", min_value=0, max_value=1, format="percent"),
                "decision": "Decision",
            },
        )
