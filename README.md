# FraudShield AI

Fraud detection and risk scoring on the IEEE-CIS Fraud Detection dataset.
Portfolio project, not connected to any real bank.

## The problem

Say someone's card details get stolen on a Sunday night. By Monday morning there are six online orders on it: headphones, a gift card, two phone chargers, and some more gift cards. None of them were made by the card owner. The shop ships the goods, the bank refunds the owner, and somebody has to pay for all of it.

Now flip it around. A college student orders a laptop at 2 am, the first big purchase they have ever made with their card. The payment gets blocked because it "looks unusual". They have to call the bank, wait on hold, and the sale might be lost.

Both of these are mistakes. The first one is a **missed fraud**, and it costs the full amount of the order. The second one is a **false alarm**: a genuine customer gets stopped, and someone has to spend time reviewing the alert.

Nobody can check every payment by hand. What I want to build is a system that gives each transaction a risk score and sends only the riskiest ones to a fraud analyst. The aim is not just "catch as much fraud as possible". It is to catch the fraud that costs the most money, without burying the analyst in false alarms.

So I measure the model the way a fraud team would think about it:

- how much fraud **money** it catches, not just how many fraud transactions
- how many of the top 50 or 100 alerts are really fraud, because an analyst can only review so many in a day
- the total cost of a review threshold: missed fraud costs its amount, and every alert costs a fixed review cost

## What I found in the data so far

The data is about 590,000 real e-commerce transactions over roughly 6 months, and only 3.5% of them are fraud.

- Transactions with device information are about 4 times more likely to be fraud (7.96%) than those without (2.10%).
- Fraud payments are only slightly larger than genuine ones (median $75 vs $68.50), so the amount alone is a weak clue.
- Product type C has 11.7% fraud. It is only 12% of transactions but holds about 39% of all fraud.
- Credit cards have 6.7% fraud against 2.4% for debit cards.
- outlook.com (9.5%) and hotmail.com (5.3%) email addresses are riskier, while att.net (0.7%) is very safe.

Fraud is not steady over time. It is lowest in the first few weeks and moves between about 3% and 5% after that, so I split the data by time and check the fraud rate in each part.

![Fraud rate per week](reports/figures/weekly_fraud_rate.png)

## Plan

Core (2 weeks): data and EDA → time-based split → Logistic Regression baseline → XGBoost → cost-based threshold → FastAPI → Streamlit → Docker / Cloud Run.

Upgrade after the core: Firestore and Pub/Sub replay.

Work in progress.

## Data

The data is not in this repo (Kaggle competition rules). To get it:

1. Join the competition and accept the rules: https://www.kaggle.com/competitions/ieee-fraud-detection/rules
2. Put your Kaggle API token in `~/.kaggle/` (kaggle.com → Settings → API).
3. Run:

```
uv run kaggle competitions download -c ieee-fraud-detection -p data/raw
```

Then unzip into `data/raw/`.
