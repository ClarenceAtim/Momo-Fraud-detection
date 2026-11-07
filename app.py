from fastapi import FastAPI
from pydantic import BaseModel
import joblib

app = FastAPI()

model = joblib.load("models/model.pkl")

class Transaction(BaseModel):
    step: int
    amount: float
    oldbalanceOrg: float
    newbalanceOrig: float
    oldbalanceDest: float
    newbalanceDest: float

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(tx: Transaction):
    orig_diff = tx.oldbalanceOrg - tx.newbalanceOrig
    dest_diff = tx.newbalanceDest - tx.oldbalanceDest

    features = [[tx.step, tx.amount, tx.oldbalanceOrg, tx.newbalanceOrig, tx.oldbalanceDest, tx.newbalanceDest, orig_diff, dest_diff]]

    prob = float(model.predict_proba(features)[0][1])
    is_fraud = prob >= 0.5

    return {"fraud_probability": prob, "is_fraud": is_fraud}