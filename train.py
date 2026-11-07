import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, classification_report
import joblib
import os

# 1. Load data - first 500,000 rows, only TRANSFER and CASH_OUT
df = pd.read_csv("data/PS_20174392719_1491204439457_log.csv", nrows=500000)
df = df[df["type"].isin(["TRANSFER", "CASH_OUT"])]

# 2. Make 2 new features from the balances
df["orig_diff"] = df["oldbalanceOrg"] - df["newbalanceOrig"]
df["dest_diff"] = df["newbalanceDest"] - df["oldbalanceDest"]

# 3. Pick the columns the model will use
features = ["step", "amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest", "orig_diff", "dest_diff"]
X = df[features]
y = df["isFraud"]

# 4. Split into train and test, keeping the same fraud rate in both
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 5. Train with MLflow tracking
mlflow.set_experiment("momo-fraud")

with mlflow.start_run():
    model = RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    auc = roc_auc_score(y_test, y_prob)

    print(classification_report(y_test, y_pred, zero_division=0))
    print(f"Precision: {precision:.3f}")
    print(f"Recall: {recall:.3f}")
    print(f"F1: {f1:.3f}")
    print(f"ROC-AUC: {auc:.3f}")

    mlflow.log_param("model", "RandomForest")
    mlflow.log_param("n_estimators", 100)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1", f1)
    mlflow.log_metric("roc_auc", auc)
   

    # 6. Save the model for the API later
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/model.pkl")
    print("Model saved to models/model.pkl")
