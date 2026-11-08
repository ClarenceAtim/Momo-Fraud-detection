# Momo-Fraud-detection
# Momo Fraud Detection

[![CI](https://github.com/ClarenceAtim/Momo-Fraud-detection/actions/workflows/ci.yml/badge.svg)](https://github.com/ClarenceAtim/Momo-Fraud-detection/actions/workflows/ci.yml)

A live fraud checker for Mobile Money style transactions (MTN MoMo / Orange Money style). Send one transaction, get a fraud probability back in under a second.

## Problem

Mobile Money fraud often empties an account to zero in one transfer, to a receiver who had zero before. This project learns those patterns from labelled past transactions and scores new ones in real time.

## Data

Synthetic PaySim data (Kaggle `ealaxi/paysim1`, CC BY-SA 4.0), 6,362,620 transactions, 8,213 fraud. Fraud is concentrated in TRANSFER and CASH_OUT, so training uses those two types only. The CSV is not in this repo, it is gitignored, download it yourself into `data/`.

This is synthetic data for learning. There is no partnership with MTN or Orange, and no real customer data is used.

## Model

RandomForest, trained on the first 500,000 rows, TRANSFER and CASH_OUT only.

Features: `step`, `amount`, `oldbalanceOrg`, `newbalanceOrig`, `oldbalanceDest`, `newbalanceDest`, plus engineered `orig_diff` and `dest_diff`.

Test set: 44,610 transactions, 47 fraud.

| Metric | Score |
| --- | --- |
| Precision (fraud) | 0.846 |
| Recall (fraud) | 0.468 |
| F1 (fraud) | 0.603 |
| ROC-AUC | 0.965 |

Accuracy is 1.00 and misleading, fraud is only 0.116% of the data, so this project is judged on precision, recall, F1 and ROC-AUC, not accuracy.

## Project structure