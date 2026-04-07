import joblib
import json
import pandas as pd
import numpy as np

from sklearn.metrics import roc_auc_score, average_precision_score, f1_score


# =============================
# CONFIG
# =============================
MODEL_PATH = "artifacts/model_20260407_191225.pkl"  
DATA_PATH  = "./data/processed/test.csv"
TARGET = "misstate"


# =============================
# METRICS
# =============================
def precision_at_k(y_true, y_prob, k_ratio=0.01):
    n = len(y_true)
    k = int(n * k_ratio)

    if k == 0:
        return 0.0

    idx = np.argsort(-y_prob)
    top_k_idx = idx[:k]
    return y_true.iloc[top_k_idx].mean()


def evaluate(y_true, y_prob, threshold):
    pred = (y_prob >= threshold).astype(int)

    return {
        "AUC": roc_auc_score(y_true, y_prob),
        "PR_AUC": average_precision_score(y_true, y_prob),
        "F1": f1_score(y_true, pred, zero_division=0),
        "Precision@1%": precision_at_k(y_true, y_prob, 0.01),
        "Precision@5%": precision_at_k(y_true, y_prob, 0.05)
    }


# =============================
# MAIN
# =============================
def main():
    print("===== LOAD MODEL =====")
    model = joblib.load(MODEL_PATH)

    print("===== LOAD DATA =====")
    df = pd.read_csv(DATA_PATH)

    features = [c for c in df.columns if c != TARGET]

    X = df[features]
    y = df[TARGET]

    print("===== PREDICT =====")
    prob = model.predict_proba(X)[:, 1]

    threshold = 0.14363636363636365

    print(f"Using threshold: {threshold}")

    metrics = evaluate(y, prob, threshold)

    print("\n===== TEST RESULT =====")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")


if __name__ == "__main__":
    main()