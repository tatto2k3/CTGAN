# =============================
# PRODUCTION-READY CTGAN PIPELINE
# =============================

import os
import json
import joblib
import logging
import argparse
import numpy as np
import pandas as pd

from datetime import datetime
from ctgan import CTGAN

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score


# =============================
# CONFIG
# =============================

def get_config():
    return {
        "train_path": "./data/processed/train.csv",
        "val_path": "./data/processed/val.csv",
        "test_path": "./data/processed/test.csv",
        "target": "misstate",
        "ratios": [0.01, 0.05, 0.1],
        "epochs": 300,
        "batch_size": 32,
        "random_state": 42
    }


# =============================
# UTILS
# =============================

def set_seed(seed):
    np.random.seed(seed)


def get_features(df, target):
    return [c for c in df.columns if c != target]


# =============================
# DATA
# =============================

def load_data(cfg):
    return (
        pd.read_csv(cfg["train_path"]),
        pd.read_csv(cfg["val_path"]),
        pd.read_csv(cfg["test_path"])
    )


# =============================
# MODEL
# =============================

def train_model(X, y, cfg):
    model = RandomForestClassifier(
        n_estimators=200,
        random_state=cfg["random_state"],
        n_jobs=-1
    )
    model.fit(X, y)
    return model


def predict(model, X):
    return model.predict_proba(X)[:, 1]


# =============================
# METRICS
# =============================

def find_best_threshold(y_true, y_prob):
    thresholds = np.linspace(0.01, 0.5, 100)
    best_t, best_f1 = 0.5, -1

    for t in thresholds:
        pred = (y_prob >= t).astype(int)
        f1 = f1_score(y_true, pred, zero_division=0)

        if f1 > best_f1:
            best_f1 = f1
            best_t = t

    return best_t, best_f1


def precision_at_k(y_true, y_prob, k_ratio=0.01):
    n = len(y_true)
    k = int(n * k_ratio)

    if k == 0:
        return 0.0

    idx = np.argsort(-y_prob)
    top_k_idx = idx[:k]
    return y_true.iloc[top_k_idx].mean()


def evaluate(y_true, y_prob, threshold=None):
    if threshold is None:
        threshold, f1 = find_best_threshold(y_true, y_prob)
    else:
        pred = (y_prob >= threshold).astype(int)
        f1 = f1_score(y_true, pred, zero_division=0)

    return {
        "AUC": roc_auc_score(y_true, y_prob),
        "PR_AUC": average_precision_score(y_true, y_prob),
        "F1": f1,
        "threshold": threshold,
        "Precision@1%": precision_at_k(y_true, y_prob, 0.01),
        "Precision@5%": precision_at_k(y_true, y_prob, 0.05)
    }


# =============================
# CTGAN
# =============================

def train_ctgan(train, cfg):
    fraud = train[train[cfg["target"]] == 1]

    ctgan = CTGAN(
        epochs=cfg["epochs"],
        batch_size=cfg["batch_size"],
        pac=4,
        verbose=True
    )

    ctgan.fit(fraud)
    return ctgan


def generate_synthetic(ctgan, n, target):
    df = ctgan.sample(n)
    df[target] = 1
    return df


# =============================
# EXPERIMENT
# =============================

def run_baseline(train, val, cfg):
    features = get_features(train, cfg["target"])

    model = train_model(train[features], train[cfg["target"]], cfg)
    prob = predict(model, val[features])

    metrics = evaluate(val[cfg["target"]], prob)

    return model, metrics


def run_ctgan_experiment(train, val, ctgan, cfg):
    features = get_features(train, cfg["target"])

    real_min = train[train[cfg["target"]] == 1]
    real_maj = train[train[cfg["target"]] == 0]

    best_ratio, best_threshold, best_score = None, None, -1

    for r in cfg["ratios"]:
        target_min = int(r * len(real_maj))
        need = target_min - len(real_min)

        if need <= 0:
            continue

        synth = generate_synthetic(ctgan, need, cfg["target"])
        train_aug = pd.concat([real_maj, real_min, synth]).sample(frac=1, random_state=cfg["random_state"])

        model = train_model(train_aug[features], train_aug[cfg["target"]], cfg)
        prob = predict(model, val[features])

        metrics = evaluate(val[cfg["target"]], prob)

        if metrics["PR_AUC"] > best_score:
            best_score = metrics["PR_AUC"]
            best_ratio = r
            best_threshold = metrics["threshold"]

    return best_ratio, best_threshold


# =============================
# FINAL TEST
# =============================

def final_test(train, test, ctgan, ratio, threshold, cfg):
    features = get_features(train, cfg["target"])

    real_min = train[train[cfg["target"]] == 1]
    real_maj = train[train[cfg["target"]] == 0]

    need = int(ratio * len(real_maj)) - len(real_min)

    synth = generate_synthetic(ctgan, need, cfg["target"])
    train_aug = pd.concat([real_maj, real_min, synth]).sample(frac=1, random_state=cfg["random_state"])

    model = train_model(train_aug[features], train_aug[cfg["target"]], cfg)
    prob = predict(model, test[features])

    metrics = evaluate(test[cfg["target"]], prob, threshold)

    return model, metrics


# =============================
# SAVE
# =============================

def save_artifacts(model, results, cfg):
    os.makedirs("artifacts", exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    joblib.dump(model, f"artifacts/model_{ts}.pkl")

    with open(f"artifacts/results_{ts}.json", "w") as f:
        json.dump({"results": results, "config": cfg}, f, indent=4)


# =============================
# MAIN
# =============================

def main():
    cfg = get_config()
    set_seed(cfg["random_state"])

    train, val, test = load_data(cfg)

    # baseline
    _, baseline_metrics = run_baseline(train, val, cfg)

    # ctgan
    ctgan = train_ctgan(train, cfg)

    best_ratio, best_threshold = run_ctgan_experiment(train, val, ctgan, cfg)

    # final
    model, test_metrics = final_test(train, test, ctgan, best_ratio, best_threshold, cfg)

    results = {
        "baseline": baseline_metrics,
        "best_ratio": best_ratio,
        "best_threshold": best_threshold,
        "test": test_metrics
    }

    save_artifacts(model, results, cfg)


if __name__ == "__main__":
    main()
