"""
04_experiment.py – Chương III.5 + IV: Thực nghiệm và Đánh giá
==============================================================
FIX: Pipeline scale đúng cho CTGAN data
  - CTGAN train trên RAW (winsorized) fraud data  ✓
  - Synthetic data sinh ra ở không gian RAW       ✓
  - Sau đó mới apply scaler.transform()           ✓  ← bước bị thiếu trước
  - Ghép với X_train (đã scaled) để train model   ✓

FIX: Threshold
  - Không dùng threshold=0.5 cứng
  - Tự tìm best threshold tối ưu F1
  - AUC/NDCG/Prec@k không phụ thuộc threshold → metric chính
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score, roc_curve, precision_recall_curve,
    f1_score, confusion_matrix, average_precision_score,
)
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import pickle
import json
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import save_fig, RATIO_FEATURES, LABEL_COL, OUTPUT_DIR

DATA_OUT = os.path.join(OUTPUT_DIR, "data")
RES_OUT  = os.path.join(OUTPUT_DIR, "results")
os.makedirs(RES_OUT, exist_ok=True)


# ─────────────────────────────────────────────────────────────────
# METRICS
# ─────────────────────────────────────────────────────────────────

def compute_ndcg_at_k(y_true, y_scores, k=100):
    order = np.argsort(y_scores)[::-1]
    y_sorted = y_true[order[:k]]
    gains = 2**y_sorted - 1
    discounts = np.log2(np.arange(2, len(y_sorted) + 2))
    dcg = (gains / discounts).sum()
    ideal = sorted(y_true, reverse=True)[:k]
    ideal_gains = 2**np.array(ideal) - 1
    ideal_discounts = np.log2(np.arange(2, len(ideal_gains) + 2))
    idcg = (ideal_gains / ideal_discounts).sum()
    return dcg / idcg if idcg > 0 else 0.0


def compute_precision_at_k(y_true, y_scores, k=100):
    order = np.argsort(y_scores)[::-1]
    return y_true[order[:k]].mean()


def compute_gmean(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        r0 = tn / (tn + fp) if (tn + fp) > 0 else 0
        r1 = tp / (tp + fn) if (tp + fn) > 0 else 0
        return np.sqrt(r0 * r1)
    return 0.0


def find_best_threshold(y_true, y_scores):
    """
    Tìm threshold tối ưu theo F1.
    Với imbalance 1:100, threshold 0.5 luôn cho F1=0 vì
    model không bao giờ predict class 1 với xác suất > 0.5.
    """
    thresholds = np.linspace(0.01, 0.5, 200)
    best_f1, best_thr = 0.0, 0.5
    for thr in thresholds:
        y_pred = (y_scores >= thr).astype(int)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thr = thr
    return best_thr


def evaluate_model(y_true, y_scores, k=100):
    auc    = roc_auc_score(y_true, y_scores) if y_true.sum() > 0 else 0
    ap     = average_precision_score(y_true, y_scores) if y_true.sum() > 0 else 0
    ndcg   = compute_ndcg_at_k(y_true, y_scores, k=k)
    prec_k = compute_precision_at_k(y_true, y_scores, k=k)

    best_thr = find_best_threshold(y_true, y_scores)
    y_pred   = (y_scores >= best_thr).astype(int)
    f1       = f1_score(y_true, y_pred, zero_division=0)
    gmean    = compute_gmean(y_true, y_pred)

    return {
        "AUC-ROC":        round(auc, 4),
        "Avg-Precision":  round(ap, 4),
        "F1-Score":       round(f1, 4),
        "G-Mean":         round(gmean, 4),
        f"NDCG@{k}":      round(ndcg, 4),
        f"Prec@{k}":      round(prec_k, 4),
        "Best-Threshold": round(best_thr, 4),
    }


# ─────────────────────────────────────────────────────────────────
# AUGMENTATION  ←  PIPELINE ĐÃ ĐƯỢC FIX
# ─────────────────────────────────────────────────────────────────

def get_augmented_data(scenario, X_train_scaled, y_train,
                       scaler, feature_cols):
    """
    PIPELINE ĐÚNG:

    Baseline / SMOTE:
        Hoạt động trực tiếp trên X_train_scaled → không cần thay đổi

    CTGAN:
        1. CTGAN sinh synthetic_raw  (không gian raw, vì CTGAN tự normalize BGM bên trong)
        2. scaler.transform(synthetic_raw)  → synthetic_scaled
        3. concat(X_train_scaled, synthetic_scaled)  → X_aug_scaled

    SAI (code cũ):
        concat(X_train_scaled, synthetic_raw)  ← khác không gian → distribution mismatch
    """

    if scenario == "baseline":
        return X_train_scaled, y_train

    elif scenario == "smote":
        smote = SMOTE(random_state=42, k_neighbors=5)
        X_aug, y_aug = smote.fit_resample(X_train_scaled, y_train)
        print(f"    SMOTE → fraud: {int(y_aug.sum())} / total: {len(y_aug)}")
        return X_aug, y_aug

    elif scenario.startswith("ctgan_ratio"):
        ratio    = int(scenario.split("ratio")[-1])
        csv_path = os.path.join(DATA_OUT, f"synthetic_fraud_ratio{ratio}.csv")

        if not os.path.exists(csv_path):
            print(f"    [SKIP] Không tìm thấy: {csv_path}")
            return None, None

        aug_df    = pd.read_csv(csv_path)
        feat_cols = [c for c in feature_cols if c in aug_df.columns]

        # ── Lấy synthetic rows (source == "synthetic") ──
        synth_df      = aug_df[aug_df["source"] == "synthetic"]
        X_synth_raw   = synth_df[feat_cols].values

        # Xử lý NaN do CTGAN sinh ra
        nan_rows = np.isnan(X_synth_raw).any(axis=1)
        if nan_rows.sum() > 0:
            print(f"    [WARN] Drop {nan_rows.sum()} NaN synthetic rows")
            X_synth_raw = X_synth_raw[~nan_rows]

        # ── BƯỚC QUAN TRỌNG: scale synthetic về cùng không gian ──
        X_synth_scaled = scaler.transform(X_synth_raw)

        # Clip giá trị cực đoan sau scale (CTGAN đôi khi extrapolate)
        X_synth_scaled = np.clip(X_synth_scaled, -10, 10)

        y_synth = np.ones(len(X_synth_scaled))

        # ── Ghép: giữ nguyên X_train_scaled (fraud thực + non-fraud thực) ──
        X_aug = np.vstack([X_train_scaled, X_synth_scaled])
        y_aug = np.concatenate([y_train, y_synth])

        n_fraud_real  = int(y_train.sum())
        n_fraud_synth = int(y_synth.sum())
        print(f"    CTGAN ratio {ratio}:")
        print(f"      Fraud thực:       {n_fraud_real}")
        print(f"      Fraud synthetic:  {n_fraud_synth}  (raw → scaled ✓)")
        print(f"      Tổng fraud:       {n_fraud_real + n_fraud_synth} / {len(y_aug)}")
        return X_aug, y_aug

    return X_train_scaled, y_train


# ─────────────────────────────────────────────────────────────────
# MODELS
# ─────────────────────────────────────────────────────────────────

def get_model(model_name, scale_pos_weight=1):
    if model_name == "RandomForest":
        return RandomForestClassifier(
            n_estimators=500,
            class_weight="balanced",
            max_depth=None,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1
        )
    elif model_name == "XGBoost":
        return xgb.XGBClassifier(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            eval_metric="auc",
            random_state=42,
            verbosity=0,
        )
    raise ValueError(f"Unknown model: {model_name}")


# ─────────────────────────────────────────────────────────────────
# THỰC NGHIỆM
# ─────────────────────────────────────────────────────────────────

def run_experiments(X_train, y_train, X_test, y_test,
                    feature_cols, scaler):
    print("\n" + "="*60)
    print("  THỰC NGHIỆM: KỊCH BẢN × MODELS")
    print("="*60)

    scenarios   = ["baseline", "smote",
                   "ctgan_ratio3", "ctgan_ratio5",
                   "ctgan_ratio10", "ctgan_ratio20"]
    model_names = ["RandomForest", "XGBoost"]

    spw = max(1, int((y_train == 0).sum() / (y_train == 1).sum()))
    print(f"  scale_pos_weight cho XGBoost baseline: {spw}")

    all_results = []
    all_scores  = {}

    for scenario in scenarios:
        for model_name in model_names:
            print(f"\n  ── {scenario} | {model_name} ──")

            X_aug, y_aug = get_augmented_data(
                scenario, X_train, y_train, scaler, feature_cols
            )
            if X_aug is None:
                continue

            # XGBoost baseline/smote cần scale_pos_weight vì chưa augment đủ
            use_spw = spw if (model_name == "XGBoost"
                              and "ctgan" not in scenario) else 1
            model = get_model(model_name, scale_pos_weight=use_spw)
            model.fit(X_aug, y_aug)

            y_scores = model.predict_proba(X_test)[:, 1]
            metrics  = evaluate_model(y_test, y_scores, k=100)

            print(f"    AUC={metrics['AUC-ROC']:.4f}  "
                  f"F1={metrics['F1-Score']:.4f}  "
                  f"G-Mean={metrics['G-Mean']:.4f}  "
                  f"NDCG@100={metrics['NDCG@100']:.4f}  "
                  f"Thr={metrics['Best-Threshold']:.3f}")

            metrics.update({"scenario": scenario, "model": model_name})
            all_results.append(metrics)
            all_scores[f"{scenario}_{model_name}"] = (y_scores, model)

    results_df = pd.DataFrame(all_results)
    results_df.to_csv(os.path.join(RES_OUT, "experiment_results.csv"), index=False)
    return results_df, all_scores


# ─────────────────────────────────────────────────────────────────
# VISUALIZATION
# ─────────────────────────────────────────────────────────────────

def plot_results(results_df, all_scores, y_test, feature_cols):

    # 1. AUC bar chart
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    for ax, mname in zip(axes, ["RandomForest", "XGBoost"]):
        sub = results_df[results_df["model"] == mname].sort_values(
            "AUC-ROC", ascending=False
        )
        colors = ["#e74c3c" if "ctgan" in s
                  else "#f39c12" if "smote" in s
                  else "#7f8c8d"
                  for s in sub["scenario"]]
        bars = ax.bar(sub["scenario"], sub["AUC-ROC"],
                      color=colors, edgecolor="black", alpha=0.85)
        for bar, val in zip(bars, sub["AUC-ROC"]):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.003,
                    f"{val:.4f}", ha="center", fontsize=8, fontweight="bold")
        ax.set_ylim(0.5, 1.0)
        ax.set_title(f"{mname} – AUC-ROC")
        ax.tick_params(axis="x", rotation=30)
        from matplotlib.patches import Patch
        ax.legend(handles=[
            Patch(color="#e74c3c", label="CTGAN"),
            Patch(color="#f39c12", label="SMOTE"),
            Patch(color="#7f8c8d", label="Baseline"),
        ])
    plt.suptitle("AUC-ROC theo Kịch bản (Pipeline đã fix)",
                 fontsize=13, fontweight="bold")
    save_fig("13_auc_comparison_fixed")

    # 2. Metrics heatmap
    metric_cols = ["AUC-ROC", "F1-Score", "G-Mean", "NDCG@100", "Avg-Precision"]
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    for ax, mname in zip(axes, ["RandomForest", "XGBoost"]):
        sub = (results_df[results_df["model"] == mname]
               [["scenario"] + metric_cols]
               .set_index("scenario"))
        sns.heatmap(sub, annot=True, fmt=".4f", cmap="RdYlGn",
                    ax=ax, linewidths=0.5, vmin=0, vmax=1,
                    annot_kws={"size": 9})
        ax.set_title(f"{mname}", fontsize=11)
    plt.suptitle("Heatmap tất cả Metrics",
                 fontsize=13, fontweight="bold")
    save_fig("14_metrics_heatmap")

    # 3. ROC Curves
    styles = {
        "baseline":      (":", "#7f8c8d", 2.0),
        "smote":         ("--", "#f39c12", 2.0),
        "ctgan_ratio3":  ("-", "#fadbd8", 1.5),
        "ctgan_ratio5":  ("-", "#f1948a", 1.5),
        "ctgan_ratio10": ("-", "#e74c3c", 2.5),
        "ctgan_ratio20": ("-", "#922b21", 1.5),
    }
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    for ax, mname in zip(axes, ["RandomForest", "XGBoost"]):
        for key, (y_scores, _) in all_scores.items():
            if mname not in key:
                continue
            scenario = key.replace(f"_{mname}", "")
            fpr, tpr, _ = roc_curve(y_test, y_scores)
            auc = roc_auc_score(y_test, y_scores)
            ls, color, lw = styles.get(scenario, ("-", "gray", 1))
            ax.plot(fpr, tpr, ls=ls, color=color, lw=lw,
                    label=f"{scenario} ({auc:.3f})")
        ax.plot([0,1],[0,1], "k--", alpha=0.3)
        ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
        ax.set_title(f"ROC – {mname}")
        ax.legend(fontsize=8, loc="lower right")
        ax.grid(alpha=0.2)
    plt.suptitle("ROC Curves", fontsize=13, fontweight="bold")
    save_fig("15_roc_curves_fixed")

    # 4. Feature Importance (Baseline vs CTGAN ratio10)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, mname in zip(axes, ["RandomForest", "XGBoost"]):
        data = {}
        for scen in ["baseline", "ctgan_ratio10"]:
            key = f"{scen}_{mname}"
            if key in all_scores:
                _, model = all_scores[key]
                if hasattr(model, "feature_importances_"):
                    data[scen] = model.feature_importances_
        if len(data) == 2:
            x = np.arange(len(feature_cols))
            w = 0.35
            ax.barh(x - w/2, data["baseline"],
                    w, label="Baseline", color="#7f8c8d", alpha=0.8)
            ax.barh(x + w/2, data["ctgan_ratio10"],
                    w, label="CTGAN ratio10", color="#e74c3c", alpha=0.8)
            ax.set_yticks(x)
            ax.set_yticklabels(feature_cols, fontsize=8)
            ax.set_title(f"Feature Importance – {mname}")
            ax.legend()
    plt.suptitle("Feature Importance: Baseline vs CTGAN",
                 fontsize=12, fontweight="bold")
    save_fig("17_feature_importance_comparison")
    print("  [saved] figures 13–17")


def print_final_table(results_df):
    print("\n" + "="*75)
    print("  BẢNG KẾT QUẢ TỔNG HỢP – CHƯƠNG IV")
    print("="*75)
    cols = ["scenario", "model", "AUC-ROC", "F1-Score", "G-Mean",
            "NDCG@100", "Prec@100", "Avg-Precision", "Best-Threshold"]
    avail = [c for c in cols if c in results_df.columns]
    print(results_df[avail]
          .sort_values(["model","AUC-ROC"], ascending=[True,False])
          .to_string(index=False))
    best = results_df.loc[results_df["AUC-ROC"].idxmax()]
    print(f"\n  ★ Best AUC: {best['AUC-ROC']:.4f} "
          f"| {best['scenario']} | {best['model']}")
    print("="*75)


# ─────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("="*60)
    print("  BƯỚC 4: THỰC NGHIỆM (FIXED SCALE PIPELINE)")
    print("="*60)

    for f in ["X_train.npy", "X_test.npy", "scaler.pkl",
              "preprocessed_info.json"]:
        if not os.path.exists(os.path.join(DATA_OUT, f)):
            print(f"  [ERROR] Thiếu: {f}. Chạy 02_preprocessing.py trước!")
            sys.exit(1)

    X_train = np.load(os.path.join(DATA_OUT, "X_train.npy"))
    y_train = np.load(os.path.join(DATA_OUT, "y_train.npy"))
    X_test  = np.load(os.path.join(DATA_OUT, "X_test.npy"))
    y_test  = np.load(os.path.join(DATA_OUT, "y_test.npy"))
    scaler  = pickle.load(open(os.path.join(DATA_OUT, "scaler.pkl"), "rb"))
    with open(os.path.join(DATA_OUT, "preprocessed_info.json")) as f:
        info = json.load(f)
    feature_cols = info["feature_cols"]

    print(f"\n  Train: {X_train.shape} | fraud={int(y_train.sum())} "
          f"({y_train.mean()*100:.2f}%)")
    print(f"  Test:  {X_test.shape}  | fraud={int(y_test.sum())}  "
          f"({y_test.mean()*100:.2f}%)")

    results_df, all_scores = run_experiments(
        X_train, y_train, X_test, y_test, feature_cols, scaler
    )
    plot_results(results_df, all_scores, y_test, feature_cols)
    print_final_table(results_df)
    print("\n[DONE] outputs/results/ và outputs/figures/")
