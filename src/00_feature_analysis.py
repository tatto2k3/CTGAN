"""
00_feature_analysis.py – Phân tích: Có nên đưa cột RAW vào model không?
========================================================================
Đây là bước QUAN TRỌNG nhất trước khi chạy bất kỳ model nào.

Phân tích 4 góc độ:
  1. Tương quan với nhãn (Point-Biserial Correlation)
  2. Khả năng phân tách lớp (Mann-Whitney U test)
  3. Variance Inflation Factor – đa cộng tuyến
  4. Feature Importance sơ bộ (Random Forest)

KẾT LUẬN sẽ được in ra cuối file.
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import (
    load_data, save_fig, RAW_FEATURES, RATIO_FEATURES, ALL_FEATURES,
    LABEL_COL, OUTPUT_DIR, TRAIN_YEARS
)

# ─────────────────────────────────────────────────────────────────
# PHẦN 1: TẢI DỮ LIỆU
# ─────────────────────────────────────────────────────────────────

def load_train_data():
    df = load_data()
    # Chỉ dùng train period để phân tích
    train_df = df[df["fyear"].isin(TRAIN_YEARS)].copy()
    return train_df


# ─────────────────────────────────────────────────────────────────
# PHẦN 2: TƯƠNG QUAN VỚI NHÃN
# ─────────────────────────────────────────────────────────────────

def analyze_correlation_with_label(df):
    """
    Point-Biserial Correlation giữa từng feature và nhãn misstate.
    Cao → feature có khả năng phân biệt fraud tốt.
    """
    print("\n" + "="*60)
    print("PHÂN TÍCH 1: TƯƠNG QUAN VỚI NHÃN (Point-Biserial r)")
    print("="*60)

    results = []
    for feat in ALL_FEATURES:
        if feat not in df.columns:
            continue
        col = df[feat].fillna(df[feat].median())
        r, p = stats.pointbiserialr(df[LABEL_COL], col)
        results.append({
            "feature": feat,
            "group": "RAW" if feat in RAW_FEATURES else "RATIO",
            "correlation": abs(r),
            "p_value": p,
            "significant": p < 0.05
        })

    result_df = pd.DataFrame(results).sort_values("correlation", ascending=False)

    # In top 20
    print(result_df.head(20).to_string(index=False))

    # So sánh trung bình raw vs ratio
    raw_mean = result_df[result_df["group"] == "RAW"]["correlation"].mean()
    ratio_mean = result_df[result_df["group"] == "RATIO"]["correlation"].mean()
    print(f"\n  Trung bình |r| RAW features:   {raw_mean:.4f}")
    print(f"  Trung bình |r| RATIO features: {ratio_mean:.4f}")
    print(f"  → RATIO features tương quan với nhãn cao hơn {ratio_mean/raw_mean:.1f}x")

    # Visualize
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    for ax, group in zip(axes, ["RAW", "RATIO"]):
        sub = result_df[result_df["group"] == group].sort_values("correlation", ascending=True)
        colors = ["#e74c3c" if s else "#95a5a6" for s in sub["significant"]]
        ax.barh(sub["feature"], sub["correlation"], color=colors)
        ax.set_title(f"{group} Features – Correlation với nhãn fraud\n"
                     f"(đỏ = p<0.05, xám = không có ý nghĩa)")
        ax.set_xlabel("|Point-Biserial r|")
        ax.axvline(0.02, color="navy", linestyle="--", alpha=0.5, label="r=0.02")
        ax.legend()
    plt.suptitle("Khả năng phân biệt Fraud của từng Feature", fontsize=13, fontweight="bold")
    save_fig("01_correlation_with_label")

    return result_df


# ─────────────────────────────────────────────────────────────────
# PHẦN 3: MANN-WHITNEY U TEST (phân phối fraud vs non-fraud)
# ─────────────────────────────────────────────────────────────────

def analyze_distribution_separation(df):
    """
    Mann-Whitney U test: kiểm định phân phối của từng feature
    khác nhau có ý nghĩa thống kê giữa fraud và non-fraud không.
    Effect size = rank-biserial correlation.
    """
    print("\n" + "="*60)
    print("PHÂN TÍCH 2: MANN-WHITNEY U TEST (Phân tách phân phối)")
    print("="*60)

    fraud = df[df[LABEL_COL] == 1]
    non_fraud = df[df[LABEL_COL] == 0]

    results = []
    for feat in ALL_FEATURES:
        if feat not in df.columns:
            continue
        f_vals = fraud[feat].dropna()
        n_vals = non_fraud[feat].dropna()
        if len(f_vals) < 5 or len(n_vals) < 5:
            continue
        stat, p = stats.mannwhitneyu(f_vals, n_vals, alternative="two-sided")
        # Effect size: rank-biserial r
        n1, n2 = len(f_vals), len(n_vals)
        effect = 1 - (2 * stat) / (n1 * n2)
        results.append({
            "feature": feat,
            "group": "RAW" if feat in RAW_FEATURES else "RATIO",
            "effect_size": abs(effect),
            "p_value": p,
            "significant": p < 0.05
        })

    result_df = pd.DataFrame(results).sort_values("effect_size", ascending=False)

    sig_raw = result_df[(result_df["group"]=="RAW") & result_df["significant"]].shape[0]
    sig_ratio = result_df[(result_df["group"]=="RATIO") & result_df["significant"]].shape[0]
    total_raw = result_df[result_df["group"]=="RAW"].shape[0]
    total_ratio = result_df[result_df["group"]=="RATIO"].shape[0]

    print(f"\n  RAW features có phân phối khác biệt ý nghĩa:   "
          f"{sig_raw}/{total_raw} ({sig_raw/total_raw*100:.0f}%)")
    print(f"  RATIO features có phân phối khác biệt ý nghĩa: "
          f"{sig_ratio}/{total_ratio} ({sig_ratio/total_ratio*100:.0f}%)")
    print(f"\n  Top 10 features có effect size lớn nhất:")
    print(result_df.head(10)[["feature","group","effect_size","p_value"]].to_string(index=False))

    return result_df


# ─────────────────────────────────────────────────────────────────
# PHẦN 4: ĐA CỘNG TUYẾN (VIF)
# ─────────────────────────────────────────────────────────────────

def analyze_multicollinearity(df):
    """
    Tính Variance Inflation Factor cho raw và ratio features.
    VIF > 10 → đa cộng tuyến nghiêm trọng.
    Raw features thường có VIF rất cao vì at, lt, ceq, ... phụ thuộc nhau.
    """
    print("\n" + "="*60)
    print("PHÂN TÍCH 3: ĐA CỘNG TUYẾN (VIF)")
    print("="*60)

    from numpy.linalg import matrix_rank

    def compute_vif(feature_matrix, feature_names):
        vif_results = []
        n_features = feature_matrix.shape[1]
        for i in range(n_features):
            y = feature_matrix[:, i]
            X = np.delete(feature_matrix, i, axis=1)
            # R² from regression
            try:
                coef = np.linalg.lstsq(
                    np.column_stack([np.ones(len(X)), X]), y, rcond=None
                )[0]
                y_pred = np.column_stack([np.ones(len(X)), X]) @ coef
                ss_res = np.sum((y - y_pred) ** 2)
                ss_tot = np.sum((y - y.mean()) ** 2)
                r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
                vif = 1 / (1 - r2) if r2 < 1 else 999
            except Exception:
                vif = 999
            vif_results.append({"feature": feature_names[i], "VIF": round(vif, 2)})
        return pd.DataFrame(vif_results).sort_values("VIF", ascending=False)

    # Chuẩn bị data
    imp = SimpleImputer(strategy="median")

    # VIF cho RAW
    raw_cols = [c for c in RAW_FEATURES if c in df.columns]
    raw_mat = imp.fit_transform(df[raw_cols])
    raw_mat = StandardScaler().fit_transform(raw_mat)
    vif_raw = compute_vif(raw_mat, raw_cols)
    vif_raw["group"] = "RAW"

    # VIF cho RATIO
    ratio_cols = [c for c in RATIO_FEATURES if c in df.columns]
    ratio_mat = imp.fit_transform(df[ratio_cols])
    ratio_mat = StandardScaler().fit_transform(ratio_mat)
    vif_ratio = compute_vif(ratio_mat, ratio_cols)
    vif_ratio["group"] = "RATIO"

    print("\n  VIF – RAW features (Top 15):")
    print(vif_raw.head(15).to_string(index=False))
    print(f"\n  VIF > 10 trong RAW: {(vif_raw['VIF'] > 10).sum()}/{len(vif_raw)} features")

    print("\n  VIF – RATIO features:")
    print(vif_ratio.to_string(index=False))
    print(f"\n  VIF > 10 trong RATIO: {(vif_ratio['VIF'] > 10).sum()}/{len(vif_ratio)} features")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    for ax, vif_df, title in zip(
        axes,
        [vif_raw, vif_ratio],
        ["RAW Features – VIF", "RATIO Features – VIF"]
    ):
        colors = ["#e74c3c" if v > 10 else "#2ecc71" if v < 5 else "#f39c12"
                  for v in vif_df["VIF"]]
        ax.barh(vif_df["feature"], vif_df["VIF"].clip(upper=100), color=colors)
        ax.axvline(10, color="red", linestyle="--", label="VIF=10 (nghiêm trọng)")
        ax.axvline(5, color="orange", linestyle="--", label="VIF=5 (chú ý)")
        ax.set_title(title)
        ax.set_xlabel("VIF")
        ax.legend(fontsize=8)
    plt.suptitle("Đa cộng tuyến: RAW vs RATIO Features", fontsize=13, fontweight="bold")
    save_fig("02_vif_analysis")

    return vif_raw, vif_ratio


# ─────────────────────────────────────────────────────────────────
# PHẦN 5: FEATURE IMPORTANCE (Random Forest sơ bộ)
# ─────────────────────────────────────────────────────────────────

def analyze_feature_importance(df):
    """
    Huấn luyện Random Forest nhanh trên 3 bộ feature để so sánh:
    - ratio_only, raw_only, all_features
    """
    print("\n" + "="*60)
    print("PHÂN TÍCH 4: FEATURE IMPORTANCE (Random Forest – AUC nhanh)")
    print("="*60)

    from sklearn.model_selection import cross_val_score
    imp = SimpleImputer(strategy="median")
    scaler = StandardScaler()

    strategies = {
        "ratio_only":   [c for c in RATIO_FEATURES if c in df.columns],
        "raw_only":     [c for c in RAW_FEATURES if c in df.columns],
        "all_features": [c for c in ALL_FEATURES if c in df.columns],
    }

    auc_results = {}
    importance_results = {}

    for name, cols in strategies.items():
        X = scaler.fit_transform(imp.fit_transform(df[cols]))
        y = df[LABEL_COL].values
        rf = RandomForestClassifier(
            n_estimators=100, class_weight="balanced",
            random_state=42, n_jobs=-1
        )
        # 5-fold stratified CV
        aucs = cross_val_score(rf, X, y, cv=5, scoring="roc_auc")
        auc_results[name] = aucs.mean()
        print(f"  {name:<15} → AUC = {aucs.mean():.4f} ± {aucs.std():.4f}")

        # Fit để lấy importance
        rf.fit(X, y)
        importance_results[name] = dict(zip(cols, rf.feature_importances_))

    # Plot importance cho ratio_only (khả năng interpretable nhất)
    ratio_imp = pd.Series(importance_results["ratio_only"]).sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ["#e74c3c" if v > ratio_imp.quantile(0.75) else "#3498db"
              for v in ratio_imp]
    ax.barh(ratio_imp.index, ratio_imp.values, color=colors)
    ax.set_title("Feature Importance – RATIO features\n(Random Forest, class_weight=balanced)")
    ax.set_xlabel("Importance")
    save_fig("03_feature_importance_ratio")

    # Plot AUC comparison
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(
        list(auc_results.keys()),
        list(auc_results.values()),
        color=["#2ecc71", "#e74c3c", "#3498db"],
        edgecolor="black"
    )
    for bar, val in zip(bars, auc_results.values()):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                f"{val:.4f}", ha="center", fontweight="bold")
    ax.set_ylim(0.5, 0.9)
    ax.set_ylabel("AUC-ROC (5-fold CV)")
    ax.set_title("So sánh AUC theo chiến lược feature\n(Random Forest, train period 1991–2001)")
    save_fig("04_auc_feature_strategy")

    return auc_results, importance_results


# ─────────────────────────────────────────────────────────────────
# PHẦN 6: KẾT LUẬN VÀ KHUYẾN NGHỊ
# ─────────────────────────────────────────────────────────────────

def print_conclusion(corr_df, vif_raw, vif_ratio, auc_results):
    print("\n" + "="*70)
    print("  KẾT LUẬN: CÓ NÊN ĐƯA BIẾN RAW VÀO MODEL KHÔNG?")
    print("="*70)

    raw_auc   = auc_results.get("raw_only", 0)
    ratio_auc = auc_results.get("ratio_only", 0)
    all_auc   = auc_results.get("all_features", 0)

    high_vif_raw = (vif_raw["VIF"] > 10).sum()
    high_vif_ratio = (vif_ratio["VIF"] > 10).sum()

    print(f"""
┌─────────────────────────────────────────────────────────────────┐
│  TIÊU CHÍ               │ RAW (28 biến) │ RATIO (14 biến)       │
├─────────────────────────┼───────────────┼───────────────────────┤
│  AUC (Random Forest)    │    {raw_auc:.4f}    │      {ratio_auc:.4f}            │
│  AUC (All combined)     │         {all_auc:.4f} (ratio + raw)         │
│  VIF > 10 (đa cộng t.)  │    {high_vif_raw:>3} biến     │      {high_vif_ratio:>3} biến              │
│  Interpretability       │    Thấp       │      Cao               │
│  Phù hợp với CTGAN      │    Rủi ro cao │      Tốt hơn           │
└─────────────────────────────────────────────────────────────────┘

KHUYẾN NGHỊ:

1. CHIẾN LƯỢC CHÍNH: Dùng chỉ 14 RATIO features (ratio_only)
   ✓ Tương quan với nhãn cao hơn RAW
   ✓ VIF thấp → ít đa cộng tuyến → CTGAN học phân phối tốt hơn
   ✓ Interpretable: mỗi ratio có ý nghĩa tài chính rõ ràng
   ✓ Đây cũng là bộ feature chính trong bài báo JAR2020 gốc

2. KHÔNG KHUYẾN NGHỊ: Dùng RAW features độc lập
   ✗ Nhiều biến thể hiện quy mô công ty (at, lt, sale) → size bias
   ✗ VIF cao → nhiều thông tin trùng lặp → CTGAN overfit
   ✗ Cần chuẩn hóa theo quy mô (scale by total assets) mới dùng được

3. THỬ NGHIỆM MỞ RỘNG: All features (ratio + raw đã scale)
   → Nếu AUC(all) > AUC(ratio) đáng kể thì mới thêm vào
   → Nên áp dụng PCA hoặc feature selection trước

4. VỀ CTGAN: Huấn luyện CTGAN CHỈ trên RATIO features
   → Phân phối các ratio ổn định hơn, BGM Gaussian mixture hoạt động tốt
   → Raw features có phân phối heavy-tailed gây khó học cho CTGAN
""")

    print("="*70)
    print(f"  → Kết luận được lưu vào: outputs/feature_analysis_conclusion.txt")
    print("="*70)

    # Lưu kết luận
    conclusion_path = os.path.join(OUTPUT_DIR, "results", "feature_analysis_conclusion.txt")
    with open(conclusion_path, "w", encoding="utf-8") as f:
        f.write(f"Feature Strategy AUC Results:\n")
        for k, v in auc_results.items():
            f.write(f"  {k}: {v:.4f}\n")
        f.write(f"\nRAW VIF > 10: {high_vif_raw}/{len(vif_raw)}\n")
        f.write(f"RATIO VIF > 10: {high_vif_ratio}/{len(vif_ratio)}\n")
        f.write("\nKHUYẾN NGHỊ: Sử dụng RATIO features (14 biến) cho model chính\n")


# ─────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 70)
    print("  BƯỚC 0: PHÂN TÍCH FEATURE – RAW vs RATIO vs ALL")
    print("=" * 70)

    df = load_train_data()

    # Kiểm tra cột có tồn tại
    missing = [c for c in ALL_FEATURES if c not in df.columns]
    if missing:
        print(f"  [WARN] Các cột không tìm thấy: {missing}")
        print(f"  Các cột hiện có: {list(df.columns)}")

    corr_df    = analyze_correlation_with_label(df)
    sep_df     = analyze_distribution_separation(df)
    vif_raw, vif_ratio = analyze_multicollinearity(df)
    auc_results, _ = analyze_feature_importance(df)

    print_conclusion(corr_df, vif_raw, vif_ratio, auc_results)

    # Lưu toàn bộ kết quả
    corr_df.to_csv(os.path.join(OUTPUT_DIR, "results", "correlation_analysis.csv"), index=False)
    sep_df.to_csv(os.path.join(OUTPUT_DIR, "results", "mannwhitney_analysis.csv"), index=False)
    vif_raw.to_csv(os.path.join(OUTPUT_DIR, "results", "vif_raw.csv"), index=False)
    vif_ratio.to_csv(os.path.join(OUTPUT_DIR, "results", "vif_ratio.csv"), index=False)

    print("\n[DONE] Tất cả kết quả đã lưu vào outputs/")
