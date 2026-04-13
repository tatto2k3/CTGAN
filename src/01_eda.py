"""
01_eda.py – Chương III.2: Phân tích dữ liệu khám phá (EDA)
============================================================
Bao gồm:
  2.1 Thống kê mô tả 28 + 14 biến
  2.2 Phân phối từng biến: fraud vs non-fraud
  2.3 Ma trận tương quan và đa cộng tuyến
  2.4 Xu hướng theo thời gian (fyear)
  2.5 Missing values
  2.6 Trực quan hóa phân tách lớp (PCA, t-SNE)
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import (
    load_data, save_fig, handle_serial_fraud, print_class_distribution,
    RAW_FEATURES, RATIO_FEATURES, RATIO_GROUPS, ALL_FEATURES,
    LABEL_COL, OUTPUT_DIR, ID_COLS
)

# ─────────────────────────────────────────────────────────────────
# 2.1 THỐNG KÊ MÔ TẢ
# ─────────────────────────────────────────────────────────────────

def descriptive_statistics(df):
    print("\n" + "="*60)
    print("EDA 2.1: THỐNG KÊ MÔ TẢ")
    print("="*60)

    print_class_distribution(df, title="Phân phối nhãn trong toàn bộ dataset")

    # Thống kê mô tả tách theo class
    ratio_cols = [c for c in RATIO_FEATURES if c in df.columns]

    for label_val, label_name in [(0, "NON-FRAUD"), (1, "FRAUD")]:
        sub = df[df[LABEL_COL] == label_val][ratio_cols]
        print(f"\n  Thống kê mô tả – {label_name} (n={len(sub)}):")
        print(sub.describe().round(4).to_string())

    # Lưu ra CSV
    stats_path = os.path.join(OUTPUT_DIR, "results", "descriptive_stats.csv")
    desc = df.groupby(LABEL_COL)[ratio_cols].describe().round(4)
    desc.to_csv(stats_path)
    print(f"\n  [saved] {stats_path}")


# ─────────────────────────────────────────────────────────────────
# 2.2 PHÂN PHỐI FRAUD vs NON-FRAUD
# ─────────────────────────────────────────────────────────────────

def plot_distribution_by_class(df):
    """
    Vẽ histogram + KDE cho 14 ratio features, tách fraud vs non-fraud.
    """
    print("\n" + "="*60)
    print("EDA 2.2: PHÂN PHỐI FRAUD vs NON-FRAUD")
    print("="*60)

    ratio_cols = [c for c in RATIO_FEATURES if c in df.columns]
    fraud = df[df[LABEL_COL] == 1]
    non_fraud = df[df[LABEL_COL] == 0]

    n_cols = 4
    n_rows = (len(ratio_cols) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, n_rows * 3.5))
    axes = axes.flatten()

    for i, feat in enumerate(ratio_cols):
        ax = axes[i]
        # Winsorize để plot dễ nhìn
        q01, q99 = df[feat].quantile([0.01, 0.99])
        nf_vals = non_fraud[feat].clip(q01, q99).dropna()
        f_vals  = fraud[feat].clip(q01, q99).dropna()

        ax.hist(nf_vals, bins=40, alpha=0.5, color="#3498db",
                density=True, label="Non-Fraud")
        ax.hist(f_vals, bins=40, alpha=0.7, color="#e74c3c",
                density=True, label="Fraud")

        # KDE
        try:
            from scipy.stats import gaussian_kde
            for vals, color in [(nf_vals, "#2980b9"), (f_vals, "#c0392b")]:
                kde = gaussian_kde(vals)
                x = np.linspace(vals.min(), vals.max(), 200)
                ax.plot(x, kde(x), color=color, linewidth=2)
        except Exception:
            pass

        ax.set_title(feat, fontsize=10, fontweight="bold")
        ax.legend(fontsize=7)
        ax.set_xlabel("")

    # Ẩn subplot thừa
    for j in range(len(ratio_cols), len(axes)):
        axes[j].set_visible(False)

    plt.suptitle("Phân phối 14 RATIO Features: Fraud vs Non-Fraud",
                 fontsize=13, fontweight="bold", y=1.01)
    save_fig("05_distribution_ratio_features")
    print("  [saved] figures/05_distribution_ratio_features.png")


# ─────────────────────────────────────────────────────────────────
# 2.3 MA TRẬN TƯƠNG QUAN
# ─────────────────────────────────────────────────────────────────

def plot_correlation_matrix(df):
    print("\n" + "="*60)
    print("EDA 2.3: MA TRẬN TƯƠNG QUAN")
    print("="*60)

    ratio_cols = [c for c in RATIO_FEATURES if c in df.columns]

    # Tương quan cho fraud và non-fraud riêng
    for label_val, label_name in [(0, "nonfraud"), (1, "fraud")]:
        sub = df[df[LABEL_COL] == label_val][ratio_cols]
        corr = sub.corr()

        fig, ax = plt.subplots(figsize=(11, 9))
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(
            corr, mask=mask, annot=True, fmt=".2f",
            cmap="RdBu_r", center=0, vmin=-1, vmax=1,
            square=True, linewidths=0.5, ax=ax,
            annot_kws={"size": 8}
        )
        ax.set_title(f"Ma trận tương quan – {label_name.upper()} observations\n"
                     f"(n={len(sub)})", fontsize=12)
        save_fig(f"06_correlation_matrix_{label_name}")

    print("  [saved] figures/06_correlation_matrix_*.png")

    # Tìm cặp tương quan cao (|r| > 0.7)
    corr_all = df[ratio_cols].corr()
    high_corr = []
    for i in range(len(ratio_cols)):
        for j in range(i+1, len(ratio_cols)):
            r = corr_all.iloc[i, j]
            if abs(r) > 0.5:
                high_corr.append((ratio_cols[i], ratio_cols[j], round(r, 3)))

    if high_corr:
        print(f"\n  Các cặp feature có |r| > 0.5:")
        for f1, f2, r in sorted(high_corr, key=lambda x: abs(x[2]), reverse=True):
            print(f"    {f1:15} × {f2:15} → r = {r}")


# ─────────────────────────────────────────────────────────────────
# 2.4 XU HƯỚNG THEO THỜI GIAN
# ─────────────────────────────────────────────────────────────────

def plot_temporal_trend(df):
    print("\n" + "="*60)
    print("EDA 2.4: XU HƯỚNG THEO THỜI GIAN")
    print("="*60)

    yearly = df.groupby("fyear").agg(
        total=("misstate", "count"),
        fraud_count=("misstate", "sum")
    ).reset_index()
    yearly["fraud_rate"] = yearly["fraud_count"] / yearly["total"] * 100

    fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True)

    # Plot 1: Số lượng
    axes[0].bar(yearly["fyear"], yearly["total"],
                color="#bdc3c7", label="Non-Fraud + Fraud", alpha=0.8)
    axes[0].bar(yearly["fyear"], yearly["fraud_count"],
                color="#e74c3c", label="Fraud", alpha=0.9)
    axes[0].set_ylabel("Số quan sát")
    axes[0].legend()
    axes[0].set_title("Số lượng quan sát theo năm")
    axes[0].axvline(2001.5, color="navy", linestyle="--",
                    alpha=0.7, label="Train/Test split")

    # Plot 2: Fraud rate
    axes[1].plot(yearly["fyear"], yearly["fraud_rate"],
                 "o-", color="#e74c3c", linewidth=2, markersize=5)
    axes[1].fill_between(yearly["fyear"], yearly["fraud_rate"],
                         alpha=0.2, color="#e74c3c")
    axes[1].set_ylabel("Fraud Rate (%)")
    axes[1].set_xlabel("Fiscal Year")
    axes[1].set_title("Tỷ lệ gian lận theo năm (%)")
    axes[1].axvline(2001.5, color="navy", linestyle="--", alpha=0.7)
    axes[1].axhline(yearly["fraud_rate"].mean(), color="green",
                    linestyle=":", label=f"Mean = {yearly['fraud_rate'].mean():.2f}%")
    axes[1].legend()

    plt.suptitle("Phân tích xu hướng gian lận kế toán theo thời gian (JAR2020)",
                 fontsize=13, fontweight="bold")
    save_fig("07_temporal_trend")

    print(yearly.to_string(index=False))
    print(f"\n  Fraud rate trung bình: {yearly['fraud_rate'].mean():.3f}%")
    print(f"  Năm có fraud rate cao nhất: {yearly.loc[yearly['fraud_rate'].idxmax(), 'fyear']:.0f}")


# ─────────────────────────────────────────────────────────────────
# 2.5 MISSING VALUES
# ─────────────────────────────────────────────────────────────────

def analyze_missing_values(df):
    print("\n" + "="*60)
    print("EDA 2.5: MISSING VALUES")
    print("="*60)

    feat_cols = [c for c in ALL_FEATURES if c in df.columns]
    missing = df[feat_cols].isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    missing_df = pd.DataFrame({
        "feature": missing.index,
        "missing_count": missing.values,
        "missing_pct": missing_pct.values,
        "group": ["RAW" if c in RAW_FEATURES else "RATIO" for c in missing.index]
    }).query("missing_count > 0").sort_values("missing_pct", ascending=False)

    if missing_df.empty:
        print("  Không có missing values trong các feature columns!")
    else:
        print(missing_df.to_string(index=False))

        fig, ax = plt.subplots(figsize=(10, max(4, len(missing_df)*0.4)))
        colors = ["#e74c3c" if g == "RAW" else "#3498db" for g in missing_df["group"]]
        ax.barh(missing_df["feature"], missing_df["missing_pct"], color=colors)
        ax.set_xlabel("Missing %")
        ax.set_title("Missing Values theo Feature")
        from matplotlib.patches import Patch
        ax.legend(handles=[
            Patch(color="#e74c3c", label="RAW"),
            Patch(color="#3498db", label="RATIO")
        ])
        save_fig("08_missing_values")

    return missing_df


# ─────────────────────────────────────────────────────────────────
# 2.6 PHÂN TÁCH LỚP: PCA VÀ T-SNE
# ─────────────────────────────────────────────────────────────────

def plot_class_separation(df, n_samples=3000):
    """
    Dùng PCA và t-SNE để trực quan hóa khả năng phân tách
    giữa fraud và non-fraud trong không gian 14 ratio features.
    """
    print("\n" + "="*60)
    print("EDA 2.6: TRỰC QUAN HÓA PHÂN TÁCH LỚP (PCA + t-SNE)")
    print("="*60)

    ratio_cols = [c for c in RATIO_FEATURES if c in df.columns]

    # Subsample để t-SNE nhanh
    fraud = df[df[LABEL_COL] == 1]
    non_fraud = df[df[LABEL_COL] == 0].sample(
        n=min(n_samples, len(non_fraud := df[df[LABEL_COL] == 0])),
        random_state=42
    )
    subset = pd.concat([fraud, non_fraud]).reset_index(drop=True)

    imp = SimpleImputer(strategy="median")
    X = StandardScaler().fit_transform(imp.fit_transform(subset[ratio_cols]))
    y = subset[LABEL_COL].values

    # PCA
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X)

    # t-SNE
    print(f"  Đang chạy t-SNE trên {len(subset)} samples... (có thể mất vài phút)")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=500)
    X_tsne = tsne.fit_transform(X)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    colors = {0: "#95a5a6", 1: "#e74c3c"}
    labels = {0: "Non-Fraud", 1: "Fraud"}
    sizes  = {0: 5, 1: 30}
    alphas = {0: 0.3, 1: 0.8}

    for cls in [0, 1]:
        mask = y == cls
        for ax, X_2d, title in zip(
            axes,
            [X_pca, X_tsne],
            [f"PCA (var explained: {pca.explained_variance_ratio_.sum()*100:.1f}%)", "t-SNE"]
        ):
            ax.scatter(
                X_2d[mask, 0], X_2d[mask, 1],
                c=colors[cls], label=labels[cls],
                s=sizes[cls], alpha=alphas[cls]
            )

    for ax, title in zip(axes, ["PCA", "t-SNE"]):
        ax.set_title(f"{title} – Phân tách Fraud vs Non-Fraud\n(14 RATIO features)")
        ax.legend()

    plt.suptitle("Trực quan hóa không gian đặc trưng (RATIO features)",
                 fontsize=13, fontweight="bold")
    save_fig("09_pca_tsne_separation")
    print("  [saved] figures/09_pca_tsne_separation.png")

    var_explained = pca.explained_variance_ratio_
    print(f"\n  PCA PC1 variance explained: {var_explained[0]*100:.1f}%")
    print(f"  PCA PC2 variance explained: {var_explained[1]*100:.1f}%")
    print(f"  Tổng 2 PC: {var_explained.sum()*100:.1f}%")


# ─────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  BƯỚC 1: EDA – PHÂN TÍCH DỮ LIỆU KHÁM PHÁ")
    print("=" * 60)

    df = load_data()
    df = handle_serial_fraud(df)

    descriptive_statistics(df)
    plot_distribution_by_class(df)
    plot_correlation_matrix(df)
    plot_temporal_trend(df)
    analyze_missing_values(df)
    plot_class_separation(df)

    print("\n[DONE] EDA hoàn tất. Kiểm tra outputs/figures/ để xem biểu đồ.")
