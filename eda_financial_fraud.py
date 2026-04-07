"""
EDA for Financial Fraud Dataset (JAR FraudDetection)

- Analyze raw variables + ratio variables
- Check missing, distribution, imbalance
- Visualize fraud vs non-fraud
- Ready for paper figures
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

# =============================
# CONFIG
# =============================
DATA_PATH = "./datasets/data_JAR2020.csv"
TARGET = "misstate"
YEAR_COL = "fyear"

# RAW FEATURES
RAW_COLS = [
    'act','ap','at','ceq','che','cogs','csho','dlc','dltis','dltt',
    'dp','ib','invt','ivao','ivst','lct','lt','ni','ppegt','pstk',
    're','rect','sale','sstk','txp','txt','xint','prcc_f'
]

OUTPUT_DIR = "./eda_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# =============================
# LOAD DATA
# =============================
df = pd.read_csv(DATA_PATH)

print("="*50)
print("DATA OVERVIEW")
print("="*50)
print(f"Shape: {df.shape}")
print(f"Columns: {len(df.columns)}")

# =============================
# DEFINE RATIO COLS
# =============================
ratio_cols = [
    col for col in df.columns
    if col not in RAW_COLS + [TARGET, YEAR_COL]
]

print(f"\nRaw cols   : {len([c for c in RAW_COLS if c in df.columns])}")
print(f"Ratio cols : {len(ratio_cols)}")

# =============================
# TARGET DISTRIBUTION
# =============================
print("\nTARGET DISTRIBUTION")
print(df[TARGET].value_counts())
print(f"Fraud rate: {df[TARGET].mean():.4f} ({df[TARGET].mean()*100:.2f}%)")

# =============================
# MISSING VALUES
# =============================
missing = df.isnull().sum()
missing_pct = (missing / len(df) * 100).round(2)

missing_df = pd.DataFrame({
    "missing_count": missing,
    "missing_pct": missing_pct
}).sort_values("missing_pct", ascending=False)

print("\nTOP 10 MISSING FEATURES")
print(missing_df.head(10))

missing_df.to_csv(f"{OUTPUT_DIR}/missing_report.csv")

# =============================
# FRAUD BY YEAR
# =============================
if YEAR_COL in df.columns:
    fraud_by_year = df.groupby(YEAR_COL)[TARGET].agg(['sum','count','mean'])
    fraud_by_year.columns = ['fraud_count','total','fraud_rate']

    print("\nFRAUD BY YEAR")
    print(fraud_by_year)

    plt.figure(figsize=(8,5))
    fraud_by_year['fraud_rate'].plot(marker='o')
    plt.title("Fraud Rate by Year")
    plt.ylabel("Fraud Rate")
    plt.grid(alpha=0.3)
    plt.savefig(f"{OUTPUT_DIR}/fraud_by_year.png", dpi=120)
    plt.close()

# =============================
# DISTRIBUTION (RAW)
# =============================
print("\nPlotting RAW distributions...")

fig, axes = plt.subplots(5, 6, figsize=(20, 16))
axes = axes.flatten()

for i, col in enumerate([c for c in RAW_COLS if c in df.columns]):
    ax = axes[i]

    df[df[TARGET]==0][col].dropna().clip(-5, 5).hist(
        bins=50, alpha=0.6, label='Non-fraud', ax=ax, density=True)

    df[df[TARGET]==1][col].dropna().clip(-5, 5).hist(
        bins=50, alpha=0.7, label='Fraud', ax=ax, density=True)

    ax.set_title(col, fontsize=9)
    ax.set_yticks([])
    ax.legend(fontsize=6)

plt.suptitle("RAW Features Distribution (Fraud vs Non-fraud)", fontsize=12)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/raw_distribution.png", dpi=120)
plt.close()

# =============================
# DISTRIBUTION (RATIO)
# =============================
print("Plotting RATIO distributions...")

n_plot = min(len(ratio_cols), 12)

fig, axes = plt.subplots(3, 4, figsize=(16, 10))
axes = axes.flatten()

for i, col in enumerate(ratio_cols[:n_plot]):
    ax = axes[i]

    df[df[TARGET]==0][col].dropna().clip(-5, 5).hist(
        bins=50, alpha=0.6, label='Non-fraud', ax=ax, density=True)

    df[df[TARGET]==1][col].dropna().clip(-5, 5).hist(
        bins=50, alpha=0.7, label='Fraud', ax=ax, density=True)

    ax.set_title(col, fontsize=9)
    ax.set_yticks([])
    ax.legend(fontsize=6)

plt.suptitle("Ratio Features Distribution (Fraud vs Non-fraud)", fontsize=12)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/ratio_distribution.png", dpi=120)
plt.close()

# =============================
# CORRELATION (RATIO)
# =============================
print("Plotting correlation heatmap...")

if len(ratio_cols) > 0:
    corr = df[ratio_cols].corr()

    mask = np.triu(np.ones_like(corr, dtype=bool))

    plt.figure(figsize=(12,10))
    sns.heatmap(
        corr,
        mask=mask,
        cmap='RdBu_r',
        center=0,
        linewidths=0.3,
        cbar_kws={"shrink": 0.8}
    )

    plt.title("Correlation Matrix (Ratio Features)")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/correlation_ratio.png", dpi=120)
    plt.close()

# =============================
# SKEWNESS CHECK (QUAN TRỌNG CHO CTGAN)
# =============================
print("\nSKEWNESS ANALYSIS")

skewness = df[RAW_COLS].skew().sort_values(ascending=False)

print(skewness.head(10))

skewness.to_csv(f"{OUTPUT_DIR}/skewness_raw.csv")

# =============================
# SUMMARY INSIGHTS
# =============================
print("\nEDA COMPLETED")
print(f"All outputs saved to: {OUTPUT_DIR}")