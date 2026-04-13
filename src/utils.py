"""
utils.py – Hằng số và hàm dùng chung cho toàn bộ project
=========================================================
Định nghĩa tên cột, nhóm feature, và các helper functions
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['figure.dpi'] = 120
matplotlib.rcParams['font.size'] = 11

# ─────────────────────────────────────────────
# ĐƯỜNG DẪN
# ─────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "figures"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "models"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "results"), exist_ok=True)

DATA_PATH = os.path.join(DATA_DIR, "data_FraudDetection_JAR2020.csv")

# ─────────────────────────────────────────────
# NHÃN VÀ ĐỊNH DANH
# ─────────────────────────────────────────────
LABEL_COL = "misstate"          # 1 = fraud, 0 = non-fraud
SERIAL_COL = "p_aaer"           # dùng xử lý serial fraud
ID_COLS = ["gvkey", "fyear", "p_aaer"]

# ─────────────────────────────────────────────
# 28 BIẾN KẾ TOÁN THÔ (RAW) TỪ COMPUSTAT
# ─────────────────────────────────────────────
RAW_FEATURES = [
    "act",      # Current Assets, Total
    "ap",       # Account Payable, Trade
    "at",       # Assets, Total
    "ceq",      # Common/Ordinary Equity, Total
    "che",      # Cash and Short-Term Investments
    "cogs",     # Cost of Goods Sold
    "csho",     # Common Shares Outstanding
    "dlc",      # Debt in Current Liabilities, Total
    "dltis",    # Long-Term Debt Issuance
    "dltt",     # Long-Term Debt, Total
    "dp",       # Depreciation and Amortization
    "ib",       # Income Before Extraordinary Items
    "invt",     # Inventories, Total
    "ivao",     # Investment and Advances, Other
    "ivst",     # Short-Term Investments, Total
    "lct",      # Current Liabilities, Total
    "lt",       # Liabilities, Total
    "ni",       # Net Income (Loss)
    "ppegt",    # Property, Plant and Equipment, Total
    "pstk",     # Preferred/Preference Stock (Capital), Total
    "re",       # Retained Earnings
    "rect",     # Receivables, Total
    "sale",     # Sales/Turnover (Net)
    "sstk",     # Sale of Common and Preferred Stock
    "txp",      # Income Taxes Payable
    "txt",      # Income Taxes, Total
    "xint",     # Interest and Related Expense, Total
    "prcc_f",   # Price Close, Annual, Fiscal
]

# ─────────────────────────────────────────────
# 14 TỶ LỆ TÀI CHÍNH (RATIO FEATURES)
# ─────────────────────────────────────────────
RATIO_FEATURES = [
    "dch_wc",       # WC accruals – thay đổi vốn lưu động
    "ch_rsst",      # RSST accruals – chỉ số accrual tổng hợp
    "dch_rec",      # Change in receivables – thay đổi khoản phải thu
    "dch_inv",      # Change in inventory – thay đổi hàng tồn kho
    "soft_assets",  # % Soft assets – tỷ lệ tài sản vô hình
    "ch_cs",        # Change in cash sales
    "ch_cm",        # Change in cash margin
    "ch_roa",       # Change in return on assets
    "issue",        # Actual issuance – phát hành cổ phiếu/nợ mới
    "bm",           # Book-to-market ratio
    "dpi",          # Depreciation index
    "reoa",         # Retained earnings over total assets
    "ebit",         # EBIT over total assets  (trong CSV có thể là 'EBIT')
    "ch_fcf",       # Change in free cash flows
]

# Nhóm theo ý nghĩa kinh tế (dùng cho phân tích)
RATIO_GROUPS = {
    "Accrual indicators": ["dch_wc", "ch_rsst", "dch_rec", "dch_inv"],
    "Performance indicators": ["ch_roa", "reoa", "ebit", "ch_fcf", "ch_cm", "ch_cs"],
    "Financing indicators": ["issue", "bm", "dpi", "soft_assets"],
}

ALL_FEATURES = RAW_FEATURES + RATIO_FEATURES

# ─────────────────────────────────────────────
# CHIẾN LƯỢC FEATURE (kết quả từ 00_feature_analysis.py)
# ─────────────────────────────────────────────
FEATURE_STRATEGIES = {
    "ratio_only":   RATIO_FEATURES,          # Chỉ 14 tỷ lệ tài chính
    "raw_only":     RAW_FEATURES,            # Chỉ 28 biến thô
    "all_features": ALL_FEATURES,            # Toàn bộ 42 biến
}

# ─────────────────────────────────────────────
# WALK-FORWARD SPLIT (theo Bao et al. 2020)
# ─────────────────────────────────────────────
# Training: 1991-2001, skip 2002 (post-Enron noise), Test: 2003-2008
TRAIN_YEARS = list(range(1991, 2002))   # 1991–2001
SKIP_YEARS  = [2002]                    # Bỏ năm SOX transition
TEST_YEARS  = list(range(2003, 2009))   # 2003–2008


# ─────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────

def load_data(path=DATA_PATH):
    """Load dataset JAR2020, chuẩn hóa tên cột."""
    df = pd.read_csv(path)
    # Chuẩn hóa tên cột về lowercase
    df.columns = [c.strip().lower() for c in df.columns]
    print(f"[load_data] Shape: {df.shape}")
    print(f"[load_data] Fraud rate: {df[LABEL_COL].mean():.4f} "
          f"({df[LABEL_COL].sum():.0f} fraud / {len(df)} total)")
    return df


def save_fig(name, tight=True):
    """Lưu figure vào outputs/figures/"""
    path = os.path.join(OUTPUT_DIR, "figures", f"{name}.png")
    if tight:
        plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    print(f"[save_fig] Saved → {path}")
    plt.close()


def handle_serial_fraud(df):
    """
    Xử lý Serial Fraud theo Bao et al. (2020):
    Nếu một công ty bị gian lận kéo dài nhiều năm (p_aaer > 0),
    chỉ giữ lại quan sát đầu tiên của mỗi đợt gian lận để tránh
    data leakage và tránh overcount.
    Trả về df đã xử lý + thống kê.
    """
    fraud_df = df[df[LABEL_COL] == 1].copy()
    non_fraud_df = df[df[LABEL_COL] == 0].copy()

    # Với các firm-year fraud: chỉ lấy năm đầu tiên của mỗi p_aaer episode
    fraud_first = (
        fraud_df[fraud_df[SERIAL_COL] > 0]
        .sort_values("fyear")
        .groupby(SERIAL_COL)
        .first()
        .reset_index()
    )
    # Các fraud không có p_aaer (isolated fraud)
    fraud_isolated = fraud_df[fraud_df[SERIAL_COL] == 0]

    fraud_cleaned = pd.concat([fraud_first, fraud_isolated], ignore_index=True)
    df_cleaned = pd.concat([non_fraud_df, fraud_cleaned], ignore_index=True)

    print(f"[handle_serial_fraud] Trước: {len(fraud_df)} fraud obs")
    print(f"[handle_serial_fraud] Sau:   {len(fraud_cleaned)} fraud obs")
    return df_cleaned


def print_class_distribution(df, label=LABEL_COL, title="Class Distribution"):
    counts = df[label].value_counts()
    total = len(df)
    print(f"\n{'='*40}")
    print(f"{title}")
    print(f"{'='*40}")
    for cls, cnt in counts.items():
        print(f"  Class {cls}: {cnt:>6,} ({cnt/total*100:.2f}%)")
    print(f"  Imbalance ratio: 1:{int(counts[0]/counts[1])}")
    print(f"{'='*40}\n")
