"""
02_preprocessing.py – Chương III.3: Tiền xử lý dữ liệu
=========================================================
3.1 Xử lý missing values (median imputation theo JAR2020)
3.2 Winsorize outliers (1%–99% per feature)
3.3 Chuẩn hóa (StandardScaler)
3.4 Walk-forward split (train 1991–2001, test 2003–2008)

Output:
  - outputs/data/X_train.npy, y_train.npy
  - outputs/data/X_test.npy, y_test.npy
  - outputs/data/X_train_fraud.npy (chỉ mẫu fraud – dùng cho CTGAN)
  - outputs/data/preprocessed_info.json
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import json
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import (
    load_data, handle_serial_fraud, print_class_distribution,
    RATIO_FEATURES, LABEL_COL, OUTPUT_DIR,
    TRAIN_YEARS, SKIP_YEARS, TEST_YEARS, ID_COLS
)

DATA_OUT = os.path.join(OUTPUT_DIR, "data")
os.makedirs(DATA_OUT, exist_ok=True)


# ─────────────────────────────────────────────────────────────────
# 3.1 XỬ LÝ MISSING VALUES
# ─────────────────────────────────────────────────────────────────

def handle_missing(df, feature_cols):
    """
    Median imputation riêng cho từng class (fraud / non-fraud).
    Lý do: median của fraud và non-fraud rất khác nhau.
    Tuy nhiên theo JAR2020, dùng median toàn bộ tập train.
    """
    print("\n" + "="*55)
    print("TIỀN XỬ LÝ 3.1: MISSING VALUES")
    print("="*55)

    before = df[feature_cols].isnull().sum().sum()
    print(f"  Tổng missing trước: {before}")

    # Imputer fit trên toàn bộ training data
    imputer = SimpleImputer(strategy="median")
    df_filled = df.copy()
    df_filled[feature_cols] = imputer.fit_transform(df[feature_cols])

    after = df_filled[feature_cols].isnull().sum().sum()
    print(f"  Tổng missing sau:   {after}")

    return df_filled, imputer


# ─────────────────────────────────────────────────────────────────
# 3.2 WINSORIZE OUTLIERS
# ─────────────────────────────────────────────────────────────────

def winsorize(df, feature_cols, lower=0.01, upper=0.99):
    """
    Clip giá trị ngoài [q1%, q99%] để giảm ảnh hưởng outliers.
    Fit trên train, apply lên cả test.
    """
    print("\n" + "="*55)
    print(f"TIỀN XỬ LÝ 3.2: WINSORIZE ({lower*100:.0f}%–{upper*100:.0f}%)")
    print("="*55)

    bounds = {}
    df_w = df.copy()

    for feat in feature_cols:
        low_val  = df[feat].quantile(lower)
        high_val = df[feat].quantile(upper)
        df_w[feat] = df[feat].clip(low_val, high_val)
        bounds[feat] = {"lower": low_val, "upper": high_val}

    # Thống kê ảnh hưởng
    affected = 0
    for feat in feature_cols:
        n_clipped = ((df[feat] < bounds[feat]["lower"]) |
                     (df[feat] > bounds[feat]["upper"])).sum()
        affected += n_clipped
    pct = affected / (len(df) * len(feature_cols)) * 100
    print(f"  Số giá trị bị clip: {affected} ({pct:.2f}% trong tất cả feature values)")

    return df_w, bounds


def apply_winsorize(df, bounds, feature_cols):
    """Apply winsorize bounds đã fit (dùng cho test set)."""
    df_w = df.copy()
    for feat in feature_cols:
        if feat in bounds:
            df_w[feat] = df[feat].clip(bounds[feat]["lower"], bounds[feat]["upper"])
    return df_w


# ─────────────────────────────────────────────────────────────────
# 3.3 CHUẨN HÓA
# ─────────────────────────────────────────────────────────────────

def scale_features(X_train, X_test):
    """StandardScaler fit trên train, transform cả hai."""
    print("\n" + "="*55)
    print("TIỀN XỬ LÝ 3.3: CHUẨN HÓA (StandardScaler)")
    print("="*55)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)
    print(f"  X_train mean sau scale: {X_train_scaled.mean():.4f} (kỳ vọng ≈ 0)")
    print(f"  X_train std  sau scale: {X_train_scaled.std():.4f}  (kỳ vọng ≈ 1)")
    return X_train_scaled, X_test_scaled, scaler


# ─────────────────────────────────────────────────────────────────
# 3.4 WALK-FORWARD SPLIT
# ─────────────────────────────────────────────────────────────────

def walk_forward_split(df, feature_cols):
    """
    Theo Bao et al. (2020):
      - Training: fiscal year 1991–2001
      - Skip: 2002 (Sarbanes-Oxley Act – nhiễu label)
      - Test: 2003–2008

    Lý do skip 2002: năm Enron/WorldCom collapse, SEC enforcement
    tăng đột biến → nhiễu nhãn nghiêm trọng cho model.
    """
    print("\n" + "="*55)
    print("TIỀN XỬ LÝ 3.4: WALK-FORWARD SPLIT")
    print("="*55)

    train_mask = df["fyear"].isin(TRAIN_YEARS)
    test_mask  = df["fyear"].isin(TEST_YEARS)
    skip_mask  = df["fyear"].isin(SKIP_YEARS)

    print(f"  Train period: {min(TRAIN_YEARS)}–{max(TRAIN_YEARS)} "
          f"→ {train_mask.sum()} quan sát")
    print(f"  Skip period:  {SKIP_YEARS} → {skip_mask.sum()} quan sát (bỏ qua)")
    print(f"  Test period:  {min(TEST_YEARS)}–{max(TEST_YEARS)} "
          f"→ {test_mask.sum()} quan sát")

    train_df = df[train_mask].copy()
    test_df  = df[test_mask].copy()

    print_class_distribution(train_df, title="Train set")
    print_class_distribution(test_df,  title="Test set")

    return train_df, test_df


# ─────────────────────────────────────────────────────────────────
# PIPELINE CHÍNH
# ─────────────────────────────────────────────────────────────────

def run_preprocessing():
    print("=" * 55)
    print("  BƯỚC 2: TIỀN XỬ LÝ DỮ LIỆU")
    print("=" * 55)

    # Load
    df = load_data()
    df = handle_serial_fraud(df)

    # Kiểm tra feature columns
    feature_cols = [c for c in RATIO_FEATURES if c in df.columns]
    missing_cols = [c for c in RATIO_FEATURES if c not in df.columns]
    if missing_cols:
        print(f"  [WARN] Thiếu columns: {missing_cols}")
    print(f"  Feature columns sử dụng: {feature_cols} ({len(feature_cols)} biến)")

    # Walk-forward split
    train_df, test_df = walk_forward_split(df, feature_cols)

    # Tiền xử lý CHỈ fit trên train
    train_filled, imputer = handle_missing(train_df, feature_cols)
    train_win, bounds     = winsorize(train_filled, feature_cols)

    # Apply lên test (không fit lại)
    test_df_filled = test_df.copy()
    test_df_filled[feature_cols] = imputer.transform(test_df[feature_cols])
    test_win = apply_winsorize(test_df_filled, bounds, feature_cols)

    # Tách X, y
    X_train_raw = train_win[feature_cols].values
    y_train     = train_win[LABEL_COL].values
    X_test_raw  = test_win[feature_cols].values
    y_test      = test_win[LABEL_COL].values

    # Scale
    X_train, X_test, scaler = scale_features(X_train_raw, X_test_raw)

    # Tách riêng fraud samples cho CTGAN (unscaled – CTGAN tự normalize)
    fraud_mask = y_train == 1
    X_train_fraud_raw = train_win[feature_cols][fraud_mask].values
    print(f"\n  Fraud samples cho CTGAN: {fraud_mask.sum()}")

    # ─── Lưu kết quả ───
    np.save(os.path.join(DATA_OUT, "X_train.npy"), X_train)
    np.save(os.path.join(DATA_OUT, "y_train.npy"), y_train)
    np.save(os.path.join(DATA_OUT, "X_test.npy"),  X_test)
    np.save(os.path.join(DATA_OUT, "y_test.npy"),  y_test)
    np.save(os.path.join(DATA_OUT, "X_train_fraud_raw.npy"), X_train_fraud_raw)

    # Lưu fraud df gốc (dùng cho CTGAN)
    fraud_df = train_win[train_win[LABEL_COL] == 1][feature_cols + [LABEL_COL]]
    fraud_df.to_csv(os.path.join(DATA_OUT, "fraud_train.csv"), index=False)

    # Lưu scaler và imputer
    with open(os.path.join(DATA_OUT, "scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)
    with open(os.path.join(DATA_OUT, "imputer.pkl"), "wb") as f:
        pickle.dump(imputer, f)

    # Lưu metadata
    info = {
        "feature_cols": feature_cols,
        "n_train": int(len(X_train)),
        "n_test":  int(len(X_test)),
        "n_train_fraud": int(fraud_mask.sum()),
        "n_train_nonfraud": int((~fraud_mask).sum()),
        "fraud_rate_train": float(y_train.mean()),
        "fraud_rate_test":  float(y_test.mean()),
        "train_years": TRAIN_YEARS,
        "test_years":  TEST_YEARS,
        "winsorize_bounds": {k: {kk: float(vv) for kk, vv in v.items()}
                             for k, v in bounds.items()},
    }
    with open(os.path.join(DATA_OUT, "preprocessed_info.json"), "w") as f:
        json.dump(info, f, indent=2)

    print("\n" + "="*55)
    print("  KẾT QUẢ TIỀN XỬ LÝ")
    print("="*55)
    print(f"  X_train shape: {X_train.shape}")
    print(f"  X_test shape:  {X_test.shape}")
    print(f"  Fraud train:   {fraud_mask.sum()} ({y_train.mean()*100:.2f}%)")
    print(f"  Fraud test:    {y_test.sum()} ({y_test.mean()*100:.2f}%)")
    print(f"\n  Files saved to: {DATA_OUT}/")
    print("  - X_train.npy, y_train.npy")
    print("  - X_test.npy, y_test.npy")
    print("  - X_train_fraud_raw.npy  ← input cho CTGAN")
    print("  - fraud_train.csv        ← input cho CTGAN (DataFrame)")
    print("  - scaler.pkl, imputer.pkl")
    print("  - preprocessed_info.json")

    return X_train, y_train, X_test, y_test, feature_cols, info


if __name__ == "__main__":
    run_preprocessing()
