"""
03_ctgan_train.py – Chương III.4: Xây dựng mô hình CTGAN
=========================================================
- Huấn luyện CTGAN CHỈ trên mẫu fraud (minority class)
- Sinh N mẫu fraud tổng hợp
- [FIX] Lọc synthetic theo Correlation Constraint
- Đánh giá chất lượng dữ liệu sinh
- Xuất file dữ liệu augmented

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[FIX] TẠI SAO CẦN CORRELATION CONSTRAINT FILTER?
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Phân tích ma trận tương quan cho thấy CTGAN học tốt
marginal distribution từng biến, nhưng KHÔNG bảo toàn
joint distribution / accounting identity:

  Cặp                  Real    Synthetic  Δ       Nguyên nhân kế toán
  ─────────────────── ──────  ─────────  ─────   ───────────────────
  reoa   ↔ ebit        0.81    0.07      0.74    Cùng mẫu số (total assets)
  ch_fcf ↔ ch_rsst    -0.78   -0.09      0.69    RSST = f(WC, NCO, FIN)
  dch_wc ↔ dch_inv     0.62   -0.04      0.59    Inventory ⊂ Working Capital
  dch_wc ↔ ch_rsst     0.50    0.10      0.41    WC accruals ⊂ RSST
  ch_roa ↔ reoa        0.30   -0.10      0.32    Cùng đo profitability/assets

Giải pháp: Sinh dư (2x–3x) rồi lọc bỏ mẫu vi phạm các
ràng buộc accounting để đạt đủ số lượng cần thiết.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import pickle
import json
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import save_fig, RATIO_FEATURES, LABEL_COL, OUTPUT_DIR

DATA_OUT  = os.path.join(OUTPUT_DIR, "data")
MODEL_OUT = os.path.join(OUTPUT_DIR, "models")
os.makedirs(MODEL_OUT, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "results"), exist_ok=True)


# ─────────────────────────────────────────────────────────────────
# CẤU HÌNH
# ─────────────────────────────────────────────────────────────────

CTGAN_CONFIG = {
    "generator_dim":       (256, 256),
    "discriminator_dim":   (256, 256),
    "batch_size":          32,
    "epochs":              300,
    "discriminator_steps": 1,
    "generator_lr":        2e-4,
    "discriminator_lr":    2e-4,
    "generator_decay":     1e-6,
    "discriminator_decay": 1e-6,
    "pac":                 8,
    "verbose":             True,
    "log_frequency":       True,
}

AUGMENT_RATIOS = [3, 5, 10]

# Hệ số sinh dư để bù cho phần bị lọc
# Ví dụ: cần 1000 mẫu, OVERSAMPLE_FACTOR=3 → sinh 3000, lọc lấy 1000
OVERSAMPLE_FACTOR = 3


# ─────────────────────────────────────────────────────────────────
# CORRELATION CONSTRAINT FILTER  ←  PHẦN MỚI THÊM
# ─────────────────────────────────────────────────────────────────

# Định nghĩa các ràng buộc tương quan từ phân tích ma trận
# Mỗi constraint: (col_a, col_b, direction, tolerance, description)
#   direction = "positive"  → a và b phải cùng chiều (tích > 0)
#   direction = "negative"  → a và b phải ngược chiều (tích < 0)
#   tolerance = cho phép sai số bao nhiêu phần trăm mẫu vi phạm nhẹ

ACCOUNTING_CONSTRAINTS = [
    # ── NGHIÊM TRỌNG (Δ > 0.5): BẮT BUỘC pass ──────────────────
    {
        "col_a":       "reoa",
        "col_b":       "ebit",
        "direction":   "positive",    # real r=0.81 → cùng chiều
        "strict":      True,          # bắt buộc
        "description": "reoa ↔ ebit: Cùng mẫu số total assets (Δ=0.74)",
    },
    {
        "col_a":       "ch_fcf",
        "col_b":       "ch_rsst",
        "direction":   "negative",    # real r=-0.78 → ngược chiều
        "strict":      True,
        "description": "ch_fcf ↔ ch_rsst: RSST=f(WC,NCO,FIN) (Δ=0.69)",
    },
    {
        "col_a":       "dch_wc",
        "col_b":       "dch_inv",
        "direction":   "positive",    # real r=0.62 → cùng chiều
        "strict":      True,
        "description": "dch_wc ↔ dch_inv: Inventory ⊂ Working Capital (Δ=0.59)",
    },
    # ── TRUNG BÌNH (Δ 0.3–0.5): MỀM (soft constraint) ──────────
    {
        "col_a":       "dch_wc",
        "col_b":       "ch_rsst",
        "direction":   "positive",    # real r=0.50 → cùng chiều
        "strict":      False,         # soft: chỉ cảnh báo, không bắt buộc
        "description": "dch_wc ↔ ch_rsst: WC accruals ⊂ RSST (Δ=0.41)",
    },
    {
        "col_a":       "ch_roa",
        "col_b":       "reoa",
        "direction":   "positive",    # real r=0.30 → cùng chiều
        "strict":      False,
        "description": "ch_roa ↔ reoa: Cùng đo profitability/assets (Δ=0.32)",
    },
]


def check_single_constraint(row, constraint):
    """
    Kiểm tra một mẫu có thỏa mãn một constraint không.
    Trả về True nếu OK, False nếu vi phạm.

    Logic:
      positive: a * b > 0  (cùng chiều → tích dương)
      negative: a * b < 0  (ngược chiều → tích âm)

    Cho phép giá trị rất gần 0 (|val| < epsilon) pass
    vì gần 0 không có chiều rõ ràng.
    """
    col_a = constraint["col_a"]
    col_b = constraint["col_b"]

    if col_a not in row.index or col_b not in row.index:
        return True  # không có cột → bỏ qua constraint này

    val_a = row[col_a]
    val_b = row[col_b]

    # Nếu một trong hai rất gần 0 → không enforce (không rõ chiều)
    epsilon = 1e-6
    if abs(val_a) < epsilon or abs(val_b) < epsilon:
        return True

    product = val_a * val_b

    if constraint["direction"] == "positive":
        return product > 0   # cùng chiều
    elif constraint["direction"] == "negative":
        return product < 0   # ngược chiều
    return True


def apply_correlation_constraints(synthetic_df, feature_cols,
                                   fraud_real_df=None, verbose=True):
    """
    Lọc synthetic samples vi phạm accounting constraints.

    Quy trình:
      1. Kiểm tra từng mẫu theo TẤT CẢ strict constraints
         → Loại bỏ nếu vi phạm BẤT KỲ strict constraint nào
      2. Kiểm tra soft constraints
         → Chỉ in cảnh báo, không loại bỏ
      3. In thống kê: bao nhiêu % bị loại, lý do gì

    Parameters:
      synthetic_df  : DataFrame chứa synthetic samples (raw space)
      feature_cols  : danh sách tên cột feature
      fraud_real_df : DataFrame fraud thực (để so sánh sau khi lọc)
      verbose       : in chi tiết hay không

    Returns:
      filtered_df   : DataFrame đã lọc
      filter_stats  : dict thống kê
    """
    n_before = len(synthetic_df)

    if verbose:
        print(f"\n  {'─'*50}")
        print(f"  CORRELATION CONSTRAINT FILTER")
        print(f"  {'─'*50}")
        print(f"  Số mẫu trước lọc: {n_before}")

    # ── Bước 1: Strict constraints ──
    strict_constraints = [c for c in ACCOUNTING_CONSTRAINTS if c["strict"]]
    soft_constraints   = [c for c in ACCOUNTING_CONSTRAINTS if not c["strict"]]

    # Tạo mask: True = giữ lại, False = loại bỏ
    keep_mask   = pd.Series(True, index=synthetic_df.index)
    violation_counts = {}

    for constraint in strict_constraints:
        col_a = constraint["col_a"]
        col_b = constraint["col_b"]

        if col_a not in synthetic_df.columns or col_b not in synthetic_df.columns:
            if verbose:
                print(f"  [SKIP] Cột không tồn tại: {col_a} hoặc {col_b}")
            continue

        # Vector hóa thay vì loop từng row (nhanh hơn nhiều)
        va = synthetic_df[col_a]
        vb = synthetic_df[col_b]
        product = va * vb
        epsilon = 1e-6

        # Mask: gần 0 → bỏ qua; ngược lại kiểm tra chiều
        near_zero   = (va.abs() < epsilon) | (vb.abs() < epsilon)

        if constraint["direction"] == "positive":
            violates = (~near_zero) & (product <= 0)
        else:  # negative
            violates = (~near_zero) & (product >= 0)

        n_violate = violates.sum()
        violation_counts[f"{col_a}↔{col_b}"] = int(n_violate)
        keep_mask = keep_mask & ~violates

        if verbose:
            pct = n_violate / n_before * 100
            status = "STRICT"
            print(f"  [{status}] {constraint['description']}")
            print(f"          Vi phạm: {n_violate}/{n_before} ({pct:.1f}%) → loại bỏ")

    # ── Bước 2: Soft constraints (chỉ báo cáo) ──
    if verbose and soft_constraints:
        print(f"\n  Soft constraints (chỉ báo cáo, không lọc):")
    for constraint in soft_constraints:
        col_a = constraint["col_a"]
        col_b = constraint["col_b"]
        if col_a not in synthetic_df.columns or col_b not in synthetic_df.columns:
            continue
        va = synthetic_df[col_a]
        vb = synthetic_df[col_b]
        product = va * vb
        epsilon = 1e-6
        near_zero = (va.abs() < epsilon) | (vb.abs() < epsilon)
        if constraint["direction"] == "positive":
            violates = (~near_zero) & (product <= 0)
        else:
            violates = (~near_zero) & (product >= 0)
        n_violate = violates.sum()
        if verbose:
            pct = n_violate / n_before * 100
            print(f"  [SOFT]  {constraint['description']}")
            print(f"          Vi phạm: {n_violate}/{n_before} ({pct:.1f}%) → chỉ ghi nhận")

    # ── Áp dụng mask ──
    filtered_df = synthetic_df[keep_mask].reset_index(drop=True)
    n_after  = len(filtered_df)
    n_removed = n_before - n_after
    pct_removed = n_removed / n_before * 100

    if verbose:
        print(f"\n  {'─'*50}")
        print(f"  Kết quả lọc:")
        print(f"    Trước: {n_before} mẫu")
        print(f"    Sau:   {n_after} mẫu")
        print(f"    Loại:  {n_removed} ({pct_removed:.1f}%)")

        # So sánh correlation sau lọc với real
        if fraud_real_df is not None:
            feat_avail = [c for c in feature_cols
                          if c in filtered_df.columns and c in fraud_real_df.columns]
            if len(filtered_df) >= 10:
                _print_correlation_improvement(
                    fraud_real_df[feat_avail],
                    synthetic_df[feat_avail],    # trước lọc
                    filtered_df[feat_avail],      # sau lọc
                )

    filter_stats = {
        "n_before":    n_before,
        "n_after":     n_after,
        "n_removed":   n_removed,
        "pct_removed": round(pct_removed, 2),
        "violations":  violation_counts,
    }

    return filtered_df, filter_stats


def _print_correlation_improvement(real_df, synth_before_df, synth_after_df):
    """In so sánh correlation của các cặp quan trọng trước/sau lọc."""
    print(f"\n  Cải thiện tương quan sau khi lọc:")
    print(f"  {'Cặp':<25} {'Real':>6} {'Trước lọc':>10} {'Sau lọc':>9} {'Cải thiện':>10}")
    print(f"  {'─'*65}")

    pairs = [
        ("reoa",   "ebit"),
        ("ch_fcf", "ch_rsst"),
        ("dch_wc", "dch_inv"),
        ("dch_wc", "ch_rsst"),
        ("ch_roa", "reoa"),
    ]

    for col_a, col_b in pairs:
        if (col_a not in real_df.columns or col_b not in real_df.columns
                or col_a not in synth_before_df.columns):
            continue
        r_real   = real_df[col_a].corr(real_df[col_b])
        r_before = synth_before_df[col_a].corr(synth_before_df[col_b])
        r_after  = synth_after_df[col_a].corr(synth_after_df[col_b]) \
                   if len(synth_after_df) >= 5 else float("nan")

        diff_before = abs(r_real - r_before)
        diff_after  = abs(r_real - r_after) if not np.isnan(r_after) else float("nan")
        improvement = diff_before - diff_after if not np.isnan(diff_after) else float("nan")

        imp_str = f"+{improvement:.3f}" if improvement > 0 else f"{improvement:.3f}"
        print(f"  {col_a+'↔'+col_b:<25} {r_real:>6.3f} {r_before:>10.3f} "
              f"{r_after:>9.3f} {imp_str:>10}")


# ─────────────────────────────────────────────────────────────────
# BƯỚC 1: HUẤN LUYỆN CTGAN
# ─────────────────────────────────────────────────────────────────

def train_ctgan(fraud_df, feature_cols):
    """
    Huấn luyện CTGAN chỉ trên mẫu fraud.
    CTGAN tự xử lý chuẩn hóa bên trong (Bayesian GMM per column).
    → Truyền vào DataFrame GỐC (chưa scale).
    """
    try:
        from ctgan import CTGAN
    except ImportError:
        print("  [ERROR] Chưa cài ctgan. Chạy: pip install ctgan")
        return None

    print("\n" + "="*55)
    print("CTGAN 3.4: HUẤN LUYỆN CTGAN TRÊN FRAUD SAMPLES")
    print("="*55)
    print(f"  Số mẫu fraud để train CTGAN: {len(fraud_df)}")
    print(f"  Features: {feature_cols}")
    print(f"  Config: epochs={CTGAN_CONFIG['epochs']}, "
          f"batch_size={CTGAN_CONFIG['batch_size']}")
    print(f"  Oversample factor khi sinh: {OVERSAMPLE_FACTOR}x "
          f"(để bù cho mẫu bị lọc sau này)")

    model = CTGAN(
        generator_dim       = CTGAN_CONFIG["generator_dim"],
        discriminator_dim   = CTGAN_CONFIG["discriminator_dim"],
        batch_size          = CTGAN_CONFIG["batch_size"],
        epochs              = CTGAN_CONFIG["epochs"],
        discriminator_steps = CTGAN_CONFIG["discriminator_steps"],
        generator_lr        = CTGAN_CONFIG["generator_lr"],
        discriminator_lr    = CTGAN_CONFIG["discriminator_lr"],
        pac                 = CTGAN_CONFIG["pac"],
        verbose             = CTGAN_CONFIG["verbose"],
        log_frequency       = CTGAN_CONFIG["log_frequency"],
    )

    model.fit(fraud_df[feature_cols], discrete_columns=[])

    model_path = os.path.join(MODEL_OUT, "ctgan_model.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    print(f"  [saved] CTGAN model → {model_path}")
    return model


# ─────────────────────────────────────────────────────────────────
# BƯỚC 2: SINH + LỌC DỮ LIỆU
# ─────────────────────────────────────────────────────────────────

def generate_and_filter(model, fraud_real_df, feature_cols,
                        n_needed, max_attempts=5):
    """
    Sinh dữ liệu và lọc theo correlation constraints.
    Lặp lại nếu sau lọc chưa đủ số lượng.

    Parameters:
      n_needed     : số mẫu cần đạt sau khi lọc
      max_attempts : số lần lặp tối đa

    Returns:
      filtered_df  : DataFrame đủ n_needed mẫu (hoặc tối đa có thể)
      all_stats    : list thống kê từng vòng lặp
    """
    collected   = []
    total_kept  = 0
    all_stats   = []

    for attempt in range(1, max_attempts + 1):
        n_remain   = n_needed - total_kept
        if n_remain <= 0:
            break

        # Sinh dư OVERSAMPLE_FACTOR lần để bù hao
        n_generate = int(n_remain * OVERSAMPLE_FACTOR)
        print(f"\n  Vòng {attempt}: cần thêm {n_remain}, "
              f"sinh {n_generate} (x{OVERSAMPLE_FACTOR})")

        raw_batch = model.sample(n_generate)
        raw_batch = raw_batch[feature_cols].copy()

        # Áp dụng constraint filter
        filtered_batch, stats = apply_correlation_constraints(
            raw_batch,
            feature_cols,
            fraud_real_df=fraud_real_df if attempt == 1 else None,
            verbose=(attempt == 1),   # chỉ in chi tiết lần đầu
        )

        if attempt > 1 and len(filtered_batch) > 0:
            print(f"    Giữ lại: {len(filtered_batch)}/{n_generate} "
                  f"({len(filtered_batch)/n_generate*100:.1f}%)")

        all_stats.append(stats)

        if len(filtered_batch) == 0:
            print(f"    [WARN] Không có mẫu nào qua lọc ở vòng {attempt}!")
            continue

        # Lấy đúng số cần
        take = min(len(filtered_batch), n_remain)
        collected.append(filtered_batch.iloc[:take])
        total_kept += take

    if not collected:
        print("  [ERROR] Không sinh được mẫu nào sau khi lọc!")
        return pd.DataFrame(columns=feature_cols), all_stats

    result = pd.concat(collected, ignore_index=True)
    print(f"\n  Tổng kết generate_and_filter:")
    print(f"    Cần: {n_needed} | Đạt được: {len(result)}")

    if len(result) < n_needed:
        pct = len(result) / n_needed * 100
        print(f"    [WARN] Chỉ đạt {pct:.1f}% mục tiêu. "
              f"Tăng OVERSAMPLE_FACTOR hoặc nới lỏng constraints.")

    return result, all_stats


def create_augmented_datasets(model, fraud_df, non_fraud_df, feature_cols):
    """
    Tạo augmented datasets cho từng ratio, có áp dụng constraint filter.
    """
    print("\n" + "="*55)
    print("CTGAN: SINH DỮ LIỆU AUGMENTED (với Constraint Filter)")
    print("="*55)

    n_real_fraud = len(fraud_df)
    augmented_datasets = {}
    all_filter_stats   = {}

    for ratio in AUGMENT_RATIOS:
        n_synthetic_needed = n_real_fraud * (ratio - 1)  # số mẫu cần sinh thêm
        print(f"\n{'─'*55}")
        print(f"  Ratio 1:{ratio} → cần sinh {n_synthetic_needed} synthetic fraud")

        if n_synthetic_needed <= 0:
            print(f"  [SKIP] ratio={ratio} không cần sinh thêm")
            continue

        # Sinh + lọc
        synthetic_filtered, stats = generate_and_filter(
            model, fraud_df, feature_cols, n_synthetic_needed
        )
        all_filter_stats[f"ratio_{ratio}"] = stats

        # Gán nhãn nguồn
        fraud_real     = fraud_df[feature_cols + [LABEL_COL]].copy()
        fraud_real["source"] = "real"

        synthetic_filtered = synthetic_filtered.copy()
        synthetic_filtered[LABEL_COL] = 1
        synthetic_filtered["source"]  = "synthetic"

        non_fraud          = non_fraud_df[feature_cols + [LABEL_COL]].copy()
        non_fraud["source"] = "real"

        combined = pd.concat(
            [non_fraud, fraud_real, synthetic_filtered],
            ignore_index=True
        )
        augmented_datasets[f"ratio_{ratio}"] = combined

        out_path = os.path.join(DATA_OUT, f"synthetic_fraud_ratio{ratio}.csv")
        combined.to_csv(out_path, index=False)

        print(f"  Kết quả cuối:")
        print(f"    Non-fraud:        {(combined[LABEL_COL]==0).sum()}")
        print(f"    Fraud thực:       {n_real_fraud}")
        print(f"    Fraud synthetic:  {len(synthetic_filtered)} "
              f"(mục tiêu: {n_synthetic_needed})")
        print(f"    [saved] {out_path}")

    # Lưu thống kê filter
    stats_path = os.path.join(OUTPUT_DIR, "results",
                               "constraint_filter_stats.json")
    with open(stats_path, "w") as f:
        json.dump(all_filter_stats, f, indent=2, default=str)
    print(f"\n  [saved] {stats_path}")

    return augmented_datasets


# ─────────────────────────────────────────────────────────────────
# BƯỚC 3: ĐÁNH GIÁ CHẤT LƯỢNG
# ─────────────────────────────────────────────────────────────────

def evaluate_synthetic_quality(fraud_real, synthetic_df,
                                feature_cols, label=""):
    """
    Đánh giá chất lượng dữ liệu sinh:
    1. KDE plot so sánh phân phối
    2. KS-test từng feature
    3. Ma trận tương quan Real vs Synthetic (trước & sau filter)
    4. t-SNE
    """
    print("\n" + "="*55)
    print(f"ĐÁNH GIÁ CHẤT LƯỢNG DỮ LIỆU SINH {label}")
    print("="*55)

    # ── KDE comparison ──
    n_cols = 4
    n_rows = (len(feature_cols) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, n_rows * 3.5))
    axes = axes.flatten()
    ks_results = []

    for i, feat in enumerate(feature_cols):
        ax = axes[i]
        real_vals  = fraud_real[feat].dropna()
        synth_vals = synthetic_df[feat].dropna()

        q01 = min(real_vals.quantile(0.01), synth_vals.quantile(0.01))
        q99 = max(real_vals.quantile(0.99), synth_vals.quantile(0.99))

        ax.hist(real_vals.clip(q01, q99), bins=30, alpha=0.5,
                color="#e74c3c", density=True, label="Real Fraud")
        ax.hist(synth_vals.clip(q01, q99), bins=30, alpha=0.5,
                color="#2ecc71", density=True, label="Synthetic")
        ax.set_title(feat, fontsize=9, fontweight="bold")
        ax.legend(fontsize=7)

        ks_stat, ks_p = stats.ks_2samp(real_vals, synth_vals)
        ks_results.append({
            "feature": feat, "ks_stat": round(ks_stat, 4),
            "p_value": round(ks_p, 4), "similar": ks_p > 0.05
        })
        color = "green" if ks_p > 0.05 else "red"
        ax.text(0.98, 0.95, f"KS={ks_stat:.3f}", transform=ax.transAxes,
                ha="right", va="top", fontsize=7, color=color, fontweight="bold")

    for j in range(len(feature_cols), len(axes)):
        axes[j].set_visible(False)

    tag = label.replace(" ", "_").lower()
    plt.suptitle(f"Real vs Synthetic Fraud – Phân phối {label}",
                 fontsize=12, fontweight="bold")
    save_fig(f"10_distribution_{tag}")

    ks_df = pd.DataFrame(ks_results).sort_values("ks_stat")
    print(f"\n  KS Test – {ks_df['similar'].mean()*100:.0f}% features "
          f"có phân phối tương tự (p>0.05)")

    # ── Correlation heatmap ──
    fig, axes = plt.subplots(1, 3, figsize=(22, 7))
    corr_real  = fraud_real[feature_cols].corr()
    corr_synth = synthetic_df[feature_cols].corr()
    corr_diff  = (corr_real - corr_synth).abs()

    for ax, corr, title, cmap, vmin, vmax in [
        (axes[0], corr_real,  "Real Fraud",      "RdBu_r", -1, 1),
        (axes[1], corr_synth, f"Synthetic {label}", "RdBu_r", -1, 1),
        (axes[2], corr_diff,  "Absolute Difference", "Reds", 0, 1),
    ]:
        sns.heatmap(corr, ax=ax, annot=True, fmt=".2f", cmap=cmap,
                    center=0 if vmin < 0 else None,
                    vmin=vmin, vmax=vmax,
                    square=True, linewidths=0.3, annot_kws={"size": 7})
        ax.set_title(title, fontsize=11)

    mad = corr_diff.values[np.triu_indices_from(corr_diff.values, k=1)].mean()
    plt.suptitle(f"Ma trận Tương quan {label} | MAD={mad:.4f}",
                 fontsize=13, fontweight="bold")
    save_fig(f"11_correlation_{tag}")
    print(f"  Mean Absolute Difference (correlation): {mad:.4f}")

    # ── t-SNE ──
    try:
        from sklearn.manifold import TSNE
        from sklearn.preprocessing import StandardScaler
        from sklearn.impute import SimpleImputer

        n_r = min(len(fraud_real), 300)
        n_s = min(len(synthetic_df), 300)
        r_sub = fraud_real[feature_cols].sample(n_r, random_state=42)
        s_sub = synthetic_df[feature_cols].sample(n_s, random_state=42)
        combined = pd.concat([r_sub, s_sub])

        imp = SimpleImputer(strategy="median")
        X = StandardScaler().fit_transform(imp.fit_transform(combined))
        labels_arr = np.array(["Real"] * n_r + ["Synthetic"] * n_s)

        print(f"\n  Chạy t-SNE ({n_r + n_s} samples)...")
        X_2d = TSNE(n_components=2, random_state=42,
                    perplexity=20, max_iter=500).fit_transform(X)

        fig, ax = plt.subplots(figsize=(9, 7))
        for lbl, color in [("Real", "#e74c3c"), ("Synthetic", "#2ecc71")]:
            m = labels_arr == lbl
            ax.scatter(X_2d[m, 0], X_2d[m, 1], c=color,
                       label=f"{lbl} Fraud", s=40, alpha=0.7)
        ax.set_title(f"t-SNE: Real vs Synthetic {label}\n"
                     "(overlap tốt → chất lượng cao)")
        ax.legend()
        save_fig(f"12_tsne_{tag}")
    except Exception as e:
        print(f"  [WARN] t-SNE skip: {e}")

    ks_df.to_csv(os.path.join(OUTPUT_DIR, "results",
                               f"ks_test_{tag}.csv"), index=False)
    return ks_df, mad


# ─────────────────────────────────────────────────────────────────
# SO SÁNH TRƯỚC/SAU LỌC
# ─────────────────────────────────────────────────────────────────

def compare_before_after_filter(model, fraud_df, feature_cols,
                                 n_eval=500):
    """
    Sinh một batch, đánh giá trước và sau khi lọc để
    minh họa hiệu quả của Correlation Constraint Filter.
    Dùng cho phần báo cáo chương IV.3.
    """
    print("\n" + "="*55)
    print("SO SÁNH TRƯỚC / SAU KHI LỌC (n_eval={})".format(n_eval))
    print("="*55)

    # Sinh dư để sau lọc vẫn còn đủ mẫu đánh giá
    raw_batch = model.sample(n_eval * OVERSAMPLE_FACTOR)
    raw_batch = raw_batch[feature_cols].copy()

    # Trước lọc
    print("\n  [Trước lọc]")
    ks_before, mad_before = evaluate_synthetic_quality(
        fraud_df, raw_batch, feature_cols, label="Trước lọc"
    )

    # Sau lọc
    filtered_batch, stats = apply_correlation_constraints(
        raw_batch, feature_cols, fraud_real_df=fraud_df, verbose=True
    )

    if len(filtered_batch) < 20:
        print("  [WARN] Quá ít mẫu sau lọc để đánh giá t-SNE")
        return

    print(f"\n  [Sau lọc] ({len(filtered_batch)} mẫu)")
    ks_after, mad_after = evaluate_synthetic_quality(
        fraud_df, filtered_batch, feature_cols, label="Sau lọc"
    )

    # Tóm tắt cải thiện
    print(f"\n  {'─'*45}")
    print(f"  TỔNG KẾT CẢI THIỆN:")
    print(f"  {'─'*45}")
    print(f"  MAD tương quan trước lọc: {mad_before:.4f}")
    print(f"  MAD tương quan sau lọc:   {mad_after:.4f}")
    delta = mad_before - mad_after
    sign  = "↓ cải thiện" if delta > 0 else "↑ xấu hơn"
    print(f"  Δ MAD: {delta:+.4f} ({sign})")
    print(f"  {'─'*45}")

    # Lưu summary
    summary = {
        "n_before_filter": len(raw_batch),
        "n_after_filter":  len(filtered_batch),
        "pct_kept":        round(len(filtered_batch)/len(raw_batch)*100, 2),
        "MAD_before":      round(mad_before, 4),
        "MAD_after":       round(mad_after, 4),
        "MAD_improvement": round(delta, 4),
        "constraints_applied": [c["description"] for c in ACCOUNTING_CONSTRAINTS
                                 if c["strict"]],
    }
    path = os.path.join(OUTPUT_DIR, "results", "filter_improvement_summary.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"  [saved] {path}")


# ─────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────

def run_ctgan():
    print("=" * 55)
    print("  BƯỚC 3: HUẤN LUYỆN CTGAN + CONSTRAINT FILTER")
    print("=" * 55)

    info_path = os.path.join(DATA_OUT, "preprocessed_info.json")
    if not os.path.exists(info_path):
        print("  [ERROR] Chạy 02_preprocessing.py trước!")
        return

    with open(info_path) as f:
        info = json.load(f)
    feature_cols = info["feature_cols"]

    fraud_df = pd.read_csv(os.path.join(DATA_OUT, "fraud_train.csv"))
    feat_available = [c for c in feature_cols if c in fraud_df.columns]
    print(f"  Fraud samples: {len(fraud_df)}")
    print(f"  Features: {feat_available}")

    # Load non-fraud (raw, chưa scale – dùng để ghép file CSV)
    X_train = np.load(os.path.join(DATA_OUT, "X_train.npy"))
    y_train = np.load(os.path.join(DATA_OUT, "y_train.npy"))
    # Lấy lại non-fraud raw từ fraud_train.csv và X_train
    # X_train đã scaled → cần raw; dùng fraud_train.csv làm tham chiếu scale
    # Cho non_fraud_df ta dùng scaled values vì chỉ cần cột feature + label + source
    non_fraud_X = X_train[y_train == 0]
    non_fraud_df = pd.DataFrame(non_fraud_X, columns=feat_available)
    non_fraud_df[LABEL_COL] = 0

    # 1. Huấn luyện
    ctgan_model = train_ctgan(fraud_df[feat_available + [LABEL_COL]]
                               if LABEL_COL in fraud_df.columns
                               else fraud_df, feat_available)
    if ctgan_model is None:
        return

    # 2. So sánh trước/sau lọc (chạy trước để có hình ảnh báo cáo)
    compare_before_after_filter(ctgan_model, fraud_df, feat_available,
                                 n_eval=min(500, len(fraud_df) * 4))

    # 3. Sinh augmented datasets
    create_augmented_datasets(ctgan_model, fraud_df, non_fraud_df,
                               feat_available)

    print("\n[DONE] Hoàn tất. Kiểm tra outputs/figures/ và outputs/results/")
    return ctgan_model


if __name__ == "__main__":
    run_ctgan()
