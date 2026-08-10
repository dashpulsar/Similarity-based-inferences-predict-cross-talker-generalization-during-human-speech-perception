#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Figure_create.py — 统一图表生成脚本
==============================================
Migrated from Nature-Human-Behavior/Figure_create.ipynb

生成的图表:
  Part 1: 单数据集 Z-Profile 图 (每个数据集一张)
  Part 2: 三数据集面板图 (1×N panel, N = 可用数据集数)
  Part 3: 多方法 Z-分布箱线图
  Part 4: 变异性分析网格图 (Xie only, 4×4)
  Part 5: 综合方法比较图 (1×N panel)
  Part 6: Similarity–Accuracy 回归图 (需要中间数据)

使用方式:
  python Figure_create.py               # 生成所有图表
  python Figure_create.py --parts 1 2   # 只生成 Part 1 和 Part 2
"""

import os
import sys
import argparse
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import seaborn as sns

warnings.filterwarnings("ignore", category=FutureWarning)

# ============================================================
# 路径配置
# ============================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
RESULTS_DIR = os.path.join(PROJECT_ROOT, "glmm_prediction", "results")
FIGURE_DIR = os.path.join(PROJECT_ROOT, "figure")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
FEATURES_DIR = os.path.join(DATA_DIR, "features")
BEHAVIORAL_DIR = os.path.join(DATA_DIR, "preprocessed data")

os.makedirs(FIGURE_DIR, exist_ok=True)

# 将 glmm_prediction 加入 sys.path 以便 Part 6 导入 project_utils
sys.path.insert(0, os.path.join(PROJECT_ROOT, "glmm_prediction"))

# ============================================================
# 层定义
# ============================================================
LAYERS = [
    "cnn_2", "cnn_3", "cnn_4", "cnn_5", "cnn_6",
    "tr_0", "tr_2", "tr_4", "tr_6", "tr_8",
    "tr_10", "tr_12", "tr_14", "tr_16", "tr_18",
    "tr_20", "tr_22", "tr_24",
]

X_LABELS = [
    "C2", "C3", "C4", "C5", "C6",
    "T0", "T2", "T4", "T6", "T8",
    "T10", "T12", "T14", "T16", "T18",
    "T20", "T22", "T24",
]

N_LAYERS = len(LAYERS)
X_POS = np.arange(N_LAYERS)

# CNN / Transformer 分界位置
CNN_TR_BOUNDARY = 4.5

# 数据集配置  (prefix, display_name)
DATASETS = [
    ("xie21", "Xie et al. (2021)"),
    ("nygaard19", "Alexander & Nygaard (2019)"),
    ("bradlow23", "Bradlow et al. (2023)"),
]

# 方法配置  (suffix, label, color, marker, linestyle)
METHOD_SPECS = [
    ("hubert",              "HuBERT + t-SNE",        "#4A90D9", "o", "-"),
    ("hubert_ft",           "HuBERT(ft) + t-SNE",    "#E8A838", "s", "-"),
    ("hubert_full_dim",     "HuBERT full-dim",       "#2ECC71", "^", "-"),
    ("hubert_cos_full_dim", "HuBERT cos full-dim",   "#E74C3C", "D", "-"),
    ("hubert_cos_full_dim_ft", "HuBERT(ft) cos full","#9B59B6", "v", "-"),
]

# ============================================================
# 工具函数
# ============================================================

def _sem(arr):
    """计算标准误差 (SEM)"""
    arr = np.asarray(arr, dtype=float)
    n = np.sum(~np.isnan(arr))
    if n <= 1:
        return 0.0
    return np.nanstd(arr, ddof=1) / np.sqrt(n)


def load_glmm_results(filepath):
    """
    加载长格式 GLMM 结果 CSV，过滤 type='corrected'，
    按 layer 分组计算 mean 和 SEM。

    CSV 列: layer, fold, type, [alpha], k/tau, z_train, z_test,
            poll_train, poll_test, optimism

    Returns:
        DataFrame with columns [layer, z_test_mean, z_test_sem,
                                z_train_mean, z_train_sem]
        or None if file not found.
    """
    if not os.path.exists(filepath):
        return None
    try:
        df = pd.read_csv(filepath)
    except Exception as e:
        print(f"  [WARN] 无法读取 {filepath}: {e}")
        return None

    # 过滤 corrected 类型
    if "type" in df.columns and "corrected" in df["type"].values:
        df = df[df["type"] == "corrected"]

    if "layer" not in df.columns or "z_test" not in df.columns:
        return None

    grouped = df.groupby("layer", sort=False).agg(
        z_test_mean=("z_test", "mean"),
        z_test_sem=("z_test", _sem),
        z_train_mean=("z_train", "mean"),
        z_train_sem=("z_train", _sem),
    ).reset_index()

    return grouped


def extract_layer_stats(grouped_df, layers=None):
    """按指定 layer 顺序提取 mean 和 sem 数组。"""
    if layers is None:
        layers = LAYERS
    if grouped_df is None:
        return np.full(len(layers), np.nan), np.full(len(layers), np.nan)

    means, sems = [], []
    for layer in layers:
        row = grouped_df[grouped_df["layer"] == layer]
        if not row.empty:
            means.append(row["z_test_mean"].values[0])
            sems.append(row["z_test_sem"].values[0])
        else:
            means.append(np.nan)
            sems.append(np.nan)
    return np.array(means), np.array(sems)


def load_baseline(prefix, baseline_name):
    """加载 baseline (mfcc / strf) 的 mean 和 sem。"""
    filepath = os.path.join(RESULTS_DIR, f"{prefix}_glmm_results_baseline.csv")
    df = load_glmm_results(filepath)
    if df is None:
        return np.nan, np.nan
    row = df[df["layer"] == baseline_name]
    if row.empty:
        return np.nan, np.nan
    return row["z_test_mean"].values[0], row["z_test_sem"].values[0]


def load_ceiling(prefix):
    """加载天花板 (ceiling) 数据。返回 (mean, sem) 或 (nan, nan)。"""
    ceiling_path = os.path.join(RESULTS_DIR, f"{prefix}_ceiling.csv")
    if not os.path.exists(ceiling_path):
        return np.nan, np.nan
    try:
        df = pd.read_csv(ceiling_path)
        vals = df["z_ceiling"].values
        return np.nanmean(vals), _sem(vals)
    except Exception:
        return np.nan, np.nan


def _has_results(prefix):
    """检查某数据集是否存在任何 GLMM 结果文件。"""
    for suffix, *_ in METHOD_SPECS:
        if os.path.exists(os.path.join(RESULTS_DIR, f"{prefix}_glmm_results_{suffix}.csv")):
            return True
    return False


def _add_ceiling(ax, prefix, x=X_POS):
    """在 ax 上绘制天花板 band。"""
    ceil_m, ceil_s = load_ceiling(prefix)
    if not np.isnan(ceil_m):
        ax.fill_between(x, ceil_m - ceil_s, ceil_m + ceil_s,
                        color="#D3D3D3", alpha=0.4, zorder=0)
        ax.axhline(ceil_m, color="gray", lw=1, zorder=0)
        return True
    return False


def _add_baselines(ax, prefix, x=X_POS):
    """在 ax 上绘制 MFCC 和 STRF baseline。"""
    handles = []
    for bname, color, ls in [("mfcc", "#8B4513", "--"), ("strf", "#228B22", "-.")]:
        bm, bs = load_baseline(prefix, bname)
        if not np.isnan(bm):
            ax.axhline(bm, color=color, linestyle=ls, lw=1.5, zorder=1)
            ax.fill_between(x, bm - bs, bm + bs, color=color, alpha=0.08, zorder=1)
            handles.append(Line2D([0], [0], color=color, ls=ls, lw=1.5,
                                  label=bname.upper()))
    return handles


# ============================================================
# Part 1: 单数据集 Z-Profile 图
# ============================================================

def plot_single_dataset_z_profile(prefix, dataset_name, save=True):
    """为单个数据集绘制 z-statistic 层级分布图。"""
    print(f"  Part 1: {dataset_name} Z-Profile ...")

    # 加载两种 HuBERT 结果
    hubert = load_glmm_results(
        os.path.join(RESULTS_DIR, f"{prefix}_glmm_results_hubert.csv"))
    hubert_ft = load_glmm_results(
        os.path.join(RESULTS_DIR, f"{prefix}_glmm_results_hubert_ft.csv"))

    if hubert is None and hubert_ft is None:
        print(f"    [SKIP] 无可用结果: {dataset_name}")
        return

    fig, ax = plt.subplots(figsize=(12, 6))

    # 天花板
    has_ceil = _add_ceiling(ax, prefix)

    # HuBERT (base) + t-SNE
    if hubert is not None:
        m, s = extract_layer_stats(hubert)
        ax.plot(X_POS, m, marker="o", color="#4A90D9", lw=2, ms=6, zorder=3,
                label="HuBERT (base) + t-SNE")
        ax.fill_between(X_POS, m - s, m + s, color="#4A90D9", alpha=0.15, zorder=2)

    # HuBERT (ft) + t-SNE
    if hubert_ft is not None:
        m, s = extract_layer_stats(hubert_ft)
        ax.plot(X_POS, m, marker="s", color="#E8A838", lw=2, ms=6, zorder=3,
                label="HuBERT (fine-tuned) + t-SNE")
        ax.fill_between(X_POS, m - s, m + s, color="#E8A838", alpha=0.15, zorder=2)

    # Baselines
    _add_baselines(ax, prefix)

    # CNN / Transformer 分界线
    ax.axvline(CNN_TR_BOUNDARY, color="black", ls=":", lw=1, alpha=0.5)

    # 坐标轴
    ax.set_xticks(X_POS)
    ax.set_xticklabels(X_LABELS, fontsize=10)
    ax.set_xlabel("Layer", fontsize=14)
    ax.set_ylabel("z-statistic (test)", fontsize=14)
    ax.set_title(f"{dataset_name} — GLMM z-statistic across HuBERT layers",
                 fontsize=16)

    # 图例
    h_extra = []
    if has_ceil:
        h_extra += [
            Line2D([0], [0], color="gray", lw=1, label="Ceiling (behavioral)"),
            mpatches.Patch(color="#D3D3D3", alpha=0.4, label="Ceiling ± SEM"),
        ]
    bl_handles = _add_baselines(ax, prefix)  # 重复调用无害，主要收集 handles
    ax.legend(fontsize=10, loc="upper left")
    ax.grid(axis="y", ls="--", alpha=0.3)
    sns.despine()
    plt.tight_layout()

    if save:
        out = os.path.join(FIGURE_DIR, f"{prefix}_z_profile.png")
        plt.savefig(out, dpi=300, bbox_inches="tight")
        print(f"    => {out}")
    plt.close()


# ============================================================
# Part 2: 三数据集面板图
# ============================================================

def plot_three_dataset_panel(save=True):
    """1×N 面板, 每个数据集一列 (自动跳过无结果的数据集)。"""
    print("  Part 2: Three-Dataset Panel ...")

    valid = [(p, n) for p, n in DATASETS if _has_results(p)]
    if not valid:
        print("    [SKIP] 没有可用的数据集结果")
        return

    fig, axes = plt.subplots(1, len(valid), figsize=(8 * len(valid), 6),
                              sharey=False, squeeze=False)
    axes = axes[0]

    for col, (prefix, title) in enumerate(valid):
        ax = axes[col]

        _add_ceiling(ax, prefix)

        # HuBERT base + t-SNE
        df = load_glmm_results(
            os.path.join(RESULTS_DIR, f"{prefix}_glmm_results_hubert.csv"))
        if df is not None:
            m, s = extract_layer_stats(df)
            ax.plot(X_POS, m, "o-", color="#4A90D9", lw=2, ms=5,
                    label="HuBERT (base) + t-SNE")
            ax.fill_between(X_POS, m - s, m + s, color="#4A90D9", alpha=0.12)

        # HuBERT ft + t-SNE
        df_ft = load_glmm_results(
            os.path.join(RESULTS_DIR, f"{prefix}_glmm_results_hubert_ft.csv"))
        if df_ft is not None:
            m, s = extract_layer_stats(df_ft)
            ax.plot(X_POS, m, "s-", color="#E8A838", lw=2, ms=5,
                    label="HuBERT (ft) + t-SNE")
            ax.fill_between(X_POS, m - s, m + s, color="#E8A838", alpha=0.12)

        _add_baselines(ax, prefix)
        ax.axvline(CNN_TR_BOUNDARY, color="black", ls=":", lw=1, alpha=0.5)
        ax.set_xticks(X_POS)
        ax.set_xticklabels(X_LABELS, fontsize=9)
        ax.set_xlabel("Layer", fontsize=12)
        if col == 0:
            ax.set_ylabel("z-statistic (test)", fontsize=13)
        ax.set_title(title, fontsize=14)
        ax.grid(axis="y", ls="--", alpha=0.3)
        sns.despine(ax=ax)

    # 共享图例
    handles = [
        Line2D([0], [0], color="gray", lw=1, label="Ceiling"),
        mpatches.Patch(color="#D3D3D3", alpha=0.4, label="Ceiling ± SEM"),
        Line2D([0], [0], color="#4A90D9", marker="o", lw=2,
               label="HuBERT (base) + t-SNE"),
        Line2D([0], [0], color="#E8A838", marker="s", lw=2,
               label="HuBERT (ft) + t-SNE"),
        Line2D([0], [0], color="#8B4513", ls="--", lw=1.5, label="MFCC"),
        Line2D([0], [0], color="#228B22", ls="-.", lw=1.5, label="STRF"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=6, fontsize=11,
               frameon=False, bbox_to_anchor=(0.5, -0.06))
    plt.tight_layout(rect=[0, 0.04, 1, 1])

    if save:
        out = os.path.join(FIGURE_DIR, "three_dataset_z_profile.png")
        plt.savefig(out, dpi=300, bbox_inches="tight")
        print(f"    => {out}")
    plt.close()


# ============================================================
# Part 3: 多方法 Z-分布箱线图
# ============================================================

def plot_z_distribution_boxplot(save=True):
    """跨方法的 z-statistic 分布箱线图 (所有 layers × folds)。"""
    print("  Part 3: Multi-Method Z-Distribution Boxplot ...")

    records = []
    for prefix, ds_name in DATASETS:
        # HuBERT 变体
        for suffix, label, *_ in METHOD_SPECS:
            fpath = os.path.join(RESULTS_DIR, f"{prefix}_glmm_results_{suffix}.csv")
            if not os.path.exists(fpath):
                continue
            df = pd.read_csv(fpath)
            if "type" in df.columns and "corrected" in df["type"].values:
                df = df[df["type"] == "corrected"]
            for _, row in df.iterrows():
                if row["layer"] in LAYERS:
                    records.append({
                        "Dataset": ds_name, "Method": label,
                        "z_test": row["z_test"],
                    })
        # Baseline
        bl_path = os.path.join(RESULTS_DIR, f"{prefix}_glmm_results_baseline.csv")
        if os.path.exists(bl_path):
            bl = pd.read_csv(bl_path)
            if "type" in bl.columns and "corrected" in bl["type"].values:
                bl = bl[bl["type"] == "corrected"]
            for _, row in bl.iterrows():
                if row["layer"] in ("mfcc", "strf"):
                    records.append({
                        "Dataset": ds_name,
                        "Method": f"Baseline ({row['layer'].upper()})",
                        "z_test": row["z_test"],
                    })

    if not records:
        print("    [SKIP] 没有找到可用的方法比较数据")
        return

    df_all = pd.DataFrame(records)
    palette = sns.color_palette("tab20", n_colors=df_all["Method"].nunique())

    fig, ax = plt.subplots(figsize=(16, 7))
    methods_order = df_all["Method"].unique().tolist()

    bp = sns.boxplot(data=df_all, x="Method", y="z_test", hue="Dataset",
                     ax=ax, palette="Set2", fliersize=0, width=0.6)
    sns.stripplot(data=df_all, x="Method", y="z_test", hue="Dataset",
                  ax=ax, dodge=True, alpha=0.5, size=3, jitter=True,
                  palette="Set2", legend=False)

    ax.set_xticklabels(ax.get_xticklabels(), rotation=35, ha="right", fontsize=9)
    ax.set_ylabel("z-statistic (test, per fold)", fontsize=13)
    ax.set_title("Cross-method z-statistic distributions (all layers × folds)",
                 fontsize=15)
    ax.grid(axis="y", ls="--", alpha=0.3)
    ax.legend(title="Dataset", fontsize=10, title_fontsize=11)
    sns.despine()
    plt.tight_layout()

    if save:
        out = os.path.join(FIGURE_DIR, "z_distribution_methods.png")
        plt.savefig(out, dpi=300, bbox_inches="tight")
        print(f"    => {out}")
    plt.close()


# ============================================================
# Part 4: 变异性分析网格图 (Xie only)
# ============================================================

def plot_variability_analysis(save=True):
    """4×4 网格，每个子图对应一种变异性测量。"""
    print("  Part 4: Variability Analysis (Xie) ...")

    prefix = "xie21"
    # 按行分组 (4 行 × 4 列，最后 3 个为空)
    grid = [
        ["WithinTokenSentence", "WithinTokenWord",
         "WithinTokenPhoneme", None],
        ["WithinTypeSentence",  "WithinTypeWord",
         "WithinTypePhoneme",  None],
        ["OrderSentence",       "OrderWord",
         "OrderPhoneme",       None],
        ["BetweenSentence",     "BetweenTypeSentence",
         "BetweenTypeWord",    "BetweenTypePhoneme"],
    ]

    # 显示名称映射 (更易读)
    display_names = {
        "WithinTokenSentence": "Within-Token\n(Sentence)",
        "WithinTokenWord":     "Within-Token\n(Word)",
        "WithinTokenPhoneme":  "Within-Token\n(Phoneme)",
        "WithinTypeSentence":  "Within-Type\n(Sentence)",
        "WithinTypeWord":      "Within-Type\n(Word)",
        "WithinTypePhoneme":   "Within-Type\n(Phoneme)",
        "OrderSentence":       "Order-Sensitive\n(Sentence)",
        "OrderWord":           "Order-Sensitive\n(Word)",
        "OrderPhoneme":        "Order-Sensitive\n(Phoneme)",
        "BetweenSentence":     "Between\n(Sentence)",
        "BetweenTypeSentence": "Between-Type\n(Sentence)",
        "BetweenTypeWord":     "Between-Type\n(Word)",
        "BetweenTypePhoneme":  "Between-Type\n(Phoneme)",
    }

    # 颜色
    palette = sns.color_palette("Set2", 13)
    color_idx = 0

    fig, axes = plt.subplots(4, 4, figsize=(22, 18), sharex=True, sharey=True)
    ceil_m, ceil_s = load_ceiling(prefix)
    has_any_data = False

    for r in range(4):
        for c in range(4):
            ax = axes[r][c]
            vtype = grid[r][c]
            if vtype is None:
                ax.set_visible(False)
                continue

            fname = f"{prefix}_tsne_ft_variability_glmm_{vtype}.csv"
            stats = load_glmm_results(os.path.join(RESULTS_DIR, fname))

            # 天花板 band
            if not np.isnan(ceil_m):
                ax.fill_between(X_POS, ceil_m - ceil_s, ceil_m + ceil_s,
                                color="#D3D3D3", alpha=0.3)
                ax.axhline(ceil_m, color="gray", lw=0.8)

            if stats is not None:
                has_any_data = True
                m, s = extract_layer_stats(stats)
                ax.plot(X_POS, m, "o-", color=palette[color_idx], lw=2, ms=5)
                ax.fill_between(X_POS, m - s, m + s,
                                color=palette[color_idx], alpha=0.15)

            ax.axvline(CNN_TR_BOUNDARY, color="black", ls=":", lw=1, alpha=0.4)
            ax.set_title(display_names.get(vtype, vtype), fontsize=11)
            ax.grid(axis="y", ls="--", alpha=0.2)
            sns.despine(ax=ax)

            if r == 3:
                ax.set_xticks(X_POS)
                ax.set_xticklabels(X_LABELS, fontsize=8)

            color_idx += 1

    if not has_any_data:
        print("    [SKIP] 没有找到变异性分析数据")
        plt.close()
        return

    fig.supxlabel("Layer", fontsize=14, y=0.02)
    fig.supylabel("z-statistic (test)", fontsize=14, x=0.02)
    fig.suptitle("Variability Analysis — Xie Dataset (HuBERT ft + t-SNE)",
                 fontsize=16, y=1.01)
    plt.tight_layout()

    if save:
        out = os.path.join(FIGURE_DIR, "variability_analysis_xie.png")
        plt.savefig(out, dpi=300, bbox_inches="tight")
        print(f"    => {out}")
    plt.close()


# ============================================================
# Part 5: 综合方法比较图
# ============================================================

def plot_comprehensive_method_comparison(save=True):
    """1×N 面板, 每列一个数据集, 叠加所有方法变体。"""
    print("  Part 5: Comprehensive Method Comparison ...")

    valid = [(p, n) for p, n in DATASETS if _has_results(p)]
    if not valid:
        print("    [SKIP] 无可用数据集")
        return

    fig, axes = plt.subplots(1, len(valid), figsize=(9 * len(valid), 7),
                              sharey=False, squeeze=False)
    axes = axes[0]

    for col, (prefix, title) in enumerate(valid):
        ax = axes[col]
        _add_ceiling(ax, prefix)

        for suffix, label, color, marker, ls in METHOD_SPECS:
            fpath = os.path.join(RESULTS_DIR,
                                 f"{prefix}_glmm_results_{suffix}.csv")
            stats = load_glmm_results(fpath)
            if stats is None:
                continue
            m, s = extract_layer_stats(stats)
            ax.plot(X_POS, m, marker=marker, color=color, lw=1.8, ms=5,
                    ls=ls, label=label)
            ax.fill_between(X_POS, m - s, m + s, color=color, alpha=0.08)

        _add_baselines(ax, prefix)
        ax.axvline(CNN_TR_BOUNDARY, color="black", ls=":", lw=1, alpha=0.5)
        ax.set_xticks(X_POS)
        ax.set_xticklabels(X_LABELS, fontsize=9)
        ax.set_xlabel("Layer", fontsize=12)
        if col == 0:
            ax.set_ylabel("z-statistic (test)", fontsize=13)
        ax.set_title(title, fontsize=14)
        ax.grid(axis="y", ls="--", alpha=0.3)
        sns.despine(ax=ax)

    # 共享图例
    handles = [
        Line2D([0], [0], color="gray", lw=1, label="Ceiling"),
        mpatches.Patch(color="#D3D3D3", alpha=0.4, label="Ceiling ± SEM"),
    ]
    for suffix, label, color, marker, ls in METHOD_SPECS:
        handles.append(Line2D([0], [0], color=color, marker=marker,
                              lw=1.8, ls=ls, label=label))
    handles.append(Line2D([0], [0], color="#8B4513", ls="--", lw=1.5,
                          label="MFCC"))
    handles.append(Line2D([0], [0], color="#228B22", ls="-.", lw=1.5,
                          label="STRF"))

    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=10,
               frameon=False, bbox_to_anchor=(0.5, -0.08))
    plt.tight_layout(rect=[0, 0.06, 1, 1])

    if save:
        out = os.path.join(FIGURE_DIR, "comprehensive_method_comparison.png")
        plt.savefig(out, dpi=300, bbox_inches="tight")
        print(f"    => {out}")
    plt.close()


# ============================================================
# Part 6: Similarity–Accuracy 回归图
# ============================================================
# 注意: 此部分需要中间数据 (similarity 值 + 行为数据的合并)。
# 中间数据保存在 results/{prefix}_similarity_accuracy_data.csv。
# 如果该文件不存在，脚本会尝试调用 compute_similarity_data() 来生成。
# 由于计算涉及加载 HDF5 特征并进行 DTW 运算，过程可能较慢。
# ============================================================

def compute_similarity_data(prefix):
    """
    计算 Similarity–Accuracy 中间数据。

    流程:
      1. 加载行为数据 (behavioral CSV)
      2. 加载 t-SNE 特征 (HDF5, layer tr_24, fine-tuned)
      3. 对每个 (exposure_talkers, test_talker) 组合:
         - 提取共有词的特征
         - 计算 DTW 距离
         - 转换为 similarity = exp(-d * k)
      4. 与行为数据合并
      5. 保存到 results/{prefix}_similarity_accuracy_data.csv

    Returns:
        bool: True if successful, False otherwise.
    """
    print(f"    [COMPUTE] 正在为 {prefix} 计算 similarity-accuracy 数据 ...")
    print(f"    [COMPUTE] 此过程需要加载 HDF5 特征文件和计算 DTW，可能需要较长时间。")

    # ---- 路径映射 ----
    beh_map = {
        "xie21":    "X21-behavioral-data.csv",
        "nygaard19": "AN19-behavioral-data.csv",
        "bradlow23": "B23-behavioral-data.csv",
    }
    h5_map = {
        "xie21":    "xie21_tsne_3d_ft.h5",
        "nygaard19": "nygaard19_tsne_3d_ft.h5",
        "bradlow23": "bradlow23_tsne_3d_ft.h5",
    }
    audio_map = {
        "xie21":    os.path.join(DATA_DIR, "raw_data", "xie_liu_jaeger21"),
        "nygaard19": os.path.join(DATA_DIR, "raw_data", "alexander_nygaard19"),
        "bradlow23": os.path.join(DATA_DIR, "raw_data", "bradlow_bassard_paller23"),
    }

    beh_file = os.path.join(BEHAVIORAL_DIR, beh_map.get(prefix, ""))
    h5_file = os.path.join(FEATURES_DIR, h5_map.get(prefix, ""))
    audio_dir = audio_map.get(prefix, "")

    if not os.path.exists(beh_file):
        print(f"    [SKIP] 行为数据文件不存在: {beh_file}")
        return False
    if not os.path.exists(h5_file):
        print(f"    [SKIP] 特征文件不存在: {h5_file}")
        return False

    try:
        import project_utils as pu
    except ImportError:
        print("    [ERROR] 无法导入 project_utils。请确保 glmm_prediction/ 在路径中。")
        return False

    try:
        # 加载行为数据
        beh_df = pd.read_csv(beh_file)
        # 仅保留 test phase
        beh_test = beh_df[beh_df["phase"] == "test"].copy()
        if beh_test.empty:
            print(f"    [SKIP] 行为数据中没有 test phase 数据")
            return False

        # 加载特征 (tr_24 层, fine-tuned)
        layer_key = "tr_24"
        print(f"    [COMPUTE] 加载特征: {h5_file}, layer={layer_key}")
        layer_data, speakers = pu.load_single_layer_from_h5(
            h5_file, audio_dir, layer_key)

        # 标准化
        layer_data = pu.standardization(layer_data)

        # 从 GLMM 结果中读取最优 k 值 (如果存在)
        glmm_path = os.path.join(RESULTS_DIR,
                                 f"{prefix}_glmm_results_hubert_ft.csv")
        k_val = 1.0  # 默认值
        if os.path.exists(glmm_path):
            glmm_df = pd.read_csv(glmm_path)
            if "type" in glmm_df.columns:
                glmm_df = glmm_df[glmm_df["type"] == "corrected"]
            tr24_rows = glmm_df[glmm_df["layer"] == "tr_24"]
            if not tr24_rows.empty and "k" in tr24_rows.columns:
                k_val = tr24_rows["k"].mean()
                print(f"    [COMPUTE] 使用 k = {k_val:.4f} (from GLMM results)")

        # 提取词级特征并计算距离
        print(f"    [COMPUTE] 构建词集并计算 DTW 距离 ...")
        word_features, feature_dict = pu.create_set(audio_dir, beh_test, layer_data)

        # 计算每个观测的 similarity
        results = []
        for idx, row in beh_test.iterrows():
            exp_talkers_str = row.get("exposure_talkers", "")
            test_talker = row.get("item_talker", "")
            if pd.isna(exp_talkers_str) or pd.isna(test_talker):
                continue
            if exp_talkers_str == "" or exp_talkers_str == "none":
                # Control 条件没有 exposure talkers
                sim = np.nan
            else:
                # 从 feature_dict 提取并计算
                exp_list = [t.strip() for t in str(exp_talkers_str).split(",")]
                # 获取 exposure 和 test 的共有词距离
                dists = []
                test_key = str(test_talker).strip()
                if test_key in feature_dict:
                    test_feats = feature_dict[test_key]
                    for exp_t in exp_list:
                        exp_key = exp_t.strip()
                        if exp_key in feature_dict:
                            exp_feats = feature_dict[exp_key]
                            # 逐句计算 DTW
                            for si in range(min(len(test_feats), len(exp_feats))):
                                for tf, ef in zip(test_feats[si], exp_feats[si]):
                                    if len(tf) > 0 and len(ef) > 0:
                                        d = pu.dtw_raw_distance(tf, ef)
                                        dists.append(d)
                if dists:
                    sim = np.exp(-np.mean(dists) * k_val)
                else:
                    sim = np.nan

            # 确定条件
            condition = _classify_condition(row, prefix)

            results.append({
                "similarity": sim,
                "accuracy": row.get("response_correct", np.nan),
                "condition": condition,
                "test_talker": test_talker,
                "participant": row.get("participant_id", ""),
                "exposure_accents": row.get("exposure_accents", ""),
                "test_accents": row.get("test_accents", ""),
            })

        out_df = pd.DataFrame(results)
        out_path = os.path.join(RESULTS_DIR,
                                f"{prefix}_similarity_accuracy_data.csv")
        out_df.to_csv(out_path, index=False)
        print(f"    [COMPUTE] 已保存: {out_path} ({len(out_df)} 行)")
        return True

    except Exception as e:
        import traceback
        print(f"    [ERROR] 计算失败: {e}")
        traceback.print_exc()
        return False


def _classify_condition(row, prefix):
    """根据行为数据行的条件列推断实验条件名称。"""
    if prefix == "xie21":
        exp_acc = row.get("exposure_condition.accent", "")
        exp_tlk = row.get("exposure_condition.talker", "")
        test_gen = row.get("test_condition.talker_generalization", "")
        if exp_acc == "control":
            return "Control"
        elif exp_tlk == "multi":
            return "Multi-talker"
        elif exp_tlk == "single" and test_gen == "cross-talker":
            return "Single-talker"
        elif exp_tlk == "single" and test_gen == "within-talker":
            return "Within-talker"
        return "Other"

    elif prefix == "nygaard19":
        # 按 exposure accent 分组
        return str(row.get("exposure_accents", "Unknown"))

    elif prefix == "bradlow23":
        return str(row.get("exposure_accents", "Unknown"))

    return "Unknown"


def plot_similarity_accuracy(save=True):
    """
    Part 6: 绘制 Similarity vs. Accuracy 回归图。

    对每个数据集:
      - 从缓存 CSV 加载 similarity-accuracy 数据
      - 如果不存在，尝试调用 compute_similarity_data() 生成
      - 绘制按条件着色的 binned scatter + logistic regression S-curves
    """
    print("  Part 6: Similarity–Accuracy Regression Plots ...")

    # 条件颜色映射
    xie_colors = {
        "Control": "gray",
        "Multi-talker": "#2ca02c",
        "Single-talker": "#1f77b4",
        "Within-talker": "#d62728",
    }

    for prefix, dataset_name in DATASETS:
        data_path = os.path.join(RESULTS_DIR,
                                 f"{prefix}_similarity_accuracy_data.csv")

        # 尝试加载或计算
        if not os.path.exists(data_path):
            print(f"    Similarity-accuracy 数据不存在: {prefix}. 尝试计算 ...")
            success = compute_similarity_data(prefix)
            if not success:
                print(f"    [SKIP] {dataset_name}: 无法生成数据")
                continue

        if not os.path.exists(data_path):
            print(f"    [SKIP] {dataset_name}: 数据文件仍不存在")
            continue

        try:
            df = pd.read_csv(data_path)
            # 丢弃 similarity 为 NaN 的行 (例如 Control 条件)
            df_valid = df.dropna(subset=["similarity", "accuracy"])
            if df_valid.empty:
                print(f"    [SKIP] {dataset_name}: 无有效数据")
                continue
        except Exception as e:
            print(f"    [ERROR] 读取失败: {e}")
            continue

        # --- 绘图 ---
        conditions = df_valid["condition"].dropna().unique()

        if prefix == "xie21":
            colors = xie_colors
        else:
            # 自动分配颜色
            palette = sns.color_palette("tab10", n_colors=len(conditions))
            colors = {c: palette[i] for i, c in enumerate(conditions)}

        fig, ax = plt.subplots(figsize=(10, 7))

        for cond in sorted(conditions):
            cond_data = df_valid[df_valid["condition"] == cond]
            if cond_data.empty:
                continue
            color = colors.get(cond, "black")

            # Binned scatter
            sns.regplot(data=cond_data, x="similarity", y="accuracy",
                        x_bins=10, fit_reg=False, ax=ax,
                        color=color,
                        scatter_kws={"alpha": 0.3, "s": 40})
            # Logistic regression S-curve
            try:
                sns.regplot(data=cond_data, x="similarity", y="accuracy",
                            scatter=False, logistic=True, ax=ax,
                            color=color,
                            line_kws={"linewidth": 3, "alpha": 0.8},
                            label=cond)
            except Exception:
                # logistic 拟合可能失败
                pass

        ax.set_ylim(-0.05, 1.05)
        ax.set_xlabel("Model Similarity (tr_24 layer)", fontsize=14)
        ax.set_ylabel("Listener Accuracy", fontsize=14)
        ax.set_title(f"{dataset_name}: Similarity vs. Accuracy", fontsize=16)
        ax.legend(title="Condition", fontsize=12, title_fontsize=12,
                  loc="lower right")
        sns.despine()
        plt.tight_layout()

        if save:
            out = os.path.join(FIGURE_DIR,
                               f"{prefix}_similarity_accuracy.png")
            plt.savefig(out, dpi=300, bbox_inches="tight")
            print(f"    => {out}")
        plt.close()


# ============================================================
# 主函数
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="生成 GLMM 分析结果的图表")
    parser.add_argument("--parts", nargs="*", type=int, default=None,
                        help="要生成的 Part 编号 (1-6)。默认全部。")
    args = parser.parse_args()

    parts = set(args.parts) if args.parts else {1, 2, 3, 4, 5, 6}

    print("=" * 60)
    print("Figure_create.py — 统一图表生成脚本")
    print(f"  项目根目录: {PROJECT_ROOT}")
    print(f"  结果目录:   {RESULTS_DIR}")
    print(f"  输出目录:   {FIGURE_DIR}")
    print("=" * 60)

    # Part 1: 单数据集 Z-Profile
    if 1 in parts:
        for prefix, name in DATASETS:
            plot_single_dataset_z_profile(prefix, name)

    # Part 2: 三数据集面板
    if 2 in parts:
        plot_three_dataset_panel()

    # Part 3: Z-分布箱线图
    if 3 in parts:
        plot_z_distribution_boxplot()

    # Part 4: 变异性分析
    if 4 in parts:
        plot_variability_analysis()

    # Part 5: 综合方法比较
    if 5 in parts:
        plot_comprehensive_method_comparison()

    # Part 6: Similarity–Accuracy 回归图
    if 6 in parts:
        plot_similarity_accuracy()

    print("=" * 60)
    print(f"图表生成完成。请查看 {FIGURE_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
