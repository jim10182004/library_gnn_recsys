"""
產生額外的圖表：
  - 系統總架構圖（fig20_pipeline.png）
  - 模型比較雷達圖（fig21_radar.png）
  - 模型比較條形圖按 R@20 排序（fig22_ranked_bars.png）
  - 冷啟動分析圖（fig23_cold_start.png）

執行：python -m src.plot_extra
"""
from __future__ import annotations
import sys
from pathlib import Path

import pandas as pd  # noqa
import pyarrow  # noqa: F401
import numpy as np
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

PROJ = Path(__file__).resolve().parent.parent
RES = PROJ / "results"
FIG = RES / "figures"
FIG.mkdir(parents=True, exist_ok=True)

plt.rcParams["font.sans-serif"] = ["Microsoft JhengHei", "SimHei", "Arial Unicode MS", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False


# ============= 1. Pipeline 架構圖 =============

def plot_pipeline():
    """資料清洗 → 圖建構 → 模型訓練 → 評估 → Demo/API 的視覺化"""
    fig, ax = plt.subplots(figsize=(15, 7))
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 7)
    ax.axis("off")

    # 配色
    PRIMARY = "#028090"
    SECONDARY = "#114b5f"
    ACCENT = "#f59e0b"
    BG = "#f4f4f4"

    stages = [
        # (x, y, w, h, title, sub, color)
        (0.3, 4.5, 2.4, 1.6, "1. 原始資料", "圖書館 4 個 Excel\n(borrows, reserves,\nusers, books)", "#94a3b8"),
        (3.1, 4.5, 2.4, 1.6, "2. 前處理", "清洗、ISBN 統一\nExcel → Parquet\nk-core >= 5 過濾", "#64748b"),
        (5.9, 4.5, 2.4, 1.6, "3. 圖建構", "雙邊圖 + 預約 weak edge\n稀疏鄰接矩陣 (sparse)\n3.6 萬讀者 + 3 萬書", PRIMARY),
        (8.7, 4.5, 2.4, 1.6, "4. 模型訓練", "16 模型對照\nBPR loss + Adam\nGPU 約 18 hr", PRIMARY),
        (11.5, 4.5, 2.4, 1.6, "5. 評估", "7 指標 × 多 seed\nstats / 公平性 / 冷啟\nOptuna 自動調參", PRIMARY),

        (3.1, 1.2, 2.4, 1.6, "6a. CLI Demo", "src/demo.py\n命令列即時推薦", ACCENT),
        (5.9, 1.2, 2.4, 1.6, "6b. Streamlit", "app_public.py\n11 個 persona\n+ 自訂模式", ACCENT),
        (8.7, 1.2, 2.4, 1.6, "6c. FastAPI", "api/main.py + HTML\n履歷展示用\nbest_model lifespan", ACCENT),
        (11.5, 1.2, 2.4, 1.6, "6d. 文件", "論文 docx\n簡報 pptx\n白話版 + Q&A", ACCENT),
    ]

    for x, y, w, h, title, sub, color in stages:
        bbox = FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.05",
            linewidth=2, edgecolor=color, facecolor=BG,
        )
        ax.add_patch(bbox)
        ax.text(x + w/2, y + h - 0.35, title,
                ha="center", va="center", fontsize=11, fontweight="bold", color=color)
        ax.text(x + w/2, y + 0.55, sub,
                ha="center", va="center", fontsize=9, color="#222222")

    # 上排箭頭：1→2→3→4→5
    for i in range(4):
        x_start = 0.3 + i * 2.8 + 2.4
        x_end = x_start + 0.4
        arrow = FancyArrowPatch(
            (x_start, 5.3), (x_end, 5.3),
            arrowstyle="->", mutation_scale=20,
            color=SECONDARY, linewidth=2,
        )
        ax.add_patch(arrow)

    # 5 → 6a/b/c/d 分支
    for i in range(4):
        x_target = 3.1 + i * 2.8 + 1.2
        arrow = FancyArrowPatch(
            (12.7, 4.45), (x_target, 2.85),
            arrowstyle="->", mutation_scale=15,
            color=SECONDARY, linewidth=1.5,
            connectionstyle="arc3,rad=-0.1",
        )
        ax.add_patch(arrow)

    # Title
    ax.text(7.5, 6.5, "圖書館 GNN 推薦系統 — 完整架構",
            ha="center", fontsize=18, fontweight="bold", color=SECONDARY)

    # 底部說明
    ax.text(7.5, 0.4,
            "* 灰：資料層    * 青：核心 ML    * 黃：對外 deliverable    * python run_all.py 一行可重現",
            ha="center", fontsize=10, color="#555555", style="italic")

    plt.tight_layout()
    out = FIG / "fig20_pipeline.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  saved: {out}")


# ============= 2. 雷達圖 =============

def plot_radar():
    """選 6 個代表性模型，6 個指標的雷達比較"""
    df = pd.read_csv(RES / "summary.csv")

    # 選代表模型
    target_models = ["popular", "bprmf", "lightgcn", "lightgcn_multi",
                     "lightgcn_multi_opt", "ngcf"]
    df = df[df["Model"].isin(target_models)].set_index("Model").reindex(target_models)
    if df.empty:
        print("  [skip] summary 沒有需要的模型")
        return

    # 選 6 個指標（要正規化到 [0,1]）
    metrics = ["recall@10", "ndcg@10", "hit@10",
               "coverage@10", "novelty@10", "mrr@10"]
    metric_labels = ["Recall@10", "NDCG@10", "Hit@10",
                     "Coverage@10", "Novelty@10", "MRR@10"]
    sub = df[metrics].copy()
    # 對每個指標 min-max normalize（雷達圖要相對比較）
    for col in metrics:
        col_min = sub[col].min()
        col_max = sub[col].max()
        if col_max > col_min:
            sub[col] = (sub[col] - col_min) / (col_max - col_min)
        else:
            sub[col] = 0.5

    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]  # close polygon

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))

    colors = ["#94a3b8", "#f59e0b", "#0ea5e9", "#22c55e", "#dc2626", "#a855f7"]
    pretty_names = {
        "popular": "Popular（基線）",
        "bprmf": "BPR-MF（傳統）",
        "lightgcn": "LightGCN",
        "lightgcn_multi": "LightGCN-Multi",
        "lightgcn_multi_opt": "LightGCN-Multi-Opt ★",
        "ngcf": "NGCF（複雜版）",
    }
    line_widths = {"lightgcn_multi_opt": 3.0}

    for i, (model, row) in enumerate(sub.iterrows()):
        values = row.tolist() + [row.tolist()[0]]
        lw = line_widths.get(model, 1.8)
        ax.plot(angles, values, "o-", linewidth=lw,
                label=pretty_names.get(model, model),
                color=colors[i % len(colors)],
                markersize=6 if model != "lightgcn_multi_opt" else 9)
        ax.fill(angles, values, alpha=0.10 if model != "lightgcn_multi_opt" else 0.18,
                color=colors[i % len(colors)])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metric_labels, fontsize=12)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["25%", "50%", "75%", "100%"], color="gray", fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.set_title("模型多面向比較（每軸 min-max 正規化）", fontsize=15,
                 fontweight="bold", color="#114b5f", pad=24)
    ax.legend(loc="upper right", bbox_to_anchor=(1.32, 1.05), fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = FIG / "fig21_radar.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  saved: {out}")


# ============= 3. R@20 排序條形圖 =============

def plot_ranked_bars():
    """16 個模型按 R@20 排序的水平條形圖"""
    df = pd.read_csv(RES / "summary.csv")
    df = df.sort_values("recall@20", ascending=True)

    pretty = {
        "popular": "Popular（基線）",
        "itemcf": "ItemCF",
        "bprmf": "BPR-MF",
        "ngcf": "NGCF",
        "lightgcn": "LightGCN",
        "lightgcn_si": "+ Side Info",
        "lightgcn_multi": "+ 預約邊",
        "lightgcn_bert": "+ BERT 書名",
        "lightgcn_hetero": "+ 作者節點",
        "lightgcn_timedecay": "+ 時間衰減",
        "lightgcn_tgn": "+ TGN",
        "lightgcn_cover": "+ Cover CNN",
        "simgcl": "SimGCL (調參)",
        "sasrec": "SASRec (序列)",
        "lightgcn_opt": "Optuna 調參",
        "lightgcn_multi_opt": "★ Optuna + Multi",
    }
    df["pretty"] = df["Model"].map(lambda x: pretty.get(x, x))

    # Highlight 最佳
    colors = ["#dc2626" if m == "lightgcn_multi_opt" else
              ("#f59e0b" if m == "lightgcn_opt" else "#0ea5e9")
              for m in df["Model"]]

    fig, ax = plt.subplots(figsize=(11, 8))
    bars = ax.barh(df["pretty"], df["recall@20"], color=colors, alpha=0.85,
                   edgecolor="#222", linewidth=0.5)

    # 標數值
    for bar, val in zip(bars, df["recall@20"]):
        ax.text(val + 0.002, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", fontsize=10, color="#222")

    ax.set_xlabel("Recall@20（越高越好）", fontsize=12)
    ax.set_title("16 個模型 Recall@20 排序", fontsize=15,
                 fontweight="bold", color="#114b5f")
    ax.set_xlim(min(df["recall@20"]) * 0.96, max(df["recall@20"]) * 1.04)
    ax.grid(axis="x", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Legend 解釋顏色
    legend_handles = [
        mpatches.Patch(color="#dc2626", label="本研究最佳"),
        mpatches.Patch(color="#f59e0b", label="Optuna 變體"),
        mpatches.Patch(color="#0ea5e9", label="其他模型"),
    ]
    ax.legend(handles=legend_handles, loc="lower right", fontsize=10)

    plt.tight_layout()
    out = FIG / "fig22_ranked_bars.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  saved: {out}")


# ============= 4. 冷啟動分箱圖 =============

def plot_cold_start():
    cold = RES / "ablation" / "cold_start_recall10.csv"
    if not cold.exists():
        print("  [skip] cold_start_recall10.csv 不存在")
        return
    df = pd.read_csv(cold)

    fig, ax = plt.subplots(figsize=(11, 6))

    # x 軸是 bin 名稱
    if "bin" not in df.columns:
        df["bin"] = df.iloc[:, 0]
    bins = df["bin"].tolist()
    n_bins = len(bins)
    x = np.arange(n_bins)

    # 找出哪幾欄是模型
    model_cols = [c for c in df.columns if c not in ("bin", "n_users")]
    width = 0.8 / max(len(model_cols), 1)

    colors = plt.cm.tab10(np.linspace(0, 1, len(model_cols)))
    for i, col in enumerate(model_cols):
        ax.bar(x + i * width, df[col], width, label=col, color=colors[i], alpha=0.85)

    ax.set_xticks(x + width * (len(model_cols) - 1) / 2)
    ax.set_xticklabels(bins, rotation=0, fontsize=11)
    ax.set_xlabel("讀者在 train 中的互動次數區間", fontsize=11)
    ax.set_ylabel("Recall@10", fontsize=11)
    ax.set_title("冷啟動分析：不同活躍度的讀者表現", fontsize=14,
                 fontweight="bold", color="#114b5f")
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(axis="y", alpha=0.3)

    # 在 x 軸下方加 n_users
    if "n_users" in df.columns:
        for i, n in enumerate(df["n_users"]):
            ax.text(x[i] + width * (len(model_cols) - 1) / 2,
                    -ax.get_ylim()[1] * 0.06,
                    f"n={int(n):,}", ha="center", fontsize=9, color="#666")

    plt.tight_layout()
    out = FIG / "fig23_cold_start.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  saved: {out}")


def main():
    print("=== 產生額外圖表 ===")
    print("\n[1/4] Pipeline 架構圖 ...")
    plot_pipeline()
    print("\n[2/4] 雷達圖 ...")
    plot_radar()
    print("\n[3/4] R@20 排序條形圖 ...")
    plot_ranked_bars()
    print("\n[4/4] 冷啟動分析圖 ...")
    plot_cold_start()
    print("\n=== 完成 ===")


if __name__ == "__main__":
    main()
