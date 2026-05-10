"""
讀取 results/ablation/*.csv，產生分析表與視覺化。
執行：python -m src.analyze_ablations
"""
from __future__ import annotations
import json
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

PROJ = Path(__file__).parent.parent
ABL = PROJ / "results" / "ablation"
FIG = PROJ / "results" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

plt.rcParams["font.sans-serif"] = ["Microsoft JhengHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False
sns.set_theme(style="whitegrid", font="Microsoft JhengHei")


def analyze_multi_seed():
    p = ABL / "multi_seed.csv"
    if not p.exists():
        print(f"[skip] {p}")
        return
    df = pd.read_parquet(p) if str(p).endswith(".parquet") else pd.read_csv(p)
    print("\n=== 多 Seed 統計（mean ± std）===")
    metrics = ["recall@10", "recall@20", "ndcg@10", "ndcg@20", "hit@10"]
    summary = (
        df.groupby("model")[metrics]
        .agg(["mean", "std"])
    )
    print(summary.round(4))

    # 寫成 markdown 表
    out = ["| 模型 | " + " | ".join(metrics) + " |", "|---" * (len(metrics) + 1) + "|"]
    for model in df["model"].unique():
        sub = df[df["model"] == model]
        row = [f"**{model}**"]
        for m in metrics:
            mean = sub[m].mean()
            std = sub[m].std()
            row.append(f"{mean:.4f} ± {std:.4f}")
        out.append("| " + " | ".join(row) + " |")
    (ABL / "multi_seed_table.md").write_text("\n".join(out), encoding="utf-8")
    print(f"[saved] {ABL / 'multi_seed_table.md'}")


def analyze_hyperparam():
    p = ABL / "hyperparam.csv"
    if not p.exists():
        print(f"[skip] {p}")
        return
    df = pd.read_csv(p)
    print("\n=== 超參數 Grid 結果（recall@20）===")
    pivot = df.pivot_table(values="recall@20", index="n_layers", columns="embed_dim")
    print(pivot.round(4))

    # heatmap
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    sns.heatmap(pivot, annot=True, fmt=".4f", cmap="YlGn", ax=axes[0])
    axes[0].set_title("Recall@20 (heatmap, embed_dim × n_layers)")
    pivot_n = df.pivot_table(values="ndcg@10", index="n_layers", columns="embed_dim")
    sns.heatmap(pivot_n, annot=True, fmt=".4f", cmap="YlGn", ax=axes[1])
    axes[1].set_title("NDCG@10 (heatmap)")
    plt.tight_layout()
    out = FIG / "fig09_hyperparam_heatmap.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"[saved] {out}")

    # 找出最佳組合
    best = df.loc[df["recall@20"].idxmax()]
    print(f"\n[BEST] embed_dim={best['embed_dim']}, n_layers={best['n_layers']}: "
          f"R@20={best['recall@20']:.4f}, NDCG@10={best['ndcg@10']:.4f}")


def analyze_side_info():
    p = ABL / "side_info.csv"
    if not p.exists():
        print(f"[skip] {p}")
        return
    df = pd.read_csv(p)
    print("\n=== Side-info Ablation ===")
    show_cols = ["config", "recall@10", "recall@20", "ndcg@10", "hit@10"]
    print(df[show_cols].to_string(index=False))

    # 長條圖
    fig, ax = plt.subplots(figsize=(12, 5))
    metrics = ["recall@10", "ndcg@10", "hit@10"]
    x = np.arange(len(df))
    width = 0.25
    for i, m in enumerate(metrics):
        ax.bar(x + i * width, df[m], width, label=m)
    ax.set_xticks(x + width)
    ax.set_xticklabels(df["config"], rotation=20)
    ax.set_title("Side-info Ablation（哪個 side info 最有效）")
    ax.set_ylabel("Score")
    ax.legend()
    plt.tight_layout()
    out = FIG / "fig10_side_info_ablation.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"[saved] {out}")


def analyze_reserve_weight():
    p = ABL / "reserve_weight.csv"
    if not p.exists():
        print(f"[skip] {p}")
        return
    df = pd.read_csv(p).sort_values("reserve_weight")
    print("\n=== Reserve Weight Ablation ===")
    print(df[["reserve_weight", "recall@10", "recall@20", "ndcg@10", "hit@10"]].to_string(index=False))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(df["reserve_weight"], df["recall@20"], "o-", linewidth=2, markersize=8)
    axes[0].set_xlabel("預約邊權重"); axes[0].set_ylabel("Recall@20")
    axes[0].set_title("Recall@20 vs 預約邊權重")
    axes[1].plot(df["reserve_weight"], df["ndcg@10"], "o-", color="orange", linewidth=2, markersize=8)
    axes[1].set_xlabel("預約邊權重"); axes[1].set_ylabel("NDCG@10")
    axes[1].set_title("NDCG@10 vs 預約邊權重")
    plt.tight_layout()
    out = FIG / "fig11_reserve_weight.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"[saved] {out}")


def analyze_cold_start():
    """從 results/*_history.json 抓 cold_start 結果"""
    print("\n=== Cold-Start 分箱（按 train 互動次數分箱）===")
    rows = []
    for name in ["popular", "itemcf", "bprmf", "lightgcn", "lightgcn_si",
                 "lightgcn_multi", "lightgcn_bert", "lightgcn_hetero",
                 "lightgcn_timedecay", "ngcf"]:
        p = PROJ / "results" / f"{name}_history.json"
        if not p.exists():
            continue
        with open(p, encoding="utf-8") as f:
            h = json.load(f)
        cs = h.get("cold_start", {})
        for label, m in cs.items():
            if m.get("n_users", 0) == 0:
                continue
            rows.append({
                "model": name, "bin": label,
                "n_users": m["n_users"],
                "recall@10": m.get("recall@10", 0),
                "ndcg@10": m.get("ndcg@10", 0),
                "hit@10": m.get("hit@10", 0),
            })
    if not rows:
        print("  (no cold_start results)")
        return
    df = pd.DataFrame(rows)
    pivot = df.pivot_table(values="recall@10", index="model", columns="bin")
    print(pivot.round(4))
    pivot.to_csv(ABL / "cold_start_recall10.csv")

    # heatmap + grouped bar
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    sns.heatmap(pivot, annot=True, fmt=".3f", cmap="YlGn", ax=axes[0])
    axes[0].set_title("Recall@10 by 模型 × 使用者活躍度")

    # grouped bar: 對前幾個關鍵模型
    key_models = ["popular", "bprmf", "lightgcn", "lightgcn_multi"]
    df_plot = df[df["model"].isin(key_models)]
    if not df_plot.empty:
        sns.barplot(data=df_plot, x="bin", y="recall@10", hue="model",
                    ax=axes[1], hue_order=key_models)
        axes[1].set_title("Recall@10 by 使用者活躍度")
        axes[1].set_xlabel("train 互動次數區間")
    plt.tight_layout()
    out = FIG / "fig12_cold_start.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"[saved] {out}")


def main():
    analyze_multi_seed()
    analyze_hyperparam()
    analyze_side_info()
    analyze_reserve_weight()
    analyze_cold_start()


if __name__ == "__main__":
    main()
