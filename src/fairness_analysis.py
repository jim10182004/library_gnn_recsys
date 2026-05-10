"""
公平性分析：把測試集讀者按性別 / 年齡分組，分別評估推薦品質
看模型是否對不同人口群體有差別待遇。

執行：python -m src.fairness_analysis
輸出：
  - results/fairness.csv
  - results/figures/fig13_fairness.png
  - results/fairness.md
"""
from __future__ import annotations
from pathlib import Path
import pandas as pd  # noqa: F401  must be imported before torch
import pyarrow  # noqa: F401
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import torch

PROJ = Path(__file__).parent.parent
PROC = PROJ / "data" / "processed"
CKPT = PROJ / "checkpoints"
RESULTS = PROJ / "results"
FIG = RESULTS / "figures"

plt.rcParams["font.sans-serif"] = ["Microsoft JhengHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False
sns.set_theme(style="whitegrid", font="Microsoft JhengHei")


def main():
    from src.dataset import load_splits
    from src.evaluate import build_user_pos, evaluate_topk
    from src.models.lightgcn import LightGCN, build_norm_adj

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {DEVICE}")

    print("[1/4] 載入 splits + users ...")
    splits = load_splits()
    users_df = pd.read_parquet(PROC / "users.parquet")

    # 對 splits 內 user 補上 gender / age
    inv_user = {v: k for k, v in splits.user_remap.items()}
    user_meta = pd.DataFrame({"u": range(splits.n_users)})
    user_meta["orig"] = user_meta["u"].map(inv_user)
    # users.parquet 的 user_id 是 OLD compact id (與 inv_user 的 value 對應)
    user_meta = user_meta.merge(users_df[["user_id", "gender", "age"]],
                                left_on="orig", right_on="user_id", how="left")
    user_meta["gender"] = user_meta["gender"].fillna("?").apply(
        lambda g: g if g in ("男", "女") else "?"
    )

    def age_bucket(a):
        if pd.isna(a): return "?"
        a = int(a)
        if a < 18: return "<18"
        if a < 25: return "18-24"
        if a < 35: return "25-34"
        if a < 50: return "35-49"
        if a < 65: return "50-64"
        return "65+"

    user_meta["age_bucket"] = user_meta["age"].apply(age_bucket)
    print(f"  gender 分布:\n{user_meta['gender'].value_counts().to_string()}")
    print(f"  age_bucket:\n{user_meta['age_bucket'].value_counts().to_string()}")
    print()

    print("[2/4] 載入 LightGCN-Multi best 模型 ...")
    from src.models.lightgcn_multi import LightGCNMulti, build_multi_edges, build_norm_adj_weighted
    from src.models.lightgcn_si import build_side_info_tensors

    books_df = pd.read_parquet(PROC / "books.parquet")
    reservations_df = pd.read_parquet(PROC / "reservations.parquet")

    # 用 LightGCN-Multi（最佳模型）
    model = LightGCNMulti(
        splits.n_users, splits.n_items,
        embed_dim=64, n_layers=3, use_side_info=True,
    ).to(DEVICE)
    sd = torch.load(CKPT / "lightgcn_multi_best.pt", map_location=DEVICE, weights_only=True)
    model.load_state_dict(sd)
    eu, ei, ew = build_multi_edges(splits, reservations_df, borrow_weight=1.0, reserve_weight=0.5)
    A_hat = build_norm_adj_weighted(eu, ei, ew, splits.n_users, splits.n_items, device=DEVICE)
    model.set_graph(A_hat)
    g, a, c, _ = build_side_info_tensors(splits, books_df, users_df)
    model.set_side_info(g.to(DEVICE), a.to(DEVICE), c.to(DEVICE))
    model.eval()

    print("[3/4] 按性別 / 年齡分組評估 ...")
    user_train_pos = build_user_pos(splits.train)
    user_test_pos = build_user_pos(splits.test)
    test_users = np.array(sorted(user_test_pos.keys()))

    # 對應 test_users 的 gender / age
    meta_idx = user_meta.set_index("u").loc[test_users]

    rows = []
    # 按 gender 分組
    for gender in ["男", "女", "?"]:
        u_in = test_users[(meta_idx["gender"].values == gender)]
        if len(u_in) < 50:
            continue
        m = evaluate_topk(model, u_in, user_train_pos, user_test_pos,
                          splits.n_items, device=DEVICE, is_torch=True)
        rows.append({
            "group_type": "gender", "group": gender, "n_users": len(u_in),
            **{k: v for k, v in m.items() if k.startswith(("recall", "ndcg", "hit", "mrr"))}
        })
        print(f"  gender={gender:3s} (n={len(u_in):5d}): R@10={m['recall@10']:.4f}  NDCG@10={m['ndcg@10']:.4f}")

    # 按年齡分組
    for ab in ["<18", "18-24", "25-34", "35-49", "50-64", "65+", "?"]:
        u_in = test_users[(meta_idx["age_bucket"].values == ab)]
        if len(u_in) < 50:
            continue
        m = evaluate_topk(model, u_in, user_train_pos, user_test_pos,
                          splits.n_items, device=DEVICE, is_torch=True)
        rows.append({
            "group_type": "age", "group": ab, "n_users": len(u_in),
            **{k: v for k, v in m.items() if k.startswith(("recall", "ndcg", "hit", "mrr"))}
        })
        print(f"  age={ab:6s} (n={len(u_in):5d}): R@10={m['recall@10']:.4f}  NDCG@10={m['ndcg@10']:.4f}")

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "fairness.csv", index=False)
    print(f"\n[saved] {RESULTS / 'fairness.csv'}")

    print("[4/4] 繪圖 ...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    # gender chart
    g_df = df[df["group_type"] == "gender"]
    axes[0].bar(g_df["group"], g_df["recall@10"], color=["#3b82f6", "#ec4899", "#9ca3af"])
    axes[0].set_title("LightGCN-Multi Recall@10 by 性別", fontsize=14)
    axes[0].set_ylabel("Recall@10")
    for i, (g, n, v) in enumerate(zip(g_df["group"], g_df["n_users"], g_df["recall@10"])):
        axes[0].text(i, v, f"{v:.4f}\n(n={n})", ha="center", va="bottom", fontsize=10)

    # age chart
    age_order = ["<18", "18-24", "25-34", "35-49", "50-64", "65+", "?"]
    a_df = df[df["group_type"] == "age"].set_index("group").reindex(age_order).dropna(subset=["recall@10"]).reset_index()
    axes[1].bar(a_df["group"], a_df["recall@10"], color=plt.cm.viridis(np.linspace(0.1, 0.9, len(a_df))))
    axes[1].set_title("LightGCN-Multi Recall@10 by 年齡", fontsize=14)
    axes[1].set_ylabel("Recall@10")
    for i, (ab, n, v) in enumerate(zip(a_df["group"], a_df["n_users"], a_df["recall@10"])):
        axes[1].text(i, v, f"{v:.4f}\n(n={int(n)})", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    plt.savefig(FIG / "fig13_fairness.png", dpi=150, bbox_inches="tight")
    print(f"[saved] {FIG / 'fig13_fairness.png'}")

    # 寫 markdown
    out = ["# 公平性分析", "",
           "**目的**：檢查 LightGCN-Multi 對不同性別 / 年齡群體的推薦品質是否一致",
           "（避免模型不公平地服務某一族群）", ""]
    out.append("## 按性別")
    out.append("")
    out.append("| 群組 | n_users | Recall@10 | NDCG@10 | Hit@10 |")
    out.append("|---|---|---|---|---|")
    for _, r in df[df["group_type"] == "gender"].iterrows():
        out.append(f"| {r['group']} | {int(r['n_users'])} | {r['recall@10']:.4f} | {r['ndcg@10']:.4f} | {r['hit@10']:.4f} |")
    out.append("")
    out.append("## 按年齡")
    out.append("")
    out.append("| 群組 | n_users | Recall@10 | NDCG@10 | Hit@10 |")
    out.append("|---|---|---|---|---|")
    for _, r in df[df["group_type"] == "age"].iterrows():
        out.append(f"| {r['group']} | {int(r['n_users'])} | {r['recall@10']:.4f} | {r['ndcg@10']:.4f} | {r['hit@10']:.4f} |")
    out.append("")
    g_df = df[df["group_type"] == "gender"]
    if len(g_df) >= 2:
        max_r = g_df["recall@10"].max()
        min_r = g_df["recall@10"].min()
        out.append(f"## 結論")
        out.append("")
        out.append(f"性別群組間最大差距：**{(max_r - min_r):.4f}** (相對 {(max_r-min_r)/min_r*100:.1f}%)")

    (RESULTS / "fairness.md").write_text("\n".join(out), encoding="utf-8")
    print(f"[saved] {RESULTS / 'fairness.md'}")


if __name__ == "__main__":
    main()
