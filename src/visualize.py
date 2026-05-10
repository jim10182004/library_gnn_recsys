"""
產生 t-SNE 視覺化圖、訓練曲線、模型比較表。
輸出：results/figures/*.png 和 results/summary.csv
"""
from __future__ import annotations
import sys
from pathlib import Path

# Windows console UTF-8
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# 先 pandas/pyarrow 後 torch
import pandas as pd
import pyarrow  # noqa: F401
import numpy as np
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

import torch
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

_PROJ_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJ_ROOT))

from src.dataset import load_splits
from src.metrics_summary import write_clean_summary
from src.models.lightgcn import LightGCN, build_norm_adj
from src.models.lightgcn_si import LightGCNSI, build_side_info_tensors


PROJECT = Path(__file__).parent.parent
PROCESSED = PROJECT / "data" / "processed"
RESULTS = PROJECT / "results"
FIG = RESULTS / "figures"
CKPT = PROJECT / "checkpoints"
FIG.mkdir(parents=True, exist_ok=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 中文字型
plt.rcParams["font.sans-serif"] = ["Microsoft JhengHei", "SimHei", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False
sns.set_theme(style="whitegrid", font="Microsoft JhengHei")


def load_lightgcn(splits, kind: str = "lightgcn"):
    """kind: 'lightgcn' or 'lightgcn_si'"""
    if kind == "lightgcn":
        m = LightGCN(splits.n_users, splits.n_items, embed_dim=64, n_layers=3).to(DEVICE)
        ckpt = CKPT / "lightgcn_best.pt"
    else:
        m = LightGCNSI(
            splits.n_users, splits.n_items,
            n_genders=3, n_age_buckets=8, n_categories=11,
            embed_dim=64, n_layers=3,
        ).to(DEVICE)
        ckpt = CKPT / "lightgcn_si_best.pt"
    state = torch.load(ckpt, map_location=DEVICE, weights_only=True)
    m.load_state_dict(state)

    train_u = torch.as_tensor(splits.train["u"].values, dtype=torch.long)
    train_i = torch.as_tensor(splits.train["i"].values, dtype=torch.long)
    A_hat = build_norm_adj(train_u, train_i, splits.n_users, splits.n_items, device=DEVICE)
    m.set_graph(A_hat)

    if kind == "lightgcn_si":
        books = pd.read_parquet(PROCESSED / "books.parquet")
        users = pd.read_parquet(PROCESSED / "users.parquet")
        g, a, c, _ = build_side_info_tensors(splits, books, users)
        m.set_side_info(g.to(DEVICE), a.to(DEVICE), c.to(DEVICE))

    m.eval()
    with torch.no_grad():
        u_emb, i_emb = m.propagate()
    return u_emb.cpu().numpy(), i_emb.cpu().numpy()


def make_tsne(X, n_sample=5000, seed=42):
    rng = np.random.default_rng(seed)
    idx = rng.choice(X.shape[0], size=min(n_sample, X.shape[0]), replace=False)
    Xs = X[idx]
    pca = PCA(n_components=min(50, Xs.shape[1])).fit_transform(Xs)
    return TSNE(n_components=2, perplexity=30, init="pca", random_state=seed, max_iter=750).fit_transform(pca), idx


def plot_user_tsne(splits, u_emb, kind="lightgcn"):
    print(f"  t-SNE on user embeddings ({kind}) ...", flush=True)
    u_2d, u_idx = make_tsne(u_emb, n_sample=5000)

    users_df = pd.read_parquet(PROCESSED / "users.parquet")
    # splits.user_remap 是 {old_compact_id -> new_compact_id}
    # users.parquet 的 user_id 也是 old_compact_id（不是 user_orig 的 raw 圖書館 ID）
    inv_user = {v: k for k, v in splits.user_remap.items()}
    user_meta = pd.DataFrame({"u": range(splits.n_users)})
    user_meta["old_compact"] = user_meta["u"].map(inv_user)
    user_meta = user_meta.merge(users_df[["user_id", "gender", "age"]],
                                left_on="old_compact", right_on="user_id", how="left")

    def age_bucket(a):
        if pd.isna(a): return "?"
        if a < 18: return "<18"
        if a < 25: return "18-24"
        if a < 35: return "25-34"
        if a < 50: return "35-49"
        if a < 65: return "50-64"
        return "65+"
    user_meta["age_bucket"] = user_meta["age"].apply(age_bucket)

    sub = user_meta.iloc[u_idx].reset_index(drop=True)
    sub["x"], sub["y"] = u_2d[:, 0], u_2d[:, 1]
    sub["gender"] = sub["gender"].fillna("?").replace({"": "?"})
    # 把不認識的 gender 都歸成 "其他"
    sub.loc[~sub["gender"].isin(["男", "女"]), "gender"] = "其他"

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    sns.scatterplot(data=sub, x="x", y="y", hue="gender",
                    palette={"男": "steelblue", "女": "palevioletred", "其他": "lightgray"},
                    s=8, alpha=0.6, ax=axes[0])
    axes[0].set_title(f"{kind} 讀者嵌入空間 (依性別)", fontsize=14)
    axes[0].set_xticks([]); axes[0].set_yticks([])

    age_order = ["<18", "18-24", "25-34", "35-49", "50-64", "65+", "?"]
    sns.scatterplot(data=sub, x="x", y="y", hue="age_bucket", hue_order=age_order,
                    palette="viridis", s=8, alpha=0.6, ax=axes[1])
    axes[1].set_title(f"{kind} 讀者嵌入空間 (依年齡)", fontsize=14)
    axes[1].set_xticks([]); axes[1].set_yticks([])
    axes[1].legend(title="年齡", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    out = FIG / f"fig05_user_tsne_{kind}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    saved: {out}")


def plot_item_tsne(splits, i_emb, kind="lightgcn"):
    print(f"  t-SNE on item embeddings ({kind}) ...", flush=True)
    i_2d, i_idx = make_tsne(i_emb, n_sample=5000)

    books_df = pd.read_parquet(PROCESSED / "books.parquet")
    inv_item = {v: k for k, v in splits.item_remap.items()}
    item_meta = pd.DataFrame({"i": range(splits.n_items)})
    item_meta["orig"] = item_meta["i"].map(inv_item)
    item_meta = item_meta.merge(books_df[["book_id", "category"]],
                                left_on="orig", right_on="book_id", how="left")
    cat_label = {
        "0": "0 總類", "1": "1 哲學", "2": "2 宗教", "3": "3 科學", "4": "4 應用科學",
        "5": "5 社會科學", "6": "6 史地中國", "7": "7 史地世界",
        "8": "8 語文文學", "9": "9 藝術",
    }
    def cat_top(c):
        if pd.isna(c): return "? 未知"
        s = str(c).strip()
        if s and s[0].isdigit():
            return cat_label.get(s[0], "? 未知")
        return "? 未知"
    item_meta["cat_label"] = item_meta["category"].apply(cat_top)

    sub = item_meta.iloc[i_idx].reset_index(drop=True)
    sub["x"], sub["y"] = i_2d[:, 0], i_2d[:, 1]

    plt.figure(figsize=(13, 9))
    cat_order = sorted([c for c in sub["cat_label"].unique() if c != "? 未知"]) + ["? 未知"]
    sns.scatterplot(data=sub, x="x", y="y", hue="cat_label", hue_order=cat_order,
                    palette="tab10", s=12, alpha=0.7)
    plt.title(f"{kind} 書籍嵌入空間 (依中圖法大類)", fontsize=14)
    plt.xticks([]); plt.yticks([])
    plt.legend(title="類別", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    out = FIG / f"fig06_item_tsne_{kind}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    saved: {out}")


def plot_training_curves():
    print("Plotting training curves ...", flush=True)
    histories = {}
    for name in ["bprmf", "lightgcn", "lightgcn_si", "lightgcn_multi",
                 "ngcf", "lightgcn_bert", "lightgcn_hetero", "lightgcn_timedecay",
                 "simgcl", "sasrec",
                 "lightgcn_opt", "lightgcn_multi_opt",
                 "lightgcn_tgn", "lightgcn_cover"]:
        p = RESULTS / f"{name}_history.json"
        if p.exists():
            with open(p, encoding="utf-8") as f:
                histories[name] = json.load(f)

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    colors = {"bprmf": "tab:orange", "lightgcn": "tab:blue", "lightgcn_si": "tab:green",
              "lightgcn_multi": "tab:red", "ngcf": "tab:purple", "lightgcn_bert": "tab:cyan",
              "lightgcn_hetero": "tab:brown", "lightgcn_timedecay": "tab:olive",
              "simgcl": "tab:pink", "sasrec": "tab:gray",
              "lightgcn_opt": "navy", "lightgcn_multi_opt": "darkred",
              "lightgcn_tgn": "darkgreen", "lightgcn_cover": "goldenrod"}
    for name, h in histories.items():
        if "history" not in h: continue
        ep = [r["epoch"] for r in h["history"]]
        ls = [r["train_loss"] for r in h["history"]]
        axes[0].plot(ep, ls, label=name, linewidth=2, color=colors.get(name))
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Train Loss")
    axes[0].set_title("Training Loss 收斂")
    axes[0].legend(fontsize=8, loc="upper right")

    for name, h in histories.items():
        if "history" not in h: continue
        ep = [r["epoch"] for r in h["history"] if "val" in r]
        rec20 = [r["val"]["recall@20"] for r in h["history"] if "val" in r]
        axes[1].plot(ep, rec20, marker="o", markersize=3, label=name, linewidth=1.5, color=colors.get(name))
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Recall@20 (val)")
    axes[1].set_title("Validation Recall@20 變化")
    axes[1].legend(fontsize=8, bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    out = FIG / "fig07_training_curves.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"    saved: {out}")


def make_summary_table():
    print("Generating summary table ...", flush=True)
    rows = []
    for name in ["popular", "itemcf", "bprmf",
                 "lightgcn", "lightgcn_si", "lightgcn_multi",
                 "ngcf", "lightgcn_bert", "lightgcn_hetero",
                 "lightgcn_timedecay", "sasrec", "simgcl",
                 "lightgcn_opt", "lightgcn_multi_opt",
                 "lightgcn_tgn", "lightgcn_cover"]:
        p = RESULTS / f"{name}_history.json"
        if not p.exists(): continue
        with open(p, encoding="utf-8") as f:
            h = json.load(f)
        test = h.get("test", {})
        if not test: continue
        rows.append({"Model": name, **test})
    if not rows:
        print("  no results found")
        return
    df = pd.DataFrame(rows)
    cols = ["Model"] + sorted([c for c in df.columns if c != "Model"])
    df = df[cols]
    out_csv = RESULTS / "summary.csv"
    df.to_csv(out_csv, index=False)
    print(f"    saved: {out_csv}")
    clean_csv = write_clean_summary(out_csv)
    print(f"    saved: {clean_csv}")
    print()
    print(df.to_string(index=False, float_format="%.4f"))

    # 同時輸出一張長條圖比較
    metrics_to_plot = ["recall@10", "recall@20", "ndcg@10", "ndcg@20", "hit@10"]
    avail = [m for m in metrics_to_plot if m in df.columns]
    fig, ax = plt.subplots(figsize=(18, 6))
    x = np.arange(len(df))
    width = 0.15
    for i, m in enumerate(avail):
        ax.bar(x + i * width, df[m].values, width, label=m)
    ax.set_xticks(x + width * (len(avail) - 1) / 2)
    ax.set_xticklabels(df["Model"].values, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Score")
    ax.set_title("各模型 Test Set 表現比較")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    out_png = FIG / "fig08_model_comparison.png"
    plt.savefig(out_png, dpi=150)
    plt.close()
    print(f"    saved: {out_png}")

    # 額外輸出一張：只比較 GNN 家族（Recall@20 排序）
    df_gnn = df[df["Model"].str.startswith(("lightgcn", "ngcf", "simgcl"))].copy()
    if len(df_gnn) > 0 and "recall@20" in df_gnn.columns:
        df_gnn = df_gnn.sort_values("recall@20", ascending=True)
        fig, ax = plt.subplots(figsize=(10, max(5, len(df_gnn) * 0.4)))
        bars = ax.barh(df_gnn["Model"], df_gnn["recall@20"], color="steelblue", alpha=0.8)
        # 標數值
        for bar, val in zip(bars, df_gnn["recall@20"]):
            ax.text(val + 0.001, bar.get_y() + bar.get_height() / 2,
                    f"{val:.4f}", va="center", fontsize=9)
        ax.set_xlabel("Recall@20")
        ax.set_title("GNN 家族 Recall@20 排序（含 Optuna / TGN / Cover）")
        ax.set_xlim(min(df_gnn["recall@20"]) * 0.97, max(df_gnn["recall@20"]) * 1.02)
        ax.grid(axis="x", alpha=0.3)
        plt.tight_layout()
        out_png2 = FIG / "fig19_gnn_family_ranking.png"
        plt.savefig(out_png2, dpi=150)
        plt.close()
        print(f"    saved: {out_png2}")


def main():
    print("=== Loading splits ===", flush=True)
    splits = load_splits()
    print(f"users={splits.n_users}  items={splits.n_items}", flush=True)

    print("\n=== LightGCN embeddings ===", flush=True)
    u_lgcn, i_lgcn = load_lightgcn(splits, kind="lightgcn")
    plot_user_tsne(splits, u_lgcn, kind="lightgcn")
    plot_item_tsne(splits, i_lgcn, kind="lightgcn")

    if (CKPT / "lightgcn_si_best.pt").exists():
        print("\n=== LightGCN-SI embeddings ===", flush=True)
        u_si, i_si = load_lightgcn(splits, kind="lightgcn_si")
        plot_user_tsne(splits, u_si, kind="lightgcn_si")
        plot_item_tsne(splits, i_si, kind="lightgcn_si")

    print("\n=== Training curves ===", flush=True)
    plot_training_curves()

    print("\n=== Summary table ===", flush=True)
    make_summary_table()

    print("\n=== 完成 ===", flush=True)
    print(f"所有圖表存到：{FIG}")


if __name__ == "__main__":
    main()
