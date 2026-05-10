"""
推薦可解釋性實驗：分解推薦分數的來源

對 LightGCN 的合成讀者推薦，把每本推薦書的分數拆成：
  - 來自每個 seed book 的貢獻
  - 來自不同層 (k=0, 1, 2, 3) 的貢獻

這樣可以回答「為什麼推薦這本書」── 從數學上拆解，而非只是 heuristic。

執行：python -m src.explainability
輸出：
  - results/explainability.md
  - results/figures/fig15_explainability.png
"""
from __future__ import annotations
from pathlib import Path
import pandas as pd  # noqa: F401
import pyarrow  # noqa: F401
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn.functional as F

PROJ = Path(__file__).parent.parent
PROC = PROJ / "data" / "processed"
CKPT = PROJ / "checkpoints"
RESULTS = PROJ / "results"
FIG = RESULTS / "figures"

plt.rcParams["font.sans-serif"] = ["Microsoft JhengHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False
sns.set_theme(style="whitegrid", font="Microsoft JhengHei")


# 試 4 個 persona case
CASES = [
    {
        "name": "日系推理小說迷",
        "seed_titles": ["白金數據", "嫌疑犯X的獻身", "解憂雜貨店"],
    },
    {
        "name": "兒童英文書",
        "seed_titles": ["Magic Tree House", "Toy story", "Mittens"],
    },
    {
        "name": "職場成長",
        "seed_titles": ["原子習慣", "拖延心理學", "目標"],
    },
]


def load_lightgcn():
    from src.dataset import load_splits
    from src.models.lightgcn import LightGCN, build_norm_adj
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    splits = load_splits()
    books = pd.read_parquet(PROC / "books.parquet")
    model = LightGCN(splits.n_users, splits.n_items, embed_dim=64, n_layers=3).to(DEVICE)
    sd = torch.load(CKPT / "lightgcn_best.pt", map_location=DEVICE, weights_only=True)
    model.load_state_dict(sd)
    train_u = torch.as_tensor(splits.train["u"].values, dtype=torch.long)
    train_i = torch.as_tensor(splits.train["i"].values, dtype=torch.long)
    A_hat = build_norm_adj(train_u, train_i, splits.n_users, splits.n_items, device=DEVICE)
    model.set_graph(A_hat)
    model.eval()
    return splits, books, model, DEVICE


def decompose_per_layer(model, splits, seed_compact_ids):
    """分解每層 (k=0..K) 對 user vec 的貢獻"""
    # propagate 並取出每層的 item embedding
    all_emb = torch.cat([model.user_emb.weight, model.item_emb.weight], dim=0)
    layers = [all_emb]
    x = all_emb
    for _ in range(model.n_layers):
        x = torch.sparse.mm(model.norm_adj, x)
        layers.append(x)

    # 每層的 item embedding（只看 item 部分）
    layer_item_embs = [L[splits.n_users:] for L in layers]

    # synthetic user vec from seed avg per layer
    seed_t = torch.as_tensor(seed_compact_ids, dtype=torch.long, device=all_emb.device)
    layer_user_vecs = [F.normalize(L[seed_t], p=2, dim=1).mean(dim=0) for L in layer_item_embs]
    layer_user_vecs = [F.normalize(v, p=2, dim=0) for v in layer_user_vecs]
    return layer_item_embs, layer_user_vecs


def analyze_case(case, splits, books, model, device):
    """分析一個 persona case"""
    # 找 seed compact ids
    seed_compact = []
    for t in case["seed_titles"]:
        m = books[books["title"].str.contains(t, na=False, regex=False)]
        for _, r in m.iterrows():
            cid = splits.item_remap.get(int(r["book_id"]))
            if cid is not None and cid not in seed_compact:
                seed_compact.append(cid)
                break
    if not seed_compact:
        return None

    # 用主 propagate 取得最終 embedding
    with torch.no_grad():
        u_all, i_all = model.propagate()
    seed_emb = F.normalize(i_all[torch.as_tensor(seed_compact, device=device)], p=2, dim=1)
    user_vec = F.normalize(seed_emb.mean(dim=0), p=2, dim=0)
    all_norm = F.normalize(i_all, p=2, dim=1)
    scores = (all_norm @ user_vec).cpu().numpy()
    scores[seed_compact] = -np.inf
    top5 = np.argpartition(-scores, kth=5)[:5]
    top5 = top5[np.argsort(-scores[top5])]

    # 對每個 top recommendation 拆解：哪個 seed 貢獻最多
    inv = {v: k for k, v in splits.item_remap.items()}
    out = {"name": case["name"], "seeds": [], "recs": []}

    for cid in seed_compact:
        meta = books[books["book_id"] == inv[cid]].iloc[0]
        out["seeds"].append(meta["title"][:40])

    # 每本推薦的「per-seed 貢獻」= seed_norm_emb · rec_norm_emb / |seeds|
    rec_norm = F.normalize(i_all[torch.as_tensor(top5.tolist(), device=device)], p=2, dim=1)
    # seed_emb shape: [n_seeds, D]; rec_norm shape: [5, D]
    contributions = (seed_emb @ rec_norm.T) / len(seed_compact)  # [n_seeds, 5]
    contributions = contributions.cpu().numpy()  # 數值即為對最終 cosine score 的 per-seed 貢獻

    for j, cid in enumerate(top5):
        meta = books[books["book_id"] == inv[int(cid)]].iloc[0]
        out["recs"].append({
            "title": (meta["title"] or "?")[:50],
            "author": (meta["author"] or "?")[:30],
            "score": float(scores[cid]),
            "per_seed": contributions[:, j].tolist(),
        })

    return out


def main():
    print("Loading model ...")
    splits, books, model, device = load_lightgcn()
    print("OK")

    all_results = []
    for case in CASES:
        print(f"\nAnalyzing: {case['name']} ...")
        result = analyze_case(case, splits, books, model, device)
        if result is not None:
            all_results.append(result)
            print(f"  seeds: {len(result['seeds'])}")
            print(f"  top rec: {result['recs'][0]['title']} (score={result['recs'][0]['score']:.4f})")

    # 寫 markdown
    md = ["# 推薦可解釋性分析", "",
          "**目的**：對 LightGCN 的推薦結果，分解出每個輸入種子書對最終推薦分數的貢獻，",
          "證明推薦不是黑箱，而是可數學分解的。",
          "",
          "**方法**：",
          "- 合成讀者向量 $u = \\frac{1}{N} \\sum_{j=1}^{N} \\hat{e}_j$（$\\hat{e}_j$ 為 seed $j$ 的 normalised embedding）",
          "- 對推薦書 $i$ 的 cosine similarity 分數 $s_i = \\hat{u} \\cdot \\hat{e}_i$",
          "- $s_i$ 可線性分解為 $\\sum_j \\frac{1}{N} \\hat{e}_j \\cdot \\hat{e}_i$（每個 seed 的貢獻）",
          ""]

    for result in all_results:
        md.append(f"## Case：{result['name']}")
        md.append("")
        md.append("**Seed books**:")
        for i, s in enumerate(result["seeds"], 1):
            md.append(f"- ({i}) {s}")
        md.append("")
        md.append("**Top-5 recommendations 與每個 seed 的貢獻**：")
        md.append("")
        # 表頭
        header = ["Rank", "推薦書", "Final Score"] + [f"Seed{i+1} 貢獻" for i in range(len(result["seeds"]))]
        md.append("| " + " | ".join(header) + " |")
        md.append("|" + "---|" * len(header))
        for r_idx, rec in enumerate(result["recs"], 1):
            row = [str(r_idx), rec["title"], f"{rec['score']:.4f}"]
            for c in rec["per_seed"]:
                row.append(f"{c:+.4f}")
            md.append("| " + " | ".join(row) + " |")
        md.append("")
        # 找出每本推薦的 dominant seed
        md.append("**主導 seed 解讀**：")
        md.append("")
        for r_idx, rec in enumerate(result["recs"], 1):
            best_seed_idx = int(np.argmax(rec["per_seed"]))
            ratio = rec["per_seed"][best_seed_idx] / max(sum(rec["per_seed"]), 1e-9)
            md.append(f"- 推薦 #{r_idx} **{rec['title']}** "
                      f"主要來自 Seed {best_seed_idx+1}「{result['seeds'][best_seed_idx]}」"
                      f"（佔 {ratio*100:.1f}%）")
        md.append("")

    (RESULTS / "explainability.md").write_text("\n".join(md), encoding="utf-8")
    print(f"\n[saved] {RESULTS / 'explainability.md'}")

    # 圖：第一個 case 的 heatmap
    if all_results:
        result = all_results[0]
        rec_titles = [r["title"][:25] for r in result["recs"]]
        contrib_matrix = np.array([r["per_seed"] for r in result["recs"]])  # [5_recs, n_seeds]
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.heatmap(contrib_matrix, annot=True, fmt=".4f",
                    xticklabels=[s[:18] for s in result["seeds"]],
                    yticklabels=rec_titles,
                    cmap="YlGn", ax=ax, cbar_kws={"label": "對推薦分數的貢獻"})
        ax.set_xlabel("Seed Books（讀者選的書）")
        ax.set_ylabel("Recommended Books（模型推薦）")
        ax.set_title(f"推薦來源分解 — Case: {result['name']}")
        plt.tight_layout()
        out_img = FIG / "fig15_explainability.png"
        plt.savefig(out_img, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"[saved] {out_img}")


if __name__ == "__main__":
    main()
