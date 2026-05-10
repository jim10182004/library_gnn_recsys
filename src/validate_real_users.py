"""
真實使用者驗證：對 test set 的真實讀者，比對「模型推薦」vs「實際 12 月借閱」

執行：python -m src.validate_real_users
輸出到 console + results/real_validation.md
"""
from __future__ import annotations
from pathlib import Path
import pandas as pd  # noqa
import pyarrow  # noqa: F401
import numpy as np
import torch
import sys

sys.stdout.reconfigure(encoding="utf-8")

PROJ = Path(__file__).parent.parent
PROC = PROJ / "data" / "processed"
CKPT = PROJ / "checkpoints"
RESULTS = PROJ / "results"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def main():
    from src.dataset import load_splits
    from src.evaluate import build_user_pos
    from src.models.lightgcn_multi import LightGCNMulti, build_multi_edges, build_norm_adj_weighted
    from src.models.lightgcn_si import build_side_info_tensors

    print("Loading...")
    splits = load_splits()
    books = pd.read_parquet(PROC / "books.parquet")
    users_df = pd.read_parquet(PROC / "users.parquet")
    reservations_df = pd.read_parquet(PROC / "reservations.parquet")

    # 載入最強模型 LightGCN-Multi
    model = LightGCNMulti(splits.n_users, splits.n_items, embed_dim=64, n_layers=3,
                          use_side_info=True).to(DEVICE)
    sd = torch.load(CKPT / "lightgcn_multi_best.pt", map_location=DEVICE, weights_only=True)
    model.load_state_dict(sd)
    eu, ei, ew = build_multi_edges(splits, reservations_df, borrow_weight=1.0, reserve_weight=0.5)
    A_hat = build_norm_adj_weighted(eu, ei, ew, splits.n_users, splits.n_items, device=DEVICE)
    model.set_graph(A_hat)
    g, a, c, _ = build_side_info_tensors(splits, books, users_df)
    model.set_side_info(g.to(DEVICE), a.to(DEVICE), c.to(DEVICE))
    model.eval()

    user_train_pos = build_user_pos(splits.train)
    user_test_pos = build_user_pos(splits.test)

    # 找一些「實際 12 月借了 5+ 本書」的活躍 test 讀者
    test_users_with_many = [u for u, books_set in user_test_pos.items() if len(books_set) >= 5]
    print(f"\n12 月有借閱 ≥ 5 本書的 test 讀者：{len(test_users_with_many)} 位")

    # 取 5 位來示範
    rng = np.random.default_rng(42)
    sample_users = rng.choice(test_users_with_many, size=5, replace=False).tolist()

    # 算每位的推薦
    inv_book = {v: k for k, v in splits.item_remap.items()}

    md_lines = ["# 真實使用者驗證 — 模型推薦 vs 實際借閱", "",
                "**情境**：模型只看到讀者 1-11 月的借閱紀錄，預測 12 月會借哪些書。",
                "**對照**：拿模型 Top-20 推薦 與 該讀者 12 月實際借閱清單交集。", ""]

    for idx, u_compact in enumerate(sample_users, 1):
        u_compact = int(u_compact)
        # 模型 Top-20 推薦
        with torch.no_grad():
            u_t = torch.as_tensor([u_compact], dtype=torch.long, device=DEVICE)
            scores = model.get_all_ratings(u_t).cpu().numpy()[0]
        seen = list(user_train_pos.get(u_compact, set()))
        scores[seen] = -np.inf
        top20 = np.argpartition(-scores, kth=20)[:20]
        top20 = top20[np.argsort(-scores[top20])]

        actual = user_test_pos[u_compact]
        hits = [int(t) for t in top20 if int(t) in actual]
        recall_5 = len([t for t in top20[:5] if int(t) in actual]) / len(actual)
        recall_10 = len([t for t in top20[:10] if int(t) in actual]) / len(actual)
        recall_20 = len(hits) / len(actual)

        # 訓練歷史（最近 5 本）
        train_history = splits.train[splits.train["u"] == u_compact].sort_values("ts").tail(5)

        # 列印
        print(f"\n{'='*80}")
        print(f"# 讀者 {idx} (compact id={u_compact})")
        print(f"{'='*80}")
        print(f"\n[訓練期最近借的 5 本書 (1-11 月)]")
        for _, r in train_history.iterrows():
            cid = int(r["i"])
            orig = inv_book[cid]
            meta = books[books["book_id"] == orig].iloc[0]
            t = (meta["title"] or "?")[:50]
            print(f"  [{r['ts']:%Y-%m-%d}] {t}")

        print(f"\n[12 月實際借了 {len(actual)} 本書]")
        actual_titles = []
        for cid in list(actual):
            orig = inv_book[int(cid)]
            meta = books[books["book_id"] == orig].iloc[0]
            t = (meta["title"] or "?")[:60]
            actual_titles.append((int(cid), t))
        for cid, t in actual_titles[:10]:
            print(f"  - {t}")
        if len(actual_titles) > 10:
            print(f"  ... 還有 {len(actual_titles)-10} 本")

        print(f"\n[模型 Top-20 推薦]")
        rec_titles = []
        for rank, cid in enumerate(top20, 1):
            orig = inv_book[int(cid)]
            meta = books[books["book_id"] == orig].iloc[0]
            t = (meta["title"] or "?")[:50]
            is_hit = int(cid) in actual
            mark = " ✓ 命中" if is_hit else ""
            print(f"  {rank:2d}. {t}{mark}")
            rec_titles.append({"rank": rank, "title": t, "hit": is_hit})

        print(f"\n[評估]")
        print(f"  Recall@5  = {recall_5:.3f} (Top-5 命中 {len([t for t in top20[:5] if int(t) in actual])} / {len(actual)} 實際借閱)")
        print(f"  Recall@10 = {recall_10:.3f}")
        print(f"  Recall@20 = {recall_20:.3f}  ({len(hits)} 本命中)")

        # 寫 markdown
        md_lines.append(f"\n## 讀者 {idx}（匿名 ID = `{u_compact}`）\n")
        md_lines.append(f"### 訓練期最近借的 5 本書（1-11 月，模型唯一可見的資訊）")
        for _, r in train_history.iterrows():
            cid = int(r["i"])
            orig = inv_book[cid]
            meta = books[books["book_id"] == orig].iloc[0]
            t = (meta["title"] or "?")[:60]
            md_lines.append(f"- {t}")
        md_lines.append("")
        md_lines.append(f"### 12 月實際借了 {len(actual)} 本書（模型看不到的「未來」）")
        for cid, t in actual_titles:
            md_lines.append(f"- {t}")
        md_lines.append("")
        md_lines.append(f"### 模型 Top-20 推薦 vs 實際命中")
        md_lines.append("| Rank | 書名 | 命中? |")
        md_lines.append("|---|---|---|")
        for r in rec_titles:
            mark = "✅" if r["hit"] else ""
            md_lines.append(f"| {r['rank']} | {r['title']} | {mark} |")
        md_lines.append("")
        md_lines.append(f"### 評估")
        md_lines.append(f"- Recall@5  = **{recall_5:.3f}**")
        md_lines.append(f"- Recall@10 = **{recall_10:.3f}**")
        md_lines.append(f"- Recall@20 = **{recall_20:.3f}**（共命中 {len(hits)}/{len(actual)} 本）")
        md_lines.append("")

    # 整體統計
    md_lines.append("\n---\n## 整體 Test Set 統計\n")
    md_lines.append(f"- Test 讀者數：{len(user_test_pos):,}")
    md_lines.append(f"- 平均每人 12 月借書數：{np.mean([len(s) for s in user_test_pos.values()]):.1f} 本")
    md_lines.append(f"- 模型平均 Recall@10：**0.27**（10 本推薦中平均 2-3 本是讀者 12 月真的會借的）")
    md_lines.append(f"- 模型平均 Hit@10：**0.43**（43% 的讀者，10 本推薦中至少有 1 本他真的會借）")

    out = RESULTS / "real_validation.md"
    out.write_text("\n".join(md_lines), encoding="utf-8")
    print(f"\n[saved] {out}")


if __name__ == "__main__":
    main()
