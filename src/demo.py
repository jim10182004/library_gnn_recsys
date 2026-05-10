"""
Demo: 輸入讀者 ID（緊湊 id 或原始 user_orig），輸出 LightGCN 推薦的 Top-K 書單。

用法：
    python src/demo.py                          # 隨機挑 5 位讀者展示
    python src/demo.py --user 100               # 指定緊湊 id
    python src/demo.py --orig 1007827           # 指定原始讀者 ID
    python src/demo.py --user 100 --k 20        # 取 Top-20
    python src/demo.py --compare                # 同時顯示 LightGCN vs BPR-MF
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

# Windows console UTF-8
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# 確保可以從專案根 import src.*
_PROJ_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJ_ROOT))

# Windows: 必須先 import pandas/pyarrow，再 import torch
import pandas as pd
import pyarrow  # noqa: F401
import numpy as np
import torch

from src.dataset import load_splits
from src.models.lightgcn import LightGCN, build_norm_adj
from src.models.baselines import BPRMF

PROJECT = Path(__file__).parent.parent
PROCESSED = PROJECT / "data" / "processed"
CKPT = PROJECT / "checkpoints"


def load_books() -> pd.DataFrame:
    return pd.read_parquet(PROCESSED / "books.parquet")


def load_lightgcn(splits, n_layers: int = 3, embed_dim: int = 64, device: str = "cuda") -> LightGCN:
    model = LightGCN(splits.n_users, splits.n_items, embed_dim, n_layers).to(device)
    state = torch.load(CKPT / "lightgcn_best.pt", map_location=device, weights_only=True)
    model.load_state_dict(state)
    # 重建鄰接矩陣
    train_u = torch.as_tensor(splits.train["u"].values, dtype=torch.long)
    train_i = torch.as_tensor(splits.train["i"].values, dtype=torch.long)
    A_hat = build_norm_adj(train_u, train_i, splits.n_users, splits.n_items, device=device)
    model.set_graph(A_hat)
    model.eval()
    return model


def load_bprmf(splits, embed_dim: int = 64, device: str = "cuda") -> BPRMF:
    model = BPRMF(splits.n_users, splits.n_items, embed_dim).to(device)
    state = torch.load(CKPT / "bprmf_best.pt", map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model


def get_user_history(splits, u: int, books: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    """讀者的借閱歷史（最近 n 本，用 train 部分）。"""
    train = splits.train
    hist = train[train["u"] == u].copy()
    if hist.empty:
        return pd.DataFrame()
    hist = hist.sort_values("ts", ascending=False).head(n)
    # 對應緊湊 i 回原書資訊
    book_map_inv = {v: k for k, v in splits.item_remap.items()}
    hist["book_id_orig"] = hist["i"].map(book_map_inv)
    out = hist.merge(
        books[["book_id", "title", "author", "category"]],
        left_on="book_id_orig",
        right_on="book_id",
        how="left",
    )
    return out[["ts", "title", "author", "category"]]


def recommend(model, user_id_compact: int, splits, k: int = 10, device: str = "cuda") -> np.ndarray:
    """傳回 Top-K 緊湊 item id。"""
    with torch.no_grad():
        u_t = torch.as_tensor([user_id_compact], dtype=torch.long, device=device)
        scores = model.get_all_ratings(u_t).cpu().numpy()[0]
    # mask 已看過的
    seen = splits.train[splits.train["u"] == user_id_compact]["i"].values
    scores[seen] = -np.inf
    top_idx = np.argpartition(-scores, kth=k)[:k]
    top_idx = top_idx[np.argsort(-scores[top_idx])]
    return top_idx


def show_recommendations(top_idx, splits, books, model_name="LightGCN", scores=None):
    """美化輸出 Top-K。"""
    book_map_inv = {v: k for k, v in splits.item_remap.items()}
    rows = []
    for rank, i in enumerate(top_idx, 1):
        orig = book_map_inv[int(i)]
        meta = books[books["book_id"] == orig].iloc[0]
        title = meta["title"] or "(無題名)"
        author = (meta["author"] or "(無作者)")[:30]
        cat = meta["category"] or "-"
        rows.append({
            "Rank": rank,
            "Title": title[:50],
            "Author": author,
            "Cat": cat,
        })
    df = pd.DataFrame(rows)
    print(f"\n=== {model_name} TOP-{len(top_idx)} 推薦 ===")
    print(df.to_string(index=False))


def show_history(hist: pd.DataFrame):
    if hist.empty:
        print("  （此讀者在 train 中沒有借閱紀錄）")
        return
    print("\n--- 借閱歷史（最近）---")
    for _, r in hist.iterrows():
        ts = r["ts"].strftime("%Y-%m-%d") if pd.notna(r["ts"]) else "?"
        title = (r["title"] or "?")[:45]
        author = (r["author"] or "?")[:25]
        print(f"  [{ts}] {title} / {author} / {r['category']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--user", type=int, default=None, help="緊湊 user id (0..n-1)")
    ap.add_argument("--orig", type=int, default=None, help="原始 user_orig")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--compare", action="store_true", help="同時跑 BPR-MF 對照")
    ap.add_argument("--n", type=int, default=5, help="隨機展示時的人數")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    print("Loading splits & books ...")
    splits = load_splits()
    books = load_books()

    print("Loading LightGCN ...")
    lgcn = load_lightgcn(splits, device=args.device)
    bpr = None
    if args.compare:
        print("Loading BPR-MF ...")
        bpr = load_bprmf(splits, device=args.device)

    # 決定要展示的 user 列表
    if args.user is not None:
        users = [args.user]
    elif args.orig is not None:
        if args.orig not in splits.user_remap:
            print(f"找不到原始讀者 ID {args.orig}（可能未通過 k-core 過濾）")
            sys.exit(1)
        users = [splits.user_remap[args.orig]]
    else:
        # 隨機挑 n 個有歷史的 user
        train_users = splits.train["u"].unique()
        rng = np.random.default_rng(42)
        users = rng.choice(train_users, size=args.n, replace=False).tolist()

    user_orig_inv = {v: k for k, v in splits.user_remap.items()}

    for u in users:
        u = int(u)
        orig = user_orig_inv.get(u, "?")
        print("\n" + "=" * 80)
        print(f"[讀者] 緊湊id={u}  原始id={orig}")
        hist = get_user_history(splits, u, books, n=5)
        show_history(hist)

        top = recommend(lgcn, u, splits, k=args.k, device=args.device)
        show_recommendations(top, splits, books, model_name="LightGCN")

        if bpr is not None:
            top_b = recommend(bpr, u, splits, k=args.k, device=args.device)
            show_recommendations(top_b, splits, books, model_name="BPR-MF")


if __name__ == "__main__":
    main()
