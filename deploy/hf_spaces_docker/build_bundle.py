"""
產生 HF Spaces Docker 部署 bundle：把模型權重 + books metadata
預先計算成可獨立執行的 bundle。

執行：python deploy/hf_spaces_docker/build_bundle.py

輸出：
  deploy/hf_spaces_docker/assets/
    item_embs.pt      # 預計算 item embeddings (n_items × 64)
    books_meta.parquet # 書本 metadata（無讀者資訊）
    item_remap.json   # original book_id <-> compact_id mapping
    metadata.json
"""
from __future__ import annotations
import sys
import json
from pathlib import Path

import pandas as pd  # noqa
import pyarrow  # noqa: F401
import torch

PROJECT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from src.dataset import load_splits
from src.models.lightgcn import LightGCN, build_norm_adj

ASSETS = Path(__file__).parent / "assets"
ASSETS.mkdir(parents=True, exist_ok=True)
PROC = PROJECT / "data" / "processed"
CKPT = PROJECT / "checkpoints"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def main():
    print("=== 載入 splits + 模型 ===")
    splits = load_splits()
    books = pd.read_parquet(PROC / "books.parquet")
    print(f"n_users={splits.n_users}, n_items={splits.n_items}, n_books_meta={len(books)}")

    print("\n=== 載入 LightGCN best checkpoint ===")
    model = LightGCN(splits.n_users, splits.n_items, embed_dim=64, n_layers=3).to(DEVICE)
    sd = torch.load(CKPT / "lightgcn_best.pt", map_location=DEVICE, weights_only=True)
    model.load_state_dict(sd)
    train_u = torch.as_tensor(splits.train["u"].values, dtype=torch.long)
    train_i = torch.as_tensor(splits.train["i"].values, dtype=torch.long)
    A = build_norm_adj(train_u, train_i, splits.n_users, splits.n_items, device=DEVICE)
    model.set_graph(A)
    model.eval()

    print("\n=== 計算 item embeddings ===")
    with torch.no_grad():
        _, item_embs = model.propagate()
    item_embs = item_embs.cpu()
    print(f"item_embs shape: {item_embs.shape}")
    print(f"記憶體大小: {item_embs.numel() * 4 / 1024 / 1024:.2f} MB")

    out_emb = ASSETS / "item_embs.pt"
    torch.save(item_embs, out_emb)
    print(f"  saved: {out_emb}")

    valid_book_ids = set(splits.item_remap.keys())
    books_in_split = books[books["book_id"].isin(valid_book_ids)].copy()
    books_min = books_in_split[["book_id", "title", "author", "pub_year",
                                "isbn_clean", "category"]].copy()
    out_books = ASSETS / "books_meta.parquet"
    books_min.to_parquet(out_books, index=False)
    print(f"  saved: {out_books}")

    out_remap = ASSETS / "item_remap.json"
    remap_serializable = {int(k): int(v) for k, v in splits.item_remap.items()}
    out_remap.write_text(json.dumps(remap_serializable), encoding="utf-8")
    print(f"  saved: {out_remap}")

    meta = {
        "n_users": int(splits.n_users),
        "n_items": int(splits.n_items),
        "embed_dim": 64,
        "n_layers": 3,
        "model": "LightGCN",
        "model_checkpoint": "lightgcn_best.pt",
        "n_books_in_bundle": int(len(books_min)),
    }
    out_meta = ASSETS / "metadata.json"
    out_meta.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"  saved: {out_meta}")

    total = sum(p.stat().st_size for p in ASSETS.iterdir() if p.is_file())
    print(f"\n=== Bundle 總大小：{total / 1024 / 1024:.2f} MB ===")


if __name__ == "__main__":
    main()
