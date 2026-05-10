"""
產生 HF Spaces 部署包：把模型權重 + books metadata 預先計算成可獨立執行的 bundle。

執行：python deploy/hf_spaces/build_bundle.py

輸出：
  deploy/hf_spaces/assets/
    item_embs.pt      # 預計算 item embeddings (n_items × 64)
    books_meta.parquet # 書本 metadata（無讀者資訊）
    item_remap.json   # original book_id <-> compact_id mapping
    README.md         # 資料說明
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

    print("\n=== 計算 item embeddings (propagate) ===")
    with torch.no_grad():
        _, item_embs = model.propagate()
    item_embs = item_embs.cpu()
    print(f"item_embs shape: {item_embs.shape}, dtype: {item_embs.dtype}")
    print(f"記憶體大小: {item_embs.numel() * 4 / 1024 / 1024:.2f} MB")

    # 1. 存 item embeddings
    out_emb = ASSETS / "item_embs.pt"
    torch.save(item_embs, out_emb)
    print(f"  saved: {out_emb} ({out_emb.stat().st_size / 1024 / 1024:.2f} MB)")

    # 2. 過濾 books — 只留有出現在 splits 的書（節省空間 + 確保 demo 完整可用）
    valid_book_ids = set(splits.item_remap.keys())
    books_in_split = books[books["book_id"].isin(valid_book_ids)].copy()
    print(f"\nbooks 過濾：原始 {len(books)} → 在 splits 中 {len(books_in_split)}")

    # 確保只保留必要欄位（最小化資料）
    books_min = books_in_split[["book_id", "title", "author", "pub_year",
                                "isbn_clean", "category"]].copy()
    out_books = ASSETS / "books_meta.parquet"
    books_min.to_parquet(out_books, index=False)
    print(f"  saved: {out_books} ({out_books.stat().st_size / 1024 / 1024:.2f} MB)")

    # 3. 存 item_remap (original book_id → compact_id)
    out_remap = ASSETS / "item_remap.json"
    # JSON 不能存 numpy int，轉成 dict[int, int]
    remap_serializable = {int(k): int(v) for k, v in splits.item_remap.items()}
    out_remap.write_text(json.dumps(remap_serializable), encoding="utf-8")
    print(f"  saved: {out_remap} ({out_remap.stat().st_size / 1024:.1f} KB)")

    # 4. 寫一份 metadata
    meta = {
        "n_users": int(splits.n_users),
        "n_items": int(splits.n_items),
        "embed_dim": 64,
        "n_layers": 3,
        "model": "LightGCN",
        "model_checkpoint": "lightgcn_best.pt",
        "n_books_in_bundle": int(len(books_min)),
        "build_date": "2026-05-11",
    }
    out_meta = ASSETS / "metadata.json"
    out_meta.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"  saved: {out_meta}")

    # 5. README 說明這些檔案
    readme_text = """# Demo Bundle

本資料夾為 HuggingFace Spaces 部署用的最小資料包，**不包含任何讀者資訊**。

## 檔案說明

| 檔案 | 大小 | 內容 |
|---|---|---|
| `item_embs.pt` | ~8 MB | 預計算的書本 64 維 embedding（LightGCN propagate 結果）|
| `books_meta.parquet` | ~9 MB | 書本 metadata：title, author, category, ISBN, pub_year |
| `item_remap.json` | ~500 KB | 原始 book_id → compact_id 對應表 |
| `metadata.json` | < 1 KB | bundle 版本與模型資訊 |

## 隱私聲明

- ❌ **無**讀者 ID
- ❌ **無**借閱事件 (borrows)
- ❌ **無**預約事件 (reservations)
- ❌ **無**讀者 demographics (gender / age)
- ✅ **僅**書本公開 metadata（書名/作者等公開資訊）+ embedding 向量

embedding 向量是模型訓練後的「書本身份卡」（64 個浮點數），無法回推任何讀者借閱歷史。

## 重新生成

```bash
python deploy/hf_spaces/build_bundle.py
```
"""
    out_readme = ASSETS / "README.md"
    out_readme.write_text(readme_text, encoding="utf-8")
    print(f"  saved: {out_readme}")

    # 總大小
    total = sum(p.stat().st_size for p in ASSETS.iterdir() if p.is_file())
    print(f"\n=== Bundle 總大小：{total / 1024 / 1024:.2f} MB ===")


if __name__ == "__main__":
    main()
