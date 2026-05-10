"""
書封多模態 pipeline：下載書封圖 → CNN 編碼 → 存 parquet
（中期工作 PoC：完整下載 30K 本要 ~12 小時 API 呼叫，本範例做 1000 本子集驗證 pipeline）

執行：
    python -m src.encode_book_covers --max-books 1000   # PoC
    python -m src.encode_book_covers --max-books 999999  # 完整版（約 12 小時）

輸出：
    data/processed/book_covers.parquet  (book_id, vec_0..vec_511)
    data/processed/book_covers_meta.csv  (book_id, isbn, has_cover, source)
"""
from __future__ import annotations
import argparse
import time
from pathlib import Path
from io import BytesIO
import urllib.request
import urllib.error

import pandas as pd
import pyarrow  # noqa: F401
import numpy as np
import torch
from PIL import Image

PROJ = Path(__file__).parent.parent
PROC = PROJ / "data" / "processed"


def fetch_cover_bytes(isbn: str, timeout: int = 5) -> bytes | None:
    """嘗試從 Open Library 抓書封"""
    if not isbn:
        return None
    url = f"https://covers.openlibrary.org/b/isbn/{isbn}-M.jpg?default=false"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "library-gnn-recsys/1.0"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            data = r.read()
        if len(data) < 1000:  # 太小通常是空圖
            return None
        return data
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError):
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-books", type=int, default=1000,
                    help="處理多少本書（PoC 預設 1000；完整 30K 要 ~12 小時）")
    ap.add_argument("--rate-limit", type=float, default=0.2,
                    help="每次 API 呼叫之間的延遲（秒），避免被 ban")
    args = ap.parse_args()

    print("[1/4] 載入 books ...")
    books = pd.read_parquet(PROC / "books.parquet")
    # 只挑有 ISBN 的書
    has_isbn = books[books["isbn_clean"].notna() & (books["isbn_clean"] != "")]
    print(f"  總書數：{len(books):,}")
    print(f"  有 ISBN：{len(has_isbn):,}")

    if args.max_books < len(has_isbn):
        # 用前 N 本測試（按 book_id 排序，可重現）
        target = has_isbn.sort_values("book_id").head(args.max_books)
    else:
        target = has_isbn
    print(f"  本次處理：{len(target):,} 本")

    print(f"[2/4] 載入 ResNet-18 (預訓練)...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    import torchvision
    model = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
    model.fc = torch.nn.Identity()  # 取 512 維 pooled feature
    model = model.to(device).eval()

    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((224, 224)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225]),
    ])

    print(f"[3/4] 下載 + 編碼 (device={device}, rate_limit={args.rate_limit}s) ...")
    embeddings = []
    book_ids = []
    metadata = []
    n_success = 0
    n_fail = 0

    t0 = time.time()
    for idx, (_, row) in enumerate(target.iterrows(), 1):
        bid = int(row["book_id"])
        isbn = str(row["isbn_clean"])
        img_bytes = fetch_cover_bytes(isbn)
        time.sleep(args.rate_limit)

        if img_bytes is None:
            n_fail += 1
            metadata.append({"book_id": bid, "isbn": isbn, "has_cover": False, "source": "none"})
            continue
        try:
            img = Image.open(BytesIO(img_bytes)).convert("RGB")
            tensor = transform(img).unsqueeze(0).to(device)
            with torch.no_grad():
                vec = model(tensor)[0].cpu().numpy().astype(np.float32)
            embeddings.append(vec)
            book_ids.append(bid)
            metadata.append({"book_id": bid, "isbn": isbn, "has_cover": True, "source": "openlibrary"})
            n_success += 1
        except Exception:
            n_fail += 1
            metadata.append({"book_id": bid, "isbn": isbn, "has_cover": False, "source": "decode_fail"})

        if idx % 50 == 0:
            elapsed = time.time() - t0
            rate = idx / elapsed
            eta = (len(target) - idx) / rate
            print(f"  [{idx}/{len(target)}] success={n_success} fail={n_fail}  "
                  f"rate={rate:.1f}/s  ETA={eta/60:.1f} min")

    print(f"\n[4/4] 寫出檔案 ...")
    if embeddings:
        embs = np.stack(embeddings)
        cols = {"book_id": book_ids}
        for i in range(embs.shape[1]):
            cols[f"v{i}"] = embs[:, i]
        out_df = pd.DataFrame(cols)
        out_df.to_parquet(PROC / "book_covers.parquet", index=False)
        print(f"  [saved] book_covers.parquet  ({len(out_df)} books × {embs.shape[1]} dims)")

    pd.DataFrame(metadata).to_csv(PROC / "book_covers_meta.csv", index=False)
    print(f"  [saved] book_covers_meta.csv  ({len(metadata)} entries)")
    print(f"\n總結：success={n_success} ({n_success/len(target)*100:.1f}%)  fail={n_fail}")


if __name__ == "__main__":
    main()
