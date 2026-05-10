"""
用中文 BERT 對所有書籍題名（含作者）編碼，產出 384 維向量。

執行：python src/encode_books_bert.py
輸出：data/processed/book_bert.parquet  (book_id, vec_0..vec_383)

模型：paraphrase-multilingual-MiniLM-L12-v2
  - 384 維、22 語言
  - 快速且支援中文
  - ~120 MB 模型檔，首次執行需下載
"""
from __future__ import annotations
import sys
from pathlib import Path

import pandas as pd
import pyarrow  # noqa: F401
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

PROJ = Path(__file__).parent.parent
PROC = PROJ / "data" / "processed"
OUT = PROC / "book_bert.parquet"
MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def main():
    print(f"[BERT] 載入書籍 ...")
    books = pd.read_parquet(PROC / "books.parquet")
    print(f"  共 {len(books):,} 本")

    # 組合 「題名 [SEP] 作者」當作輸入文字
    def make_text(row):
        title = (row["title"] or "").strip()
        author = (row["author"] or "").strip()
        if not title and not author:
            return "未知"
        if not title:
            return author
        if not author:
            return title
        return f"{title}。作者：{author}"

    texts = books.apply(make_text, axis=1).tolist()
    book_ids = books["book_id"].values.astype(np.int64)

    print(f"[BERT] 載入模型 {MODEL_NAME} (device={DEVICE}) ...")
    model = SentenceTransformer(MODEL_NAME, device=DEVICE)
    dim = model.get_sentence_embedding_dimension()
    print(f"  embedding dim = {dim}")

    print("[BERT] 開始編碼（batch_size=64）...")
    embs = model.encode(
        texts, batch_size=64, show_progress_bar=True, convert_to_numpy=True,
        normalize_embeddings=True,
    )
    print(f"  完成，shape={embs.shape}")

    # 存成 parquet：long format 太大，改 wide format（book_id + vec_*）
    cols = {"book_id": book_ids}
    for i in range(dim):
        cols[f"v{i}"] = embs[:, i].astype(np.float32)
    out_df = pd.DataFrame(cols)
    out_df.to_parquet(OUT, index=False)
    print(f"[BERT] 已存到 {OUT}  ({OUT.stat().st_size/1024/1024:.1f} MB)")

    # 也快速驗證：找出與某本書最相似的 5 本
    print("\n[Sanity check] 「紅豆綠豆碰. 5, 學習好好玩」最相似的 5 本：")
    target_idx = books[books["title"].str.contains("紅豆綠豆碰", na=False)].index
    if len(target_idx) > 0:
        ti = target_idx[0]
        sim = embs @ embs[ti]
        top = np.argpartition(-sim, kth=6)[:6]
        top = top[np.argsort(-sim[top])]
        for r, idx in enumerate(top):
            t = books.iloc[idx]["title"] or "?"
            print(f"  {r+1}. ({sim[idx]:.3f}) {t}")


if __name__ == "__main__":
    main()
