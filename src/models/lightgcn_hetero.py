"""
LightGCN-Hetero: 三種節點的異質圖推薦
  Nodes: User、Book、Author
  Edges: User—Book (借閱)、Book—Author (撰寫)

訊息傳遞：
  - book embedding 同時聚合（讀者鄰居 + 作者鄰居）
  - author embedding 聚合相關書籍
  - user embedding 聚合借閱書籍

實作上仍然用單一 (N+M+P) x (N+M+P) 鄰接矩陣，
不同邊類型用不同權重區分。
"""
from __future__ import annotations
from collections import Counter
import re
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


def extract_authors(books_df: pd.DataFrame, max_per_book: int = 3) -> dict:
    """
    從 books_df 抽取作者並建立索引：
      - 把作者欄拆成主作者（取譯者前的部分）
      - 同一本書最多 max_per_book 位作者
    回傳：(author_to_id, book_to_authors)
    """
    book_to_authors: dict[int, list[int]] = {}
    author_to_id: dict[str, int] = {}

    def parse_authors(s) -> list[str]:
        if not s or pd.isna(s):
            return []
        s = str(s)
        # 切掉「; ...譯」這類附註
        s = re.split(r"[;；]\s*[^;；]*?譯", s)[0]
        # 拆分作者
        names = re.split(r"[,，;；]|\s+著\s*|\s+作\s*|\s+編\s*", s)
        out = []
        for n in names:
            n = n.strip()
            n = re.sub(r"[\(\)（）].*$", "", n).strip()
            if n and len(n) <= 30 and len(n) >= 1:
                out.append(n)
        return out[:max_per_book]

    for _, row in books_df.iterrows():
        authors = parse_authors(row["author"])
        ids = []
        for a in authors:
            if a not in author_to_id:
                author_to_id[a] = len(author_to_id)
            ids.append(author_to_id[a])
        if ids:
            book_to_authors[int(row["book_id"])] = ids

    return author_to_id, book_to_authors


def build_hetero_adj(
    splits,
    books_df: pd.DataFrame,
    *,
    user_book_weight: float = 1.0,
    book_author_weight: float = 0.3,
    device: str = "cpu",
) -> tuple[torch.sparse.Tensor, int, dict]:
    """
    建構異質圖的對稱正規化鄰接矩陣。
    節點順序：[users (n_users) | books (n_items) | authors (n_authors)]
    """
    n_u = splits.n_users
    n_i = splits.n_items

    print("[Hetero] 抽取作者 ...")
    author_to_id, book_to_authors = extract_authors(books_df)
    n_a = len(author_to_id)
    print(f"  作者數：{n_a:,}")

    # User-Book 邊（借閱）
    bu = splits.train["u"].values.astype(np.int64)
    bi = splits.train["i"].values.astype(np.int64)
    ub_w = np.full(len(bu), user_book_weight, dtype=np.float32)

    # Book-Author 邊：原始 book_id → splits 緊湊 i
    inv_item = {orig: i for orig, i in splits.item_remap.items()}
    ba_b = []
    ba_a = []
    for orig_book_id, author_ids in book_to_authors.items():
        if orig_book_id in inv_item:
            i = inv_item[orig_book_id]
            for a in author_ids:
                ba_b.append(i)
                ba_a.append(a)
    ba_b = np.array(ba_b, dtype=np.int64)
    ba_a = np.array(ba_a, dtype=np.int64)
    ba_w = np.full(len(ba_b), book_author_weight, dtype=np.float32)
    print(f"  Book-Author 邊：{len(ba_b):,}")

    n = n_u + n_i + n_a
    # User—Book 雙向
    src1 = np.concatenate([bu, bi + n_u])
    dst1 = np.concatenate([bi + n_u, bu])
    w1 = np.concatenate([ub_w, ub_w])
    # Book—Author 雙向
    src2 = np.concatenate([ba_b + n_u, ba_a + n_u + n_i])
    dst2 = np.concatenate([ba_a + n_u + n_i, ba_b + n_u])
    w2 = np.concatenate([ba_w, ba_w])

    src = np.concatenate([src1, src2])
    dst = np.concatenate([dst1, dst2])
    w = np.concatenate([w1, w2])

    indices = torch.from_numpy(np.stack([src, dst]))
    values = torch.from_numpy(w)
    A = torch.sparse_coo_tensor(indices, values, (n, n)).coalesce()

    deg = torch.sparse.sum(A, dim=1).to_dense()
    d_inv_sqrt = deg.pow(-0.5)
    d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.0
    row, col = A.indices()
    norm_vals = A.values() * d_inv_sqrt[row] * d_inv_sqrt[col]
    A_hat = torch.sparse_coo_tensor(A.indices(), norm_vals, (n, n)).coalesce().to(device)

    return A_hat, n_a, author_to_id


class LightGCNHetero(nn.Module):
    """LightGCN 在 [user | item | author] 三類節點上的傳播。"""

    def __init__(
        self,
        n_users: int,
        n_items: int,
        n_authors: int,
        embed_dim: int = 64,
        n_layers: int = 3,
    ):
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.n_authors = n_authors
        self.embed_dim = embed_dim
        self.n_layers = n_layers

        self.user_emb = nn.Embedding(n_users, embed_dim)
        self.item_emb = nn.Embedding(n_items, embed_dim)
        self.author_emb = nn.Embedding(max(n_authors, 1), embed_dim)
        for emb in (self.user_emb, self.item_emb, self.author_emb):
            nn.init.normal_(emb.weight, std=0.1)

        self.norm_adj: torch.sparse.Tensor | None = None

    def set_graph(self, norm_adj: torch.sparse.Tensor):
        self.norm_adj = norm_adj

    def propagate(self) -> tuple[torch.Tensor, torch.Tensor]:
        assert self.norm_adj is not None
        all_emb = torch.cat([
            self.user_emb.weight,
            self.item_emb.weight,
            self.author_emb.weight,
        ], dim=0)
        embs = [all_emb]
        x = all_emb
        for _ in range(self.n_layers):
            x = torch.sparse.mm(self.norm_adj, x)
            embs.append(x)
        out = torch.stack(embs, dim=0).mean(dim=0)
        users = out[: self.n_users]
        items = out[self.n_users : self.n_users + self.n_items]
        return users, items

    def bpr_loss(self, users, pos_items, neg_items, decay=1e-4):
        u_emb_all, i_emb_all = self.propagate()
        u = u_emb_all[users]
        pi = i_emb_all[pos_items]
        ni = i_emb_all[neg_items]
        pos_scores = (u * pi).sum(-1)
        neg_scores = (u * ni).sum(-1)
        bpr = -F.logsigmoid(pos_scores - neg_scores).mean()
        u0 = self.user_emb(users)
        pi0 = self.item_emb(pos_items)
        ni0 = self.item_emb(neg_items)
        reg = (u0.norm(2).pow(2) + pi0.norm(2).pow(2) + ni0.norm(2).pow(2)) / users.size(0)
        return bpr + decay * reg, bpr.detach()

    @torch.no_grad()
    def get_all_ratings(self, batch_users: torch.Tensor) -> torch.Tensor:
        u_emb_all, i_emb_all = self.propagate()
        u = u_emb_all[batch_users]
        return u @ i_emb_all.T
