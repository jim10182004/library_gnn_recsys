"""
LightGCN-BERT: 結合 BERT 書名語意 + 多邊圖 + 側資訊

書籍初始 embedding：
    e_i^(0) = e_i^id + e_i^category + W_bert(bert_emb_i)

讀者保持與 LightGCN-Multi 一樣（id + gender + age）。
之後一樣做 K 層圖卷積。

W_bert：384 → embed_dim 的線性投影，與整個模型一起訓練。
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class LightGCNBert(nn.Module):
    def __init__(
        self,
        n_users: int,
        n_items: int,
        bert_dim: int = 384,
        embed_dim: int = 64,
        n_layers: int = 3,
        n_genders: int = 3,
        n_age_buckets: int = 8,
        n_categories: int = 11,
    ):
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.embed_dim = embed_dim
        self.n_layers = n_layers

        self.user_emb = nn.Embedding(n_users, embed_dim)
        self.item_emb = nn.Embedding(n_items, embed_dim)
        self.gender_emb = nn.Embedding(n_genders, embed_dim)
        self.age_emb = nn.Embedding(n_age_buckets, embed_dim)
        self.cat_emb = nn.Embedding(n_categories, embed_dim)
        for emb in (self.user_emb, self.item_emb, self.gender_emb, self.age_emb, self.cat_emb):
            nn.init.normal_(emb.weight, std=0.1)

        # BERT 投影
        self.bert_proj = nn.Linear(bert_dim, embed_dim, bias=False)
        nn.init.xavier_uniform_(self.bert_proj.weight)

        # 註冊 buffer
        self.register_buffer("user_gender", torch.zeros(n_users, dtype=torch.long), persistent=False)
        self.register_buffer("user_age_bucket", torch.zeros(n_users, dtype=torch.long), persistent=False)
        self.register_buffer("item_cat", torch.zeros(n_items, dtype=torch.long), persistent=False)
        self.register_buffer("item_bert", torch.zeros(n_items, bert_dim, dtype=torch.float32), persistent=False)
        self._side_info_set = False
        self._bert_set = False

        self.norm_adj: torch.sparse.Tensor | None = None

    def set_graph(self, norm_adj: torch.sparse.Tensor):
        self.norm_adj = norm_adj

    def set_side_info(self, user_gender, user_age_bucket, item_cat):
        self.user_gender.copy_(user_gender)
        self.user_age_bucket.copy_(user_age_bucket)
        self.item_cat.copy_(item_cat)
        self._side_info_set = True

    def set_bert(self, item_bert: torch.Tensor):
        assert item_bert.shape == (self.n_items, self.item_bert.shape[1])
        self.item_bert.copy_(item_bert)
        self._bert_set = True

    def _initial_embeddings(self):
        assert self._side_info_set
        assert self._bert_set
        u0 = self.user_emb.weight + self.gender_emb(self.user_gender) + self.age_emb(self.user_age_bucket)
        bert_proj = self.bert_proj(self.item_bert)
        i0 = self.item_emb.weight + self.cat_emb(self.item_cat) + bert_proj
        return u0, i0

    def propagate(self) -> tuple[torch.Tensor, torch.Tensor]:
        assert self.norm_adj is not None
        u0, i0 = self._initial_embeddings()
        all_emb = torch.cat([u0, i0], dim=0)
        embs = [all_emb]
        x = all_emb
        for _ in range(self.n_layers):
            x = torch.sparse.mm(self.norm_adj, x)
            embs.append(x)
        out = torch.stack(embs, dim=0).mean(dim=0)
        return out[: self.n_users], out[self.n_users :]

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


def load_bert_tensor(parquet_path, n_items: int, item_remap: dict) -> torch.Tensor:
    """從 book_bert.parquet 載入並對齊到 splits 的 i 編號。"""
    import pandas as pd
    import numpy as np
    df = pd.read_parquet(parquet_path)
    vec_cols = [c for c in df.columns if c.startswith("v")]
    bert_dim = len(vec_cols)
    out = np.zeros((n_items, bert_dim), dtype=np.float32)
    df_idx = df.set_index("book_id")
    for orig_book_id, i in item_remap.items():
        if orig_book_id in df_idx.index:
            out[i] = df_idx.loc[orig_book_id, vec_cols].values.astype(np.float32)
    return torch.from_numpy(out)
