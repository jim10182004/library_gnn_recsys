"""
LightGCN-Cover: 多模態 LightGCN，整合書封 CNN feature
與 LightGCN-BERT 概念相同，但用視覺特徵（512 維 ResNet-18）取代語意特徵。

設計：
  e_i^(0) = e_i^id + W_cover(cover_emb_i)

對於沒有書封的書，cover_emb_i = 0（自動 fallback）。
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


class LightGCNCover(nn.Module):
    def __init__(
        self,
        n_users: int,
        n_items: int,
        cover_dim: int = 512,
        embed_dim: int = 64,
        n_layers: int = 3,
    ):
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.embed_dim = embed_dim
        self.n_layers = n_layers

        self.user_emb = nn.Embedding(n_users, embed_dim)
        self.item_emb = nn.Embedding(n_items, embed_dim)
        nn.init.normal_(self.user_emb.weight, std=0.1)
        nn.init.normal_(self.item_emb.weight, std=0.1)

        self.cover_proj = nn.Linear(cover_dim, embed_dim, bias=False)
        nn.init.xavier_uniform_(self.cover_proj.weight)

        self.register_buffer("item_cover", torch.zeros(n_items, cover_dim, dtype=torch.float32),
                            persistent=False)
        self.register_buffer("has_cover", torch.zeros(n_items, dtype=torch.float32),
                            persistent=False)
        self.norm_adj: torch.sparse.Tensor | None = None
        self._cover_set = False

    def set_graph(self, norm_adj: torch.sparse.Tensor):
        self.norm_adj = norm_adj

    def set_covers(self, item_cover: torch.Tensor, has_cover: torch.Tensor):
        self.item_cover.copy_(item_cover)
        self.has_cover.copy_(has_cover)
        self._cover_set = True

    def propagate(self) -> tuple[torch.Tensor, torch.Tensor]:
        assert self.norm_adj is not None
        assert self._cover_set
        cover_proj = self.cover_proj(self.item_cover) * self.has_cover.unsqueeze(1)
        i0 = self.item_emb.weight + cover_proj
        all_emb = torch.cat([self.user_emb.weight, i0], dim=0)
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


def load_cover_tensors(parquet_path, n_items: int, item_remap: dict) -> tuple[torch.Tensor, torch.Tensor]:
    """從 book_covers.parquet 載入並對齊到 splits 的 i 編號
    回傳 (item_cover [n_items, 512], has_cover [n_items] {0,1})
    """
    df = pd.read_parquet(parquet_path)
    vec_cols = [c for c in df.columns if c.startswith("v")]
    cover_dim = len(vec_cols)
    out = np.zeros((n_items, cover_dim), dtype=np.float32)
    has = np.zeros(n_items, dtype=np.float32)
    df_idx = df.set_index("book_id")
    for orig_book_id, i in item_remap.items():
        if orig_book_id in df_idx.index:
            out[i] = df_idx.loc[orig_book_id, vec_cols].values.astype(np.float32)
            has[i] = 1.0
    return torch.from_numpy(out), torch.from_numpy(has)
