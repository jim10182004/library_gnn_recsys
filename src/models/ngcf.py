"""
NGCF: Neural Graph Collaborative Filtering (Wang et al., SIGIR 2019)
原文：https://arxiv.org/abs/1905.08108

與 LightGCN 對照：NGCF 保留 GCN 的線性轉換 W 與激活函數 LeakyReLU，
另外多了「element-wise interaction」這個 second-order 訊號。
LightGCN 證明這些其實對推薦有害，但作為對照組仍重要。

每層公式：
    e_u^{(k+1)} = LeakyReLU( W_1 * msg_1 + W_2 * msg_2 )
    msg_1 = sum_{i ∈ N(u)} (1/√(|N(u)||N(i)|)) * e_i^{(k)}
    msg_2 = sum_{i ∈ N(u)} (1/√(|N(u)||N(i)|)) * (e_u^{(k)} ⊙ e_i^{(k)})

最終 embedding：把所有層 (含第 0 層) concat 起來。
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class NGCF(nn.Module):
    def __init__(
        self,
        n_users: int,
        n_items: int,
        embed_dim: int = 64,
        n_layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.embed_dim = embed_dim
        self.n_layers = n_layers
        self.dropout = dropout

        self.user_emb = nn.Embedding(n_users, embed_dim)
        self.item_emb = nn.Embedding(n_items, embed_dim)
        nn.init.xavier_uniform_(self.user_emb.weight)
        nn.init.xavier_uniform_(self.item_emb.weight)

        # 每層兩組 W
        self.W1 = nn.ModuleList([nn.Linear(embed_dim, embed_dim, bias=True) for _ in range(n_layers)])
        self.W2 = nn.ModuleList([nn.Linear(embed_dim, embed_dim, bias=True) for _ in range(n_layers)])
        for layer in (*self.W1, *self.W2):
            nn.init.xavier_uniform_(layer.weight)

        self.norm_adj: torch.sparse.Tensor | None = None

    def set_graph(self, norm_adj: torch.sparse.Tensor):
        self.norm_adj = norm_adj

    def propagate(self) -> tuple[torch.Tensor, torch.Tensor]:
        assert self.norm_adj is not None
        all_emb = torch.cat([self.user_emb.weight, self.item_emb.weight], dim=0)
        embs = [all_emb]
        x = all_emb
        for k in range(self.n_layers):
            # msg_1 = A_hat @ x
            msg1 = torch.sparse.mm(self.norm_adj, x)
            # msg_2 = A_hat @ (x ⊙ x_self) — 但 self_emb 是節點自己，
            # 在 matrix form 等價於 element-wise 乘以 x，再 propagate；
            # 這裡簡化為：先 propagate 取鄰居，再 element-wise 乘 x
            interact = torch.sparse.mm(self.norm_adj, x) * x
            x = F.leaky_relu(self.W1[k](msg1) + self.W2[k](interact), negative_slope=0.2)
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = F.normalize(x, p=2, dim=1)
            embs.append(x)
        # NGCF 是 concat，不是 mean
        out = torch.cat(embs, dim=1)
        users, items = out[: self.n_users], out[self.n_users :]
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
