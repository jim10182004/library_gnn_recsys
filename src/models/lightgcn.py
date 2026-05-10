"""
LightGCN: He et al., SIGIR 2020 - "LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation"
原文: https://arxiv.org/abs/2002.02126

核心想法：
  1. 不用線性轉換 W、不用激活函數，只做純鄰居平均
  2. 最終 embedding = 各層 embedding 的加權平均（α_k）
  3. score(u, i) = e_u · e_i  (內積)
  4. BPR loss 訓練

實作方式：使用 normalized adjacency matrix 直接相乘（PyTorch sparse），
比逐層 message passing 快很多，且符合原論文 closed-form。
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class LightGCN(nn.Module):
    def __init__(
        self,
        n_users: int,
        n_items: int,
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

        self.norm_adj: torch.sparse.Tensor | None = None  # 由外部設定

    def set_graph(self, norm_adj: torch.sparse.Tensor):
        """注入正規化的鄰接矩陣 ((N+M) x (N+M)) sparse。"""
        self.norm_adj = norm_adj

    def propagate(self) -> tuple[torch.Tensor, torch.Tensor]:
        """執行 K 層圖卷積，回傳每個 node 的最終 embedding。"""
        assert self.norm_adj is not None, "請先呼叫 set_graph(norm_adj)"
        all_emb = torch.cat([self.user_emb.weight, self.item_emb.weight], dim=0)
        embs = [all_emb]
        x = all_emb
        for _ in range(self.n_layers):
            x = torch.sparse.mm(self.norm_adj, x)
            embs.append(x)
        # 各層平均（α_k = 1/(K+1)）
        out = torch.stack(embs, dim=0).mean(dim=0)
        users, items = out[: self.n_users], out[self.n_users :]
        return users, items

    def bpr_loss(
        self,
        users: torch.Tensor,
        pos_items: torch.Tensor,
        neg_items: torch.Tensor,
        decay: float = 1e-4,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        u_emb_all, i_emb_all = self.propagate()

        u = u_emb_all[users]
        pi = i_emb_all[pos_items]
        ni = i_emb_all[neg_items]

        pos_scores = (u * pi).sum(dim=-1)
        neg_scores = (u * ni).sum(dim=-1)

        bpr = -F.logsigmoid(pos_scores - neg_scores).mean()

        # L2 regularisation 只用初始 embedding（LightGCN 標準做法）
        u0 = self.user_emb(users)
        pi0 = self.item_emb(pos_items)
        ni0 = self.item_emb(neg_items)
        reg = (u0.norm(2).pow(2) + pi0.norm(2).pow(2) + ni0.norm(2).pow(2)) / users.size(0)

        loss = bpr + decay * reg
        return loss, bpr.detach()

    @torch.no_grad()
    def get_all_ratings(self, batch_users: torch.Tensor) -> torch.Tensor:
        """回傳一批 user 對所有 item 的預測分數，shape: [B, n_items]"""
        u_emb_all, i_emb_all = self.propagate()
        u = u_emb_all[batch_users]
        return u @ i_emb_all.T


def build_norm_adj(
    train_u: torch.Tensor,
    train_i: torch.Tensor,
    n_users: int,
    n_items: int,
    device: str = "cpu",
) -> torch.sparse.Tensor:
    """
    建構對稱正規化的鄰接矩陣 A_hat = D^(-1/2) A D^(-1/2)
    A 是 (N+M)x(N+M) 的二部圖鄰接矩陣
    """
    n = n_users + n_items
    # user 與 item 的雙向邊
    src = torch.cat([train_u, train_i + n_users])
    dst = torch.cat([train_i + n_users, train_u])
    indices = torch.stack([src, dst], dim=0)
    values = torch.ones(indices.size(1), dtype=torch.float32)

    A = torch.sparse_coo_tensor(indices, values, (n, n)).coalesce()

    # 度
    deg = torch.sparse.sum(A, dim=1).to_dense()
    d_inv_sqrt = deg.pow(-0.5)
    d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.0

    # D^-1/2 A D^-1/2 (對 sparse 操作）
    row, col = A.indices()
    norm_vals = A.values() * d_inv_sqrt[row] * d_inv_sqrt[col]
    A_hat = torch.sparse_coo_tensor(
        A.indices(), norm_vals, (n, n)
    ).coalesce().to(device)
    return A_hat
