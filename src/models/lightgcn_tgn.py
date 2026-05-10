"""
LightGCN-TGN: Time-aware LightGCN inspired by TGN (Rossi et al., 2020)
完整 TGN 包含 memory module 與 temporal sampling，學習門檻較高。
本實作為「TGN 簡化版」：保留 TGN 的核心創新 ── time encoding，
但移除 memory module 與 temporal neighbor sampler，可在標準 LightGCN
基礎上以最小修改取得時序感知能力。

核心：Time2Vec encoding (Kazemi et al. 2019)
    φ(t) = [t · ω_0 + φ_0,
            sin(ω_1 t + φ_1), sin(ω_2 t + φ_2), ..., sin(ω_{d-1} t + φ_{d-1})]

每條邊（借閱事件）有 timestamp t，計算 time encoding 後與 item embedding 相加，
等效於「時間越近的借閱影響力越強」。

對照 LightGCN-TimeDecay（已實作）：
    - TimeDecay：用固定衰減函數 exp(-λ Δt) 加權邊
    - TGN-Lite：用「學習得到」的 sinusoidal embedding 編碼時間
    - TGN-Lite 理論上更靈活，能學到非線性時間模式
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


class Time2Vec(nn.Module):
    """Kazemi et al. 2019: Time2Vec
    將 scalar time t 編碼成 d 維向量
    """
    def __init__(self, embed_dim: int = 64):
        super().__init__()
        self.embed_dim = embed_dim
        # 第 0 維線性，其餘 d-1 維用 sinusoidal
        self.w0 = nn.Parameter(torch.randn(1))
        self.b0 = nn.Parameter(torch.randn(1))
        self.w = nn.Parameter(torch.randn(embed_dim - 1))
        self.b = nn.Parameter(torch.randn(embed_dim - 1))

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """t: [N] tensor of normalized timestamps in [0, 1]
           回傳：[N, embed_dim]
        """
        linear = (t * self.w0 + self.b0).unsqueeze(-1)        # [N, 1]
        sinus = torch.sin(t.unsqueeze(-1) * self.w + self.b)   # [N, embed_dim - 1]
        return torch.cat([linear, sinus], dim=-1)


class LightGCNTGN(nn.Module):
    """LightGCN with learned time encoding."""

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

        self.time_enc = Time2Vec(embed_dim)
        # 把 time encoding 投影回原 embedding 空間（殘差連接）
        self.time_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        nn.init.zeros_(self.time_proj.weight)  # 初始化為 0：純 LightGCN，再學起來

        self.norm_adj: torch.sparse.Tensor | None = None
        self.register_buffer("user_recency", torch.zeros(n_users), persistent=False)
        self.register_buffer("item_recency", torch.zeros(n_items), persistent=False)
        self._recency_set = False

    def set_graph(self, norm_adj: torch.sparse.Tensor):
        self.norm_adj = norm_adj

    def set_recency(self, user_recency: torch.Tensor, item_recency: torch.Tensor):
        """每個 user / item 的最近一次互動時間（normalized 到 [0,1]）"""
        self.user_recency.copy_(user_recency)
        self.item_recency.copy_(item_recency)
        self._recency_set = True

    def propagate(self) -> tuple[torch.Tensor, torch.Tensor]:
        assert self.norm_adj is not None
        assert self._recency_set, "請先呼叫 set_recency()"

        # 把時間編碼加到初始 embedding（殘差）
        u_time = self.time_proj(self.time_enc(self.user_recency))   # [n_users, D]
        i_time = self.time_proj(self.time_enc(self.item_recency))   # [n_items, D]
        u0 = self.user_emb.weight + u_time
        i0 = self.item_emb.weight + i_time

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


def compute_recency(splits) -> tuple[torch.Tensor, torch.Tensor]:
    """從 splits.train 算每個 user / item 最近的互動時間，normalize 到 [0, 1]
    （t=1 表示 train cutoff，t=0 表示 train 開始）
    """
    train = splits.train.copy()
    train["ts"] = pd.to_datetime(train["ts"])
    t_min = train["ts"].min()
    t_max = train["ts"].max()
    span = (t_max - t_min).total_seconds()
    train["t_norm"] = (train["ts"] - t_min).dt.total_seconds() / max(span, 1.0)

    user_rec = train.groupby("u")["t_norm"].max()
    item_rec = train.groupby("i")["t_norm"].max()

    u_arr = np.zeros(splits.n_users, dtype=np.float32)
    i_arr = np.zeros(splits.n_items, dtype=np.float32)
    u_arr[user_rec.index.values] = user_rec.values
    i_arr[item_rec.index.values] = item_rec.values
    return torch.from_numpy(u_arr), torch.from_numpy(i_arr)
