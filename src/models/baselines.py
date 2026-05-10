"""
Baselines:
  1. Popular        - 推薦最熱門的書（不個人化）
  2. ItemCF         - Item-based Collaborative Filtering（cosine similarity）
  3. BPR-MF         - Bayesian Personalised Ranking + Matrix Factorisation
"""
from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.sparse import csr_matrix
from sklearn.preprocessing import normalize


class PopularRecommender:
    """所有人都推薦借閱次數最多的 N 本書。"""

    def __init__(self):
        self.item_pop: np.ndarray | None = None  # shape: [n_items]

    def fit(self, train_u: np.ndarray, train_i: np.ndarray, n_items: int):
        pop = np.bincount(train_i, minlength=n_items).astype(np.float32)
        self.item_pop = pop

    def get_all_ratings(self, batch_users: np.ndarray) -> np.ndarray:
        return np.broadcast_to(self.item_pop, (len(batch_users), len(self.item_pop))).copy()


class ItemCF:
    """Item-based CF：相似度 = cosine(item_i_users, item_j_users)。"""

    def __init__(self, top_sim: int = 200):
        self.top_sim = top_sim
        self.user_item: csr_matrix | None = None
        self.item_sim: csr_matrix | None = None

    def fit(self, train_u: np.ndarray, train_i: np.ndarray, n_users: int, n_items: int):
        # user x item 互動矩陣
        data = np.ones(len(train_u), dtype=np.float32)
        ui = csr_matrix((data, (train_u, train_i)), shape=(n_users, n_items))
        self.user_item = ui
        # item-item 相似度（cosine）
        norm_iu = normalize(ui.T, norm="l2", axis=1)  # item x user (L2 normalised)
        sim = norm_iu @ norm_iu.T  # item x item
        sim.setdiag(0)
        sim.eliminate_zeros()
        self.item_sim = sim

    def get_all_ratings(self, batch_users: np.ndarray) -> np.ndarray:
        # rating(u, i) = sum_j sim(i, j) * has_borrowed(u, j)
        ui_batch = self.user_item[batch_users]
        scores = ui_batch @ self.item_sim
        return np.asarray(scores.todense())


class BPRMF(nn.Module):
    """傳統矩陣分解 + BPR loss，是 LightGCN 直接的對照組（沒有圖卷積）。"""

    def __init__(self, n_users: int, n_items: int, embed_dim: int = 64):
        super().__init__()
        self.user_emb = nn.Embedding(n_users, embed_dim)
        self.item_emb = nn.Embedding(n_items, embed_dim)
        nn.init.normal_(self.user_emb.weight, std=0.1)
        nn.init.normal_(self.item_emb.weight, std=0.1)

    def bpr_loss(self, users, pos_items, neg_items, decay=1e-4):
        u = self.user_emb(users)
        pi = self.item_emb(pos_items)
        ni = self.item_emb(neg_items)
        pos_scores = (u * pi).sum(-1)
        neg_scores = (u * ni).sum(-1)
        bpr = -F.logsigmoid(pos_scores - neg_scores).mean()
        reg = (u.norm(2).pow(2) + pi.norm(2).pow(2) + ni.norm(2).pow(2)) / users.size(0)
        return bpr + decay * reg, bpr.detach()

    @torch.no_grad()
    def get_all_ratings(self, batch_users: torch.Tensor) -> torch.Tensor:
        u = self.user_emb(batch_users)
        return u @ self.item_emb.weight.T
