"""
SimGCL: Simple Graph Contrastive Learning (Yu et al., SIGIR 2022)
論文：https://arxiv.org/abs/2112.08679

LightGCL（Cai et al., ICLR 2023）的同源簡化版：
- 沿用 LightGCN 的純圖卷積
- 在 embedding 層加隨機噪音產生 augmented view
- BPR loss + InfoNCE 對比 loss

公式：
  e_view1, e_view2 = LightGCN_propagate() + noise_1, + noise_2
  L = L_BPR + λ * InfoNCE(e_view1, e_view2)

InfoNCE 拉近同一節點兩 view 的 embedding，推遠不同節點。
這帶來 implicit data augmentation，提升表現與 robustness。

本實作為畢業專題對照組，目的：
- 證明 LightGCN 在我們的資料上仍有競爭力 (SimGCL 不一定贏)
- 與 2022+ SOTA 方法做公平比較
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class SimGCL(nn.Module):
    def __init__(
        self,
        n_users: int,
        n_items: int,
        embed_dim: int = 64,
        n_layers: int = 3,
        eps: float = 0.02,            # 經 sweep 找到的最佳值（原本默認 0.1 太高）
        cl_weight: float = 0.001,     # 經 sweep：cl_weight 越小越好（避免主導 BPR）
        cl_temperature: float = 0.2,  # InfoNCE 溫度
    ):
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.embed_dim = embed_dim
        self.n_layers = n_layers
        self.eps = eps
        self.cl_weight = cl_weight
        self.cl_temp = cl_temperature

        self.user_emb = nn.Embedding(n_users, embed_dim)
        self.item_emb = nn.Embedding(n_items, embed_dim)
        nn.init.normal_(self.user_emb.weight, std=0.1)
        nn.init.normal_(self.item_emb.weight, std=0.1)
        self.norm_adj: torch.sparse.Tensor | None = None

    def set_graph(self, norm_adj: torch.sparse.Tensor):
        self.norm_adj = norm_adj

    def _propagate(self, perturb: bool = False):
        """LightGCN 風格傳播；perturb=True 時每層加隨機噪音"""
        all_emb = torch.cat([self.user_emb.weight, self.item_emb.weight], dim=0)
        embs = [all_emb]
        x = all_emb
        for _ in range(self.n_layers):
            x = torch.sparse.mm(self.norm_adj, x)
            if perturb:
                # 加上方向隨機的噪音（SimGCL 原文做法）
                noise = torch.rand_like(x)
                noise = F.normalize(noise, p=2, dim=1) * self.eps
                # noise 與原 embedding 同方向（避免完全隨機破壞訊息）
                noise = noise * torch.sign(x)
                x = x + noise
            embs.append(x)
        out = torch.stack(embs, dim=0).mean(dim=0)
        return out[: self.n_users], out[self.n_users :]

    def propagate(self):
        """推論用 — 不加噪音"""
        return self._propagate(perturb=False)

    def _info_nce(self, z1: torch.Tensor, z2: torch.Tensor):
        """InfoNCE loss between two views of same nodes"""
        z1 = F.normalize(z1, p=2, dim=1)
        z2 = F.normalize(z2, p=2, dim=1)
        # 正樣本：同 index；負樣本：所有其他 index
        sim = z1 @ z2.T / self.cl_temp           # [B, B]
        labels = torch.arange(z1.size(0), device=z1.device)
        return F.cross_entropy(sim, labels)

    def bpr_loss(self, users, pos_items, neg_items, decay=1e-4):
        # 主 loss：用乾淨 embedding
        u_emb_all, i_emb_all = self.propagate()
        u = u_emb_all[users]
        pi = i_emb_all[pos_items]
        ni = i_emb_all[neg_items]
        pos_scores = (u * pi).sum(-1)
        neg_scores = (u * ni).sum(-1)
        bpr = -F.logsigmoid(pos_scores - neg_scores).mean()

        # 對比 loss：兩個 perturbed views
        u1_all, i1_all = self._propagate(perturb=True)
        u2_all, i2_all = self._propagate(perturb=True)
        cl_user = self._info_nce(u1_all[users], u2_all[users])
        cl_item = self._info_nce(i1_all[pos_items], i2_all[pos_items])
        cl = cl_user + cl_item

        u0 = self.user_emb(users)
        pi0 = self.item_emb(pos_items)
        ni0 = self.item_emb(neg_items)
        reg = (u0.norm(2).pow(2) + pi0.norm(2).pow(2) + ni0.norm(2).pow(2)) / users.size(0)
        return bpr + decay * reg + self.cl_weight * cl, bpr.detach()

    @torch.no_grad()
    def get_all_ratings(self, batch_users: torch.Tensor) -> torch.Tensor:
        u_emb_all, i_emb_all = self.propagate()
        u = u_emb_all[batch_users]
        return u @ i_emb_all.T
