"""
LightGCN-Multi: 多邊型版本，將「預約」資料當作較弱訊號加入圖

設計：
  - 借閱邊權重 = 1.0  (強訊號：實際借走)
  - 預約邊權重 = 0.5  (弱訊號：有興趣但可能沒借)

實作上仍用同一張正規化鄰接矩陣，只是邊的初始權重不一樣，
之後對稱正規化會自動處理。

也支援 side-info（共用 LightGCN-SI 的設計）。
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


def build_norm_adj_weighted(
    edges_u: torch.Tensor,
    edges_i: torch.Tensor,
    edges_w: torch.Tensor,
    n_users: int,
    n_items: int,
    device: str = "cpu",
) -> torch.sparse.Tensor:
    """加權鄰接矩陣 + 對稱正規化"""
    n = n_users + n_items
    src = torch.cat([edges_u, edges_i + n_users])
    dst = torch.cat([edges_i + n_users, edges_u])
    w = torch.cat([edges_w, edges_w])
    indices = torch.stack([src, dst], dim=0)
    A = torch.sparse_coo_tensor(indices, w, (n, n)).coalesce()

    deg = torch.sparse.sum(A, dim=1).to_dense()
    d_inv_sqrt = deg.pow(-0.5)
    d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.0

    row, col = A.indices()
    norm_vals = A.values() * d_inv_sqrt[row] * d_inv_sqrt[col]
    return torch.sparse_coo_tensor(A.indices(), norm_vals, (n, n)).coalesce().to(device)


class LightGCNMulti(nn.Module):
    """LightGCN with multi-edge weighted graph + optional side info."""

    def __init__(
        self,
        n_users: int,
        n_items: int,
        embed_dim: int = 64,
        n_layers: int = 3,
        use_side_info: bool = True,
        n_genders: int = 3,
        n_age_buckets: int = 8,
        n_categories: int = 11,
    ):
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.embed_dim = embed_dim
        self.n_layers = n_layers
        self.use_side_info = use_side_info

        self.user_emb = nn.Embedding(n_users, embed_dim)
        self.item_emb = nn.Embedding(n_items, embed_dim)
        nn.init.normal_(self.user_emb.weight, std=0.1)
        nn.init.normal_(self.item_emb.weight, std=0.1)

        if use_side_info:
            self.gender_emb = nn.Embedding(n_genders, embed_dim)
            self.age_emb = nn.Embedding(n_age_buckets, embed_dim)
            self.cat_emb = nn.Embedding(n_categories, embed_dim)
            for emb in (self.gender_emb, self.age_emb, self.cat_emb):
                nn.init.normal_(emb.weight, std=0.1)
            self.register_buffer("user_gender", torch.zeros(n_users, dtype=torch.long), persistent=False)
            self.register_buffer("user_age_bucket", torch.zeros(n_users, dtype=torch.long), persistent=False)
            self.register_buffer("item_cat", torch.zeros(n_items, dtype=torch.long), persistent=False)
            self._side_info_set = False

        self.norm_adj: torch.sparse.Tensor | None = None

    def set_graph(self, norm_adj: torch.sparse.Tensor):
        self.norm_adj = norm_adj

    def set_side_info(self, user_gender, user_age_bucket, item_cat):
        assert self.use_side_info
        self.user_gender.copy_(user_gender)
        self.user_age_bucket.copy_(user_age_bucket)
        self.item_cat.copy_(item_cat)
        self._side_info_set = True

    def _initial_embeddings(self):
        if self.use_side_info:
            assert self._side_info_set
            u0 = self.user_emb.weight + self.gender_emb(self.user_gender) + self.age_emb(self.user_age_bucket)
            i0 = self.item_emb.weight + self.cat_emb(self.item_cat)
        else:
            u0 = self.user_emb.weight
            i0 = self.item_emb.weight
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


def build_multi_edges(splits, reservations_df, borrow_weight: float = 1.0, reserve_weight: float = 0.5):
    """
    把 train 借閱 + 預約資料合併成加權邊集合。
    只保留在 splits.user_remap / item_remap 中出現的 user/item。

    回傳：edges_u, edges_i, edges_w (LongTensor, LongTensor, FloatTensor)
    """
    import numpy as np

    # 借閱邊
    bu = splits.train["u"].values.astype(np.int64)
    bi = splits.train["i"].values.astype(np.int64)
    bw = np.full(len(bu), borrow_weight, dtype=np.float32)

    # 預約邊（先過濾在 user_remap / item_remap 中出現的）
    user_map = splits.user_remap
    item_map = splits.item_remap
    res = reservations_df[
        reservations_df["user_id"].isin(user_map) & reservations_df["book_id"].isin(item_map)
    ].copy()
    res["u"] = res["user_id"].map(user_map).astype("int64")
    res["i"] = res["book_id"].map(item_map).astype("int64")

    # 也要過濾時間（只用 train 期間之前的預約，避免洩漏）
    train_max_ts = splits.train["ts"].max() if "ts" in splits.train.columns else None
    if train_max_ts is not None:
        res["ts"] = pd.to_datetime(res["ts"])
        res = res[res["ts"] <= train_max_ts]

    # 去重 (u, i) — 同一人預約同本書多次不重複加邊
    res = res.drop_duplicates(subset=["u", "i"])

    # 也去掉與借閱重複的（已經是借閱邊就不重加）
    borrow_set = set(zip(bu.tolist(), bi.tolist()))
    res = res[~res.apply(lambda r: (r["u"], r["i"]) in borrow_set, axis=1)]

    ru = res["u"].values.astype(np.int64)
    ri = res["i"].values.astype(np.int64)
    rw = np.full(len(ru), reserve_weight, dtype=np.float32)

    edges_u = np.concatenate([bu, ru])
    edges_i = np.concatenate([bi, ri])
    edges_w = np.concatenate([bw, rw])

    print(f"[Multi-edge] borrow={len(bu):,}  reserve(extra)={len(ru):,}  total={len(edges_u):,}")

    return (
        torch.from_numpy(edges_u),
        torch.from_numpy(edges_i),
        torch.from_numpy(edges_w),
    )


# 因為要用 pandas，避免 lightgcn_multi 依賴循環，把 import 放這裡
import pandas as pd  # noqa: E402
