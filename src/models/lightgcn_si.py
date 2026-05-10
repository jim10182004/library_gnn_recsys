"""
LightGCN-SI: 加入側資訊（Side Information）的 LightGCN

加入特徵：
  - User 端：性別 (Embedding)、年齡分箱 (Embedding)
  - Item 端：中圖法首碼 (Embedding)

與原版 LightGCN 的差異：
  - 初始 embedding e_u^(0) = e_u_id + e_u_gender + e_u_age_bucket
  - 初始 embedding e_i^(0) = e_i_id + e_i_category
  - 之後一樣做 K 層圖卷積、各層平均

這樣的設計：
  ✔ 解冷啟動：新 user 至少有 gender/age 訊號
  ✔ 加入語意：同分類書籍會有共同訊號
  ✔ 不破壞 LightGCN 的「輕量」精神：只是加總幾個 lookup
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class LightGCNSI(nn.Module):
    def __init__(
        self,
        n_users: int,
        n_items: int,
        n_genders: int,        # 性別類別數（含 unknown）
        n_age_buckets: int,    # 年齡分箱數
        n_categories: int,     # 分類號類別數
        embed_dim: int = 64,
        n_layers: int = 3,
    ):
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.embed_dim = embed_dim
        self.n_layers = n_layers

        # ID embedding
        self.user_emb = nn.Embedding(n_users, embed_dim)
        self.item_emb = nn.Embedding(n_items, embed_dim)
        # Side info embeddings
        self.gender_emb = nn.Embedding(n_genders, embed_dim)
        self.age_emb = nn.Embedding(n_age_buckets, embed_dim)
        self.cat_emb = nn.Embedding(n_categories, embed_dim)

        for emb in (self.user_emb, self.item_emb, self.gender_emb, self.age_emb, self.cat_emb):
            nn.init.normal_(emb.weight, std=0.1)

        # 須由外部設定（使用 register_buffer 統一註冊，這裡先放佔位）
        self.norm_adj: torch.sparse.Tensor | None = None
        self.register_buffer("user_gender", torch.zeros(n_users, dtype=torch.long), persistent=False)
        self.register_buffer("user_age_bucket", torch.zeros(n_users, dtype=torch.long), persistent=False)
        self.register_buffer("item_cat", torch.zeros(n_items, dtype=torch.long), persistent=False)
        self._side_info_set = False

    def set_graph(self, norm_adj: torch.sparse.Tensor):
        self.norm_adj = norm_adj

    def set_side_info(
        self,
        user_gender: torch.Tensor,
        user_age_bucket: torch.Tensor,
        item_cat: torch.Tensor,
    ):
        # 直接複製到既有 buffer（已和模型同 device）
        self.user_gender.copy_(user_gender)
        self.user_age_bucket.copy_(user_age_bucket)
        self.item_cat.copy_(item_cat)
        self._side_info_set = True

    def _initial_embeddings(self):
        """e^(0) = id_emb + side_info_embs"""
        u0 = self.user_emb.weight + self.gender_emb(self.user_gender) + self.age_emb(self.user_age_bucket)
        i0 = self.item_emb.weight + self.cat_emb(self.item_cat)
        return u0, i0

    def propagate(self) -> tuple[torch.Tensor, torch.Tensor]:
        assert self.norm_adj is not None, "請先呼叫 set_graph(...)"
        assert self._side_info_set, "請先呼叫 set_side_info(...)"
        u0, i0 = self._initial_embeddings()
        all_emb = torch.cat([u0, i0], dim=0)
        embs = [all_emb]
        x = all_emb
        for _ in range(self.n_layers):
            x = torch.sparse.mm(self.norm_adj, x)
            embs.append(x)
        out = torch.stack(embs, dim=0).mean(dim=0)
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
        # 正則化只對 ID embedding（side embedding 共享，不重複罰）
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


# ----- 工具函式：把 split 中的 user/item 對應到 side info -----

def build_side_info_tensors(splits, books_df, users_df):
    """
    回傳：
        user_gender: LongTensor [n_users]   (0=男, 1=女, 2=unknown)
        user_age:    LongTensor [n_users]   (0~7 共 8 個年齡段)
        item_cat:    LongTensor [n_items]   (0~10, 0~9 對應中圖大類, 10=unknown)
        meta dict
    """
    import numpy as np

    inv_user = {v: k for k, v in splits.user_remap.items()}
    inv_item = {v: k for k, v in splits.item_remap.items()}

    # 把原始 user / item 反查
    n_users = splits.n_users
    n_items = splits.n_items

    # users
    users_idx = users_df.set_index("user_orig")
    gender_map = {"男": 0, "女": 1}
    user_gender = np.full(n_users, 2, dtype=np.int64)
    user_age = np.full(n_users, 7, dtype=np.int64)  # default "unknown"
    age_bins = [0, 12, 18, 25, 35, 50, 65, 200]      # 7 個區間，第 8 個 (idx=7) 留給 missing
    for u_compact in range(n_users):
        orig = inv_user[u_compact]
        if orig in users_idx.index:
            row = users_idx.loc[orig]
            g = row["gender"]
            if g in gender_map:
                user_gender[u_compact] = gender_map[g]
            a = row["age"]
            if a is not None and not np.isnan(a):
                bucket = np.searchsorted(age_bins, a, side="right") - 1
                user_age[u_compact] = max(0, min(6, bucket))

    # items
    books_idx = books_df.set_index("book_id")
    item_cat = np.full(n_items, 10, dtype=np.int64)  # 10 = unknown
    for i_compact in range(n_items):
        orig = inv_item[i_compact]
        if orig in books_idx.index:
            cat = books_idx.loc[orig]["category"]
            if cat is not None:
                s = str(cat).strip()
                if s and s[0].isdigit():
                    item_cat[i_compact] = int(s[0])

    import torch
    return (
        torch.from_numpy(user_gender),
        torch.from_numpy(user_age),
        torch.from_numpy(item_cat),
        {"n_genders": 3, "n_age_buckets": 8, "n_categories": 11},
    )
