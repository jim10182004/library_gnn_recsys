"""
SASRec: Self-Attentive Sequential Recommendation (Kang & McAuley, ICDM 2018)
原文：https://arxiv.org/abs/1808.09781

把每位讀者的借閱歷史視為「序列」，用 Transformer encoder 學習序列表示，
預測「下一本可能借的書」。

與 LightGCN 不同：
  - 利用借閱「順序」資訊（GNN 視為無向圖、無順序）
  - 對近期借閱權重隱式較高（attention）
  - 推論時：給定使用者的歷史 → 對所有 item 算分

訓練設定：
  - 每位讀者最後 max_len 條借閱
  - 預測 shifted-by-one 的下個 item
  - 採樣負樣本，BPR loss
"""
from __future__ import annotations
import math
from collections import defaultdict
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


class SASRec(nn.Module):
    def __init__(
        self,
        n_items: int,
        embed_dim: int = 64,
        max_len: int = 50,
        n_blocks: int = 2,
        n_heads: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.n_items = n_items
        self.embed_dim = embed_dim
        self.max_len = max_len

        # item_emb: 0 = padding
        self.item_emb = nn.Embedding(n_items + 1, embed_dim, padding_idx=0)
        self.pos_emb = nn.Embedding(max_len, embed_dim)
        nn.init.xavier_uniform_(self.item_emb.weight)
        nn.init.xavier_uniform_(self.pos_emb.weight)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=n_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout, activation="gelu",
            batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_blocks)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(embed_dim)

    def _seq_repr(self, seq: torch.Tensor) -> torch.Tensor:
        """seq: [B, L] (item ids, 0 = padding 在前面). 回傳 [B, L, D]"""
        B, L = seq.shape
        positions = torch.arange(L, device=seq.device).unsqueeze(0).expand(B, L)
        x = self.item_emb(seq) + self.pos_emb(positions)
        x = self.dropout(x)
        # causal mask: 不可看未來
        mask = torch.triu(torch.ones(L, L, device=seq.device, dtype=torch.bool), diagonal=1)
        # padding mask
        pad_mask = (seq == 0)
        out = self.encoder(x, mask=mask, src_key_padding_mask=pad_mask)
        out = self.norm(out)
        return out

    def forward_train(self, seq: torch.Tensor, pos: torch.Tensor, neg: torch.Tensor):
        """
        seq: [B, L]    輸入序列
        pos: [B, L]    每個位置的正樣本（next item）
        neg: [B, L]    每個位置的負樣本
        """
        out = self._seq_repr(seq)
        pos_emb = self.item_emb(pos)
        neg_emb = self.item_emb(neg)
        pos_logits = (out * pos_emb).sum(-1)
        neg_logits = (out * neg_emb).sum(-1)

        # 只計算有效位置的 loss (pos 不為 0)
        mask = (pos != 0).float()
        loss = -(F.logsigmoid(pos_logits - neg_logits) * mask).sum() / mask.sum().clamp(min=1)
        return loss

    @torch.no_grad()
    def get_user_repr(self, seq: torch.Tensor) -> torch.Tensor:
        """seq: [B, L] → user vec [B, D]：取最後一個非 padding 位置的輸出"""
        out = self._seq_repr(seq)
        # 找每位 user 最後一個非 padding 位置
        valid = (seq != 0)
        last_idx = (valid.long().cumsum(dim=1) == valid.long().sum(dim=1, keepdim=True))
        last_idx = (last_idx & valid).float().argmax(dim=1)
        gather = last_idx.view(-1, 1, 1).expand(-1, 1, self.embed_dim)
        return out.gather(1, gather).squeeze(1)

    @torch.no_grad()
    def get_all_ratings(self, batch_users: torch.Tensor) -> torch.Tensor:
        """要透過 self.user_seq tensor 對應每個 user 的歷史"""
        seq = self.user_seq[batch_users]  # [B, L]
        u_rep = self.get_user_repr(seq)
        # item embeddings (包含 padding 0，要跳掉)
        all_items = self.item_emb.weight[1:]  # [n_items, D]
        scores = u_rep @ all_items.T
        return scores  # [B, n_items]

    def set_user_sequences(self, user_seq: torch.Tensor):
        """user_seq: [n_users, max_len] LongTensor with item ids (0 = padding)"""
        self.register_buffer("user_seq", user_seq, persistent=False)


def build_sequences(splits, max_len: int = 50) -> tuple[torch.Tensor, dict[int, list[int]]]:
    """
    從 train 中建構每位 user 的借閱序列（按時間排序，最舊在前）。
    回傳：
      user_seq:    [n_users, max_len] (右對齊；前面 padding=0；item id = 原 i + 1)
      raw_seq:     dict u → list of item ids（原 i，未 +1）（給訓練用）
    """
    train = splits.train.sort_values("ts")
    raw: dict[int, list[int]] = defaultdict(list)
    for u, i in zip(train["u"].values, train["i"].values):
        raw[int(u)].append(int(i))

    user_seq = np.zeros((splits.n_users, max_len), dtype=np.int64)
    for u, seq in raw.items():
        if len(seq) > max_len:
            seq = seq[-max_len:]
        # 右對齊：放到尾部，前面 0 padding
        user_seq[u, max_len - len(seq):] = np.array(seq) + 1  # +1 因為 0 是 padding
    return torch.from_numpy(user_seq), dict(raw)


class SASRecDataset(torch.utils.data.Dataset):
    def __init__(self, raw_seq: dict[int, list[int]], n_items: int, max_len: int = 50):
        self.users = sorted([u for u, s in raw_seq.items() if len(s) >= 2])
        self.raw = raw_seq
        self.n_items = n_items
        self.max_len = max_len

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        u = self.users[idx]
        seq = self.raw[u]
        if len(seq) > self.max_len + 1:
            seq = seq[-(self.max_len + 1):]
        # 輸入 = seq[:-1], 目標 = seq[1:]
        L = self.max_len
        input_ids = np.zeros(L, dtype=np.int64)
        pos_ids = np.zeros(L, dtype=np.int64)
        neg_ids = np.zeros(L, dtype=np.int64)

        # 右對齊
        in_part = seq[:-1]
        pos_part = seq[1:]
        if len(in_part) > L:
            in_part = in_part[-L:]
            pos_part = pos_part[-L:]
        offset = L - len(in_part)
        input_ids[offset:] = np.array(in_part) + 1
        pos_ids[offset:] = np.array(pos_part) + 1

        seen = set(seq)
        for j in range(offset, L):
            while True:
                ni = np.random.randint(0, self.n_items)
                if ni not in seen:
                    neg_ids[j] = ni + 1
                    break
        return input_ids, pos_ids, neg_ids
