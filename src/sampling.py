"""
進階負例採樣 (Negative Sampling) 策略。

預設 BPR 用「均勻隨機負例」：對每個 (u, pos_i) 隨機抽 1 本沒看過的 j 當負例。
這對「熱門書 vs 冷門書」沒區別，學到的訊號可能稀疏。

本模組提供 3 種進階策略：

1. PopularityNegativeSampler — 按熱門度開根號採樣
   - 熱門書被抽中為負例的機率 ∝ pop^α
   - 直覺：熱門書更可能是「真負例」（讀者看過但沒借）
   - 對「Coverage / 長尾」有幫助

2. HardNegativeSampler — Hard negative mining
   - 從目前模型給「分數高但讀者沒看過」的書當負例
   - 直覺：這些是模型容易誤判的書，學它們才能進步
   - 對「Recall / NDCG」有幫助但訓練較慢

3. CategoryAwareNegativeSampler — 同類別負例
   - 從「跟正例同類別但讀者沒看過」抽
   - 直覺：強迫模型學會「為什麼是 A 不是 B」（細粒度區分）
"""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset


# ============================================================
# 1. 均勻負例（baseline，等同於原 BPRDataset）
# ============================================================

class UniformNegSampler(Dataset):
    """每個 (u, pi) 從未互動過的 item 中均勻抽 1 個負例"""

    def __init__(self, train_u, train_i, n_items, user_pos):
        self.u = np.asarray(train_u, dtype=np.int64)
        self.i = np.asarray(train_i, dtype=np.int64)
        self.n_items = n_items
        self.user_pos = user_pos
        self.name = "uniform"

    def __len__(self):
        return len(self.u)

    def __getitem__(self, idx):
        u = self.u[idx]
        pi = self.i[idx]
        seen = self.user_pos[int(u)]
        while True:
            ni = np.random.randint(0, self.n_items)
            if ni not in seen:
                return u, pi, ni


# ============================================================
# 2. Popularity-aware Negative Sampler
# ============================================================

class PopularityNegSampler(Dataset):
    """按 pop^alpha 抽負例。alpha=0 等於均勻，alpha=1 等於完全按熱門度。

    經驗值 alpha=0.75（word2vec 同款）。
    """

    def __init__(self, train_u, train_i, n_items, user_pos, item_pop, alpha=0.75):
        self.u = np.asarray(train_u, dtype=np.int64)
        self.i = np.asarray(train_i, dtype=np.int64)
        self.n_items = n_items
        self.user_pos = user_pos
        self.alpha = alpha
        self.name = f"pop-aware(α={alpha})"

        # 預先算機率分布
        pop_powered = np.asarray(item_pop, dtype=np.float64) ** alpha
        pop_powered[pop_powered <= 0] = 1.0  # 沒被借過的書也給機會
        self.probs = pop_powered / pop_powered.sum()
        # 預先生一大堆候選（accelerate 抽樣）
        self._cache = np.random.choice(n_items, size=200_000, p=self.probs)
        self._cache_idx = 0

    def _draw_neg(self):
        # 從 cache 抽（快），用完重新生
        if self._cache_idx >= len(self._cache):
            self._cache = np.random.choice(self.n_items, size=200_000, p=self.probs)
            self._cache_idx = 0
        n = self._cache[self._cache_idx]
        self._cache_idx += 1
        return n

    def __len__(self):
        return len(self.u)

    def __getitem__(self, idx):
        u = self.u[idx]
        pi = self.i[idx]
        seen = self.user_pos[int(u)]
        for _ in range(20):
            ni = self._draw_neg()
            if ni not in seen:
                return u, pi, ni
        # fallback：均勻抽
        while True:
            ni = np.random.randint(0, self.n_items)
            if ni not in seen:
                return u, pi, ni


# ============================================================
# 3. Hard Negative Sampler
# ============================================================

class HardNegSampler(Dataset):
    """每個 (u, pi)：從目前 model 給高分但 user 沒互動過的 K 本中抽 1 本。

    需要每個 epoch 重算 user 的 top scores（cost 高），所以
    本實作用「先 sample n_pool 個隨機 negative，挑分數最高的」近似。

    pool_size = 100：每個 user 在 100 個 random neg 裡挑 model 分數最高 1 個當 hard neg。
    """

    def __init__(self, train_u, train_i, n_items, user_pos, model=None, pool_size=100, device="cuda"):
        self.u = np.asarray(train_u, dtype=np.int64)
        self.i = np.asarray(train_i, dtype=np.int64)
        self.n_items = n_items
        self.user_pos = user_pos
        self.pool_size = pool_size
        self.model = model
        self.device = device
        self.name = f"hard-neg(pool={pool_size})"

    def set_model(self, model):
        """每個 epoch 後可更新模型（讓 hard neg 跟著進步）"""
        self.model = model

    def __len__(self):
        return len(self.u)

    def __getitem__(self, idx):
        u = self.u[idx]
        pi = self.i[idx]
        seen = self.user_pos[int(u)]

        # Step 1: 抽 pool_size 個隨機候選（過濾 seen）
        candidates = []
        attempts = 0
        while len(candidates) < self.pool_size and attempts < self.pool_size * 3:
            ni = np.random.randint(0, self.n_items)
            if ni not in seen:
                candidates.append(ni)
            attempts += 1

        if not candidates:
            return u, pi, np.random.randint(0, self.n_items)

        # Step 2: 模型給分，挑最高的
        if self.model is None:
            return u, pi, candidates[0]

        with torch.no_grad():
            u_t = torch.tensor([int(u)], dtype=torch.long, device=self.device)
            scores = self.model.get_all_ratings(u_t).cpu().numpy()[0]
            cand_scores = scores[candidates]
            best = candidates[int(np.argmax(cand_scores))]
        return u, pi, int(best)


# ============================================================
# 4. Category-aware Negative Sampler
# ============================================================

class CategoryNegSampler(Dataset):
    """從「跟 pos_i 同類別但 user 沒看過」抽 hard-ish negative。

    需要 item_category: np.ndarray shape (n_items,)，整數 category id。
    若 user 在該類別沒未互動 item，fallback 到均勻負例。
    """

    def __init__(self, train_u, train_i, n_items, user_pos, item_category, prob_same_cat=0.7):
        self.u = np.asarray(train_u, dtype=np.int64)
        self.i = np.asarray(train_i, dtype=np.int64)
        self.n_items = n_items
        self.user_pos = user_pos
        self.item_cat = np.asarray(item_category, dtype=np.int64)
        self.prob = prob_same_cat
        self.name = f"category-aware(p={prob_same_cat})"

        # 預先算每個 category 的 item 列表
        from collections import defaultdict
        cat_items = defaultdict(list)
        for i, c in enumerate(self.item_cat):
            cat_items[int(c)].append(i)
        self.cat_items = {c: np.asarray(v, dtype=np.int64) for c, v in cat_items.items()}

    def __len__(self):
        return len(self.u)

    def __getitem__(self, idx):
        u = self.u[idx]
        pi = self.i[idx]
        seen = self.user_pos[int(u)]
        same_cat = int(self.item_cat[pi])

        # 機率 prob_same_cat 走 category-aware；否則均勻
        if np.random.random() < self.prob and same_cat in self.cat_items:
            cat_pool = self.cat_items[same_cat]
            for _ in range(10):
                ni = int(np.random.choice(cat_pool))
                if ni not in seen and ni != pi:
                    return u, pi, ni

        while True:
            ni = np.random.randint(0, self.n_items)
            if ni not in seen:
                return u, pi, ni


# ============================================================
# Factory
# ============================================================

def get_sampler(name: str, train_u, train_i, n_items, user_pos, **kwargs):
    """factory 函數 — name ∈ {uniform, pop, hard, category}"""
    if name == "uniform":
        return UniformNegSampler(train_u, train_i, n_items, user_pos)
    elif name == "pop":
        item_pop = kwargs.get("item_pop")
        alpha = kwargs.get("alpha", 0.75)
        return PopularityNegSampler(train_u, train_i, n_items, user_pos, item_pop, alpha=alpha)
    elif name == "hard":
        model = kwargs.get("model")
        pool_size = kwargs.get("pool_size", 100)
        device = kwargs.get("device", "cuda")
        return HardNegSampler(train_u, train_i, n_items, user_pos, model=model,
                              pool_size=pool_size, device=device)
    elif name == "category":
        item_cat = kwargs.get("item_category")
        prob = kwargs.get("prob_same_cat", 0.7)
        return CategoryNegSampler(train_u, train_i, n_items, user_pos, item_cat,
                                  prob_same_cat=prob)
    else:
        raise ValueError(f"unknown sampler: {name}")
