"""
推薦結果的「重排序 (Re-ranking)」模組。

LightGCN 給出 Top-100 的 raw 分數後，這層用其他標準重新排序前 K 本：
  - 多樣性 (diversity)：避免推薦清單全是同類別
  - 反熱門 (de-popularize)：稍微降低熱門書權重
  - 作者上限 (author cap)：同作者最多 N 本
  - 新穎性 (novelty)：偏好冷門但仍相關的書

核心：MMR (Maximal Marginal Relevance, Carbonell & Goldstein 1998) 的變體。
  re_score(j) = λ × relevance(j) - (1-λ) × max_sim(j, already_picked)

實際做法：
  1. 模型先給 Top-100 raw scores
  2. iterative selection：每步找「relevance 高 + 跟已選擇的書差異大」的書

執行：python -m src.reranker  # 自帶 demo
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Optional


class MMRReranker:
    """Maximal Marginal Relevance 多樣性重排序"""

    def __init__(
        self,
        item_category: np.ndarray,
        item_author: np.ndarray | None = None,
        item_pop: np.ndarray | None = None,
        *,
        diversity_lambda: float = 0.7,
        depopularize_alpha: float = 0.0,
        author_cap: int = 3,
        category_cap: int = 6,
        novelty_weight: float = 0.0,
    ):
        """
        Args:
            item_category: shape (n_items,)，每本書的類別 id（integer）
            item_author: shape (n_items,) 或 None，每本書的作者 hash id
            item_pop: shape (n_items,) 訓練集中該書的借閱次數
            diversity_lambda: MMR 的 λ。1.0 = 完全 relevance，0.0 = 完全 diversity
            depopularize_alpha: 熱門懲罰強度。score' = score - α × log(pop+1)
            author_cap: 同作者最多選幾本
            category_cap: 同類別最多選幾本
            novelty_weight: 加給冷門書的 bonus
        """
        self.item_cat = np.asarray(item_category, dtype=np.int64)
        self.item_author = np.asarray(item_author, dtype=np.int64) if item_author is not None else None
        self.item_pop = np.asarray(item_pop, dtype=np.float32) if item_pop is not None else None
        self.lam = diversity_lambda
        self.alpha = depopularize_alpha
        self.author_cap = author_cap
        self.category_cap = category_cap
        self.nov_w = novelty_weight

    def rerank(
        self,
        candidate_ids: np.ndarray,
        candidate_scores: np.ndarray,
        k: int = 10,
    ) -> np.ndarray:
        """
        Args:
            candidate_ids: 模型選出的 Top-N 候選 (N >= k，建議 N = 5k 以上)
            candidate_scores: 對應的 raw score
            k: 最終要回的數量

        Returns:
            shape (k,) 重排序後的 item ids
        """
        candidates = np.asarray(candidate_ids, dtype=np.int64)
        raw_scores = np.asarray(candidate_scores, dtype=np.float32)

        # Step 1：對 raw_scores 做去熱門 + novelty 修正（global，不看 picked）
        adjusted = raw_scores.copy()
        if self.alpha > 0 and self.item_pop is not None:
            pop = self.item_pop[candidates]
            adjusted -= self.alpha * np.log(pop + 1.0)
        if self.nov_w > 0 and self.item_pop is not None:
            pop = self.item_pop[candidates]
            max_log_pop = np.log(self.item_pop.max() + 1.0)
            novelty = 1.0 - np.log(pop + 1.0) / max_log_pop
            adjusted += self.nov_w * novelty

        # Step 2：iterative MMR with caps
        picked: list[int] = []
        picked_idx: list[int] = []  # 在 candidates 中的位置
        author_count: dict[int, int] = {}
        cat_count: dict[int, int] = {}

        # 按 adjusted 排序當作候選順序
        order = np.argsort(-adjusted)

        for it in range(min(k, len(candidates))):
            best_score = -np.inf
            best_idx = -1
            for cand_idx in order:
                if cand_idx in picked_idx:
                    continue
                cid = int(candidates[cand_idx])

                # cap check
                cat = int(self.item_cat[cid])
                if cat_count.get(cat, 0) >= self.category_cap:
                    continue
                if self.item_author is not None:
                    auth = int(self.item_author[cid])
                    if auth >= 0 and author_count.get(auth, 0) >= self.author_cap:
                        continue

                relevance = float(adjusted[cand_idx])

                # MMR：diversity penalty
                if picked:
                    sim = self._max_similarity(cid, picked)
                    mmr_score = self.lam * relevance - (1 - self.lam) * sim
                else:
                    mmr_score = relevance

                if mmr_score > best_score:
                    best_score = mmr_score
                    best_idx = int(cand_idx)

            if best_idx < 0:
                # 全部被 cap 擋掉了，放寬
                for cand_idx in order:
                    if cand_idx not in picked_idx:
                        best_idx = int(cand_idx)
                        break

            if best_idx < 0:
                break

            cid = int(candidates[best_idx])
            picked.append(cid)
            picked_idx.append(best_idx)
            cat = int(self.item_cat[cid])
            cat_count[cat] = cat_count.get(cat, 0) + 1
            if self.item_author is not None:
                auth = int(self.item_author[cid])
                if auth >= 0:
                    author_count[auth] = author_count.get(auth, 0) + 1

        return np.asarray(picked, dtype=np.int64)

    def _max_similarity(self, candidate: int, picked: list[int]) -> float:
        """簡單的「相似度」：類別相同或作者相同 ⇒ 1.0；否則 0.0
        多本相同的話取 max。"""
        cat = int(self.item_cat[candidate])
        sim = 0.0
        for p in picked:
            if int(self.item_cat[p]) == cat:
                sim = max(sim, 0.7)
            if self.item_author is not None:
                if int(self.item_author[candidate]) >= 0 and \
                   int(self.item_author[candidate]) == int(self.item_author[p]):
                    sim = max(sim, 1.0)  # 同作者罰最重
        return sim


# ============================================================
# Demo
# ============================================================

def _demo():
    """跑一個簡單 demo 證明邏輯正確"""
    print("=== MMR Reranker Demo ===\n")

    # 假裝有 20 本書，5 個類別，前 4 名都是「類別 A 的東野圭吾」
    n_items = 20
    item_cat = np.array([0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 0, 1, 2, 3, 4, 4, 4, 0, 1, 2])
    item_author = np.array([1, 1, 1, 1, 2, 2, 3, 3, 4, 4, 5, 6, 7, 8, 9, 9, 9, 1, 2, 3])
    item_pop = np.array([100, 80, 60, 50, 30, 25, 20, 18, 15, 12, 10, 8, 6, 5, 4, 3, 2, 90, 70, 40])

    candidates = np.arange(n_items)
    scores = np.array([0.95, 0.93, 0.91, 0.89, 0.87, 0.85, 0.83, 0.81,
                       0.79, 0.77, 0.75, 0.73, 0.71, 0.69, 0.67, 0.65,
                       0.63, 0.61, 0.59, 0.57])

    print("[原始 Top-10] 直接按 raw score 排序：")
    raw_top = candidates[np.argsort(-scores)][:10]
    for i, c in enumerate(raw_top):
        print(f"  {i+1:2d}. item {c:2d}  cat={item_cat[c]}  author={item_author[c]}  pop={item_pop[c]}")

    rerank = MMRReranker(
        item_category=item_cat,
        item_author=item_author,
        item_pop=item_pop,
        diversity_lambda=0.7,
        author_cap=2,
        category_cap=3,
        depopularize_alpha=0.05,
    )
    reranked = rerank.rerank(candidates, scores, k=10)
    print(f"\n[MMR 重排序 Top-10] (λ=0.7, author_cap=2, cat_cap=3, depop=0.05)：")
    for i, c in enumerate(reranked):
        c = int(c)
        print(f"  {i+1:2d}. item {c:2d}  cat={item_cat[c]}  author={item_author[c]}  pop={item_pop[c]}")

    print("\n→ 注意：原本前 4 名都是 author=1 / cat=0，重排序後同作者只剩 2 本、類別更分散")


if __name__ == "__main__":
    _demo()
