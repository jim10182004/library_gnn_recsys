"""
測試推薦邏輯：
  - 推薦結果不應該包含 user 已經借過的書（mask_seen 正確運作）
  - top-k 確實 sorted by score
  - 分數越高的越在前
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT = Path(__file__).resolve().parent.parent
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from src.evaluate import evaluate_topk


class StaticModel:
    """固定 score 的測試 model，方便驗證行為"""

    def __init__(self, scores: np.ndarray):
        self.scores = scores  # shape (n_users, n_items)

    def get_all_ratings(self, users: np.ndarray) -> np.ndarray:
        return self.scores[users].copy()


def test_seen_items_are_masked():
    """已借過的書（在 user_train_pos）絕不出現在 Top-K"""
    # 4 個 item，user 0 借過 item 0、1
    scores = np.array([[0.9, 0.95, 0.7, 0.5]], dtype=np.float32)
    model = StaticModel(scores)

    metrics, topk = evaluate_topk(
        model,
        eval_users=np.array([0]),
        user_train_pos={0: {0, 1}},  # 已借過 0, 1
        user_eval_pos={0: {2}},        # 真實要命中 2
        n_items=4,
        k_list=(2,),
        is_torch=False,
        item_pop=np.array([10, 5, 3, 1], dtype=np.float32),
        return_topk=True,
    )

    rec = topk[0]
    # 0、1 都不應該在 top-K
    assert 0 not in rec, f"item 0 已借過，不應在推薦中: {rec}"
    assert 1 not in rec, f"item 1 已借過，不應在推薦中: {rec}"
    # 2、3 都會被推（因為只剩這兩個）
    assert set(rec.tolist()) == {2, 3}


def test_topk_sorted_by_score_desc():
    """Top-K 排序：分數高的在前"""
    # 3 個 user，每個有不同的 score 分布
    scores = np.array(
        [
            [0.1, 0.9, 0.5, 0.8, 0.3],
            [0.7, 0.2, 0.6, 0.4, 0.5],
            [0.5, 0.5, 0.5, 0.5, 0.5],  # 全相等
        ],
        dtype=np.float32,
    )
    model = StaticModel(scores)

    _, topk = evaluate_topk(
        model,
        eval_users=np.array([0, 1, 2]),
        user_train_pos={0: set(), 1: set(), 2: set()},
        user_eval_pos={0: {1}, 1: {0}, 2: {0}},
        n_items=5,
        k_list=(3,),
        is_torch=False,
        item_pop=np.array([10, 5, 3, 1, 1], dtype=np.float32),
        return_topk=True,
    )

    # user 0: 應該是 [1, 3, 2]（按 0.9, 0.8, 0.5 排序）
    rec0 = topk[0][:3]
    assert rec0[0] == 1, f"user 0 第一名應該是 item 1，實際 {rec0}"
    assert rec0[1] == 3
    assert rec0[2] == 2

    # user 1: 應該是 [0, 2, 4]（按 0.7, 0.6, 0.5 排序）
    rec1 = topk[1][:3]
    assert rec1[0] == 0
    assert rec1[1] == 2


def test_recall_max_when_all_hits():
    """所有 ground truth 都在 top-K 時 recall = 1.0"""
    scores = np.array([[0.9, 0.8, 0.7, 0.1]], dtype=np.float32)
    metrics = evaluate_topk(
        StaticModel(scores),
        eval_users=np.array([0]),
        user_train_pos={0: set()},
        user_eval_pos={0: {0, 1}},  # 兩個 ground truth
        n_items=4,
        k_list=(2,),
        is_torch=False,
        item_pop=np.array([10, 5, 3, 1], dtype=np.float32),
    )
    assert metrics["recall@2"] == 1.0
    assert metrics["hit@2"] == 1.0


def test_no_ground_truth_user_skipped():
    """user 沒有 ground truth (user_eval_pos 空) 不影響平均"""
    scores = np.array(
        [
            [0.9, 0.8, 0.7, 0.1],
            [0.5, 0.5, 0.5, 0.5],
        ],
        dtype=np.float32,
    )
    metrics = evaluate_topk(
        StaticModel(scores),
        eval_users=np.array([0, 1]),
        user_train_pos={0: set(), 1: set()},
        user_eval_pos={0: {0}, 1: set()},  # user 1 沒 ground truth
        n_items=4,
        k_list=(2,),
        is_torch=False,
        item_pop=np.array([10, 5, 3, 1], dtype=np.float32),
    )
    # 只算 user 0：item 0 在 top-2 → recall=1.0
    assert metrics["recall@2"] == 1.0


def test_coverage_grows_with_k():
    """Coverage@20 ≥ Coverage@10"""
    rng = np.random.default_rng(42)
    scores = rng.random((50, 100)).astype(np.float32)
    metrics = evaluate_topk(
        StaticModel(scores),
        eval_users=np.arange(50),
        user_train_pos={i: set() for i in range(50)},
        user_eval_pos={i: {(i + 1) % 100} for i in range(50)},
        n_items=100,
        k_list=(10, 20),
        is_torch=False,
        item_pop=np.arange(100, dtype=np.float32),
    )
    assert metrics["coverage@20"] >= metrics["coverage@10"]


def test_novelty_in_valid_range():
    """Novelty 應該在 [0, 1]"""
    rng = np.random.default_rng(42)
    scores = rng.random((10, 50)).astype(np.float32)
    metrics = evaluate_topk(
        StaticModel(scores),
        eval_users=np.arange(10),
        user_train_pos={i: set() for i in range(10)},
        user_eval_pos={i: {(i + 1) % 50} for i in range(10)},
        n_items=50,
        k_list=(5,),
        is_torch=False,
        item_pop=np.arange(50, dtype=np.float32) + 1,
    )
    assert 0.0 <= metrics["novelty@5"] <= 1.0


def test_mrr_nonnegative_and_bounded():
    """MRR ∈ [0, 1]"""
    scores = np.array([[0.1, 0.9, 0.8, 0.7]], dtype=np.float32)
    metrics = evaluate_topk(
        StaticModel(scores),
        eval_users=np.array([0]),
        user_train_pos={0: set()},
        user_eval_pos={0: {1}},  # item 1 是 ground truth，應該排第 1
        n_items=4,
        k_list=(3,),
        is_torch=False,
        item_pop=np.array([10, 5, 3, 1], dtype=np.float32),
    )
    # item 1 score 0.9 是最高 → 排第 1 → MRR = 1/1 = 1.0
    assert metrics["mrr@3"] == 1.0
