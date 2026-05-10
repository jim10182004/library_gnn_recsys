from __future__ import annotations

import numpy as np

from src.evaluate import evaluate_topk


class StaticModel:
    def __init__(self, scores: np.ndarray):
        self.scores = scores

    def get_all_ratings(self, users: np.ndarray) -> np.ndarray:
        return self.scores[users].copy()


def test_evaluate_topk_masks_training_items_and_scores_hits():
    model = StaticModel(
        np.array(
            [
                [0.9, 0.8, 0.7, 0.1],
                [0.1, 0.9, 0.8, 0.7],
            ],
            dtype=np.float32,
        )
    )

    metrics = evaluate_topk(
        model,
        eval_users=np.array([0, 1]),
        user_train_pos={0: {0}, 1: {1}},
        user_eval_pos={0: {1}, 1: {2}},
        n_items=4,
        k_list=(1, 2),
        is_torch=False,
        item_pop=np.array([10, 5, 3, 1], dtype=np.float32),
    )

    assert metrics["recall@1"] == 1.0
    assert metrics["precision@1"] == 1.0
    assert metrics["hit@1"] == 1.0
    assert metrics["ndcg@1"] == 1.0
    assert metrics["mrr@1"] == 1.0
    assert "coverage@1" in metrics
    assert "novelty@1" in metrics
