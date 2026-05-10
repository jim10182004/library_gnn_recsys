from __future__ import annotations

from types import SimpleNamespace

import api.main as api_main


def test_api_stats_uses_summary_helper(monkeypatch):
    monkeypatch.setattr(
        api_main,
        "best_model",
        lambda metric: {
            "model": "lightgcn_multi_opt",
            "metric": metric,
            "recall@10": 0.2706,
            "recall@20": 0.3015,
            "ndcg@10": 0.2232,
            "ndcg@20": 0.2315,
        },
    )
    api_main.state.splits = SimpleNamespace(
        n_users=10,
        n_items=20,
        train=[1, 2, 3],
        val=[1],
        test=[1, 2],
    )

    stats = api_main.get_stats()

    assert stats["best_model"] == "lightgcn_multi_opt"
    assert stats["best_recall_at_10"] == 0.2706
    assert stats["n_train"] == 3
