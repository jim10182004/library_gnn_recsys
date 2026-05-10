from __future__ import annotations

from pathlib import Path

from src.metrics_summary import best_model, clean_summary, write_clean_summary


def test_best_model_and_clean_summary(tmp_path: Path):
    summary = tmp_path / "summary.csv"
    summary.write_text(
        "\n".join(
            [
                "Model,recall@10,ndcg@10,coverage@10",
                "popular,0.10,0.05,",
                "lightgcn,0.20,0.08,0.30",
            ]
        ),
        encoding="utf-8",
    )

    best = best_model("recall@10", path=summary)
    assert best["model"] == "lightgcn"
    assert best["recall@10"] == 0.20

    clean = clean_summary(summary)
    assert clean.loc[clean["Model"] == "popular", "status"].item() == "partial_metrics"
    assert clean.loc[clean["Model"] == "lightgcn", "status"].item() == "partial_metrics"

    out = write_clean_summary(summary)
    assert out.exists()
    assert "NA" in out.read_text(encoding="utf-8")
