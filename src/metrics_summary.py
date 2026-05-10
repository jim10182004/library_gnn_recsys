from __future__ import annotations

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
SUMMARY_CSV = RESULTS_DIR / "summary.csv"

PRIMARY_METRICS = ("recall@10", "recall@20", "ndcg@10", "ndcg@20", "hit@10")
OPTIONAL_METRICS = ("coverage@10", "coverage@20", "novelty@10", "novelty@20", "mrr@10", "mrr@20")
SUMMARY_COLUMNS = (
    "Model",
    "status",
    *PRIMARY_METRICS,
    "precision@10",
    "precision@20",
    *OPTIONAL_METRICS,
)


def load_summary(path: Path = SUMMARY_CSV) -> pd.DataFrame:
    """Load the experiment summary and coerce metric columns to numeric values."""
    if not path.exists():
        return pd.DataFrame(columns=SUMMARY_COLUMNS)

    df = pd.read_csv(path)
    if "Model" not in df.columns:
        raise ValueError(f"{path} is missing required column: Model")

    for col in df.columns:
        if col != "Model":
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def best_model(metric: str = "recall@10", path: Path = SUMMARY_CSV) -> dict:
    """Return the best row for a metric as a plain dict for API/UI use."""
    df = load_summary(path)
    if df.empty or metric not in df.columns:
        return {}

    ranked = df.dropna(subset=[metric]).sort_values(metric, ascending=False)
    if ranked.empty:
        return {}

    row = ranked.iloc[0]
    return {
        "model": row["Model"],
        "metric": metric,
        "value": float(row[metric]),
        "recall@10": _metric_value(row, "recall@10"),
        "recall@20": _metric_value(row, "recall@20"),
        "ndcg@10": _metric_value(row, "ndcg@10"),
        "ndcg@20": _metric_value(row, "ndcg@20"),
    }


def clean_summary(path: Path = SUMMARY_CSV) -> pd.DataFrame:
    """
    Produce a display/report-friendly summary.

    Missing secondary metrics are kept as NA instead of being fabricated. The
    status column makes it obvious which rows need a full evaluation rerun.
    """
    df = load_summary(path)
    if df.empty:
        return df

    for col in SUMMARY_COLUMNS:
        if col not in df.columns and col not in ("Model", "status"):
            df[col] = pd.NA

    metric_cols = [c for c in df.columns if c != "Model"]
    df["status"] = df[metric_cols].isna().any(axis=1).map(
        {True: "partial_metrics", False: "complete"}
    )
    cols = [c for c in SUMMARY_COLUMNS if c in df.columns]
    extras = [c for c in df.columns if c not in cols]
    return df[cols + extras]


def write_clean_summary(
    path: Path = SUMMARY_CSV,
    out_path: Path | None = None,
) -> Path:
    if out_path is None:
        out_path = path.with_name("summary_clean.csv")
    df = clean_summary(path)
    df.to_csv(out_path, index=False, na_rep="NA")
    return out_path


def _metric_value(row: pd.Series, metric: str) -> float | None:
    if metric not in row or pd.isna(row[metric]):
        return None
    return float(row[metric])


if __name__ == "__main__":
    out = write_clean_summary()
    print(f"Wrote {out}")
