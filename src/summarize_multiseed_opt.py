"""
彙總 lightgcn_multi_opt 的 3 個 seed 結果，計算 mean ± std。

輸入：
  - results/lightgcn_multi_opt_history.json (seed 42, 主實驗)
  - results/lightgcn_multi_opt_seed123_history.json
  - results/lightgcn_multi_opt_seed2024_history.json

輸出：
  - results/ablation/multi_seed_optuna.csv
  - 列印 mean ± std 表給 docx 用

執行：python -m src.summarize_multiseed_opt
"""
from __future__ import annotations
import sys
import json
from pathlib import Path

import pandas as pd  # noqa
import pyarrow  # noqa: F401
import numpy as np

sys.stdout.reconfigure(encoding="utf-8")

PROJECT = Path(__file__).resolve().parent.parent
RES = PROJECT / "results"
RES_AB = RES / "ablation"
RES_AB.mkdir(parents=True, exist_ok=True)

CONFIGS = [
    ("seed 42",   "lightgcn_multi_opt_history.json"),
    ("seed 123",  "lightgcn_multi_opt_seed123_history.json"),
    ("seed 2024", "lightgcn_multi_opt_seed2024_history.json"),
]


def main():
    rows = []
    for label, fname in CONFIGS:
        p = RES / fname
        if not p.exists():
            print(f"[skip] {fname} 還沒存在")
            continue
        with open(p, encoding="utf-8") as f:
            d = json.load(f)
        test = d.get("test", {})
        if not test:
            continue
        rows.append({"seed": label, **test})

    if not rows:
        print("沒有任何 seed 結果可彙總。")
        return

    df = pd.DataFrame(rows)
    out_csv = RES_AB / "multi_seed_optuna.csv"
    df.to_csv(out_csv, index=False)
    print(f"[saved] {out_csv}\n")

    # 計算 mean ± std
    metric_cols = [c for c in df.columns if c != "seed"]
    print("== 各 seed 結果 ==")
    print(df[["seed"] + ["recall@10", "recall@20", "ndcg@10", "coverage@10",
             "novelty@10", "hit@10", "mrr@10"]].to_string(index=False, float_format="%.4f"))

    if len(rows) >= 2:
        print("\n== Mean ± Std ==")
        for m in ["recall@10", "recall@20", "ndcg@10", "coverage@10",
                  "novelty@10", "hit@10", "mrr@10"]:
            vals = df[m].values
            print(f"  {m:14s} = {vals.mean():.4f} ± {vals.std(ddof=1):.4f}")

    # 寫一份 markdown 摘要供 docx 引用
    md_lines = ["# LightGCN-Multi-Opt Multi-seed 結果\n"]
    md_lines.append(f"| 指標 | " + " | ".join([r["seed"] for r in rows]) + " | Mean ± Std |")
    md_lines.append("|---|" + "|".join(["---"] * (len(rows) + 1)) + "|")
    for m in ["recall@10", "recall@20", "ndcg@10", "coverage@10",
              "novelty@10", "hit@10", "mrr@10"]:
        vals = df[m].values
        cells = [f"{v:.4f}" for v in vals]
        if len(vals) >= 2:
            cells.append(f"**{vals.mean():.4f} ± {vals.std(ddof=1):.4f}**")
        else:
            cells.append("(only 1 seed)")
        md_lines.append(f"| {m} | " + " | ".join(cells) + " |")

    out_md = RES_AB / "multi_seed_optuna.md"
    out_md.write_text("\n".join(md_lines), encoding="utf-8")
    print(f"\n[saved] {out_md}")


if __name__ == "__main__":
    main()
