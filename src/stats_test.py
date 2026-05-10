"""
統計顯著性檢驗：對 multi_seed 結果做 paired t-test
（每個 seed 都跑了 LightGCN / LightGCN-SI / LightGCN-Multi，
 同 seed 是 paired sample，做 paired t-test）

執行：python -m src.stats_test
輸出：results/ablation/stats_test.md
"""
from __future__ import annotations
from pathlib import Path
import pandas as pd
import numpy as np
from scipy import stats

PROJ = Path(__file__).parent.parent
ABL = PROJ / "results" / "ablation"


def main():
    df = pd.read_csv(ABL / "multi_seed.csv")
    print(f"Loaded {len(df)} runs (model × seed)")
    print()

    metrics = ["recall@10", "recall@20", "ndcg@10", "ndcg@20", "hit@10"]
    models = ["lightgcn", "lightgcn_si", "lightgcn_multi"]
    pairs = [
        ("lightgcn", "lightgcn_si", "SI 是否顯著優於純 LightGCN"),
        ("lightgcn_si", "lightgcn_multi", "Multi 是否顯著優於 SI"),
        ("lightgcn", "lightgcn_multi", "Multi 是否顯著優於純 LightGCN"),
    ]

    print("=" * 80)
    print("Mean ± Std (n=3 seeds)")
    print("=" * 80)
    summary = df.groupby("model")[metrics].agg(["mean", "std"])
    print(summary.round(4))
    print()

    out_md = ["# 統計顯著性檢驗結果", ""]
    out_md.append(f"資料：3 個 seed × 3 個模型，共 9 次完整訓練（multi_seed.csv）。")
    out_md.append("")
    out_md.append("## Mean ± Std (n=3)")
    out_md.append("")
    out_md.append("| 模型 | " + " | ".join(metrics) + " |")
    out_md.append("|" + "---|" * (len(metrics) + 1))
    for m in models:
        sub = df[df["model"] == m]
        row = [f"**{m}**"]
        for met in metrics:
            row.append(f"{sub[met].mean():.4f} ± {sub[met].std():.4f}")
        out_md.append("| " + " | ".join(row) + " |")
    out_md.append("")

    print("=" * 80)
    print("Paired t-test results")
    print("=" * 80)
    out_md.append("## Paired t-test (one-tailed: B > A)")
    out_md.append("")
    out_md.append("- 同 seed 配對 → paired t-test")
    out_md.append("- 顯著性閾值：p < 0.05")
    out_md.append("- 假設方向：B 模型 > A 模型（單尾）")
    out_md.append("")
    out_md.append("| 比較 | 指標 | mean diff (B-A) | t | p (one-tail) | 顯著? |")
    out_md.append("|---|---|---|---|---|---|")

    for a, b, desc in pairs:
        print(f"\n--- {desc}: {a} vs {b} ---")
        for met in metrics:
            xs = df[df["model"] == a].sort_values("seed")[met].values
            ys = df[df["model"] == b].sort_values("seed")[met].values
            assert len(xs) == len(ys) == 3
            diff = ys - xs
            mean_diff = diff.mean()
            t_stat, p_two = stats.ttest_rel(ys, xs)
            # one-tailed: 如果 mean diff > 0 且 t > 0，則 p_one = p_two/2
            p_one = p_two / 2 if mean_diff > 0 else 1 - p_two / 2
            sig_md = "**[YES]** p<0.05" if p_one < 0.05 else "no"
            sig = "[*] p<0.05" if p_one < 0.05 else "(ns)"
            print(f"  {met}: mean_diff={mean_diff:+.4f}  t={t_stat:.3f}  p={p_one:.4f}  {sig}")
            out_md.append(f"| {a} -> {b} | {met} | {mean_diff:+.4f} | {t_stat:.3f} | {p_one:.4f} | {sig_md} |")

    out_md.append("")
    out_md.append("## 解讀")
    out_md.append("")
    out_md.append("- **p < 0.05**：拒絕「兩模型相同」的虛無假設，B 顯著優於 A")
    out_md.append("- **p ≥ 0.05**：差距可能是隨機波動")
    out_md.append("- 由於 n=3 樣本數小，統計檢定力較弱；趨勢一致性 + p 值需綜合判斷")

    (ABL / "stats_test.md").write_text("\n".join(out_md), encoding="utf-8")
    print(f"\n[saved] {ABL / 'stats_test.md'}")


if __name__ == "__main__":
    main()
