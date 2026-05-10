"""
一鍵重現所有實驗結果（Reproducibility Script）

執行：python run_all.py
時間估計：~3-4 小時（含全部訓練、ablation、視覺化、文件產生）

可選參數：
    --skip preprocess   略過 Excel→Parquet 轉換（已有 processed/）
    --skip splits       略過資料切分
    --skip baselines    略過 popular/itemcf/bprmf
    --skip lightgcn     略過 LightGCN 系列訓練
    --skip ablations    略過 ablation 實驗
    --skip advanced     略過 NGCF/BERT/Hetero/TimeDecay/SASRec
    --skip analysis     略過 stats_test / fairness
    --skip docs         略過文件產生
    --quick             所有訓練只跑 10 epoch（測試用）
"""
from __future__ import annotations
import argparse
import subprocess
import time
import sys
from pathlib import Path

PROJ = Path(__file__).parent

# 各階段的指令
STAGES = [
    # (key, description, command)
    ("preprocess",  "1/8 資料前處理（Excel → Parquet）",       ["python", "src/preprocess.py"]),
    ("splits",      "2/8 資料切分（k-core + train/val/test）", ["python", "-m", "src.dataset"]),
    ("reclean",     "2.5/8 重新清理 gender 資訊",              ["python", "-m", "src.reclean_gender"]),
    ("baselines",   "3/8 baseline 模型 (popular/itemcf)",      ["python", "run.py", "--model", "popular"]),
    ("baselines2",  "3.5/8 baseline (bprmf)",                   ["python", "run.py", "--model", "bprmf", "--epochs", "30"]),
    ("baselines3",  "3.6/8 baseline (itemcf)",                  ["python", "run.py", "--model", "itemcf"]),
    ("lightgcn",    "4/8 LightGCN 主模型",                      ["python", "run.py", "--model", "lightgcn", "--epochs", "60"]),
    ("lightgcn_si", "4.2/8 LightGCN-SI",                        ["python", "run.py", "--model", "lightgcn_si", "--epochs", "60"]),
    ("lightgcn_multi", "4.4/8 LightGCN-Multi",                  ["python", "run.py", "--model", "lightgcn_multi", "--epochs", "60"]),
    ("ngcf",        "5/8 NGCF 對照",                            ["python", "run.py", "--model", "ngcf", "--epochs", "40"]),
    ("timedecay",   "5.2/8 LightGCN-TimeDecay",                 ["python", "run.py", "--model", "lightgcn_timedecay", "--epochs", "40"]),
    ("bert_encode", "5.4/8 BERT 編碼書名（CPU 約 1 分鐘）",      ["python", "src/encode_books_bert.py"]),
    ("bert",        "5.5/8 LightGCN-BERT",                      ["python", "run.py", "--model", "lightgcn_bert", "--epochs", "40"]),
    ("hetero",      "5.6/8 LightGCN-Hetero",                    ["python", "run.py", "--model", "lightgcn_hetero", "--epochs", "40"]),
    ("sasrec",      "5.7/8 SASRec 序列模型",                    ["python", "-m", "src.train_sasrec", "--epochs", "40"]),
    ("ablations",   "6/8 4 個 ablation suites (multi-seed/hyperparam/side-info/reserve_weight)",
                    ["python", "-m", "src.run_experiments", "--suite", "all", "--epochs", "40"]),
    ("stats",       "7/8 統計顯著性檢驗",                       ["python", "-m", "src.stats_test"]),
    ("fairness",    "7.2/8 公平性分析",                         ["python", "-m", "src.fairness_analysis"]),
    ("ablation_analysis", "7.4/8 Ablation 分析圖表",            ["python", "-m", "src.analyze_ablations"]),
    ("visualize",   "7.6/8 t-SNE + 訓練曲線 + summary",         ["python", "src/visualize.py"]),
    ("docx",        "8/8 產生 Word 論文",                       ["python", "docs/build_docx.py"]),
    ("pptx",        "8.5/8 產生 PowerPoint 簡報",               ["python", "docs/build_pptx.py"]),
]

# 哪些 stage 屬於哪個 --skip 群組
SKIP_GROUPS = {
    "preprocess": ["preprocess"],
    "splits":     ["splits", "reclean"],
    "baselines":  ["baselines", "baselines2", "baselines3"],
    "lightgcn":   ["lightgcn", "lightgcn_si", "lightgcn_multi"],
    "advanced":   ["ngcf", "timedecay", "bert_encode", "bert", "hetero", "sasrec"],
    "ablations":  ["ablations"],
    "analysis":   ["stats", "fairness", "ablation_analysis", "visualize"],
    "docs":       ["docx", "pptx"],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip", action="append", default=[],
                    choices=list(SKIP_GROUPS.keys()),
                    help="略過某類階段（可多次指定）")
    ap.add_argument("--quick", action="store_true",
                    help="所有訓練只跑 10 epoch（用於測試流程）")
    ap.add_argument("--dry-run", action="store_true", help="只列出將執行的指令，不真的執行")
    args = ap.parse_args()

    skip_keys = set()
    for grp in args.skip:
        skip_keys.update(SKIP_GROUPS[grp])

    print("=" * 70)
    print(" 一鍵重現所有實驗 (run_all.py)")
    print("=" * 70)
    print(f" 略過：{args.skip if args.skip else '(無)'}")
    print(f" Quick mode：{args.quick}")
    print(f" Dry run：{args.dry_run}")
    print()

    total_start = time.time()
    failed = []
    for key, desc, cmd in STAGES:
        if key in skip_keys:
            print(f"[SKIP]  {desc}")
            continue
        # quick mode：訓練 epoch 改 10
        if args.quick:
            cmd = [c if c != "60" and c != "40" and c != "30" else "10" for c in cmd]
        print(f"\n{'#' * 70}\n# {desc}\n# $ {' '.join(cmd)}\n{'#' * 70}")
        if args.dry_run:
            continue
        t0 = time.time()
        try:
            subprocess.check_call(cmd, cwd=str(PROJ))
            print(f"\n[OK] {desc} ({time.time() - t0:.0f}s)")
        except subprocess.CalledProcessError as e:
            print(f"\n[FAIL] {desc} (exit {e.returncode})", file=sys.stderr)
            failed.append(desc)

    total = time.time() - total_start
    print("\n" + "=" * 70)
    print(f" 總耗時：{total / 60:.1f} 分鐘")
    if failed:
        print(f" 失敗階段：{len(failed)}")
        for f in failed:
            print(f"   - {f}")
    else:
        print(" 全部完成")
    print("=" * 70)


if __name__ == "__main__":
    main()
