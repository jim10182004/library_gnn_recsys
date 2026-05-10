"""
依序訓練進階模型：NGCF / TimeDecay / BERT / Hetero / SASRec。
跑在 ablation 完成之後。

執行：python -m src.train_advanced_models
"""
from __future__ import annotations
import subprocess
import time
import sys
from pathlib import Path

PROJ = Path(__file__).parent.parent

# (name, command_args)
MODELS = [
    ("NGCF",            ["python", "-u", "run.py", "--model", "ngcf", "--epochs", "40"]),
    ("LightGCN-TimeDecay", ["python", "-u", "run.py", "--model", "lightgcn_timedecay", "--epochs", "40"]),
    ("LightGCN-BERT",   ["python", "-u", "run.py", "--model", "lightgcn_bert", "--epochs", "40"]),
    ("LightGCN-Hetero", ["python", "-u", "run.py", "--model", "lightgcn_hetero", "--epochs", "40"]),
    ("SASRec",          ["python", "-u", "-m", "src.train_sasrec", "--epochs", "40"]),
]

def main():
    for name, cmd in MODELS:
        print(f"\n{'#'*70}\n# {name}\n{'#'*70}", flush=True)
        t0 = time.time()
        try:
            subprocess.check_call(cmd, cwd=str(PROJ))
            print(f"\n[{name}] DONE in {time.time()-t0:.0f}s")
        except subprocess.CalledProcessError as e:
            print(f"\n[{name}] FAILED with exit code {e.returncode}", file=sys.stderr)
            # 繼續跑下一個
            continue

if __name__ == "__main__":
    main()
