"""
頂層執行腳本

範例：
    python run.py --model popular
    python run.py --model itemcf
    python run.py --model bprmf --epochs 30
    python run.py --model lightgcn --epochs 50 --n-layers 3
"""
# Windows: 必須先 import pandas/pyarrow，再 import torch，否則 read_parquet crash
import pandas  # noqa: F401
import pyarrow  # noqa: F401

from src.train import main

if __name__ == "__main__":
    main()
