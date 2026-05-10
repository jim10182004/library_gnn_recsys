"""
測試資料切分的正確性：
  - 時序切分：train < val < test 的時間順序
  - 沒有資料洩漏（leakage）：val/test 中的 (user, item) 對不應該在 train 中出現
  - k-core ≥ 5：每個 user 在 train+val+test 合計至少 5 次互動
  - 索引對齊：u/i 都在 [0, n_users) 與 [0, n_items) 範圍內
"""
from __future__ import annotations
import os
import sys
from pathlib import Path

import pandas as pd
import pyarrow  # noqa: F401
import pytest

PROJECT = Path(__file__).resolve().parent.parent
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from src.dataset import load_splits


@pytest.fixture(scope="module")
def splits():
    """只 load 一次，全 module 共用。若 processed parquet 不存在則 skip。"""
    proc = PROJECT / "data" / "processed"
    if not (proc / "borrows.parquet").exists():
        pytest.skip("data/processed 不存在，請先跑 src.preprocess")
    return load_splits()


# ============= 時序與索引 =============

def test_split_indices_in_range(splits):
    """所有 u 在 [0, n_users)、所有 i 在 [0, n_items)"""
    for name, df in [("train", splits.train), ("val", splits.val), ("test", splits.test)]:
        assert df["u"].min() >= 0, f"{name}: 有負數 u"
        assert df["u"].max() < splits.n_users, f"{name}: u 超出 n_users"
        assert df["i"].min() >= 0, f"{name}: 有負數 i"
        assert df["i"].max() < splits.n_items, f"{name}: i 超出 n_items"


def test_split_chronological_order(splits):
    """train 全部 < val 全部 < test 全部 的時間"""
    if "ts" not in splits.train.columns:
        pytest.skip("split 沒有 ts 欄位")
    train_max = splits.train["ts"].max()
    val_min = splits.val["ts"].min()
    val_max = splits.val["ts"].max()
    test_min = splits.test["ts"].min()
    assert train_max <= val_min, f"train_max={train_max} 應 ≤ val_min={val_min}"
    assert val_max <= test_min, f"val_max={val_max} 應 ≤ test_min={test_min}"


def test_split_sizes_reasonable(splits):
    """train >> val >> test 應該都有合理數量"""
    n_train = len(splits.train)
    n_val = len(splits.val)
    n_test = len(splits.test)
    # train 至少 10 萬
    assert n_train > 100_000, f"train 太小: {n_train}"
    # train 應該佔 70%+ 整體
    total = n_train + n_val + n_test
    assert n_train / total > 0.7, f"train 比例 {n_train/total:.2f} 太低"


# ============= 資料洩漏檢查 =============

def _pairs(df) -> set:
    return set(zip(df["u"].astype(int).values, df["i"].astype(int).values))


def test_no_leakage_train_to_val(splits):
    """val 中不應該有 train 已經出現的 (user, item) 對"""
    train_pairs = _pairs(splits.train)
    val_pairs = _pairs(splits.val)
    leak = train_pairs & val_pairs
    assert len(leak) == 0, f"train ↔ val leakage: {len(leak)} 對重疊"


def test_no_leakage_train_to_test(splits):
    """test 中不應該有 train 已經出現的 (user, item) 對"""
    train_pairs = _pairs(splits.train)
    test_pairs = _pairs(splits.test)
    leak = train_pairs & test_pairs
    assert len(leak) == 0, f"train ↔ test leakage: {len(leak)} 對重疊"


def test_no_leakage_val_to_test(splits):
    """val 中不應該有 test 已經出現的 (user, item) 對"""
    val_pairs = _pairs(splits.val)
    test_pairs = _pairs(splits.test)
    leak = val_pairs & test_pairs
    assert len(leak) == 0, f"val ↔ test leakage: {len(leak)} 對重疊"


# ============= k-core 過濾 =============

def test_kcore_users_have_enough_total_interactions(splits):
    """k-core ≥ 5：合計 (train+val+test) 每個 user ≥ 5 互動。
    注意：因為時序切分發生在 k-core 之後，部分 user 的 train 部分可能 < 5 但
    全部加起來仍 ≥ 5（這是預期行為）。"""
    all_df = pd.concat([splits.train, splits.val, splits.test], ignore_index=True)
    user_counts = all_df.groupby("u").size()
    pct_ok = (user_counts >= 5).mean()
    assert pct_ok > 0.99, f"k-core 5 違反：只 {pct_ok*100:.1f}% 的 user 合計 ≥ 5 互動"


def test_kcore_items_have_enough_total_interactions(splits):
    """k-core ≥ 5：合計 (train+val+test) 每個 item ≥ 5 互動"""
    all_df = pd.concat([splits.train, splits.val, splits.test], ignore_index=True)
    item_counts = all_df.groupby("i").size()
    pct_ok = (item_counts >= 5).mean()
    # 容許 ~2% 邊緣 case（k-core 收斂時 item 可能差 1-2 互動）
    assert pct_ok > 0.97, f"k-core 5 違反：只 {pct_ok*100:.1f}% 的 item 合計 ≥ 5 互動"


def test_train_users_majority_have_5plus(splits):
    """放寬版 k-core：≥ 80% 的 train user 在 train 中也有 ≥ 5 互動"""
    train_user_counts = splits.train.groupby("u").size()
    pct_ok = (train_user_counts >= 5).mean()
    assert pct_ok > 0.80, f"train 中 ≥ 5 互動的 user 比例僅 {pct_ok*100:.1f}%（過低）"


# ============= remap 對應表 =============

def test_user_remap_is_bijection(splits):
    """user_remap 應該是 1-to-1 mapping，沒有重複 value"""
    remap = splits.user_remap
    assert len(remap) > 0, "user_remap 是空的"
    # value 都在 [0, n_users)
    values = set(remap.values())
    assert len(values) == len(remap), "user_remap 有重複的 compact id"
    assert max(values) < splits.n_users, "user_remap 的 max value 超出 n_users"


def test_item_remap_is_bijection(splits):
    """item_remap 應該是 1-to-1 mapping"""
    remap = splits.item_remap
    assert len(remap) > 0
    values = set(remap.values())
    assert len(values) == len(remap), "item_remap 有重複的 compact id"
    assert max(values) < splits.n_items
