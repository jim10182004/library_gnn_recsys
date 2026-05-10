"""
Dataset：載入借閱資料、過濾稀疏使用者/書籍、做 train/val/test 切分。

切分策略：時間序列切分（leakage-free）
  train: 2025-01 ~ 2025-10
  val:   2025-11
  test:  2025-12

過濾：
  - 移除借閱次數 < MIN_INTERACTIONS 的使用者
  - 移除被借次數 < MIN_INTERACTIONS 的書
  - 反覆過濾直到穩定（k-core）
"""
from __future__ import annotations
import json
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).parent.parent / "data"
PROCESSED = DATA_DIR / "processed"
SPLITS = DATA_DIR / "splits"
SPLITS.mkdir(parents=True, exist_ok=True)

MIN_INTERACTIONS = 5  # k-core 過濾門檻


@dataclass
class Splits:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    n_users: int
    n_items: int
    user_remap: dict        # 原 user_id -> 緊湊新 id
    item_remap: dict


def load_borrows() -> pd.DataFrame:
    df = pd.read_parquet(PROCESSED / "borrows.parquet")
    # 同一位讀者重複借同一本書，當作一條互動（保最早一次）
    df = df.sort_values("ts").drop_duplicates(subset=["user_id", "book_id"], keep="first")
    return df


def k_core_filter(df: pd.DataFrame, k: int = MIN_INTERACTIONS) -> pd.DataFrame:
    """反覆過濾，使每個 user 至少 k 個 item，每個 item 至少 k 個 user。"""
    prev_len = -1
    while len(df) != prev_len:
        prev_len = len(df)
        u_cnt = df["user_id"].value_counts()
        i_cnt = df["book_id"].value_counts()
        keep_u = u_cnt[u_cnt >= k].index
        keep_i = i_cnt[i_cnt >= k].index
        df = df[df["user_id"].isin(keep_u) & df["book_id"].isin(keep_i)]
    return df.reset_index(drop=True)


def remap_ids(df: pd.DataFrame) -> tuple[pd.DataFrame, dict, dict]:
    """把 user_id / book_id 重新編號為 0..N-1（緊湊）。"""
    u_unique = sorted(df["user_id"].unique())
    i_unique = sorted(df["book_id"].unique())
    user_remap = {u: i for i, u in enumerate(u_unique)}
    item_remap = {b: i for i, b in enumerate(i_unique)}
    df = df.copy()
    df["u"] = df["user_id"].map(user_remap).astype("int32")
    df["i"] = df["book_id"].map(item_remap).astype("int32")
    return df, user_remap, item_remap


def time_split(
    df: pd.DataFrame,
    val_start: str = "2025-11-01",
    test_start: str = "2025-12-01",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = df.copy()
    df["ts"] = pd.to_datetime(df["ts"])
    train = df[df["ts"] < val_start]
    val = df[(df["ts"] >= val_start) & (df["ts"] < test_start)]
    test = df[df["ts"] >= test_start]
    return train, val, test


def make_splits(
    k: int = MIN_INTERACTIONS,
    val_start: str = "2025-11-01",
    test_start: str = "2025-12-01",
    save: bool = True,
) -> Splits:
    print("[Dataset] 載入借閱資料 ...")
    df = load_borrows()
    print(f"   原始（去重後同一 user-item 互動）：{len(df):,}")

    print(f"[Dataset] k-core 過濾 (k={k}) ...")
    df = k_core_filter(df, k=k)
    print(f"   過濾後：{len(df):,}  users={df['user_id'].nunique():,}  items={df['book_id'].nunique():,}")

    df, user_remap, item_remap = remap_ids(df)

    train, val, test = time_split(df, val_start=val_start, test_start=test_start)
    print(f"[Split] train={len(train):,}  val={len(val):,}  test={len(test):,}")

    # 為了讓 val/test 可評估，僅保留 train 中見過的 user/item
    train_u = set(train["u"].unique())
    train_i = set(train["i"].unique())
    val = val[val["u"].isin(train_u) & val["i"].isin(train_i)].reset_index(drop=True)
    test = test[test["u"].isin(train_u) & test["i"].isin(train_i)].reset_index(drop=True)
    print(f"[Split] (僅留 train 見過的 u/i) val={len(val):,}  test={len(test):,}")

    splits = Splits(
        train=train.reset_index(drop=True),
        val=val,
        test=test,
        n_users=len(user_remap),
        n_items=len(item_remap),
        user_remap=user_remap,
        item_remap=item_remap,
    )

    if save:
        train[["u", "i", "ts"]].to_parquet(SPLITS / "train.parquet", index=False)
        val[["u", "i", "ts"]].to_parquet(SPLITS / "val.parquet", index=False)
        test[["u", "i", "ts"]].to_parquet(SPLITS / "test.parquet", index=False)
        meta = {
            "n_users": splits.n_users,
            "n_items": splits.n_items,
            "k_core": k,
            "val_start": val_start,
            "test_start": test_start,
        }
        with open(SPLITS / "meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)
        # remap 表存起來方便 demo 查 user/book 原 id
        pd.DataFrame(
            {"orig_user_id": list(user_remap.keys()), "u": list(user_remap.values())}
        ).to_parquet(SPLITS / "user_remap.parquet", index=False)
        pd.DataFrame(
            {"orig_book_id": list(item_remap.keys()), "i": list(item_remap.values())}
        ).to_parquet(SPLITS / "item_remap.parquet", index=False)
        print(f"[Save] 已存到 {SPLITS}")

    return splits


def load_splits() -> Splits:
    """從磁碟載入已切好的 splits（不含 user_remap，但有 n_users/n_items）。"""
    with open(SPLITS / "meta.json", encoding="utf-8") as f:
        meta = json.load(f)
    train = pd.read_parquet(SPLITS / "train.parquet")
    val = pd.read_parquet(SPLITS / "val.parquet")
    test = pd.read_parquet(SPLITS / "test.parquet")
    user_df = pd.read_parquet(SPLITS / "user_remap.parquet")
    item_df = pd.read_parquet(SPLITS / "item_remap.parquet")
    return Splits(
        train=train,
        val=val,
        test=test,
        n_users=meta["n_users"],
        n_items=meta["n_items"],
        user_remap=dict(zip(user_df["orig_user_id"], user_df["u"])),
        item_remap=dict(zip(item_df["orig_book_id"], item_df["i"])),
    )


if __name__ == "__main__":
    make_splits()
