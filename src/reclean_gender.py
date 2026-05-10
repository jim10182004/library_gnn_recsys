"""
重新清理 users.parquet 的性別欄位。
原本 users.parquet 是用 drop_duplicates(keep='first') 建立，
若一位讀者第一次出現時 gender 欄為 None，後來其他借閱記錄即使有 gender 也不會用。

改進：對每位 user_orig，找全部記錄中最常見（mode）的非空 gender。
"""
from __future__ import annotations
from pathlib import Path
from collections import Counter
import pandas as pd
import pyarrow  # noqa: F401

PROJ = Path(__file__).parent.parent
PROC = PROJ / "data" / "processed"


def main():
    print("[Re-clean] 載入借閱、預約 ...")
    borrows = pd.read_parquet(PROC / "borrows.parquet")
    reservations = pd.read_parquet(PROC / "reservations.parquet")

    # 把所有借閱+預約紀錄合併，找每位 user_id 最常見的 gender
    cols = ["user_id", "gender", "age"]
    all_records = pd.concat([borrows[cols], reservations[cols]], ignore_index=True)

    print(f"[Re-clean] 全部紀錄：{len(all_records):,}")
    print(f"[Re-clean] 含非空 gender 的紀錄：{all_records['gender'].notna().sum():,}")

    # mode (非空) gender per user
    def best_gender(gs):
        gs_clean = [g for g in gs if pd.notna(g) and g in ("男", "女")]
        if not gs_clean:
            return None
        return Counter(gs_clean).most_common(1)[0][0]

    def best_age(ages):
        valid = [a for a in ages if pd.notna(a)]
        if not valid:
            return None
        # 取最常見值，age 通常一樣（同一人）
        return Counter(valid).most_common(1)[0][0]

    print("[Re-clean] 計算每位讀者的最常見 gender / age（這需要一些時間）...")
    # 注意：borrows/reservations 中的 user_id 已是「緊湊 id」(compact)
    grouped = all_records.groupby("user_id").agg({
        "gender": best_gender,
        "age": best_age,
    }).reset_index()
    # 為了對映回 users.parquet，這裡的 user_id 就是 users.parquet 的 user_id（緊湊 id）

    # 比較改進
    old_users = pd.read_parquet(PROC / "users.parquet")
    print(f"\n[改進前] gender 非空 user 數：{old_users['gender'].notna().sum():,} / {len(old_users):,}")
    print(f"[改進後 (compact)] gender 非空 user 數：{grouped['gender'].notna().sum():,} / {len(grouped):,}")

    # 用 user_id (緊湊) 對映回 users.parquet
    out = old_users[["user_orig", "user_id"]].merge(
        grouped[["user_id", "gender", "age"]],
        on="user_id", how="left",
    )
    print(f"[最終] gender 非空 user 數：{out['gender'].notna().sum():,} / {len(out):,}")
    delta = out['gender'].notna().sum() - old_users['gender'].notna().sum()
    print(f"[最終] 多救回 {delta:,} 位 user 的 gender")
    out.to_parquet(PROC / "users.parquet", index=False)
    print(f"\n[Re-clean] 已寫回 {PROC / 'users.parquet'}")


if __name__ == "__main__":
    main()
