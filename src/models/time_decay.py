"""
時間衰減 edge weight：較近的借閱獲得較高權重。

權重函數：w(t) = exp(-λ * (t_cutoff - t) / 86400)
  λ=0 → 不衰減（等於 LightGCN-Multi）
  λ=0.01 → 半衰期約 70 天
  λ=0.005 → 半衰期約 140 天
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import torch


def build_time_decayed_edges(
    splits,
    reservations_df: pd.DataFrame,
    *,
    decay_lambda: float = 0.005,
    borrow_base: float = 1.0,
    reserve_base: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """回傳 (edges_u, edges_i, edges_w) 加權邊。"""
    train = splits.train.copy()
    train["ts"] = pd.to_datetime(train["ts"])
    cutoff = train["ts"].max()

    # 借閱邊
    days_diff = (cutoff - train["ts"]).dt.total_seconds() / 86400.0
    bw = borrow_base * np.exp(-decay_lambda * days_diff.values)

    bu = train["u"].values.astype(np.int64)
    bi = train["i"].values.astype(np.int64)

    # 預約邊（沿用 build_multi_edges 邏輯，但加時間衰減）
    res = reservations_df[
        reservations_df["user_id"].isin(splits.user_remap)
        & reservations_df["book_id"].isin(splits.item_remap)
    ].copy()
    res["u"] = res["user_id"].map(splits.user_remap).astype("int64")
    res["i"] = res["book_id"].map(splits.item_remap).astype("int64")
    res["ts"] = pd.to_datetime(res["ts"])
    res = res[res["ts"] <= cutoff]
    res = res.drop_duplicates(subset=["u", "i"])

    # 去掉與借閱重複的
    borrow_set = set(zip(bu.tolist(), bi.tolist()))
    res = res[~res.apply(lambda r: (r["u"], r["i"]) in borrow_set, axis=1)]

    if len(res) > 0:
        days_r = (cutoff - res["ts"]).dt.total_seconds() / 86400.0
        rw = reserve_base * np.exp(-decay_lambda * days_r.values)
        ru = res["u"].values.astype(np.int64)
        ri = res["i"].values.astype(np.int64)
    else:
        ru = np.array([], dtype=np.int64)
        ri = np.array([], dtype=np.int64)
        rw = np.array([], dtype=np.float32)

    edges_u = np.concatenate([bu, ru])
    edges_i = np.concatenate([bi, ri])
    edges_w = np.concatenate([bw, rw]).astype(np.float32)

    if len(rw):
        print(f"[time-decay] lambda={decay_lambda}  borrow_w in [{bw.min():.3f},{bw.max():.3f}]  "
              f"reserve_w in [{rw.min():.3f},{rw.max():.3f}]")
    else:
        print(f"[time-decay] lambda={decay_lambda}  borrow_w in [{bw.min():.3f},{bw.max():.3f}]")

    return (
        torch.from_numpy(edges_u),
        torch.from_numpy(edges_i),
        torch.from_numpy(edges_w),
    )
