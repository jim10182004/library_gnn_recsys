"""
評估指標：
  - Recall@K, Precision@K, NDCG@K, HitRate@K  (per-user 平均)
  - MRR@K  (Mean Reciprocal Rank)
  - Coverage@K  (推薦覆蓋了多少 % 的 catalog)
  - Diversity@K  (item 多樣性，由 1 - 平均共現相似度近似)
所有指標都會 mask 掉「使用者在 train 中已經借過的書」(避免推薦已看過的)。
"""
from __future__ import annotations
from collections import defaultdict
import numpy as np
import torch
import pandas as pd


def build_user_pos(df: pd.DataFrame) -> dict[int, set[int]]:
    """user -> set of item_ids"""
    out: dict[int, set[int]] = defaultdict(set)
    for u, i in zip(df["u"].values, df["i"].values):
        out[int(u)].add(int(i))
    return dict(out)


def evaluate_topk(
    model,
    eval_users: np.ndarray,
    user_train_pos: dict[int, set[int]],
    user_eval_pos: dict[int, set[int]],
    n_items: int,
    k_list: tuple[int, ...] = (10, 20),
    batch_size: int = 256,
    device: str = "cuda",
    is_torch: bool = True,
    *,
    item_pop: np.ndarray | None = None,  # for novelty
    return_topk: bool = False,           # 是否額外回傳 user → top-K item
) -> dict[str, float] | tuple[dict[str, float], dict[int, np.ndarray]]:
    """
    對 eval_users 計算 Top-K 指標。
    model 須有 get_all_ratings(batch_users) 方法。
    """
    max_k = max(k_list)
    metrics = {f"recall@{k}": [] for k in k_list}
    metrics.update({f"precision@{k}": [] for k in k_list})
    metrics.update({f"ndcg@{k}": [] for k in k_list})
    metrics.update({f"hit@{k}": [] for k in k_list})
    metrics.update({f"mrr@{k}": [] for k in k_list})

    # 為了算 Coverage 與 Diversity，需要記錄推薦過的所有 item
    recommended_items: dict[int, set[int]] = {k: set() for k in k_list}
    # 為了算 Novelty (1 - log popularity)
    novelty_records: dict[int, list[float]] = {k: [] for k in k_list}

    user_topk: dict[int, np.ndarray] = {}

    eval_users = np.asarray(eval_users)
    n = len(eval_users)
    for start in range(0, n, batch_size):
        users = eval_users[start : start + batch_size]
        if is_torch:
            u_t = torch.as_tensor(users, dtype=torch.long, device=device)
            scores = model.get_all_ratings(u_t).cpu().numpy()
        else:
            scores = model.get_all_ratings(users)

        for row, u in enumerate(users):
            seen = user_train_pos.get(int(u), set())
            if seen:
                scores[row, list(seen)] = -np.inf

        top_idx = np.argpartition(-scores, kth=max_k - 1, axis=1)[:, :max_k]
        order = np.argsort(-np.take_along_axis(scores, top_idx, axis=1), axis=1)
        top_idx_sorted = np.take_along_axis(top_idx, order, axis=1)

        for row, u in enumerate(users):
            gt = user_eval_pos.get(int(u), set())
            preds = top_idx_sorted[row]
            user_topk[int(u)] = preds

            if not gt:
                continue
            for k in k_list:
                pred_k = preds[:k]
                recommended_items[k].update(int(p) for p in pred_k)
                hits = np.array([p in gt for p in pred_k], dtype=np.float32)
                n_hits = int(hits.sum())
                metrics[f"recall@{k}"].append(n_hits / len(gt))
                metrics[f"precision@{k}"].append(n_hits / k)
                metrics[f"hit@{k}"].append(1.0 if n_hits > 0 else 0.0)
                # NDCG@K
                gains = hits / np.log2(np.arange(2, k + 2))
                dcg = gains.sum()
                idcg = (1.0 / np.log2(np.arange(2, min(len(gt), k) + 2))).sum()
                metrics[f"ndcg@{k}"].append(dcg / idcg if idcg > 0 else 0.0)
                # MRR@K
                rr = 0.0
                for rank, p in enumerate(pred_k, 1):
                    if p in gt:
                        rr = 1.0 / rank
                        break
                metrics[f"mrr@{k}"].append(rr)
                # Novelty: 1 - log(popularity_rank+1) / log(n_items)
                if item_pop is not None:
                    pop = item_pop[pred_k]
                    nov = 1.0 - np.log(pop + 2) / np.log(item_pop.max() + 2)
                    novelty_records[k].extend(nov.tolist())

    # 平均 per-user 指標
    out = {m: float(np.mean(v)) if v else 0.0 for m, v in metrics.items()}

    # Coverage@K：推薦過的不同 item 數 / 總 item 數
    for k in k_list:
        out[f"coverage@{k}"] = len(recommended_items[k]) / max(1, n_items)
    # Novelty@K
    if item_pop is not None:
        for k in k_list:
            out[f"novelty@{k}"] = float(np.mean(novelty_records[k])) if novelty_records[k] else 0.0

    if return_topk:
        return out, user_topk
    return out


def evaluate_cold_start_bins(
    model,
    eval_users: np.ndarray,
    user_train_pos: dict[int, set[int]],
    user_eval_pos: dict[int, set[int]],
    n_items: int,
    *,
    bins: tuple[tuple[int, int], ...] = ((1, 5), (6, 15), (16, 50), (51, 99999)),
    bin_labels: tuple[str, ...] | None = None,
    k_list: tuple[int, ...] = (10, 20),
    batch_size: int = 256,
    device: str = "cuda",
    is_torch: bool = True,
) -> dict[str, dict[str, float]]:
    """根據使用者在 train 中互動次數分箱，分別評估。"""
    if bin_labels is None:
        bin_labels = tuple(f"{lo}-{hi if hi < 99999 else '+'}" for lo, hi in bins)
    # 每個 user 的 train 互動數
    train_count = {u: len(s) for u, s in user_train_pos.items()}
    bin_users: dict[str, list[int]] = {b: [] for b in bin_labels}
    for u in eval_users:
        c = train_count.get(int(u), 0)
        for (lo, hi), label in zip(bins, bin_labels):
            if lo <= c <= hi:
                bin_users[label].append(int(u))
                break

    out: dict[str, dict[str, float]] = {}
    for label, users in bin_users.items():
        if len(users) == 0:
            out[label] = {"n_users": 0}
            continue
        m = evaluate_topk(model, np.array(users), user_train_pos, user_eval_pos,
                          n_items, k_list=k_list, batch_size=batch_size,
                          device=device, is_torch=is_torch)
        m["n_users"] = len(users)
        out[label] = m
    return out


def format_metrics(m: dict[str, float]) -> str:
    """主要指標的精簡顯示（不顯示 coverage/novelty 等）。"""
    keys = sorted(
        [k for k in m.keys() if any(k.startswith(p) for p in ("recall", "precision", "ndcg", "hit", "mrr"))],
        key=lambda k: (k.split("@")[0], int(k.split("@")[1])),
    )
    return "  ".join(f"{k}={m[k]:.4f}" for k in keys)
