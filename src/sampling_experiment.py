"""
比較不同負例採樣策略對 LightGCN 訓練的影響。

跑 4 種策略 × 1 個 seed × 25 epochs（縮短 epoch 加快實驗）：
  - uniform（baseline）
  - pop（α=0.75，按熱門度開根號）
  - category-aware（70% 機率抽同類別）
  - hard（pool=100，每個 epoch 重算）

輸出：results/ablation/sampling_strategies.csv

執行：python -m src.sampling_experiment
"""
from __future__ import annotations
import sys
import json
import time
from pathlib import Path

import pandas as pd  # noqa
import pyarrow  # noqa: F401
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.stdout.reconfigure(encoding="utf-8")

PROJECT = Path(__file__).resolve().parent.parent
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from src.dataset import load_splits
from src.evaluate import build_user_pos, evaluate_topk, format_metrics
from src.models.lightgcn import LightGCN, build_norm_adj
from src.sampling import get_sampler

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PROC = PROJECT / "data" / "processed"
RES = PROJECT / "results"
RES_AB = RES / "ablation"
RES_AB.mkdir(parents=True, exist_ok=True)


def train_with_sampler(splits, sampler_name, epochs=25, lr=1e-3, batch_size=4096,
                       embed_dim=64, n_layers=3, decay=1e-4, seed=42, **sampler_kwargs):
    torch.manual_seed(seed)
    np.random.seed(seed)

    train_u = splits.train["u"].values
    train_i = splits.train["i"].values
    user_train_pos = build_user_pos(splits.train)
    user_val_pos = build_user_pos(splits.val)
    user_test_pos = build_user_pos(splits.test)
    item_pop = np.bincount(train_i, minlength=splits.n_items).astype(np.float32)

    model = LightGCN(splits.n_users, splits.n_items, embed_dim, n_layers).to(DEVICE)
    A = build_norm_adj(
        torch.as_tensor(train_u, dtype=torch.long),
        torch.as_tensor(train_i, dtype=torch.long),
        splits.n_users, splits.n_items, device=DEVICE,
    )
    model.set_graph(A)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    # build sampler
    extra = {"item_pop": item_pop, "device": DEVICE, "model": model}
    extra.update(sampler_kwargs)
    sampler = get_sampler(sampler_name, train_u, train_i, splits.n_items,
                          user_train_pos, **extra)
    loader = DataLoader(sampler, batch_size=batch_size, shuffle=True, num_workers=0)

    print(f"\n=== {sampler.name} ===")
    t_start = time.time()

    for ep in range(1, epochs + 1):
        model.train()
        loss_sum = 0.0
        n_batch = 0
        for u, pi, ni in loader:
            u = u.to(DEVICE); pi = pi.to(DEVICE); ni = ni.to(DEVICE)
            loss, bpr = model.bpr_loss(u, pi, ni, decay=decay)
            opt.zero_grad()
            loss.backward()
            opt.step()
            loss_sum += loss.item()
            n_batch += 1

        # for hard-neg：每 5 epoch 更新模型 reference
        if sampler_name == "hard" and ep % 5 == 0 and hasattr(sampler, "set_model"):
            sampler.set_model(model)

        if ep % 5 == 0:
            train_loss = loss_sum / n_batch
            print(f"  ep {ep:3d}: loss={train_loss:.4f}")

    elapsed = time.time() - t_start

    # 最終 test 評估
    model.eval()
    eval_users = np.array(sorted(user_test_pos.keys()))
    test_m = evaluate_topk(
        model, eval_users, user_train_pos, user_test_pos,
        splits.n_items, device=DEVICE, is_torch=True,
        item_pop=item_pop,
    )
    print(f"  [TEST] {format_metrics(test_m)}  cov@10={test_m.get('coverage@10', 0):.4f}")
    print(f"  耗時：{elapsed:.0f} 秒")

    return {
        "sampler": sampler.name,
        "elapsed_sec": elapsed,
        **{k: float(v) for k, v in test_m.items()},
    }


def get_item_category(splits, books_path):
    """從 books.parquet 取得每個 compact_id 的 category（取第一個 digit）"""
    books = pd.read_parquet(books_path)
    inv_remap = {v: k for k, v in splits.item_remap.items()}
    cats = np.full(splits.n_items, 0, dtype=np.int64)
    for cid in range(splits.n_items):
        orig = inv_remap.get(cid)
        if orig is None:
            continue
        m = books[books["book_id"] == orig]
        if m.empty:
            continue
        c = str(m.iloc[0].get("category") or "").strip()
        if c and c[0].isdigit():
            cats[cid] = int(c[0])
    return cats


def main():
    print("=== Loading splits ===")
    splits = load_splits()
    item_cat = get_item_category(splits, PROC / "books.parquet")

    rows = []
    for name, kwargs in [
        ("uniform", {}),
        ("pop", {"alpha": 0.75}),
        ("category", {"item_category": item_cat, "prob_same_cat": 0.7}),
        # hard 跑得太慢（每 sample 1 個都要 forward 一次），跳過
        # ("hard", {"pool_size": 50}),
    ]:
        try:
            row = train_with_sampler(splits, name, epochs=25, **kwargs)
            rows.append(row)
        except Exception as e:
            print(f"[ERROR {name}] {e}")
            import traceback; traceback.print_exc()

    df = pd.DataFrame(rows)
    out = RES_AB / "sampling_strategies.csv"
    df.to_csv(out, index=False)
    print(f"\n[saved] {out}")
    print()
    print(df[["sampler", "recall@10", "recall@20", "ndcg@10", "coverage@10",
             "novelty@10", "elapsed_sec"]].to_string(index=False, float_format="%.4f"))


if __name__ == "__main__":
    main()
