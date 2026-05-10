"""
SimGCL 超參數調參：試多組 (eps, cl_weight, cl_temp) 看能否打敗預設值。

預設值（之前跑的）：eps=0.1, cl_weight=0.1, cl_temp=0.2 → 表現很差
    Test R@10 = 0.1506

這次試降低 noise 強度 + 對比 loss 權重。

執行：python -m src.simgcl_sweep
"""
from __future__ import annotations
import json
import time
from pathlib import Path
import pandas as pd  # noqa: F401
import pyarrow  # noqa: F401
import numpy as np
import torch
from torch.utils.data import DataLoader

PROJ = Path(__file__).parent.parent
ABL = PROJ / "results" / "ablation"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def main():
    from src.dataset import load_splits
    from src.evaluate import build_user_pos, evaluate_topk
    from src.models.simgcl import SimGCL
    from src.models.lightgcn import build_norm_adj
    from src.train import BPRDataset, set_all_seeds

    print("Loading splits ...")
    splits = load_splits()
    user_train_pos = build_user_pos(splits.train)
    user_val_pos = build_user_pos(splits.val)
    train_u = splits.train["u"].values
    train_i = splits.train["i"].values

    A_hat = build_norm_adj(
        torch.as_tensor(train_u, dtype=torch.long),
        torch.as_tensor(train_i, dtype=torch.long),
        splits.n_users, splits.n_items, device=DEVICE,
    )

    # 試的組合
    CONFIGS = [
        # (eps, cl_weight, cl_temp)
        (0.1, 0.1, 0.2),  # 預設（baseline）
        (0.05, 0.05, 0.2),
        (0.02, 0.02, 0.2),
        (0.01, 0.01, 0.5),
        (0.05, 0.01, 0.2),
        (0.1, 0.01, 0.5),
        (0.02, 0.001, 0.2),
    ]

    results = []
    for cfg_idx, (eps, cw, temp) in enumerate(CONFIGS):
        print(f"\n[{cfg_idx+1}/{len(CONFIGS)}] eps={eps} cl_weight={cw} cl_temp={temp}")
        set_all_seeds(42)
        model = SimGCL(
            splits.n_users, splits.n_items,
            embed_dim=64, n_layers=3,
            eps=eps, cl_weight=cw, cl_temperature=temp,
        ).to(DEVICE)
        model.set_graph(A_hat)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)

        ds = BPRDataset(train_u, train_i, splits.n_items, user_train_pos)
        loader = DataLoader(ds, batch_size=4096, shuffle=True, num_workers=0)

        t0 = time.time()
        for ep in range(20):  # 短訓練 20 epoch 看趨勢
            model.train()
            for u, pi, ni in loader:
                u = u.to(DEVICE); pi = pi.to(DEVICE); ni = ni.to(DEVICE)
                loss, _ = model.bpr_loss(u, pi, ni, decay=1e-4)
                opt.zero_grad(); loss.backward(); opt.step()

        model.eval()
        eval_users = np.array(sorted(user_val_pos.keys()))
        m = evaluate_topk(model, eval_users, user_train_pos, user_val_pos,
                          splits.n_items, device=DEVICE, is_torch=True)
        dt = time.time() - t0
        print(f"  R@20={m['recall@20']:.4f}  NDCG@10={m['ndcg@10']:.4f}  ({dt:.0f}s)")
        results.append({
            "eps": eps, "cl_weight": cw, "cl_temp": temp,
            "recall@10": m["recall@10"],
            "recall@20": m["recall@20"],
            "ndcg@10": m["ndcg@10"],
            "hit@10": m["hit@10"],
            "time_sec": dt,
        })

    df = pd.DataFrame(results).sort_values("recall@20", ascending=False)
    df.to_csv(ABL / "simgcl_sweep.csv", index=False)
    print()
    print("=" * 70)
    print("Sweep 結果（按 R@20 排序）")
    print("=" * 70)
    print(df.to_string(index=False))
    print()
    best = df.iloc[0]
    print(f"最佳：eps={best['eps']} cl_weight={best['cl_weight']} cl_temp={best['cl_temp']}")
    print(f"  R@20 = {best['recall@20']:.4f}")
    print(f"  vs LightGCN baseline (R@20 = 0.2977): {(best['recall@20'] - 0.2977)*100:+.2f}%")


if __name__ == "__main__":
    main()
