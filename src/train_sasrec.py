"""
SASRec 訓練腳本（與 train.py 相容的評估）

執行：python -m src.train_sasrec
"""
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path

import pandas as pd  # noqa: F401
import pyarrow  # noqa: F401
import numpy as np
import torch
from torch.utils.data import DataLoader

from src.dataset import load_splits
from src.evaluate import build_user_pos, evaluate_topk, evaluate_cold_start_bins, format_metrics
from src.models.sasrec import SASRec, SASRecDataset, build_sequences

PROJ = Path(__file__).parent.parent
CKPT = PROJ / "checkpoints"
RESULT = PROJ / "results"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--embed-dim", type=int, default=64)
    ap.add_argument("--n-blocks", type=int, default=2)
    ap.add_argument("--n-heads", type=int, default=2)
    ap.add_argument("--max-len", type=int, default=50)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--eval-every", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    np.random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)

    print(f"=== Loading splits ===")
    splits = load_splits()
    print(f"users={splits.n_users:,}  items={splits.n_items:,}")

    # 建立序列
    user_seq, raw_seq = build_sequences(splits, max_len=args.max_len)
    user_train_pos = build_user_pos(splits.train)
    user_val_pos = build_user_pos(splits.val)
    user_test_pos = build_user_pos(splits.test)
    train_i = splits.train["i"].values
    item_pop = np.bincount(train_i, minlength=splits.n_items).astype(np.float32)

    DEV = args.device
    model = SASRec(
        n_items=splits.n_items, embed_dim=args.embed_dim,
        max_len=args.max_len, n_blocks=args.n_blocks,
        n_heads=args.n_heads, dropout=args.dropout,
    ).to(DEV)
    model.set_user_sequences(user_seq.to(DEV))

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    ds = SASRecDataset(raw_seq, splits.n_items, max_len=args.max_len)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=0)

    history = []
    best_val = -1.0
    best_state = None

    for ep in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        loss_sum, n_batch = 0.0, 0
        for inp, pos, neg in loader:
            inp = inp.to(DEV); pos = pos.to(DEV); neg = neg.to(DEV)
            loss = model.forward_train(inp, pos, neg)
            opt.zero_grad(); loss.backward(); opt.step()
            loss_sum += loss.item(); n_batch += 1
        train_loss = loss_sum / max(1, n_batch)
        rec = {"epoch": ep, "train_loss": train_loss, "epoch_sec": time.time() - t0}

        if ep % args.eval_every == 0 or ep == args.epochs:
            model.eval()
            eval_users = np.array(sorted(user_val_pos.keys()))
            val_m = evaluate_topk(model, eval_users, user_train_pos, user_val_pos,
                                  splits.n_items, device=DEV, is_torch=True, item_pop=item_pop)
            rec["val"] = val_m
            print(f"[ep {ep:3d}] loss={train_loss:.4f}  {format_metrics(val_m)}  ({rec['epoch_sec']:.1f}s)")
            if val_m["recall@20"] > best_val:
                best_val = val_m["recall@20"]
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            print(f"[ep {ep:3d}] loss={train_loss:.4f}  ({rec['epoch_sec']:.1f}s)")
        history.append(rec)

    if best_state is not None:
        model.load_state_dict(best_state)
        torch.save(best_state, CKPT / "sasrec_best.pt")
    model.eval()
    eval_users = np.array(sorted(user_test_pos.keys()))
    test_m = evaluate_topk(model, eval_users, user_train_pos, user_test_pos,
                           splits.n_items, device=DEV, is_torch=True, item_pop=item_pop)
    cs = evaluate_cold_start_bins(model, eval_users, user_train_pos, user_test_pos,
                                  splits.n_items, device=DEV, is_torch=True)
    print(f"\n[TEST] {format_metrics(test_m)}")
    print(f"[TEST] coverage@10={test_m.get('coverage@10', 0):.4f}")

    out = {"history": history, "test": test_m, "cold_start": cs}
    with open(RESULT / "sasrec_history.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
