"""
訓練 LightGCN 與 baselines，並評估在 val/test 上的表現。

執行：
    python run.py --model lightgcn --epochs 100
    python run.py --model lightgcn_si --side-info gender,age,category
    python run.py --model lightgcn --seed 42
    python run.py --model lightgcn --embed-dim 128 --n-layers 4 --tag d128_l4
"""
from __future__ import annotations
import argparse
import json
import random
import sys
import time
from pathlib import Path

# Windows: 必須先 import pandas/pyarrow，再 import torch，否則 read_parquet crash
import pandas as pd  # noqa: F401
import pyarrow  # noqa: F401
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from src.dataset import load_splits
from src.evaluate import build_user_pos, evaluate_topk, evaluate_cold_start_bins, format_metrics
from src.models.lightgcn import LightGCN, build_norm_adj
from src.models.lightgcn_si import LightGCNSI, build_side_info_tensors
from src.models.lightgcn_multi import LightGCNMulti, build_multi_edges, build_norm_adj_weighted
from src.models.lightgcn_bert import LightGCNBert, load_bert_tensor
from src.models.lightgcn_hetero import LightGCNHetero, build_hetero_adj
from src.models.ngcf import NGCF
from src.models.simgcl import SimGCL
from src.models.lightgcn_tgn import LightGCNTGN, compute_recency
from src.models.lightgcn_cover import LightGCNCover, load_cover_tensors
from src.models.time_decay import build_time_decayed_edges
from src.models.baselines import PopularRecommender, ItemCF, BPRMF


CKPT_DIR = Path(__file__).parent.parent / "checkpoints"
RESULT_DIR = Path(__file__).parent.parent / "results"
CKPT_DIR.mkdir(exist_ok=True)
RESULT_DIR.mkdir(exist_ok=True)


def set_all_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class BPRDataset(Dataset):
    """每筆樣本 = (user, pos_item, neg_item)，neg_item 從未互動過的書隨機採。"""

    def __init__(self, train_u: np.ndarray, train_i: np.ndarray, n_items: int, user_pos: dict[int, set[int]]):
        self.u = train_u.astype(np.int64)
        self.i = train_i.astype(np.int64)
        self.n_items = n_items
        self.user_pos = user_pos

    def __len__(self):
        return len(self.u)

    def __getitem__(self, idx):
        u = self.u[idx]
        pi = self.i[idx]
        seen = self.user_pos[int(u)]
        while True:
            ni = np.random.randint(0, self.n_items)
            if ni not in seen:
                return u, pi, ni


def train_neural(
    model,
    splits,
    *,
    epochs: int = 50,
    lr: float = 1e-3,
    batch_size: int = 4096,
    decay: float = 1e-4,
    eval_every: int = 5,
    device: str = "cuda",
    extra_setup=None,
    name: str = "model",
    cold_start_eval: bool = True,
    item_pop: np.ndarray | None = None,
):
    train_u = splits.train["u"].values
    train_i = splits.train["i"].values
    user_train_pos = build_user_pos(splits.train)
    user_val_pos = build_user_pos(splits.val)
    user_test_pos = build_user_pos(splits.test)

    if item_pop is None:
        item_pop = np.bincount(train_i, minlength=splits.n_items).astype(np.float32)

    if extra_setup is not None:
        extra_setup(model, train_u, train_i, splits.n_users, splits.n_items, device)

    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    ds = BPRDataset(train_u, train_i, splits.n_items, user_train_pos)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0)

    best_val = -1.0
    best_state = None
    history = []

    for ep in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        loss_sum = 0.0
        bpr_sum = 0.0
        n_batch = 0
        for u, pi, ni in loader:
            u = u.to(device)
            pi = pi.to(device)
            ni = ni.to(device)
            loss, bpr = model.bpr_loss(u, pi, ni, decay=decay)
            opt.zero_grad()
            loss.backward()
            opt.step()
            loss_sum += loss.item()
            bpr_sum += bpr.item()
            n_batch += 1

        train_loss = loss_sum / n_batch
        train_bpr = bpr_sum / n_batch
        rec = {
            "epoch": ep,
            "train_loss": train_loss,
            "train_bpr": train_bpr,
            "epoch_sec": time.time() - t0,
        }

        if ep % eval_every == 0 or ep == epochs:
            model.eval()
            eval_users = np.array(sorted(user_val_pos.keys()))
            val_m = evaluate_topk(
                model, eval_users, user_train_pos, user_val_pos,
                splits.n_items, device=device, is_torch=True,
                item_pop=item_pop,
            )
            rec["val"] = val_m
            print(f"[ep {ep:3d}] loss={train_loss:.4f}  bpr={train_bpr:.4f}  {format_metrics(val_m)}  ({rec['epoch_sec']:.1f}s)")

            if val_m["recall@20"] > best_val:
                best_val = val_m["recall@20"]
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            print(f"[ep {ep:3d}] loss={train_loss:.4f}  bpr={train_bpr:.4f}  ({rec['epoch_sec']:.1f}s)")

        history.append(rec)

    if best_state is not None:
        model.load_state_dict(best_state)
        torch.save(best_state, CKPT_DIR / f"{name}_best.pt")
    model.eval()
    eval_users = np.array(sorted(user_test_pos.keys()))
    test_m = evaluate_topk(
        model, eval_users, user_train_pos, user_test_pos,
        splits.n_items, device=device, is_torch=True, item_pop=item_pop,
    )
    print(f"\n[TEST] {format_metrics(test_m)}")
    print(f"[TEST] coverage@10={test_m.get('coverage@10', 0):.4f}  "
          f"novelty@10={test_m.get('novelty@10', 0):.4f}")

    out = {"history": history, "test": test_m}

    if cold_start_eval:
        cs = evaluate_cold_start_bins(
            model, eval_users, user_train_pos, user_test_pos,
            splits.n_items, device=device, is_torch=True,
        )
        out["cold_start"] = cs
        print("\n[COLD-START 分箱（按 train 互動次數）]")
        for label, m in cs.items():
            n = m.get("n_users", 0)
            if n == 0:
                print(f"  {label:10s}: (no users)")
            else:
                print(f"  {label:10s} (n={n:5d}): "
                      f"R@10={m.get('recall@10', 0):.4f}  "
                      f"NDCG@10={m.get('ndcg@10', 0):.4f}  "
                      f"Hit@10={m.get('hit@10', 0):.4f}")

    with open(RESULT_DIR / f"{name}_history.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    return model, test_m


def setup_lightgcn_graph(model, train_u, train_i, n_users, n_items, device):
    A_hat = build_norm_adj(
        torch.as_tensor(train_u, dtype=torch.long),
        torch.as_tensor(train_i, dtype=torch.long),
        n_users, n_items, device=device,
    )
    model.set_graph(A_hat)


def setup_lightgcn_si(model, train_u, train_i, n_users, n_items, device, *, splits, books_df, users_df):
    setup_lightgcn_graph(model, train_u, train_i, n_users, n_items, device)
    g, a, c, _ = build_side_info_tensors(splits, books_df, users_df)
    model.set_side_info(g.to(device), a.to(device), c.to(device))


def run_classical(name: str, splits, *, item_pop=None):
    user_train_pos = build_user_pos(splits.train)
    user_val_pos = build_user_pos(splits.val)
    user_test_pos = build_user_pos(splits.test)
    train_u = splits.train["u"].values
    train_i = splits.train["i"].values

    if item_pop is None:
        item_pop = np.bincount(train_i, minlength=splits.n_items).astype(np.float32)

    if name == "popular":
        model = PopularRecommender()
        model.fit(train_u, train_i, splits.n_items)
    elif name == "itemcf":
        model = ItemCF()
        model.fit(train_u, train_i, splits.n_users, splits.n_items)
    else:
        raise ValueError(name)

    eval_users = np.array(sorted(user_test_pos.keys()))
    test_m = evaluate_topk(
        model, eval_users, user_train_pos, user_test_pos,
        splits.n_items, is_torch=False, item_pop=item_pop,
    )
    cs = evaluate_cold_start_bins(
        model, eval_users, user_train_pos, user_test_pos,
        splits.n_items, is_torch=False,
    )
    print(f"[TEST] {name}: {format_metrics(test_m)}")
    print(f"[TEST] coverage@10={test_m.get('coverage@10', 0):.4f}  "
          f"novelty@10={test_m.get('novelty@10', 0):.4f}")
    print("\n[COLD-START 分箱]")
    for label, m in cs.items():
        n = m.get("n_users", 0)
        if n == 0:
            continue
        print(f"  {label:10s} (n={n:5d}): "
              f"R@10={m.get('recall@10', 0):.4f}  "
              f"NDCG@10={m.get('ndcg@10', 0):.4f}")

    with open(RESULT_DIR / f"{name}_history.json", "w", encoding="utf-8") as f:
        json.dump({"test": test_m, "cold_start": cs}, f, indent=2, ensure_ascii=False)
    return test_m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default="lightgcn",
                    choices=["lightgcn", "lightgcn_si", "lightgcn_multi", "lightgcn_bert",
                             "lightgcn_timedecay", "lightgcn_hetero", "ngcf", "simgcl",
                             "lightgcn_tgn", "lightgcn_cover",
                             "bprmf", "itemcf", "popular"])
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--embed-dim", type=int, default=64)
    ap.add_argument("--n-layers", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int, default=4096)
    ap.add_argument("--decay", type=float, default=1e-4)
    ap.add_argument("--eval-every", type=int, default=5)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tag", type=str, default="",
                    help="附加在輸出檔名上的標籤，例如 'seed1' 或 'd128_l4'")
    ap.add_argument("--side-info", type=str, default="gender,age,category",
                    help="LightGCN-SI 中啟用哪些 side info（逗號分隔）")
    ap.add_argument("--reserve-weight", type=float, default=0.5,
                    help="LightGCN-Multi 中預約邊的權重")
    ap.add_argument("--decay-lambda", type=float, default=0.005,
                    help="LightGCN-TimeDecay 中時間衰減的 λ")
    args = ap.parse_args()

    set_all_seeds(args.seed)

    print(f"=== Loading splits (seed={args.seed}) ===")
    splits = load_splits()
    print(f"users={splits.n_users:,}  items={splits.n_items:,}")
    print(f"train={len(splits.train):,}  val={len(splits.val):,}  test={len(splits.test):,}")

    name = args.model + (f"_{args.tag}" if args.tag else "")

    if args.model in ("popular", "itemcf"):
        # baselines 不需要 seed/tag
        run_classical(args.model, splits)
        return

    if args.model == "lightgcn":
        model = LightGCN(splits.n_users, splits.n_items, args.embed_dim, args.n_layers)
        extra = setup_lightgcn_graph
    elif args.model == "simgcl":
        model = SimGCL(splits.n_users, splits.n_items, args.embed_dim, args.n_layers)
        extra = setup_lightgcn_graph
    elif args.model == "lightgcn_tgn":
        model = LightGCNTGN(splits.n_users, splits.n_items, args.embed_dim, args.n_layers)
        def extra(m, tu, ti, nu, ni, dev):
            setup_lightgcn_graph(m, tu, ti, nu, ni, dev)
            ur, ir = compute_recency(splits)
            m.set_recency(ur.to(dev), ir.to(dev))
    elif args.model == "lightgcn_cover":
        model = LightGCNCover(splits.n_users, splits.n_items,
                              cover_dim=512, embed_dim=args.embed_dim, n_layers=args.n_layers)
        def extra(m, tu, ti, nu, ni, dev):
            setup_lightgcn_graph(m, tu, ti, nu, ni, dev)
            from pathlib import Path as _P
            cover, has = load_cover_tensors(
                _P(__file__).parent.parent / "data" / "processed" / "book_covers.parquet",
                ni, splits.item_remap)
            m.set_covers(cover.to(dev), has.to(dev))
    elif args.model == "lightgcn_si":
        from pathlib import Path as _P
        proc = _P(__file__).parent.parent / "data" / "processed"
        books_df = pd.read_parquet(proc / "books.parquet")
        users_df = pd.read_parquet(proc / "users.parquet")
        # 解析 side-info 啟用旗標
        enabled = {x.strip() for x in args.side_info.split(",") if x.strip()}
        use_g = "gender" in enabled
        use_a = "age" in enabled
        use_c = "category" in enabled
        model = LightGCNSI(
            splits.n_users, splits.n_items,
            n_genders=3 if use_g else 1,
            n_age_buckets=8 if use_a else 1,
            n_categories=11 if use_c else 1,
            embed_dim=args.embed_dim, n_layers=args.n_layers,
        )
        def extra(m, tu, ti, nu, ni, dev):
            setup_lightgcn_graph(m, tu, ti, nu, ni, dev)
            g, a, c, _ = build_side_info_tensors(splits, books_df, users_df)
            # 若 disabled，把對應索引全設 0（單一類別 → embedding 等價於常數，幾乎無作用）
            if not use_g:
                g = torch.zeros_like(g)
            if not use_a:
                a = torch.zeros_like(a)
            if not use_c:
                c = torch.zeros_like(c)
            m.set_side_info(g.to(dev), a.to(dev), c.to(dev))
    elif args.model == "lightgcn_multi":
        from pathlib import Path as _P
        proc = _P(__file__).parent.parent / "data" / "processed"
        books_df = pd.read_parquet(proc / "books.parquet")
        users_df = pd.read_parquet(proc / "users.parquet")
        reservations_df = pd.read_parquet(proc / "reservations.parquet")
        model = LightGCNMulti(
            splits.n_users, splits.n_items,
            embed_dim=args.embed_dim, n_layers=args.n_layers,
            use_side_info=True,
        )
        rw = args.reserve_weight
        def extra(m, tu, ti, nu, ni, dev):
            eu, ei, ew = build_multi_edges(splits, reservations_df,
                                           borrow_weight=1.0, reserve_weight=rw)
            A_hat = build_norm_adj_weighted(eu, ei, ew, nu, ni, device=dev)
            m.set_graph(A_hat)
            g, a, c, _ = build_side_info_tensors(splits, books_df, users_df)
            m.set_side_info(g.to(dev), a.to(dev), c.to(dev))
    elif args.model == "lightgcn_timedecay":
        from pathlib import Path as _P
        proc = _P(__file__).parent.parent / "data" / "processed"
        books_df = pd.read_parquet(proc / "books.parquet")
        users_df = pd.read_parquet(proc / "users.parquet")
        reservations_df = pd.read_parquet(proc / "reservations.parquet")
        model = LightGCNMulti(
            splits.n_users, splits.n_items,
            embed_dim=args.embed_dim, n_layers=args.n_layers,
            use_side_info=True,
        )
        decay_lambda = args.decay_lambda
        def extra(m, tu, ti, nu, ni, dev):
            eu, ei, ew = build_time_decayed_edges(
                splits, reservations_df, decay_lambda=decay_lambda,
            )
            A_hat = build_norm_adj_weighted(eu, ei, ew, nu, ni, device=dev)
            m.set_graph(A_hat)
            g, a, c, _ = build_side_info_tensors(splits, books_df, users_df)
            m.set_side_info(g.to(dev), a.to(dev), c.to(dev))
    elif args.model == "lightgcn_bert":
        from pathlib import Path as _P
        proc = _P(__file__).parent.parent / "data" / "processed"
        books_df = pd.read_parquet(proc / "books.parquet")
        users_df = pd.read_parquet(proc / "users.parquet")
        bert_path = proc / "book_bert.parquet"
        if not bert_path.exists():
            print(f"[ERROR] {bert_path} 不存在，請先執行 python src/encode_books_bert.py")
            sys.exit(1)
        # 嘗試 load 一條 vec 取得 bert_dim
        sample = pd.read_parquet(bert_path).iloc[:1]
        bert_dim = len([c for c in sample.columns if c.startswith("v")])
        model = LightGCNBert(
            splits.n_users, splits.n_items,
            bert_dim=bert_dim,
            embed_dim=args.embed_dim, n_layers=args.n_layers,
        )
        def extra(m, tu, ti, nu, ni, dev):
            setup_lightgcn_graph(m, tu, ti, nu, ni, dev)
            g, a, c, _ = build_side_info_tensors(splits, books_df, users_df)
            m.set_side_info(g.to(dev), a.to(dev), c.to(dev))
            bert_t = load_bert_tensor(bert_path, ni, splits.item_remap)
            m.set_bert(bert_t.to(dev))
    elif args.model == "ngcf":
        model = NGCF(splits.n_users, splits.n_items,
                     embed_dim=args.embed_dim, n_layers=args.n_layers)
        extra = setup_lightgcn_graph
    elif args.model == "lightgcn_hetero":
        from pathlib import Path as _P
        proc = _P(__file__).parent.parent / "data" / "processed"
        books_df = pd.read_parquet(proc / "books.parquet")
        # 先建異質圖以拿到 n_authors
        A_hat_tmp, n_authors, _ = build_hetero_adj(splits, books_df, device="cpu")
        model = LightGCNHetero(
            splits.n_users, splits.n_items, n_authors,
            embed_dim=args.embed_dim, n_layers=args.n_layers,
        )
        def extra(m, tu, ti, nu, ni, dev):
            A_hat, _, _ = build_hetero_adj(splits, books_df, device=dev)
            m.set_graph(A_hat)
    else:  # bprmf
        model = BPRMF(splits.n_users, splits.n_items, args.embed_dim)
        extra = None

    train_neural(
        model, splits,
        epochs=args.epochs, lr=args.lr,
        batch_size=args.batch_size, decay=args.decay,
        eval_every=args.eval_every, device=args.device,
        extra_setup=extra, name=name,
    )


if __name__ == "__main__":
    main()
