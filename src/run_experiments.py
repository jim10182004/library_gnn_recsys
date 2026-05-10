"""
自動跑批次實驗：multi-seed、超參數、side-info ablation。

執行：
    python -m src.run_experiments --suite multi_seed
    python -m src.run_experiments --suite hyperparam
    python -m src.run_experiments --suite side_info
    python -m src.run_experiments --suite reserve_weight
    python -m src.run_experiments --suite all
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

from src.dataset import load_splits
from src.evaluate import build_user_pos, evaluate_topk, format_metrics
from src.models.lightgcn import LightGCN, build_norm_adj
from src.models.lightgcn_si import LightGCNSI, build_side_info_tensors
from src.models.lightgcn_multi import LightGCNMulti, build_multi_edges, build_norm_adj_weighted
from src.train import set_all_seeds, train_neural, setup_lightgcn_graph

PROJ = Path(__file__).parent.parent
PROC = PROJ / "data" / "processed"
RESULTS = PROJ / "results"
ABLATION_DIR = RESULTS / "ablation"
ABLATION_DIR.mkdir(parents=True, exist_ok=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def make_lightgcn(splits, embed_dim=64, n_layers=3):
    return LightGCN(splits.n_users, splits.n_items, embed_dim, n_layers), setup_lightgcn_graph


def make_lightgcn_si(splits, books_df, users_df, embed_dim=64, n_layers=3,
                    use_g=True, use_a=True, use_c=True):
    m = LightGCNSI(
        splits.n_users, splits.n_items,
        n_genders=3 if use_g else 1,
        n_age_buckets=8 if use_a else 1,
        n_categories=11 if use_c else 1,
        embed_dim=embed_dim, n_layers=n_layers,
    )
    def extra(model, tu, ti, nu, ni, dev):
        setup_lightgcn_graph(model, tu, ti, nu, ni, dev)
        g, a, c, _ = build_side_info_tensors(splits, books_df, users_df)
        if not use_g: g = torch.zeros_like(g)
        if not use_a: a = torch.zeros_like(a)
        if not use_c: c = torch.zeros_like(c)
        model.set_side_info(g.to(dev), a.to(dev), c.to(dev))
    return m, extra


def make_lightgcn_multi(splits, books_df, users_df, reservations_df,
                        embed_dim=64, n_layers=3, reserve_weight=0.5):
    m = LightGCNMulti(
        splits.n_users, splits.n_items,
        embed_dim=embed_dim, n_layers=n_layers,
        use_side_info=True,
    )
    def extra(model, tu, ti, nu, ni, dev):
        eu, ei, ew = build_multi_edges(splits, reservations_df,
                                       borrow_weight=1.0, reserve_weight=reserve_weight)
        A_hat = build_norm_adj_weighted(eu, ei, ew, nu, ni, device=dev)
        model.set_graph(A_hat)
        g, a, c, _ = build_side_info_tensors(splits, books_df, users_df)
        model.set_side_info(g.to(dev), a.to(dev), c.to(dev))
    return m, extra


def run_one(name, model, extra, splits, *, epochs=40, eval_every=10):
    print(f"\n{'='*70}\n[RUN] {name}\n{'='*70}")
    t0 = time.time()
    _, test_m = train_neural(
        model, splits,
        epochs=epochs, eval_every=eval_every,
        extra_setup=extra, name=name,
        device=DEVICE, cold_start_eval=True,
    )
    dt = time.time() - t0
    print(f"[RUN] {name} done in {dt:.0f}s")
    return test_m


def suite_multi_seed(splits, books_df, users_df, reservations_df, *, epochs=40, seeds=(42, 123, 2024)):
    print(f"\n###### SUITE: multi-seed (3 seeds × 3 models) ######")
    rows = []
    for seed in seeds:
        for model_name in ("lightgcn", "lightgcn_si", "lightgcn_multi"):
            set_all_seeds(seed)
            tag = f"{model_name}_seed{seed}"
            if model_name == "lightgcn":
                m, e = make_lightgcn(splits)
            elif model_name == "lightgcn_si":
                m, e = make_lightgcn_si(splits, books_df, users_df)
            else:
                m, e = make_lightgcn_multi(splits, books_df, users_df, reservations_df)
            test_m = run_one(tag, m, e, splits, epochs=epochs)
            test_m["model"] = model_name
            test_m["seed"] = seed
            rows.append(test_m)
    pd.DataFrame(rows).to_csv(ABLATION_DIR / "multi_seed.csv", index=False)
    print(f"\n[saved] {ABLATION_DIR / 'multi_seed.csv'}")


def suite_hyperparam(splits, books_df, users_df, *, epochs=40):
    print(f"\n###### SUITE: hyperparam (embed_dim × n_layers) ######")
    rows = []
    for d in (32, 64, 128):
        for L in (1, 2, 3, 4):
            set_all_seeds(42)
            tag = f"lightgcn_d{d}_L{L}"
            m, e = make_lightgcn(splits, embed_dim=d, n_layers=L)
            test_m = run_one(tag, m, e, splits, epochs=epochs)
            test_m["embed_dim"] = d
            test_m["n_layers"] = L
            rows.append(test_m)
    pd.DataFrame(rows).to_csv(ABLATION_DIR / "hyperparam.csv", index=False)
    print(f"\n[saved] {ABLATION_DIR / 'hyperparam.csv'}")


def suite_side_info(splits, books_df, users_df, *, epochs=40):
    print(f"\n###### SUITE: side-info ablation ######")
    configs = [
        ("none",          False, False, False),
        ("gender_only",   True,  False, False),
        ("age_only",      False, True,  False),
        ("category_only", False, False, True),
        ("g+a",           True,  True,  False),
        ("g+c",           True,  False, True),
        ("a+c",           False, True,  True),
        ("all",           True,  True,  True),
    ]
    rows = []
    for tag_suffix, ug, ua, uc in configs:
        set_all_seeds(42)
        tag = f"lgcn_si_{tag_suffix}"
        m, e = make_lightgcn_si(splits, books_df, users_df,
                                use_g=ug, use_a=ua, use_c=uc)
        test_m = run_one(tag, m, e, splits, epochs=epochs)
        test_m["config"] = tag_suffix
        test_m["use_gender"] = ug
        test_m["use_age"] = ua
        test_m["use_category"] = uc
        rows.append(test_m)
    pd.DataFrame(rows).to_csv(ABLATION_DIR / "side_info.csv", index=False)
    print(f"\n[saved] {ABLATION_DIR / 'side_info.csv'}")


def suite_reserve_weight(splits, books_df, users_df, reservations_df, *, epochs=40):
    print(f"\n###### SUITE: reserve_weight ablation ######")
    rows = []
    for w in (0.0, 0.3, 0.5, 0.7, 1.0):
        set_all_seeds(42)
        tag = f"lgcn_multi_rw{int(w*10):02d}"
        m, e = make_lightgcn_multi(splits, books_df, users_df, reservations_df, reserve_weight=w)
        test_m = run_one(tag, m, e, splits, epochs=epochs)
        test_m["reserve_weight"] = w
        rows.append(test_m)
    pd.DataFrame(rows).to_csv(ABLATION_DIR / "reserve_weight.csv", index=False)
    print(f"\n[saved] {ABLATION_DIR / 'reserve_weight.csv'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", type=str, required=True,
                    choices=["multi_seed", "hyperparam", "side_info", "reserve_weight", "all"])
    ap.add_argument("--epochs", type=int, default=40)
    args = ap.parse_args()

    print(f"=== Loading splits ===")
    splits = load_splits()
    books_df = pd.read_parquet(PROC / "books.parquet")
    users_df = pd.read_parquet(PROC / "users.parquet")
    reservations_df = pd.read_parquet(PROC / "reservations.parquet")
    print(f"users={splits.n_users:,}  items={splits.n_items:,}")

    if args.suite in ("multi_seed", "all"):
        suite_multi_seed(splits, books_df, users_df, reservations_df, epochs=args.epochs)
    if args.suite in ("hyperparam", "all"):
        suite_hyperparam(splits, books_df, users_df, epochs=args.epochs)
    if args.suite in ("side_info", "all"):
        suite_side_info(splits, books_df, users_df, epochs=args.epochs)
    if args.suite in ("reserve_weight", "all"):
        suite_reserve_weight(splits, books_df, users_df, reservations_df, epochs=args.epochs)


if __name__ == "__main__":
    main()
