"""
重算所有模型的完整指標（coverage / novelty / mrr 等）。

部分舊 checkpoint 在訓練時 evaluate_topk 沒帶 item_pop，
所以 history.json 的 test 區塊缺 coverage/novelty/mrr。
這支腳本載入既有 checkpoint，重新評估，補齊指標後覆寫 history JSON。

執行：python -m src.recompute_full_metrics
       python -m src.recompute_full_metrics --models lightgcn,lightgcn_si
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

# Windows: 必須先 import pandas/pyarrow
import pandas as pd  # noqa: F401
import pyarrow  # noqa: F401
import numpy as np
import torch

sys.stdout.reconfigure(encoding="utf-8")

PROJECT = Path(__file__).resolve().parent.parent
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from src.dataset import load_splits
from src.evaluate import build_user_pos, evaluate_topk
from src.models.lightgcn import LightGCN, build_norm_adj
from src.models.lightgcn_si import LightGCNSI, build_side_info_tensors
from src.models.lightgcn_multi import (
    LightGCNMulti, build_multi_edges, build_norm_adj_weighted,
)
from src.models.lightgcn_bert import LightGCNBert, load_bert_tensor
from src.models.lightgcn_hetero import LightGCNHetero, build_hetero_adj
from src.models.baselines import PopularRecommender, ItemCF, BPRMF
from src.models.time_decay import build_time_decayed_edges

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PROC = PROJECT / "data" / "processed"
CKPT = PROJECT / "checkpoints"
RES = PROJECT / "results"


def build_simple_graph(model, splits):
    train_u = torch.as_tensor(splits.train["u"].values, dtype=torch.long)
    train_i = torch.as_tensor(splits.train["i"].values, dtype=torch.long)
    A = build_norm_adj(train_u, train_i, splits.n_users, splits.n_items, device=DEVICE)
    model.set_graph(A)


def build_si_assets(splits, books, users):
    g, a, c, _ = build_side_info_tensors(splits, books, users)
    return g.to(DEVICE), a.to(DEVICE), c.to(DEVICE)


def load_lightgcn(splits):
    m = LightGCN(splits.n_users, splits.n_items, embed_dim=64, n_layers=3).to(DEVICE)
    sd = torch.load(CKPT / "lightgcn_best.pt", map_location=DEVICE, weights_only=True)
    m.load_state_dict(sd)
    build_simple_graph(m, splits)
    return m


def load_lightgcn_si(splits, books, users):
    m = LightGCNSI(
        splits.n_users, splits.n_items,
        n_genders=3, n_age_buckets=8, n_categories=11,
        embed_dim=64, n_layers=3,
    ).to(DEVICE)
    sd = torch.load(CKPT / "lightgcn_si_best.pt", map_location=DEVICE, weights_only=True)
    m.load_state_dict(sd)
    build_simple_graph(m, splits)
    g, a, c = build_si_assets(splits, books, users)
    m.set_side_info(g, a, c)
    return m


def load_lightgcn_multi(splits, books, users, reservations, *, ckpt_name="lightgcn_multi_best.pt",
                        embed_dim=64, n_layers=3, reserve_weight=0.5):
    m = LightGCNMulti(
        splits.n_users, splits.n_items,
        embed_dim=embed_dim, n_layers=n_layers, use_side_info=True,
    ).to(DEVICE)
    sd = torch.load(CKPT / ckpt_name, map_location=DEVICE, weights_only=True)
    m.load_state_dict(sd)
    eu, ei, ew = build_multi_edges(splits, reservations,
                                   borrow_weight=1.0, reserve_weight=reserve_weight)
    A = build_norm_adj_weighted(eu, ei, ew, splits.n_users, splits.n_items, device=DEVICE)
    m.set_graph(A)
    g, a, c = build_si_assets(splits, books, users)
    m.set_side_info(g, a, c)
    return m


def load_lightgcn_timedecay(splits, books, users, reservations):
    m = LightGCNMulti(
        splits.n_users, splits.n_items,
        embed_dim=64, n_layers=3, use_side_info=True,
    ).to(DEVICE)
    sd = torch.load(CKPT / "lightgcn_timedecay_best.pt", map_location=DEVICE, weights_only=True)
    m.load_state_dict(sd)
    eu, ei, ew = build_time_decayed_edges(splits, reservations, decay_lambda=0.05)
    A = build_norm_adj_weighted(eu, ei, ew, splits.n_users, splits.n_items, device=DEVICE)
    m.set_graph(A)
    g, a, c = build_si_assets(splits, books, users)
    m.set_side_info(g, a, c)
    return m


def load_lightgcn_bert(splits, books, users):
    bert_path = PROC / "book_bert.parquet"
    sample = pd.read_parquet(bert_path).iloc[:1]
    bert_dim = len([c for c in sample.columns if c.startswith("v")])
    m = LightGCNBert(
        splits.n_users, splits.n_items,
        bert_dim=bert_dim, embed_dim=64, n_layers=3,
    ).to(DEVICE)
    sd = torch.load(CKPT / "lightgcn_bert_best.pt", map_location=DEVICE, weights_only=True)
    m.load_state_dict(sd)
    build_simple_graph(m, splits)
    g, a, c = build_si_assets(splits, books, users)
    m.set_side_info(g, a, c)
    bert_t = load_bert_tensor(bert_path, splits.n_items, splits.item_remap)
    m.set_bert(bert_t.to(DEVICE))
    return m


def load_lightgcn_hetero(splits, books):
    A_tmp, n_authors, _ = build_hetero_adj(splits, books, device="cpu")
    m = LightGCNHetero(
        splits.n_users, splits.n_items, n_authors,
        embed_dim=64, n_layers=3,
    ).to(DEVICE)
    sd = torch.load(CKPT / "lightgcn_hetero_best.pt", map_location=DEVICE, weights_only=True)
    m.load_state_dict(sd)
    A_hat, _, _ = build_hetero_adj(splits, books, device=DEVICE)
    m.set_graph(A_hat)
    return m


def load_bprmf(splits):
    m = BPRMF(splits.n_users, splits.n_items, 64).to(DEVICE)
    sd = torch.load(CKPT / "bprmf_best.pt", map_location=DEVICE, weights_only=True)
    m.load_state_dict(sd)
    return m


def load_itemcf(splits):
    train_u = splits.train["u"].values
    train_i = splits.train["i"].values
    m = ItemCF()
    m.fit(train_u, train_i, splits.n_users, splits.n_items)
    return m


def load_popular(splits):
    train_u = splits.train["u"].values
    train_i = splits.train["i"].values
    m = PopularRecommender()
    m.fit(train_u, train_i, splits.n_items)
    return m


# {model_name: (loader_fn, is_torch)}
def get_loaders(splits, books, users, reservations):
    return {
        "popular": (lambda: load_popular(splits), False),
        "itemcf": (lambda: load_itemcf(splits), False),
        "bprmf": (lambda: load_bprmf(splits), True),
        "lightgcn": (lambda: load_lightgcn(splits), True),
        "lightgcn_si": (lambda: load_lightgcn_si(splits, books, users), True),
        "lightgcn_multi": (lambda: load_lightgcn_multi(splits, books, users, reservations,
                                                       reserve_weight=0.5), True),
        "lightgcn_bert": (lambda: load_lightgcn_bert(splits, books, users), True),
        "lightgcn_hetero": (lambda: load_lightgcn_hetero(splits, books), True),
        "lightgcn_timedecay": (lambda: load_lightgcn_timedecay(splits, books, users, reservations), True),
    }


def history_needs_recompute(name: str) -> bool:
    p = RES / f"{name}_history.json"
    if not p.exists():
        return False
    with open(p, encoding="utf-8") as f:
        d = json.load(f)
    test = d.get("test", {})
    needed = ["coverage@10", "coverage@20", "novelty@10", "novelty@20", "mrr@10", "mrr@20"]
    return any(k not in test for k in needed)


def update_history(name: str, new_test: dict[str, float]):
    p = RES / f"{name}_history.json"
    if not p.exists():
        return
    with open(p, encoding="utf-8") as f:
        d = json.load(f)
    d["test"] = new_test
    with open(p, "w", encoding="utf-8") as f:
        json.dump(d, f, indent=2, ensure_ascii=False)


def regenerate_summary():
    """重建 summary.csv，順便 trigger summary_clean.csv"""
    rows = []
    for name in ["popular", "itemcf", "bprmf",
                 "lightgcn", "lightgcn_si", "lightgcn_multi",
                 "ngcf", "lightgcn_bert", "lightgcn_hetero",
                 "lightgcn_timedecay", "sasrec", "simgcl",
                 "lightgcn_opt", "lightgcn_multi_opt",
                 "lightgcn_tgn", "lightgcn_cover"]:
        p = RES / f"{name}_history.json"
        if not p.exists():
            continue
        with open(p, encoding="utf-8") as f:
            h = json.load(f)
        test = h.get("test", {})
        if not test:
            continue
        rows.append({"Model": name, **test})
    if rows:
        df = pd.DataFrame(rows)
        cols = ["Model"] + sorted([c for c in df.columns if c != "Model"])
        df = df[cols]
        df.to_csv(RES / "summary.csv", index=False)
        print(f"\n[saved] {RES / 'summary.csv'}")
        try:
            from src.metrics_summary import write_clean_summary
            out = write_clean_summary()
            print(f"[saved] {out}")
        except Exception as e:
            print(f"  (warn) write_clean_summary failed: {e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", type=str, default="auto",
                    help="comma-separated, or 'auto' for models with missing metrics, or 'all'")
    args = ap.parse_args()

    print("=== Loading splits ===")
    splits = load_splits()
    books = pd.read_parquet(PROC / "books.parquet")
    users = pd.read_parquet(PROC / "users.parquet")
    reservations = pd.read_parquet(PROC / "reservations.parquet")

    train_i = splits.train["i"].values
    item_pop = np.bincount(train_i, minlength=splits.n_items).astype(np.float32)

    user_train_pos = build_user_pos(splits.train)
    user_test_pos = build_user_pos(splits.test)
    eval_users = np.array(sorted(user_test_pos.keys()))

    loaders = get_loaders(splits, books, users, reservations)

    if args.models == "auto":
        targets = [n for n in loaders if history_needs_recompute(n)]
    elif args.models == "all":
        targets = list(loaders.keys())
    else:
        targets = [x.strip() for x in args.models.split(",") if x.strip()]

    if not targets:
        print("沒有需要重算的模型。")
        regenerate_summary()
        return

    print(f"\n要重算的模型：{targets}\n")

    for name in targets:
        if name not in loaders:
            print(f"[skip] {name}：未知或無 loader")
            continue
        loader_fn, is_torch = loaders[name]
        try:
            print(f"\n--- {name} ---")
            model = loader_fn()
            if is_torch:
                model.eval()
            with torch.no_grad():
                test_m = evaluate_topk(
                    model, eval_users, user_train_pos, user_test_pos,
                    splits.n_items, device=DEVICE, is_torch=is_torch,
                    item_pop=item_pop,
                )
            metrics_str = "  ".join([f"{k}={v:.4f}" for k, v in sorted(test_m.items())])
            print(f"  {metrics_str}")
            update_history(name, test_m)
            print(f"  [updated] {RES / f'{name}_history.json'}")
            # 釋放
            if is_torch:
                del model
                torch.cuda.empty_cache()
        except Exception as e:
            print(f"  [error] {name}: {e}")
            import traceback
            traceback.print_exc()

    regenerate_summary()
    print("\n=== 完成 ===")


if __name__ == "__main__":
    main()
