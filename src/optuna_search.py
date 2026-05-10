"""
用 Optuna 做 LightGCN 超參數隨機/貝氏搜尋
（vs 我們之前的 grid search — 隨機/TPE 通常更有效率）

執行：python -m src.optuna_search --n-trials 20
輸出：
  - results/ablation/optuna.csv
  - results/ablation/optuna_best.json
  - results/figures/fig16_optuna.png
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import pandas as pd  # noqa: F401
import pyarrow  # noqa: F401
import numpy as np
import torch
import optuna

PROJ = Path(__file__).parent.parent
ABL = PROJ / "results" / "ablation"
FIG = PROJ / "results" / "figures"
ABL.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def objective(trial: optuna.Trial, splits, n_epochs: int = 20):
    """單一 trial：取超參數、訓練、回傳 val Recall@20"""
    from src.models.lightgcn import LightGCN, build_norm_adj
    from src.evaluate import build_user_pos, evaluate_topk
    from torch.utils.data import DataLoader
    from src.train import BPRDataset, set_all_seeds

    # 採樣超參數
    embed_dim = trial.suggest_categorical("embed_dim", [32, 64, 128])
    n_layers = trial.suggest_int("n_layers", 1, 4)
    lr = trial.suggest_float("lr", 1e-4, 5e-3, log=True)
    decay = trial.suggest_float("decay", 1e-6, 1e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [2048, 4096, 8192])

    set_all_seeds(42)
    user_train_pos = build_user_pos(splits.train)
    user_val_pos = build_user_pos(splits.val)
    train_u = splits.train["u"].values
    train_i = splits.train["i"].values

    model = LightGCN(splits.n_users, splits.n_items, embed_dim, n_layers).to(DEVICE)
    A_hat = build_norm_adj(
        torch.as_tensor(train_u, dtype=torch.long),
        torch.as_tensor(train_i, dtype=torch.long),
        splits.n_users, splits.n_items, device=DEVICE,
    )
    model.set_graph(A_hat)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    ds = BPRDataset(train_u, train_i, splits.n_items, user_train_pos)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0)

    for ep in range(n_epochs):
        model.train()
        for u, pi, ni in loader:
            u = u.to(DEVICE); pi = pi.to(DEVICE); ni = ni.to(DEVICE)
            loss, _ = model.bpr_loss(u, pi, ni, decay=decay)
            opt.zero_grad(); loss.backward(); opt.step()

    model.eval()
    eval_users = np.array(sorted(user_val_pos.keys()))
    val_m = evaluate_topk(model, eval_users, user_train_pos, user_val_pos,
                          splits.n_items, device=DEVICE, is_torch=True)
    return val_m["recall@20"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-trials", type=int, default=20)
    ap.add_argument("--epochs-per-trial", type=int, default=20,
                    help="每個 trial 的訓練 epoch 數（少一點省時間）")
    args = ap.parse_args()

    from src.dataset import load_splits
    print("Loading splits ...")
    splits = load_splits()

    print(f"Starting Optuna search: {args.n_trials} trials × {args.epochs_per_trial} epochs")
    sampler = optuna.samplers.TPESampler(seed=42)
    study = optuna.create_study(direction="maximize", sampler=sampler,
                                study_name="lightgcn_search")
    study.optimize(lambda t: objective(t, splits, args.epochs_per_trial),
                   n_trials=args.n_trials, show_progress_bar=False)

    print()
    print("=" * 70)
    print("Optuna 結果")
    print("=" * 70)
    print(f"Best Recall@20: {study.best_value:.4f}")
    print(f"Best params: {study.best_params}")

    # 存所有 trials
    trials_df = study.trials_dataframe()
    trials_df.to_csv(ABL / "optuna.csv", index=False)
    print(f"[saved] {ABL / 'optuna.csv'}")

    # 存 best
    with open(ABL / "optuna_best.json", "w", encoding="utf-8") as f:
        json.dump({
            "best_value_recall20": study.best_value,
            "best_params": study.best_params,
            "n_trials": len(study.trials),
        }, f, indent=2)
    print(f"[saved] {ABL / 'optuna_best.json'}")

    # 視覺化
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Microsoft JhengHei"]
    plt.rcParams["axes.unicode_minus"] = False

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    # 左：trial 進展
    values = [t.value for t in study.trials if t.value is not None]
    cumulative_best = np.maximum.accumulate(values)
    axes[0].plot(values, "o-", alpha=0.4, label="Each trial", color="#0d9488")
    axes[0].plot(cumulative_best, "-", linewidth=2.5, label="Best so far", color="#f59e0b")
    axes[0].axhline(0.3034, color="red", linestyle="--", alpha=0.5,
                    label="Grid search 最佳 (d=128, L=2)")
    axes[0].set_xlabel("Trial")
    axes[0].set_ylabel("Validation Recall@20")
    axes[0].set_title(f"Optuna search 進展（{args.n_trials} trials, TPE sampler）")
    axes[0].legend()

    # 右：parameter importance（只看連續變數）
    try:
        importance = optuna.importance.get_param_importances(study)
        names = list(importance.keys())
        vals = list(importance.values())
        axes[1].barh(names, vals, color="#0d9488")
        axes[1].set_xlabel("Importance")
        axes[1].set_title("超參數重要性")
        axes[1].invert_yaxis()
    except Exception as e:
        axes[1].text(0.5, 0.5, f"無法計算 importance：{e}",
                     ha="center", va="center", transform=axes[1].transAxes)

    plt.tight_layout()
    out = FIG / "fig16_optuna.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"[saved] {out}")


if __name__ == "__main__":
    main()
