"""
畫長尾分布圖（圖書館借閱資料的核心 EDA 圖）
輸出：results/figures/fig17_long_tail.png + fig18_top20.png
"""
from __future__ import annotations
from pathlib import Path
import pandas as pd  # noqa
import pyarrow  # noqa: F401
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

PROJ = Path(__file__).parent.parent
PROC = PROJ / "data" / "processed"
FIG = PROJ / "results" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

plt.rcParams["font.sans-serif"] = ["Microsoft JhengHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False
sns.set_theme(style="whitegrid", font="Microsoft JhengHei")


def main():
    print("Loading borrows ...")
    borrows = pd.read_parquet(PROC / "borrows.parquet")
    books = pd.read_parquet(PROC / "books.parquet")
    counts = borrows["book_id"].value_counts()
    sorted_counts = counts.sort_values(ascending=False).values

    n_books = len(counts)
    total = sorted_counts.sum()
    cum = np.cumsum(sorted_counts) / total

    # ============ Plot 1: log-log long tail ============
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: log-log scatter
    ranks = np.arange(1, n_books + 1)
    axes[0].loglog(ranks, sorted_counts, "-", linewidth=1.5, color="#0d9488", alpha=0.8)

    # 標記 top 1%
    p01 = int(n_books * 0.01)
    axes[0].axvline(p01, color="#dc2626", linestyle="--", alpha=0.7, label=f"Top 1% = {p01:,} 本")
    axes[0].axvline(int(n_books * 0.10), color="#f59e0b", linestyle="--", alpha=0.7,
                    label=f"Top 10% = {int(n_books * 0.10):,} 本")
    axes[0].set_xlabel("書籍排名（按借閱次數降冪）")
    axes[0].set_ylabel("借閱次數")
    axes[0].set_title("圖書館借閱資料的長尾分布 (log-log)")
    axes[0].legend(loc="upper right")
    axes[0].grid(True, alpha=0.3)
    # 標註 top 5
    titles_top5 = []
    top5_books = counts.head(5).reset_index()
    top5_books.columns = ["book_id", "n"]
    top5_books = top5_books.merge(books[["book_id", "title"]], on="book_id", how="left")
    for _, r in top5_books.iterrows():
        titles_top5.append((r["title"] or "?")[:18])
    for i, (rank, count) in enumerate(zip(range(1, 6), sorted_counts[:5])):
        axes[0].annotate(titles_top5[i], (rank, count),
                         textcoords="offset points", xytext=(8, -3),
                         fontsize=9, color="#dc2626")

    # Right: cumulative %
    axes[1].plot(ranks / n_books * 100, cum * 100, linewidth=2.5, color="#0d9488")
    axes[1].axvline(1, color="#dc2626", linestyle="--", alpha=0.7)
    axes[1].axvline(10, color="#f59e0b", linestyle="--", alpha=0.7)
    axes[1].axhline(58.7, color="#dc2626", linestyle=":", alpha=0.5)
    axes[1].axhline(74.1, color="#f59e0b", linestyle=":", alpha=0.5)
    axes[1].set_xlabel("書籍排名（百分比，按熱門度排序）")
    axes[1].set_ylabel("累積借閱占比 (%)")
    axes[1].set_title("Top X% 的書佔了多少 % 借閱")
    # 標註
    axes[1].annotate("Top 1% 占 58.7%", xy=(1, 58.7), xytext=(15, 50),
                     arrowprops=dict(arrowstyle="->", color="#dc2626"),
                     fontsize=11, color="#dc2626", fontweight="bold")
    axes[1].annotate("Top 10% 占 74.1%", xy=(10, 74.1), xytext=(30, 80),
                     arrowprops=dict(arrowstyle="->", color="#f59e0b"),
                     fontsize=11, color="#f59e0b", fontweight="bold")
    axes[1].set_xlim(0, 100)
    axes[1].set_ylim(0, 100)

    plt.tight_layout()
    out = FIG / "fig17_long_tail.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"[saved] {out}")

    # ============ Plot 2: Top 20 horizontal bar ============
    top20 = counts.head(20).reset_index()
    top20.columns = ["book_id", "borrow_count"]
    top20 = top20.merge(books[["book_id", "title", "category"]], on="book_id", how="left")

    def cat_color(c):
        if not c: return "#9ca3af"
        s = str(c).strip()
        if not s or not s[0].isdigit(): return "#9ca3af"
        colors = {"0":"#6b7280","1":"#8b5cf6","2":"#ec4899","3":"#06b6d4","4":"#0ea5e9",
                  "5":"#10b981","6":"#f59e0b","7":"#ef4444","8":"#3b82f6","9":"#a855f7"}
        return colors.get(s[0], "#9ca3af")

    top20["color"] = top20["category"].apply(cat_color)
    top20["short_title"] = top20["title"].fillna("?").str.slice(0, 30)

    fig, ax = plt.subplots(figsize=(11, 8))
    y_pos = np.arange(len(top20))
    ax.barh(y_pos, top20["borrow_count"], color=top20["color"], alpha=0.85)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([f"{i+1}. {t}" for i, t in enumerate(top20["short_title"])], fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel("借閱次數")
    ax.set_title("Top 20 熱門借閱書（顏色按中圖法分類）", fontsize=14)
    # 數字標註
    for i, v in enumerate(top20["borrow_count"]):
        ax.text(v + total * 0.005, i, f"{v:,}", va="center", fontsize=9, color="#374151")
    ax.set_xscale("log")
    plt.tight_layout()
    out = FIG / "fig18_top20.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"[saved] {out}")


if __name__ == "__main__":
    main()
