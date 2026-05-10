"""
產生資料 schema 的 ER 圖（用 matplotlib，不需要 graphviz）
輸出：results/figures/fig14_er_diagram.png
"""
from __future__ import annotations
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

FIG = Path(__file__).parent.parent / "results" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

plt.rcParams["font.sans-serif"] = ["Microsoft JhengHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False


def draw_table(ax, x, y, w, h, title, fields, color="#0d9488"):
    # 外框
    box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02",
                          linewidth=2, edgecolor=color, facecolor="white", zorder=2)
    ax.add_patch(box)
    # 標題列
    title_h = 0.4
    title_box = FancyBboxPatch((x, y + h - title_h), w, title_h,
                                boxstyle="round,pad=0.02",
                                linewidth=0, facecolor=color, zorder=3)
    ax.add_patch(title_box)
    ax.text(x + w / 2, y + h - title_h / 2, title,
            ha="center", va="center", color="white", fontsize=12, fontweight="bold", zorder=4)

    # 欄位
    for i, (name, ftype, is_pk, is_fk) in enumerate(fields):
        fy = y + h - title_h - 0.3 - i * 0.28
        marker = ""
        if is_pk: marker = "🔑 "
        elif is_fk: marker = "🔗 "
        text = f"{marker}{name}"
        ax.text(x + 0.1, fy, text, ha="left", va="center", fontsize=9.5, zorder=5)
        ax.text(x + w - 0.1, fy, ftype, ha="right", va="center",
                fontsize=8.5, color="#6b7280", style="italic", zorder=5)


def draw_link(ax, x1, y1, x2, y2, label="N : 1", color="#0d9488"):
    arrow = FancyArrowPatch((x1, y1), (x2, y2),
                            arrowstyle="-|>", color=color,
                            mutation_scale=15, linewidth=2, zorder=1)
    ax.add_patch(arrow)
    ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 0.1, label,
            ha="center", va="center", fontsize=9, fontweight="bold",
            color=color,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="none"))


def main():
    fig, ax = plt.subplots(figsize=(13, 7))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)
    ax.axis("off")

    # USERS
    draw_table(ax, 0.5, 4.5, 3.5, 3,
               "users",
               [
                   ("user_id",   "int (PK)",   True,  False),
                   ("user_orig", "int",        False, False),
                   ("gender",    "str (男/女)", False, False),
                   ("age",       "int",        False, False),
               ],
               color="#0d9488")

    # BOOKS
    draw_table(ax, 10, 4.5, 3.5, 3,
               "books",
               [
                   ("book_id",    "int (PK)",   True,  False),
                   ("book_key",   "str",        False, False),
                   ("title",      "str",        False, False),
                   ("author",     "str",        False, False),
                   ("isbn_clean", "str",        False, False),
                   ("category",   "str (中圖)",  False, False),
                   ("pub_year",   "int",        False, False),
               ],
               color="#0d9488")

    # BORROWS
    draw_table(ax, 5, 4.5, 4, 3,
               "borrows (1.3M 筆)",
               [
                   ("user_id",   "int (FK)",   False, True),
                   ("book_id",   "int (FK)",   False, True),
                   ("ts",        "datetime",   False, False),
                   ("return_ts", "datetime",   False, False),
                   ("gender",    "str",        False, False),
                   ("age",       "int",        False, False),
                   ("category",  "str",        False, False),
               ],
               color="#f59e0b")

    # RESERVATIONS
    draw_table(ax, 5, 0.5, 4, 3,
               "reservations (320K 筆)",
               [
                   ("user_id",  "int (FK)",   False, True),
                   ("book_id",  "int (FK)",   False, True),
                   ("ts",       "datetime",   False, False),
                   ("gender",   "str",        False, False),
                   ("age",      "int",        False, False),
                   ("category", "str",        False, False),
               ],
               color="#a855f7")

    # 連線
    # users → borrows
    draw_link(ax, 4, 6, 5, 6, "1 : N", "#0d9488")
    # books → borrows
    draw_link(ax, 10, 6, 9, 6, "1 : N", "#0d9488")
    # users → reservations
    draw_link(ax, 4, 5, 5, 2.5, "1 : N", "#a855f7")
    # books → reservations
    draw_link(ax, 10, 5, 9, 2.5, "1 : N", "#a855f7")

    # Title and legend
    ax.text(7, 7.7, "圖書館資料 ER Diagram", ha="center", fontsize=16, fontweight="bold")
    ax.text(7, 0.15,
            "註：🔑 = Primary Key   🔗 = Foreign Key   "
            "讀者匿名化處理（user_orig 經對照表轉換）",
            ha="center", fontsize=9, color="#6b7280", style="italic")

    plt.tight_layout()
    out = FIG / "fig14_er_diagram.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"[saved] {out}")


if __name__ == "__main__":
    main()
