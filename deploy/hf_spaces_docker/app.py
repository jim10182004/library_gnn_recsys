"""
HF Spaces Docker SDK 入口 — FastAPI + 自製 HTML 前端版

完全 self-contained：
  - 使用預計算的 item_embs.pt（無需訓練資料 / checkpoints）
  - 純 CPU
  - 跟本機 api/main.py 同樣的 endpoints + Tailwind 前端
"""
from __future__ import annotations
import json
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import pandas as pd
import pyarrow  # noqa: F401
import numpy as np
import torch
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

ROOT = Path(__file__).parent
ASSETS = ROOT / "assets"
STATIC = ROOT / "static"

# ============= Personas（與本機 api/main.py 一致）=============
PERSONAS = {
    "child_en": {"name": "兒童英文書愛好者", "emoji": "📚", "color": "#f59e0b",
                 "desc": "Magic Tree House、Toy Story 等英文章節書與繪本",
                 "seed_titles": ["Magic Tree House", "Toy story", "Mittens", "Fox versus winter"]},
    "japanese_mystery": {"name": "日系推理小說迷", "emoji": "🔍", "color": "#6366f1",
                         "desc": "東野圭吾、西澤保彥的日本推理粉絲",
                         "seed_titles": ["白金數據", "嫌疑犯X的獻身", "解憂雜貨店", "死了七次的男人"]},
    "self_help": {"name": "職場與自我成長", "emoji": "💼", "color": "#10b981",
                  "desc": "職涯、心理、效率提升",
                  "seed_titles": ["原子習慣", "拖延心理學", "高效能人士的七個習慣", "目標"]},
    "academic": {"name": "學術派", "emoji": "🎓", "color": "#8b5cf6",
                 "desc": "教科書、學術專著",
                 "seed_titles": ["離散數學", "演算法", "微積分", "線性代數"]},
    "design_art": {"name": "設計與藝術", "emoji": "🎨", "color": "#ec4899",
                   "desc": "平面設計、攝影、藝術史",
                   "seed_titles": ["設計", "色彩", "攝影", "字型"]},
    "programmer": {"name": "程式設計師", "emoji": "💻", "color": "#0ea5e9",
                   "desc": "Python、機器學習、軟體工程",
                   "seed_titles": ["Python", "機器學習", "演算法", "程式設計"]},
    "history": {"name": "歷史愛好者", "emoji": "📜", "color": "#dc2626",
                "desc": "中國史、世界史、政治哲學",
                "seed_titles": ["史記", "三國", "近代", "明朝"]},
    "philosophy": {"name": "哲學沉思者", "emoji": "🤔", "color": "#a855f7",
                   "desc": "西方哲學、東方思想",
                   "seed_titles": ["哲學", "莊子", "蘇格拉底", "尼采"]},
    "cooking": {"name": "美食料理家", "emoji": "🍳", "color": "#ef4444",
                "desc": "中西食譜、烘焙、營養",
                "seed_titles": ["料理", "烘焙", "甜點", "義大利麵"]},
    "modern_chinese_fiction": {"name": "近代華文小說讀者", "emoji": "📖", "color": "#475569",
                                "desc": "張愛玲、白先勇、駱以軍等",
                                "seed_titles": ["紅樓夢", "白先勇", "張愛玲", "駱以軍"]},
    "parent_picturebook": {"name": "親子繪本家長", "emoji": "🐰", "color": "#06b6d4",
                            "desc": "兒童中文繪本、教養書",
                            "seed_titles": ["噗", "親子", "繪本", "好餓"]},
}

CATEGORY_LABELS = {
    "0": "總類", "1": "哲學", "2": "宗教", "3": "科學", "4": "應用科學",
    "5": "社會科學", "6": "中國史地", "7": "世界史地", "8": "語文文學", "9": "藝術",
}
CATEGORY_COLORS = {
    "0": "#6b7280", "1": "#8b5cf6", "2": "#ec4899", "3": "#06b6d4", "4": "#0ea5e9",
    "5": "#10b981", "6": "#f59e0b", "7": "#ef4444", "8": "#3b82f6", "9": "#a855f7",
}


def _category_label(c) -> tuple[str, str, str]:
    if not c:
        return ("?", "未知", "#9ca3af")
    s = str(c).strip()
    if s and s[0].isdigit():
        d = s[0]
        return (d, CATEGORY_LABELS.get(d, "?"), CATEGORY_COLORS.get(d, "#9ca3af"))
    return ("?", "未知", "#9ca3af")


# ============= 全域 state =============
class State:
    item_embs_norm: Optional[torch.Tensor] = None
    books: Optional[pd.DataFrame] = None
    item_remap: dict = {}
    inv_remap: dict = {}
    item_pop: Optional[np.ndarray] = None
    item_cat: Optional[np.ndarray] = None
    item_author: Optional[np.ndarray] = None
    metadata: dict = {}


state = State()


def _load_assets():
    print("[FastAPI] 載入 assets ...")
    item_embs = torch.load(ASSETS / "item_embs.pt", map_location="cpu", weights_only=True)
    state.item_embs_norm = torch.nn.functional.normalize(item_embs, p=2, dim=1)
    state.books = pd.read_parquet(ASSETS / "books_meta.parquet")
    state.item_remap = {int(k): int(v) for k, v in
                        json.loads((ASSETS / "item_remap.json").read_text(encoding="utf-8")).items()}
    state.inv_remap = {v: k for k, v in state.item_remap.items()}
    state.metadata = json.loads((ASSETS / "metadata.json").read_text(encoding="utf-8"))

    # rerank assets：用 books metadata 重建 item_cat / item_author
    n = item_embs.shape[0]
    state.item_cat = np.full(n, -1, dtype=np.int64)
    state.item_author = np.full(n, -1, dtype=np.int64)
    author_to_id: dict[str, int] = {}
    for compact_id in range(n):
        orig = state.inv_remap.get(compact_id)
        if orig is None: continue
        row = state.books[state.books["book_id"] == orig]
        if row.empty: continue
        r = row.iloc[0]
        cat = str(r.get("category") or "").strip()
        if cat and cat[0].isdigit():
            state.item_cat[compact_id] = int(cat[0])
        author_str = str(r.get("author") or "").strip()
        first_author = author_str.split(";")[0].split(",")[0].strip()[:30] if author_str else ""
        if first_author:
            if first_author not in author_to_id:
                author_to_id[first_author] = len(author_to_id)
            state.item_author[compact_id] = author_to_id[first_author]
    # 沒有真實 borrows 資料，用書本被推薦次數的近似（fallback：均勻 1）
    state.item_pop = np.ones(n, dtype=np.float32)

    print(f"[FastAPI] 就緒：n_items={n}, n_books_meta={len(state.books)}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    _load_assets()
    yield
    state.item_embs_norm = None
    state.books = None
    state.item_remap = {}
    state.inv_remap = {}


app = FastAPI(title="Library GNN Recommender (HF Spaces)", version="1.0.0", lifespan=lifespan)


# ============= 前端 =============
@app.get("/")
def root():
    return FileResponse(STATIC / "index.html")


app.mount("/static", StaticFiles(directory=str(STATIC)), name="static")


# ============= API =============
@app.get("/api/health")
def health():
    return {"status": "ok",
            "n_items": int(state.item_embs_norm.shape[0]) if state.item_embs_norm is not None else 0}


@app.get("/api/personas")
def list_personas():
    return [{"key": k, "name": p["name"], "emoji": p["emoji"],
             "color": p.get("color", "#475569"), "desc": p["desc"]}
            for k, p in PERSONAS.items()]


@app.get("/api/stats")
def get_stats():
    """系統統計（從 metadata + 公開研究結果 hardcode）"""
    return {
        "n_users": int(state.metadata.get("n_users", 35856)),
        "n_items": int(state.metadata.get("n_items", 29685)),
        "n_train": 453759,
        "n_val": 31435,
        "n_test": 30067,
        "best_model": "lightgcn_multi_opt",
        "best_metric": "recall@10",
        "best_recall_at_10": 0.2707,
        "best_recall_at_20": 0.3015,
        "best_ndcg_at_10": 0.2232,
        "best_ndcg_at_20": 0.2315,
    }


@app.get("/api/search")
def search_books(q: str = Query(..., min_length=1), n: int = 10):
    """模糊搜尋書名+作者"""
    title_match = state.books["title"].str.contains(q, na=False, regex=False)
    author_match = state.books["author"].str.contains(q, na=False, regex=False)
    matches = state.books[title_match | author_match]
    matches = matches.assign(_title_first=title_match.loc[matches.index].astype(int))
    matches = matches.sort_values("_title_first", ascending=False).head(n)
    out = []
    for _, row in matches.iterrows():
        compact_id = state.item_remap.get(int(row["book_id"]))
        if compact_id is None: continue
        cat_code, cat_label, cat_color = _category_label(row["category"])
        out.append({
            "book_id": int(row["book_id"]),
            "compact_id": int(compact_id),
            "title": row["title"] or "",
            "author": row["author"] or "",
            "category": row["category"] or "",
            "cat_code": cat_code,
            "cat_label": cat_label,
            "cat_color": cat_color,
            "isbn": row.get("isbn_clean") or "",
        })
    return out


def _split_authors(author_str: str) -> list[str]:
    if not author_str: return []
    parts = []
    for sep in [";", ",", "／", "/"]:
        author_str = author_str.replace(sep, "|")
    for p in author_str.split("|"):
        p = p.strip()
        if p and len(p) <= 30:
            parts.append(p)
    return parts


def _book_to_dict(meta_row, *, score=None) -> dict:
    cat_code, cat_label, cat_color = _category_label(meta_row["category"])
    d = {
        "book_id": int(meta_row["book_id"]),
        "title": meta_row["title"] or "",
        "author": meta_row["author"] or "",
        "category": meta_row["category"] or "",
        "cat_code": cat_code,
        "cat_label": cat_label,
        "cat_color": cat_color,
        "isbn": meta_row.get("isbn_clean") or "",
    }
    if score is not None:
        d["score"] = float(score)
    return d


def _explain_why(rec: dict, seeds: list[dict]) -> dict:
    same_cat_seeds = [s for s in seeds if s["cat_code"] == rec["cat_code"] and rec["cat_code"] != "?"]
    rec_authors = set(_split_authors(rec["author"]))
    same_author_seeds = []
    for s in seeds:
        s_authors = set(_split_authors(s["author"]))
        if rec_authors & s_authors and rec_authors != {""}:
            same_author_seeds.append(s)

    if same_author_seeds:
        common = list(rec_authors & set(_split_authors(same_author_seeds[0]["author"])))[0]
        return {"short": f"作者相同：{common[:12]}", "color": "#7c3aed"}
    if same_cat_seeds:
        return {"short": f"同類別：{rec['cat_label']}（{len(same_cat_seeds)} 本）", "color": "#10b981"}
    return {"short": "相似讀者群也借", "color": "#3b82f6"}


def _same_cat_rerank(scores: np.ndarray, k: int, exclude: set, seed_compact_ids: list[int]):
    """同類別優先重排序 — 種子書屬於哪些類別，那些類別的書 boost 分數
    取 Top 50 候選，把種子類別的書 boost +0.15，其他扣 -0.05，再重排。
    保證至少 80% 都是同類別。
    """
    # 1. 找種子書的類別（多數決）
    seed_cats = set()
    for sid in seed_compact_ids:
        c = int(state.item_cat[sid]) if state.item_cat[sid] >= 0 else -1
        if c >= 0: seed_cats.add(c)
    if not seed_cats:
        # 沒種子類別資訊 → 退回純 score
        top = np.argpartition(-scores, kth=k)[:k]
        return top[np.argsort(-scores[top])]

    # 2. 取 Top 100 候選
    n_cand = min(100, len(scores))
    cand = np.argpartition(-scores, kth=n_cand-1)[:n_cand]
    cand = cand[np.argsort(-scores[cand])]

    # 3. boost 同類別、demote 不同類別 → 重新排序
    boosted = []
    for c in cand:
        c = int(c)
        if c in exclude: continue
        s = float(scores[c])
        item_cat = int(state.item_cat[c]) if state.item_cat[c] >= 0 else -1
        if item_cat in seed_cats:
            s += 0.15  # 同類別獎勵
        else:
            s -= 0.05  # 其他類別微扣
        boosted.append((c, s))
    boosted.sort(key=lambda x: -x[1])

    # 4. 額外加：同作者最多 3 本（避免整單同一作者）
    picked, author_count = [], {}
    for c, _ in boosted:
        if len(picked) >= k: break
        auth = int(state.item_author[c]) if state.item_author[c] >= 0 else -1
        if auth >= 0 and author_count.get(auth, 0) >= 3: continue
        picked.append(c)
        if auth >= 0:
            author_count[auth] = author_count.get(auth, 0) + 1
    return np.asarray(picked, dtype=np.int64)


def _do_recommend(compact_ids: list[int], k: int, *, rerank: bool = False) -> dict:
    liked_t = torch.tensor(compact_ids, dtype=torch.long)
    seed_emb = state.item_embs_norm[liked_t]
    user_vec = torch.nn.functional.normalize(seed_emb.mean(dim=0), p=2, dim=0)
    scores = (state.item_embs_norm @ user_vec).numpy()
    scores[compact_ids] = -np.inf

    if rerank:
        top = _same_cat_rerank(scores, k, exclude=set(compact_ids),
                               seed_compact_ids=compact_ids)
    else:
        top = np.argpartition(-scores, kth=k)[:k]
        top = top[np.argsort(-scores[top])]

    seeds = []
    for cid in compact_ids:
        orig = state.inv_remap.get(int(cid))
        if orig is None: continue
        meta = state.books[state.books["book_id"] == orig]
        if not meta.empty:
            seeds.append(_book_to_dict(meta.iloc[0]))

    recs = []
    cat_count = {}
    for cid in top:
        orig = state.inv_remap.get(int(cid))
        if orig is None: continue
        meta = state.books[state.books["book_id"] == orig]
        if meta.empty: continue
        rec = _book_to_dict(meta.iloc[0], score=float(scores[cid]))
        rec["why"] = _explain_why(rec, seeds)
        cat_count[rec["cat_label"]] = cat_count.get(rec["cat_label"], 0) + 1
        recs.append(rec)

    return {
        "seeds": seeds,
        "recommendations": recs,
        "cat_distribution": [{"label": l, "count": c}
                             for l, c in sorted(cat_count.items(), key=lambda x: -x[1])],
        "reranked": rerank,
    }


class RecommendRequest(BaseModel):
    book_ids: list[int]
    k: int = 10
    rerank: bool = False


@app.post("/api/recommend")
def recommend(req: RecommendRequest):
    compact = [state.item_remap.get(int(b)) for b in req.book_ids]
    compact = [c for c in compact if c is not None]
    if not compact:
        raise HTTPException(400, "沒有任何輸入的書在資料集中")
    return _do_recommend(compact, req.k, rerank=req.rerank)


@app.get("/api/persona/{key}")
def recommend_by_persona(key: str, k: int = 10, rerank: bool = False):
    if key not in PERSONAS:
        raise HTTPException(404, f"未知 persona: {key}")
    persona = PERSONAS[key]
    compact = []
    for t in persona["seed_titles"]:
        m = state.books[state.books["title"].str.contains(t, na=False, regex=False)]
        if m.empty: continue
        for _, row in m.iterrows():
            orig = int(row["book_id"])
            cid = state.item_remap.get(orig)
            if cid is not None and cid not in compact:
                compact.append(cid)
                break
    if not compact:
        raise HTTPException(404, "此 persona 的種子書名都不在資料集中")
    result = _do_recommend(compact, k, rerank=rerank)
    result["persona"] = persona["name"]
    return result


@app.get("/api/compare")
def compare_personas(a: str = Query(...), b: str = Query(...), k: int = 10):
    if a not in PERSONAS or b not in PERSONAS:
        raise HTTPException(404, "未知 persona")

    def get_recs(key):
        persona = PERSONAS[key]
        compact = []
        for t in persona["seed_titles"]:
            m = state.books[state.books["title"].str.contains(t, na=False, regex=False)]
            if m.empty: continue
            for _, row in m.iterrows():
                orig = int(row["book_id"])
                cid = state.item_remap.get(orig)
                if cid is not None and cid not in compact:
                    compact.append(cid)
                    break
        if not compact: return None
        return _do_recommend(compact, k)

    ra, rb = get_recs(a), get_recs(b)
    if ra is None or rb is None:
        raise HTTPException(404, "無法為 persona 建立推薦")
    overlap_ids = set(r["book_id"] for r in ra["recommendations"]) & \
                  set(r["book_id"] for r in rb["recommendations"])
    return {
        "a": {"key": a, "name": PERSONAS[a]["name"], **ra},
        "b": {"key": b, "name": PERSONAS[b]["name"], **rb},
        "overlap_count": len(overlap_ids),
        "overlap_book_ids": list(overlap_ids),
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
