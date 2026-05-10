"""
FastAPI 後端 — 圖書館 GNN 推薦系統 REST API

啟動：
    uvicorn api.main:app --reload --port 8000

API 端點：
    GET  /                          — 服務前端 HTML
    GET  /api/personas              — 列出預設 personas
    GET  /api/search?q=...&n=10     — 模糊搜尋書名
    POST /api/recommend             — Body: {"book_ids": [int], "k": 10} 回傳推薦
    GET  /api/persona/{key}?k=10    — 用 persona 取得推薦
    GET  /api/health                — 健康檢查
"""
from __future__ import annotations
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import pandas as pd
import pyarrow  # noqa: F401
import numpy as np

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import torch

_PROJ = Path(__file__).resolve().parent.parent
if str(_PROJ) not in sys.path:
    sys.path.insert(0, str(_PROJ))

from src.dataset import load_splits
from src.metrics_summary import best_model
from src.models.lightgcn import LightGCN, build_norm_adj


PROCESSED = _PROJ / "data" / "processed"
CKPT = _PROJ / "checkpoints"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ============= Personas =============
PERSONAS = {
    "child_en": {
        "name": "兒童英文書愛好者",
        "emoji": "📚",
        "color": "#f59e0b",
        "desc": "Magic Tree House、Toy Story 等英文章節書與繪本",
        "seed_titles": ["Magic Tree House", "Toy story", "Mittens", "Fox versus winter"],
    },
    "japanese_mystery": {
        "name": "日系推理小說迷",
        "emoji": "🔍",
        "color": "#6366f1",
        "desc": "東野圭吾、西澤保彥的日本推理粉絲",
        "seed_titles": ["白金數據", "嫌疑犯X的獻身", "解憂雜貨店", "死了七次的男人"],
    },
    "self_help": {
        "name": "職場與自我成長",
        "emoji": "💼",
        "color": "#10b981",
        "desc": "職涯、心理、效率提升",
        "seed_titles": ["原子習慣", "拖延心理學", "高效能人士的七個習慣", "目標"],
    },
    "academic": {
        "name": "學術派",
        "emoji": "🎓",
        "color": "#8b5cf6",
        "desc": "教科書、論文工具書、研究方法",
        "seed_titles": ["深度學習", "經濟學", "心理學", "研究方法"],
    },
    "design_art": {
        "name": "設計與藝術",
        "emoji": "🎨",
        "color": "#ec4899",
        "desc": "美學、設計、攝影",
        "seed_titles": ["設計", "字型", "色彩", "攝影"],
    },
    "programming": {
        "name": "程式設計師",
        "emoji": "💻",
        "color": "#0ea5e9",
        "desc": "Python、機器學習、軟體工程",
        "seed_titles": ["Python", "機器學習", "演算法", "程式設計"],
    },
    "history": {
        "name": "歷史愛好者",
        "emoji": "📜",
        "color": "#a16207",
        "desc": "中國史、世界史、傳記",
        "seed_titles": ["明朝", "三國", "羅馬", "民國"],
    },
    "philosophy": {
        "name": "哲學沉思者",
        "emoji": "🧠",
        "color": "#475569",
        "desc": "東西方哲學、思想史",
        "seed_titles": ["哲學", "存在", "尼采", "莊子"],
    },
    "cooking": {
        "name": "美食料理家",
        "emoji": "🍳",
        "color": "#dc2626",
        "desc": "食譜、烘焙、家常菜",
        "seed_titles": ["食譜", "烘焙", "麵包", "料理"],
    },
    "novel": {
        "name": "近代華文小說讀者",
        "emoji": "📖",
        "color": "#7c3aed",
        "desc": "村上春樹、東野圭吾、近代華文長篇小說",
        "seed_titles": ["挪威", "海邊", "嫌疑犯", "解憂"],
    },
    "kids_picture": {
        "name": "親子繪本家長",
        "emoji": "🐰",
        "color": "#06b6d4",
        "desc": "兒童中文繪本、教養書",
        "seed_titles": ["噗", "親子", "繪本", "好餓"],
    },
}


# ============= 全域物件（啟動時載入一次）=============
class State:
    splits = None
    books = None
    item_embs: Optional[torch.Tensor] = None
    book_map_inv: dict = {}
    rerank_assets: Optional[dict] = None  # {reranker, item_pop, item_cat, item_author}


state = State()


def _build_rerank_assets(splits, books) -> dict:
    """為 MMR reranker 準備所需資料：item_category / item_author / item_pop"""
    from src.reranker import MMRReranker

    # 1. item_category 取分類號第 1 個數字
    n_items = splits.n_items
    item_cat = np.full(n_items, -1, dtype=np.int64)
    item_author = np.full(n_items, -1, dtype=np.int64)

    inv_remap = {v: k for k, v in splits.item_remap.items()}
    author_to_id: dict[str, int] = {}

    for compact_id in range(n_items):
        orig = inv_remap.get(compact_id)
        if orig is None:
            continue
        meta = books[books["book_id"] == orig]
        if meta.empty:
            continue
        row = meta.iloc[0]
        cat = str(row.get("category") or "").strip()
        if cat and cat[0].isdigit():
            item_cat[compact_id] = int(cat[0])

        author_str = str(row.get("author") or "").strip()
        # 取第一段作者當識別 (簡單 hash)
        first_author = author_str.split(";")[0].split(",")[0].strip()[:30] if author_str else ""
        if first_author:
            if first_author not in author_to_id:
                author_to_id[first_author] = len(author_to_id)
            item_author[compact_id] = author_to_id[first_author]

    # 2. item_pop 從 train
    train_i = splits.train["i"].values
    item_pop = np.bincount(train_i, minlength=n_items).astype(np.float32)

    reranker = MMRReranker(
        item_category=item_cat,
        item_author=item_author,
        item_pop=item_pop,
        diversity_lambda=0.7,
        depopularize_alpha=0.05,
        author_cap=3,
        category_cap=6,
        novelty_weight=0.0,
    )
    return {
        "reranker": reranker,
        "item_pop": item_pop,
        "item_cat": item_cat,
        "item_author": item_author,
    }


def _load_model_into_state() -> None:
    print("[FastAPI] 載入模型 ...")
    splits = load_splits()
    books = pd.read_parquet(PROCESSED / "books.parquet")
    model = LightGCN(splits.n_users, splits.n_items, embed_dim=64, n_layers=3).to(DEVICE)
    sd = torch.load(CKPT / "lightgcn_best.pt", map_location=DEVICE, weights_only=True)
    model.load_state_dict(sd)
    train_u = torch.as_tensor(splits.train["u"].values, dtype=torch.long)
    train_i = torch.as_tensor(splits.train["i"].values, dtype=torch.long)
    A_hat = build_norm_adj(train_u, train_i, splits.n_users, splits.n_items, device=DEVICE)
    model.set_graph(A_hat)
    model.eval()
    with torch.no_grad():
        _, item_embs = model.propagate()
    state.splits = splits
    state.books = books
    state.item_embs = item_embs
    state.book_map_inv = {v: k for k, v in splits.item_remap.items()}
    print(f"[FastAPI] 模型就緒：n_users={splits.n_users}, n_items={splits.n_items}, device={DEVICE}")
    print("[FastAPI] 準備 MMR reranker ...")
    state.rerank_assets = _build_rerank_assets(splits, books)
    print(f"[FastAPI] reranker 就緒：{len(set(state.rerank_assets['item_cat']))} 個類別, "
          f"{len(set(state.rerank_assets['item_author']))} 個作者")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # startup
    _load_model_into_state()
    yield
    # shutdown — 釋放 GPU 記憶體
    state.item_embs = None
    state.splits = None
    state.books = None
    state.book_map_inv = {}
    state.rerank_assets = None
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("[FastAPI] 已釋放模型資源")


app = FastAPI(
    title="Library GNN Recommender API",
    version="1.0.0",
    lifespan=lifespan,
)


# ============= 前端 =============
@app.get("/")
def root():
    return FileResponse(_PROJ / "api" / "static" / "index.html")


app.mount("/static", StaticFiles(directory=str(_PROJ / "api" / "static")), name="static")


# ============= API =============
@app.get("/api/health")
def health():
    return {"status": "ok", "n_items": state.splits.n_items if state.splits else 0}


@app.get("/api/personas")
def list_personas():
    """列出所有預設 personas"""
    out = []
    for key, p in PERSONAS.items():
        out.append({
            "key": key,
            "name": p["name"],
            "emoji": p["emoji"],
            "color": p.get("color", "#475569"),
            "desc": p["desc"],
        })
    return out


@app.get("/api/stats")
def get_stats():
    """系統統計資料（給 hero 區用）"""
    best = best_model("recall@10")
    return {
        "n_users": state.splits.n_users,
        "n_items": state.splits.n_items,
        "n_train": len(state.splits.train),
        "n_val": len(state.splits.val),
        "n_test": len(state.splits.test),
        "best_model": best.get("model"),
        "best_metric": best.get("metric"),
        "best_recall_at_10": best.get("recall@10"),
        "best_recall_at_20": best.get("recall@20"),
        "best_ndcg_at_10": best.get("ndcg@10"),
        "best_ndcg_at_20": best.get("ndcg@20"),
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
        compact_id = state.splits.item_remap.get(int(row["book_id"]))
        if compact_id is None:
            continue
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


class RecommendRequest(BaseModel):
    book_ids: list[int]  # 原始 book_id 列表（從 /api/search 拿到的 book_id）
    k: int = 10
    rerank: bool = False  # 啟用 MMR 重排序


@app.post("/api/recommend")
def recommend(req: RecommendRequest):
    """基於使用者選的書，回傳 Top-K 推薦"""
    # 把原始 book_id 轉成 compact id
    compact = []
    for bid in req.book_ids:
        cid = state.splits.item_remap.get(int(bid))
        if cid is not None:
            compact.append(cid)
    if not compact:
        raise HTTPException(400, "沒有任何輸入的書在訓練集中")
    return _do_recommend(compact, req.k, rerank=req.rerank)


@app.get("/api/persona/{key}")
def recommend_by_persona(key: str, k: int = 10, rerank: bool = False):
    """用 persona 取得推薦

    Args:
        rerank: 啟用 MMR 重排序（多樣性 + 反熱門 + 作者上限）
    """
    if key not in PERSONAS:
        raise HTTPException(404, f"未知 persona: {key}")
    persona = PERSONAS[key]
    # 找對應的書：每個 seed_title 找所有匹配，取第一個有在 k-core 內的
    compact = []
    for t in persona["seed_titles"]:
        m = state.books[state.books["title"].str.contains(t, na=False, regex=False)]
        if m.empty:
            continue
        for _, row in m.iterrows():
            orig = int(row["book_id"])
            cid = state.splits.item_remap.get(orig)
            if cid is not None and cid not in compact:
                compact.append(cid)
                break  # 找到一個就夠
    if not compact:
        raise HTTPException(404, "此 persona 的種子書名都不在訓練集中")
    result = _do_recommend(compact, k, rerank=rerank)
    result["persona"] = persona["name"]
    return result


CATEGORY_LABELS = {
    "0": "總類", "1": "哲學", "2": "宗教", "3": "科學", "4": "應用科學",
    "5": "社會科學", "6": "中國史地", "7": "世界史地", "8": "語文文學", "9": "藝術",
}
CATEGORY_COLORS = {
    "0": "#6b7280", "1": "#8b5cf6", "2": "#ec4899", "3": "#06b6d4", "4": "#0ea5e9",
    "5": "#10b981", "6": "#f59e0b", "7": "#ef4444", "8": "#3b82f6", "9": "#a855f7",
}

def _category_label(c) -> tuple[str, str, str]:
    """回傳 (代號, 標籤, 顏色)"""
    if not c:
        return ("?", "未知", "#9ca3af")
    s = str(c).strip()
    if s and s[0].isdigit():
        d = s[0]
        return (d, CATEGORY_LABELS.get(d, "?"), CATEGORY_COLORS.get(d, "#9ca3af"))
    return ("?", "未知", "#9ca3af")


def _book_to_dict(meta_row, *, score=None) -> dict:
    """統一把 books DataFrame 一列轉成 API dict（含分類顏色 + ISBN）"""
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
    """根據 rec 與 seeds 的關聯，產生「為什麼推薦」說明。
    回傳 {short_tag, full_text} 格式。
    """
    same_cat_seeds = [s for s in seeds if s["cat_code"] == rec["cat_code"] and rec["cat_code"] != "?"]
    rec_authors = set(_split_authors(rec["author"]))
    same_author_seeds = []
    for s in seeds:
        s_authors = set(_split_authors(s["author"]))
        if rec_authors & s_authors and rec_authors != {""}:
            same_author_seeds.append(s)

    if same_author_seeds:
        common = list(rec_authors & set(_split_authors(same_author_seeds[0]["author"])))[0]
        return {
            "short": f"作者相同：{common[:12]}",
            "color": "#7c3aed",  # purple — strongest signal
        }
    if same_cat_seeds:
        return {
            "short": f"同類別：{rec['cat_label']}（{len(same_cat_seeds)} 本）",
            "color": rec["cat_color"],
        }
    return {
        "short": "相似讀者群也借",
        "color": "#0d9488",  # teal — collaborative
    }


def _split_authors(s: str) -> list[str]:
    """把作者欄拆成單一作者名"""
    if not s:
        return []
    import re
    s = re.split(r"[;；]\s*[^;；]*?譯", str(s))[0]
    parts = re.split(r"[,，;；]|\s+著\s*|\s+作\s*|\s+編\s*", s)
    out = []
    for p in parts:
        p = re.sub(r"[\(\)（）].*$", "", p).strip()
        if p and len(p) <= 30:
            out.append(p)
    return out


def _do_recommend(compact_ids: list[int], k: int, *, rerank: bool = False) -> dict:
    """合成讀者向量 → 用 cosine similarity 推薦

    Args:
        compact_ids: 種子書的 compact id
        k: 回傳的推薦數
        rerank: 是否啟用 MMR 重排序（多樣性 + 作者上限 + 反熱門）
    """
    liked_t = torch.as_tensor(compact_ids, dtype=torch.long, device=state.item_embs.device)
    seed_emb = state.item_embs[liked_t]
    seed_emb = torch.nn.functional.normalize(seed_emb, p=2, dim=1)
    user_vec = seed_emb.mean(dim=0)
    user_vec = torch.nn.functional.normalize(user_vec, p=2, dim=0)
    all_norm = torch.nn.functional.normalize(state.item_embs, p=2, dim=1)
    scores = (all_norm @ user_vec).cpu().numpy()
    scores[compact_ids] = -np.inf

    # 若啟用 reranker，先取 Top-N (N = 5k) 作為候選池
    if rerank and state.rerank_assets is not None:
        n_candidates = min(max(50, k * 5), 200)
        cand_idx = np.argpartition(-scores, kth=n_candidates)[:n_candidates]
        cand_idx = cand_idx[np.argsort(-scores[cand_idx])]
        cand_scores = scores[cand_idx]
        try:
            top = state.rerank_assets["reranker"].rerank(cand_idx, cand_scores, k=k)
        except Exception as e:
            print(f"[rerank failed, fallback to raw] {e}")
            top = np.argpartition(-scores, kth=k)[:k]
            top = top[np.argsort(-scores[top])]
    else:
        top = np.argpartition(-scores, kth=k)[:k]
        top = top[np.argsort(-scores[top])]

    seeds = []
    for cid in compact_ids:
        orig = state.book_map_inv[int(cid)]
        meta = state.books[state.books["book_id"] == orig]
        if meta.empty:
            continue
        seeds.append(_book_to_dict(meta.iloc[0]))

    recs = []
    cat_count = {}
    for cid in top:
        orig = state.book_map_inv[int(cid)]
        meta = state.books[state.books["book_id"] == orig]
        if meta.empty:
            continue
        rec = _book_to_dict(meta.iloc[0], score=float(scores[cid]))
        rec["why"] = _explain_why(rec, seeds)
        cat_count[rec["cat_label"]] = cat_count.get(rec["cat_label"], 0) + 1
        recs.append(rec)

    cat_distribution = [
        {"label": label, "count": count}
        for label, count in sorted(cat_count.items(), key=lambda x: -x[1])
    ]
    return {
        "seeds": seeds,
        "recommendations": recs,
        "cat_distribution": cat_distribution,
        "reranked": rerank,
    }


@app.get("/api/compare")
def compare_personas(a: str = Query(...), b: str = Query(...), k: int = 10):
    """並排比較兩個 persona 的推薦"""
    if a not in PERSONAS:
        raise HTTPException(404, f"未知 persona: {a}")
    if b not in PERSONAS:
        raise HTTPException(404, f"未知 persona: {b}")

    def get_recs(key):
        persona = PERSONAS[key]
        compact = []
        for t in persona["seed_titles"]:
            m = state.books[state.books["title"].str.contains(t, na=False, regex=False)]
            for _, row in m.iterrows():
                cid = state.splits.item_remap.get(int(row["book_id"]))
                if cid is not None and cid not in compact:
                    compact.append(cid)
                    break
        if not compact:
            return None
        result = _do_recommend(compact, k)
        result["persona_name"] = persona["name"]
        result["persona_emoji"] = persona["emoji"]
        result["persona_color"] = persona.get("color", "#475569")
        return result

    ra = get_recs(a)
    rb = get_recs(b)
    if ra is None or rb is None:
        raise HTTPException(404, "其中一個 persona 種子書找不到")

    # 找重疊的書（兩邊都推薦的）
    a_ids = {r["book_id"] for r in ra["recommendations"]}
    b_ids = {r["book_id"] for r in rb["recommendations"]}
    common_ids = a_ids & b_ids

    return {
        "a": ra,
        "b": rb,
        "common_book_ids": list(common_ids),
        "n_common": len(common_ids),
    }
