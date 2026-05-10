"""
HuggingFace Spaces 部署版 — 圖書館 GNN 推薦系統 Demo

特色：
  - 完全 self-contained（不需要訓練/原始資料）
  - 使用預計算的 item embeddings（assets/item_embs.pt）
  - CPU 即可跑（推薦延遲 ~50 ms）
  - 11 個人物原型 + 自訂喜好書 + 比較模式 + MMR 重排序
"""
from __future__ import annotations
import json
from pathlib import Path

import pandas as pd
import numpy as np
import streamlit as st
import torch

ASSETS = Path(__file__).parent / "assets"

# ============= 11 個人物原型（與本機 demo 一致）=============
PERSONAS = {
    "child_en": {
        "name": "兒童英文書愛好者", "emoji": "📚",
        "desc": "Magic Tree House、Toy Story 等英文章節書與繪本",
        "seed_titles": ["Magic Tree House", "Toy story", "Mittens", "Fox versus winter"],
    },
    "japanese_mystery": {
        "name": "日系推理小說迷", "emoji": "🔍",
        "desc": "東野圭吾、西澤保彥的日本推理粉絲",
        "seed_titles": ["白金數據", "嫌疑犯X的獻身", "解憂雜貨店", "死了七次的男人"],
    },
    "self_help": {
        "name": "職場與自我成長", "emoji": "💼",
        "desc": "職涯、心理、效率提升",
        "seed_titles": ["原子習慣", "拖延心理學", "高效能人士的七個習慣", "目標"],
    },
    "academic": {
        "name": "學術派", "emoji": "🎓",
        "desc": "教科書、學術專著",
        "seed_titles": ["離散數學", "演算法", "微積分", "線性代數"],
    },
    "design_art": {
        "name": "設計與藝術", "emoji": "🎨",
        "desc": "平面設計、攝影、藝術史",
        "seed_titles": ["設計", "色彩", "攝影", "字型"],
    },
    "programmer": {
        "name": "程式設計師", "emoji": "💻",
        "desc": "Python、機器學習、軟體工程",
        "seed_titles": ["Python", "機器學習", "演算法", "程式設計"],
    },
    "history": {
        "name": "歷史愛好者", "emoji": "📜",
        "desc": "中國史、世界史、政治哲學",
        "seed_titles": ["史記", "三國", "近代", "明朝"],
    },
    "philosophy": {
        "name": "哲學沉思者", "emoji": "🤔",
        "desc": "西方哲學、東方思想",
        "seed_titles": ["哲學", "莊子", "蘇格拉底", "尼采"],
    },
    "cooking": {
        "name": "美食料理家", "emoji": "🍳",
        "desc": "中西食譜、烘焙、營養",
        "seed_titles": ["料理", "烘焙", "甜點", "義大利麵"],
    },
    "modern_chinese_fiction": {
        "name": "近代華文小說讀者", "emoji": "📖",
        "desc": "張愛玲、白先勇、駱以軍等",
        "seed_titles": ["紅樓夢", "白先勇", "張愛玲", "駱以軍"],
    },
    "parent_picturebook": {
        "name": "親子繪本家長", "emoji": "🐰",
        "desc": "兒童中文繪本、教養書",
        "seed_titles": ["噗", "親子", "繪本", "好餓"],
    },
}

CAT_LABELS = {
    "0": "0 總類", "1": "1 哲學", "2": "2 宗教", "3": "3 科學", "4": "4 應用科學",
    "5": "5 社會科學", "6": "6 中國史地", "7": "7 世界史地",
    "8": "8 語文文學", "9": "9 藝術",
}
CAT_COLORS = {
    "0": "#6b7280", "1": "#8b5cf6", "2": "#ec4899", "3": "#06b6d4", "4": "#0ea5e9",
    "5": "#10b981", "6": "#f59e0b", "7": "#ef4444", "8": "#3b82f6", "9": "#a855f7",
}


@st.cache_resource(show_spinner="正在載入模型 embedding...")
def load_assets():
    item_embs = torch.load(ASSETS / "item_embs.pt", map_location="cpu", weights_only=True)
    item_embs_norm = torch.nn.functional.normalize(item_embs, p=2, dim=1)
    books = pd.read_parquet(ASSETS / "books_meta.parquet")
    item_remap = {int(k): int(v) for k, v in
                  json.loads((ASSETS / "item_remap.json").read_text(encoding="utf-8")).items()}
    inv_remap = {v: k for k, v in item_remap.items()}
    return item_embs_norm, books, item_remap, inv_remap


def find_seed_compact_ids(seed_titles, books, item_remap):
    """從書名找對應的 compact_id"""
    out = []
    for t in seed_titles:
        m = books[books["title"].str.contains(t, na=False, regex=False)]
        if m.empty:
            continue
        for _, row in m.iterrows():
            orig = int(row["book_id"])
            cid = item_remap.get(orig)
            if cid is not None and cid not in out:
                out.append(cid)
                break
    return out


def category_label(c) -> tuple[str, str, str]:
    if not c:
        return ("?", "未知", "#9ca3af")
    s = str(c).strip()
    if s and s[0].isdigit():
        d = s[0]
        return (d, CAT_LABELS.get(d, d), CAT_COLORS.get(d, "#9ca3af"))
    return ("?", "未知", "#9ca3af")


def recommend(seed_compact_ids, item_embs_norm, k=10, *, mmr=False,
              books=None, inv_remap=None):
    """合成虛擬讀者向量 → cosine similarity → top-K"""
    if not seed_compact_ids:
        return None
    seed_emb = item_embs_norm[seed_compact_ids]
    user_vec = seed_emb.mean(dim=0)
    user_vec = torch.nn.functional.normalize(user_vec, p=2, dim=0)
    scores = (item_embs_norm @ user_vec).numpy()
    scores[seed_compact_ids] = -np.inf

    if mmr and books is not None:
        # 簡化版 MMR：取 top 50 → 同類別最多 6 本
        cand = np.argpartition(-scores, kth=50)[:50]
        cand = cand[np.argsort(-scores[cand])]
        picked = []
        cat_count = {}
        for c in cand:
            if len(picked) >= k:
                break
            orig = inv_remap.get(int(c))
            if orig is None:
                continue
            row = books[books["book_id"] == orig]
            if row.empty:
                continue
            cat_code, _, _ = category_label(row.iloc[0]["category"])
            if cat_count.get(cat_code, 0) >= 6:
                continue
            picked.append(int(c))
            cat_count[cat_code] = cat_count.get(cat_code, 0) + 1
        return picked, scores
    else:
        top = np.argpartition(-scores, kth=k)[:k]
        top = top[np.argsort(-scores[top])]
        return top.tolist(), scores


def render_book_card(idx, cid, score, books, inv_remap):
    orig = inv_remap.get(int(cid))
    if orig is None:
        return
    row = books[books["book_id"] == orig]
    if row.empty:
        return
    r = row.iloc[0]
    cat_code, cat_label, cat_color = category_label(r.get("category"))
    title = r.get("title") or "(無題)"
    author = r.get("author") or "(未知作者)"

    st.markdown(f"""
    <div style="border:1px solid #e5e7eb; border-radius:8px; padding:12px; margin:8px 0;
                background:linear-gradient(to right, white, #f8fafc);">
        <div style="display:flex; justify-content:space-between; align-items:start;">
            <div>
                <span style="background:{cat_color}; color:white; padding:2px 8px; border-radius:4px;
                              font-size:0.75rem; margin-right:8px;">{cat_label}</span>
                <span style="color:#6b7280; font-size:0.85rem;">#{idx}</span>
            </div>
            <div style="color:#6b7280; font-size:0.85rem;">分數 {score:.3f}</div>
        </div>
        <div style="font-weight:600; font-size:1.05rem; margin-top:6px;">{title}</div>
        <div style="color:#6b7280; font-size:0.85rem; margin-top:2px;">{author}</div>
    </div>
    """, unsafe_allow_html=True)


# ============= 主畫面 =============
st.set_page_config(
    page_title="圖書館 GNN 推薦系統 Demo",
    page_icon="📚",
    layout="wide",
)

st.markdown("""
<style>
    .stButton button { width: 100%; }
    h1 { color: #028090; }
</style>
""", unsafe_allow_html=True)

st.markdown("# 📚 圖書館借閱資料 GNN 推薦系統")
st.markdown("""
> 基於 **LightGCN**（Graph Neural Network）的個人化書籍推薦
>
> 🎓 畢業專題 demo | 130 萬筆借閱資料 | 16 個模型對照 | Optuna 自動調參
> [📖 GitHub: jim10182004/library_gnn_recsys](https://github.com/jim10182004/library_gnn_recsys)
""")

with st.expander("ℹ️ 隱私聲明", expanded=False):
    st.info("""
    **本 demo 完全不包含讀者資訊**。
    - ❌ 無讀者 ID / 無借閱事件 / 無讀者 demographics
    - ✅ 僅書本公開 metadata + 預訓練的 64 維書本向量
    - 📂 [資料說明](https://github.com/jim10182004/library_gnn_recsys/blob/main/DATA_CARD.md)
    - 🤖 [模型卡片](https://github.com/jim10182004/library_gnn_recsys/blob/main/MODEL_CARD.md)
    """)

item_embs_norm, books, item_remap, inv_remap = load_assets()

st.markdown("---")
st.markdown("### 選擇一個讀者類型，看模型如何推薦")

tab1, tab2, tab3 = st.tabs(["🎭 11 個人物原型", "✏️ 自訂喜歡的書", "🆚 比較兩個原型"])

with tab1:
    cols = st.columns(4)
    persona_options = list(PERSONAS.items())
    for i, (key, p) in enumerate(persona_options):
        with cols[i % 4]:
            if st.button(f"{p['emoji']} {p['name']}", key=f"p_{key}", help=p["desc"]):
                st.session_state.selected_persona = key

    if "selected_persona" in st.session_state:
        key = st.session_state.selected_persona
        p = PERSONAS[key]
        st.markdown(f"## {p['emoji']} {p['name']}")
        st.caption(p["desc"])

        seed_cids = find_seed_compact_ids(p["seed_titles"], books, item_remap)
        if not seed_cids:
            st.error("找不到該 persona 的種子書 — 可能 demo 資料太少")
        else:
            col_a, col_b = st.columns([1, 2])
            with col_a:
                st.markdown("**種子書（虛擬「他喜歡的書」）**")
                for cid in seed_cids:
                    orig = inv_remap.get(int(cid))
                    if orig is None: continue
                    row = books[books["book_id"] == orig]
                    if not row.empty:
                        r = row.iloc[0]
                        st.markdown(f"- {r.get('title')}")

            with col_b:
                k = st.slider("推薦本數", 5, 30, 10, key=f"k_{key}")
                use_mmr = st.checkbox("✨ 啟用 MMR 多樣性重排序", value=False, key=f"mmr_{key}",
                                       help="每個類別最多 6 本，避免推薦清單同質化")
                top_ids, scores = recommend(seed_cids, item_embs_norm, k=k, mmr=use_mmr,
                                            books=books, inv_remap=inv_remap)
                st.markdown(f"**模型推薦 Top-{k}**" + (" (MMR 重排序)" if use_mmr else ""))
                for idx, cid in enumerate(top_ids, 1):
                    render_book_card(idx, cid, float(scores[cid]), books, inv_remap)

with tab2:
    st.markdown("輸入你看過、喜歡的 3-5 本書，模型會找類似的書推薦。")
    query = st.text_input("搜尋書名（部分字也可）", "東野圭吾")
    matches = books[books["title"].str.contains(query, na=False, regex=False)].head(20) if query else pd.DataFrame()

    if not matches.empty:
        chosen = st.multiselect(
            "選擇你喜歡的書（建議 3-5 本）",
            options=matches["book_id"].tolist(),
            format_func=lambda bid: matches[matches["book_id"]==bid].iloc[0]["title"],
        )
        if chosen and st.button("🚀 取得推薦", type="primary"):
            seed_cids = [item_remap.get(int(bid)) for bid in chosen if item_remap.get(int(bid)) is not None]
            if seed_cids:
                top_ids, scores = recommend(seed_cids, item_embs_norm, k=10)
                st.markdown("### 推薦結果")
                for idx, cid in enumerate(top_ids, 1):
                    render_book_card(idx, cid, float(scores[cid]), books, inv_remap)

with tab3:
    st.markdown("選兩個 persona，看模型推薦的差異與重疊")
    col1, col2 = st.columns(2)
    with col1:
        a_key = st.selectbox("Persona A", list(PERSONAS.keys()),
                             format_func=lambda k: f"{PERSONAS[k]['emoji']} {PERSONAS[k]['name']}",
                             index=1, key="cmp_a")
    with col2:
        b_key = st.selectbox("Persona B", list(PERSONAS.keys()),
                             format_func=lambda k: f"{PERSONAS[k]['emoji']} {PERSONAS[k]['name']}",
                             index=5, key="cmp_b")

    if st.button("並排比較", type="primary"):
        a_seeds = find_seed_compact_ids(PERSONAS[a_key]["seed_titles"], books, item_remap)
        b_seeds = find_seed_compact_ids(PERSONAS[b_key]["seed_titles"], books, item_remap)
        a_top, a_scores = recommend(a_seeds, item_embs_norm, k=10)
        b_top, b_scores = recommend(b_seeds, item_embs_norm, k=10)
        overlap = set(a_top) & set(b_top)

        st.info(f"重疊書本：**{len(overlap)} 本**（10/10 中）" +
                (" ← 兩個 persona 的興趣高度重疊" if len(overlap) > 3 else
                 " ← 模型成功區分這兩種讀者群" if len(overlap) <= 1 else ""))
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown(f"### {PERSONAS[a_key]['emoji']} {PERSONAS[a_key]['name']}")
            for idx, cid in enumerate(a_top, 1):
                is_overlap = cid in overlap
                if is_overlap:
                    st.markdown(f"<div style='border:2px solid #f59e0b; border-radius:6px; padding:4px;'>",
                                unsafe_allow_html=True)
                render_book_card(idx, cid, float(a_scores[cid]), books, inv_remap)
                if is_overlap:
                    st.markdown("</div>", unsafe_allow_html=True)
        with col_b:
            st.markdown(f"### {PERSONAS[b_key]['emoji']} {PERSONAS[b_key]['name']}")
            for idx, cid in enumerate(b_top, 1):
                is_overlap = cid in overlap
                if is_overlap:
                    st.markdown(f"<div style='border:2px solid #f59e0b; border-radius:6px; padding:4px;'>",
                                unsafe_allow_html=True)
                render_book_card(idx, cid, float(b_scores[cid]), books, inv_remap)
                if is_overlap:
                    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("---")
st.caption(
    f"Model: LightGCN (n_items={len(books):,}, embed=64, layers=3) | "
    "Inference: CPU cosine similarity (~50 ms) | "
    "[更多細節](https://github.com/jim10182004/library_gnn_recsys)"
)
