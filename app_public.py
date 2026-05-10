"""
公開安全版 Streamlit Demo

設計原則：
  - 不顯示真實讀者 ID
  - 提供「人物原型 (Persona)」demo：4-5 種預設讀者類型
  - 提供「自訂讀者」模式：使用者選 3~5 本書作為「我喜歡的書」，模型即時推薦
  - 顯眼的去識別化聲明

執行方式：
    streamlit run app_public.py

部署：
    - Hugging Face Spaces：直接放到 repo 根目錄
    - Streamlit Cloud：同上
    - 本機測試：streamlit run app_public.py
"""
from __future__ import annotations
import sys
from pathlib import Path

import pandas as pd
import pyarrow  # noqa: F401
import numpy as np
import streamlit as st

_PROJ = Path(__file__).resolve().parent
if str(_PROJ) not in sys.path:
    sys.path.insert(0, str(_PROJ))

import torch
from src.dataset import load_splits
from src.metrics_summary import best_model
from src.models.lightgcn import LightGCN, build_norm_adj


PROCESSED = _PROJ / "data" / "processed"
CKPT = _PROJ / "checkpoints"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ============= Personas（預設讀者原型） =============
# 每個 persona 用「他喜歡的書」(book_id 列表) 來表達

PERSONAS = {
    "child_en": {
        "name": "兒童英文書愛好者",
        "emoji": "📚",
        "desc": "喜歡 Magic Tree House、Toy Story 等英文章節書與繪本",
        "seed_titles": [
            "Magic Tree House",
            "Toy story",
            "Mittens",
            "Fox versus winter",
        ],
    },
    "japanese_mystery": {
        "name": "日系推理小說迷",
        "emoji": "🔍",
        "desc": "東野圭吾、西澤保彥、宮部美幸的日本推理粉絲",
        "seed_titles": [
            "白金數據",
            "嫌疑犯X的獻身",
            "解憂雜貨店",
            "死了七次的男人",
        ],
    },
    "self_help": {
        "name": "職場/自我成長",
        "emoji": "💼",
        "desc": "閱讀職涯、心理、效率提升相關書籍",
        "seed_titles": [
            "原子習慣",
            "拖延心理學",
            "高效能人士的七個習慣",
            "目標 : 簡單有效的常識管理",
        ],
    },
    "academic": {
        "name": "學術派",
        "emoji": "🎓",
        "desc": "教科書、論文工具書、專業領域書籍",
        "seed_titles": [
            "Introduction to Probability and Statistics",
            "深度學習",
            "Pattern Recognition",
            "計量經濟學",
        ],
    },
    "design_art": {
        "name": "設計與藝術",
        "emoji": "🎨",
        "desc": "美學、攝影、平面設計相關書籍",
        "seed_titles": [
            "設計的法則",
            "字型之不思議",
            "色彩學",
            "攝影師之眼",
        ],
    },
}


# ============= 載入資料與模型（只載入一次） =============

@st.cache_resource(show_spinner=True)
def get_data_and_model():
    splits = load_splits()
    books = pd.read_parquet(PROCESSED / "books.parquet")
    model = LightGCN(splits.n_users, splits.n_items, embed_dim=64, n_layers=3).to(DEVICE)
    state = torch.load(CKPT / "lightgcn_best.pt", map_location=DEVICE, weights_only=True)
    model.load_state_dict(state)
    train_u = torch.as_tensor(splits.train["u"].values, dtype=torch.long)
    train_i = torch.as_tensor(splits.train["i"].values, dtype=torch.long)
    A_hat = build_norm_adj(train_u, train_i, splits.n_users, splits.n_items, device=DEVICE)
    model.set_graph(A_hat)
    model.eval()

    # 預先計算所有 item 的 embedding
    with torch.no_grad():
        _, item_embs = model.propagate()
    return splits, books, model, item_embs


def find_book_indices_by_titles(books: pd.DataFrame, splits, titles: list[str]) -> list[int]:
    """根據書名（部分匹配）找出 compact item id。"""
    out = []
    for t in titles:
        # 模糊匹配：找包含 t 的書
        matches = books[books["title"].str.contains(t, na=False, regex=False)]
        if matches.empty:
            continue
        # 取第一個匹配
        orig_book_id = int(matches.iloc[0]["book_id"])
        if orig_book_id in splits.item_remap:
            out.append(splits.item_remap[orig_book_id])
    return out


def synthetic_user_recommend(item_embs: torch.Tensor, liked_book_ids: list[int], k: int = 10) -> tuple[np.ndarray, np.ndarray]:
    """
    合成讀者推薦（cosine similarity 版）：
      1. 取喜歡的書 embedding 平均 → 合成 user vector（normalise）
      2. 對所有書計算 cosine 相似度
      3. mask 掉喜歡的，回傳 Top-K

    為什麼用 cosine 而非內積：避免高頻書籍因 norm 大而被偏好推薦
    """
    if not liked_book_ids:
        return np.array([]), np.array([])
    liked_t = torch.as_tensor(liked_book_ids, dtype=torch.long, device=item_embs.device)
    seed_emb = torch.nn.functional.normalize(item_embs[liked_t], p=2, dim=1)
    user_vec = seed_emb.mean(dim=0)
    user_vec = torch.nn.functional.normalize(user_vec, p=2, dim=0)
    all_norm = torch.nn.functional.normalize(item_embs, p=2, dim=1)
    scores = (all_norm @ user_vec).cpu().numpy()
    scores[liked_book_ids] = -np.inf
    top = np.argpartition(-scores, kth=k)[:k]
    top = top[np.argsort(-scores[top])]
    return top, scores[top]


def items_to_df(item_ids, scores, splits, books):
    book_map_inv = {v: k for k, v in splits.item_remap.items()}
    rows = []
    for rank, (i, s) in enumerate(zip(item_ids, scores), 1):
        orig = book_map_inv[int(i)]
        meta = books[books["book_id"] == orig]
        if meta.empty:
            continue
        meta = meta.iloc[0]
        rows.append({
            "排名": rank,
            "分數": f"{s:.3f}",
            "書名": (meta["title"] or "?")[:60],
            "作者": (meta["author"] or "?")[:30],
            "分類": meta["category"] or "?",
        })
    return pd.DataFrame(rows)


def format_metric(value):
    return "NA" if value is None else f"{value:.3f}"


# ============= UI =============

st.set_page_config(page_title="圖書館 GNN 推薦系統 Demo", page_icon=None, layout="wide")

# Header
col1, col2 = st.columns([3, 1])
with col1:
    st.title("圖書館書籍推薦系統")
    st.caption("基於圖神經網路（LightGCN, SIGIR 2020）的個人化推薦")
with col2:
    st.markdown(
        '<div style="background:#fef9e7;padding:10px;border-radius:8px;'
        'border-left:4px solid #f0b400;font-size:13px">'
        '<b>隱私聲明</b><br>本 demo 使用之資料來源已去識別化，'
        '不顯示任何真實讀者資訊。'
        '</div>', unsafe_allow_html=True)

# 載入
with st.spinner("載入模型 ..."):
    splits, books, model, item_embs = get_data_and_model()
best = best_model("recall@10")

# Sidebar
st.sidebar.header("Demo 模式")
mode = st.sidebar.radio(
    "選擇互動方式",
    ["人物原型 (Persona)", "自訂喜歡的書", "關於系統"],
)
k = st.sidebar.slider("Top-K 推薦數", 5, 30, 10)

st.sidebar.markdown("---")
st.sidebar.markdown("### 模型資訊")
st.sidebar.markdown(f"- 架構：**LightGCN**\n- 讀者數：{splits.n_users:,}\n- 書籍數：{splits.n_items:,}\n- 訓練互動：{len(splits.train):,}")
if best:
    st.sidebar.markdown(
        f"- 最佳模型：**{best['model']}**\n"
        f"- Recall@10：**{format_metric(best['recall@10'])}**\n"
        f"- NDCG@10：**{format_metric(best['ndcg@10'])}**"
    )
else:
    st.sidebar.markdown("- 評估：尚未產生 `results/summary.csv`")
st.sidebar.markdown("---")
st.sidebar.markdown("[GitHub](https://github.com) | 畢業專題")

# ------ Mode 1: Persona ------
if mode.startswith("人物原型"):
    st.subheader("選擇一個讀者原型，看模型會推薦什麼")
    cols = st.columns(len(PERSONAS))
    selected = None
    for i, (key, persona) in enumerate(PERSONAS.items()):
        with cols[i]:
            with st.container(border=True):
                st.markdown(f"#### {persona['emoji']}　{persona['name']}")
                st.caption(persona["desc"])
                if st.button("看推薦", key=f"persona_{key}", use_container_width=True):
                    st.session_state.selected_persona = key

    if "selected_persona" in st.session_state:
        key = st.session_state.selected_persona
        persona = PERSONAS[key]
        st.markdown("---")
        st.markdown(f"### {persona['emoji']} {persona['name']}")

        # 找對應的書
        liked = find_book_indices_by_titles(books, splits, persona["seed_titles"])
        if not liked:
            st.error("此 persona 的種子書名在資料中找不到（可能訓練時 k-core 過濾掉了）")
        else:
            st.markdown(f"**該讀者喜歡的 {len(liked)} 本書（種子）**")
            seed_df = items_to_df(liked, [0.0] * len(liked), splits, books)
            st.dataframe(seed_df.drop(columns=["分數", "排名"]),
                         use_container_width=True, hide_index=True)

            st.markdown(f"**LightGCN 推薦 Top-{k}**")
            top, scores = synthetic_user_recommend(item_embs, liked, k=k)
            rec_df = items_to_df(top, scores, splits, books)
            st.dataframe(rec_df, use_container_width=True, hide_index=True)

# ------ Mode 2: Custom ------
elif mode.startswith("自訂"):
    st.subheader("挑 3-5 本你喜歡的書，看推薦")
    st.caption("輸入書名（部分文字即可）—— 系統會找到第一個匹配的，"
               "再用合成 user vector 算出推薦")

    n_inputs = st.slider("你想輸入幾本書？", 1, 5, 3)
    inputs = []
    cols = st.columns(min(n_inputs, 3))
    for i in range(n_inputs):
        with cols[i % len(cols)]:
            v = st.text_input(f"書 {i+1}", key=f"book_input_{i}",
                              placeholder="例：原子習慣")
            inputs.append(v.strip())

    if st.button("開始推薦", type="primary", use_container_width=True):
        liked = find_book_indices_by_titles(books, splits, [t for t in inputs if t])
        if not liked:
            st.error("找不到任何輸入的書 (可能未在訓練集中)")
        else:
            book_map_inv = {v: k for k, v in splits.item_remap.items()}
            st.success(f"找到 {len(liked)} 本書")
            seed_df = items_to_df(liked, [0.0] * len(liked), splits, books)
            st.markdown("**你選的書**")
            st.dataframe(seed_df.drop(columns=["分數", "排名"]),
                         use_container_width=True, hide_index=True)

            st.markdown(f"**LightGCN 推薦 Top-{k}**")
            top, scores = synthetic_user_recommend(item_embs, liked, k=k)
            rec_df = items_to_df(top, scores, splits, books)
            st.dataframe(rec_df, use_container_width=True, hide_index=True)

# ------ Mode 3: About ------
else:
    st.markdown("""
### 關於這個系統

本系統為畢業專題之研究成果，使用 **圖神經網路 (Graph Neural Network)** 中的 **LightGCN** 模型，
在某市立圖書館 2025 年完整年度借閱與預約資料上訓練，建立個人化書籍推薦系統。

#### 技術細節
- **資料**：去識別化的 130 萬筆借閱 + 32 萬筆預約
- **規模**：35,856 位讀者 × 29,685 本書（k-core 過濾後）
- **模型**：LightGCN（3 層、64 維 embedding）+ 多種擴充版本
- **比較對象**：Popular、ItemCF、BPR-MF、NGCF

#### 主要結果
| 模型 | Recall@10 | NDCG@10 |
|------|-----------|---------|
| Popular | 0.2532 | 0.2169 |
| BPR-MF | 0.2544 | 0.2087 |
| **LightGCN** | **0.2648** | **0.2178** |

#### 推薦邏輯（本 Demo）
為了保護隱私，本 demo 不直接使用任何真實讀者，而是：
1. 你選擇「喜歡的書」
2. 系統取這些書的 embedding 平均，作為「合成讀者」
3. 計算合成讀者與所有書籍的相似度，回傳 Top-K

這同時也展示了模型對「冷啟動讀者」的處理能力。

#### 資料隱私
- 所有讀者 ID 已透過對照表匿名化
- Demo 不顯示任何真實讀者識別資訊
- 圖書館名稱已隱去
""")

st.markdown("---")
st.caption("基於圖神經網路之市立圖書館書籍推薦系統 | 畢業專題 | 2026")
