"""
Streamlit 互動式 Demo

啟動方式：
    streamlit run app.py

功能：
  - 輸入讀者 ID 或從清單挑選
  - 顯示借閱歷史
  - 顯示 4 個模型的 Top-10 推薦並排比較
  - 顯示讀者人口統計（性別、年齡）
"""
from __future__ import annotations
import sys
from pathlib import Path

# Windows: pandas/pyarrow 必須在 torch 之前
import pandas as pd
import pyarrow  # noqa: F401
import numpy as np

import streamlit as st

# 專案 import 路徑
_PROJ_ROOT = Path(__file__).resolve().parent
if str(_PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJ_ROOT))

import torch
from src.dataset import load_splits
from src.models.lightgcn import LightGCN, build_norm_adj
from src.models.lightgcn_si import LightGCNSI, build_side_info_tensors
from src.models.baselines import BPRMF, PopularRecommender


PROJECT = Path(__file__).parent
PROCESSED = PROJECT / "data" / "processed"
CKPT = PROJECT / "checkpoints"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------- 快取載入

@st.cache_resource
def get_splits():
    return load_splits()

@st.cache_resource
def get_books():
    return pd.read_parquet(PROCESSED / "books.parquet")

@st.cache_resource
def get_users():
    return pd.read_parquet(PROCESSED / "users.parquet")

@st.cache_resource
def get_lightgcn(_splits):
    m = LightGCN(_splits.n_users, _splits.n_items, embed_dim=64, n_layers=3).to(DEVICE)
    state = torch.load(CKPT / "lightgcn_best.pt", map_location=DEVICE, weights_only=True)
    m.load_state_dict(state)
    train_u = torch.as_tensor(_splits.train["u"].values, dtype=torch.long)
    train_i = torch.as_tensor(_splits.train["i"].values, dtype=torch.long)
    A_hat = build_norm_adj(train_u, train_i, _splits.n_users, _splits.n_items, device=DEVICE)
    m.set_graph(A_hat)
    m.eval()
    return m

@st.cache_resource
def get_lightgcn_si(_splits, _books_df, _users_df):
    if not (CKPT / "lightgcn_si_best.pt").exists():
        return None
    m = LightGCNSI(
        _splits.n_users, _splits.n_items,
        n_genders=3, n_age_buckets=8, n_categories=11,
        embed_dim=64, n_layers=3,
    ).to(DEVICE)
    state = torch.load(CKPT / "lightgcn_si_best.pt", map_location=DEVICE, weights_only=True)
    m.load_state_dict(state)
    train_u = torch.as_tensor(_splits.train["u"].values, dtype=torch.long)
    train_i = torch.as_tensor(_splits.train["i"].values, dtype=torch.long)
    A_hat = build_norm_adj(train_u, train_i, _splits.n_users, _splits.n_items, device=DEVICE)
    m.set_graph(A_hat)
    g, a, c, _ = build_side_info_tensors(_splits, _books_df, _users_df)
    m.set_side_info(g.to(DEVICE), a.to(DEVICE), c.to(DEVICE))
    m.eval()
    return m

@st.cache_resource
def get_bprmf(_splits):
    if not (CKPT / "bprmf_best.pt").exists():
        return None
    m = BPRMF(_splits.n_users, _splits.n_items, embed_dim=64).to(DEVICE)
    state = torch.load(CKPT / "bprmf_best.pt", map_location=DEVICE, weights_only=True)
    m.load_state_dict(state)
    m.eval()
    return m

@st.cache_resource
def get_popular(_splits):
    m = PopularRecommender()
    m.fit(_splits.train["u"].values, _splits.train["i"].values, _splits.n_items)
    return m

# ---------------------------------------------------------------- 推薦邏輯

def recommend_torch(model, user_id, splits, k=10):
    with torch.no_grad():
        u_t = torch.as_tensor([user_id], dtype=torch.long, device=DEVICE)
        scores = model.get_all_ratings(u_t).cpu().numpy()[0]
    seen = splits.train[splits.train["u"] == user_id]["i"].values
    scores[seen] = -np.inf
    top = np.argpartition(-scores, kth=k)[:k]
    top = top[np.argsort(-scores[top])]
    return top, scores[top]

def recommend_popular(model, user_id, splits, k=10):
    scores = model.get_all_ratings(np.array([user_id]))[0].astype(np.float64)
    seen = splits.train[splits.train["u"] == user_id]["i"].values
    scores[seen] = -np.inf
    top = np.argpartition(-scores, kth=k)[:k]
    top = top[np.argsort(-scores[top])]
    return top, scores[top]

def get_history(splits, u, books, n=10):
    train = splits.train
    hist = train[train["u"] == u].sort_values("ts", ascending=False).head(n)
    book_map_inv = {v: k for k, v in splits.item_remap.items()}
    hist = hist.copy()
    hist["book_id_orig"] = hist["i"].map(book_map_inv)
    out = hist.merge(books[["book_id", "title", "author", "category"]],
                     left_on="book_id_orig", right_on="book_id", how="left")
    return out[["ts", "title", "author", "category"]]

def items_to_df(item_ids, scores, splits, books):
    book_map_inv = {v: k for k, v in splits.item_remap.items()}
    rows = []
    for rank, (i, s) in enumerate(zip(item_ids, scores), 1):
        orig = book_map_inv[int(i)]
        meta = books[books["book_id"] == orig]
        if meta.empty:
            rows.append({"#": rank, "Score": f"{s:.3f}", "Title": "?", "Author": "?", "Cat": "?"})
            continue
        meta = meta.iloc[0]
        rows.append({
            "#": rank,
            "Score": f"{s:.3f}",
            "Title": (meta["title"] or "?")[:50],
            "Author": (meta["author"] or "?")[:30],
            "Cat": meta["category"] or "?",
        })
    return pd.DataFrame(rows)

# ---------------------------------------------------------------- UI

st.set_page_config(page_title="圖書館 GNN 推薦系統", layout="wide")
st.title("圖書館 GNN 推薦系統 Demo")
st.caption("LightGCN（圖神經網路） vs 傳統推薦法的比較 | 畢業專題")

with st.spinner("載入資料與模型 ..."):
    splits = get_splits()
    books = get_books()
    users = get_users()
    lgcn = get_lightgcn(splits)
    lgcn_si = get_lightgcn_si(splits, books, users)
    bprmf = get_bprmf(splits)
    popular = get_popular(splits)

st.success(
    f"模型已載入：n_users={splits.n_users:,}, n_items={splits.n_items:,}, "
    f"訓練互動={len(splits.train):,}, 設備={DEVICE}"
)

# Sidebar
st.sidebar.header("選擇讀者")
mode = st.sidebar.radio("輸入方式", ["隨機抽取", "輸入緊湊 ID", "輸入原始 user_orig"])

if mode == "隨機抽取":
    if st.sidebar.button("重新抽一位"):
        st.session_state.random_seed = np.random.randint(10000)
    seed = st.session_state.get("random_seed", 42)
    rng = np.random.default_rng(seed)
    train_users = splits.train["u"].unique()
    user_id = int(rng.choice(train_users))
elif mode == "輸入緊湊 ID":
    user_id = st.sidebar.number_input(
        "緊湊 user id (0 ~ n-1)", min_value=0, max_value=splits.n_users - 1, value=100,
    )
else:
    orig = st.sidebar.number_input("原始 user_orig", min_value=0, value=1007827)
    if orig in splits.user_remap:
        user_id = splits.user_remap[int(orig)]
    else:
        st.sidebar.error("此原始 ID 未通過 k-core 過濾，無模型可推薦")
        st.stop()

k = st.sidebar.slider("Top-K", 5, 30, 10)
st.sidebar.markdown("---")
st.sidebar.markdown("### 顯示模型")
show_popular = st.sidebar.checkbox("Popular", value=True)
show_bprmf = st.sidebar.checkbox("BPR-MF", value=True)
show_lgcn = st.sidebar.checkbox("LightGCN", value=True)
show_lgcn_si = st.sidebar.checkbox("LightGCN-SI", value=True)

# 主畫面
inv_user = {v: k for k, v in splits.user_remap.items()}
orig = inv_user.get(user_id, "?")

# 讀者資訊
user_info = users[users["user_orig"] == orig]
if not user_info.empty:
    info = user_info.iloc[0]
    st.markdown(f"### 讀者 #{user_id}（原始 ID: {orig}）　|　性別：{info['gender'] or '?'}　|　年齡：{int(info['age']) if pd.notna(info['age']) else '?'}")
else:
    st.markdown(f"### 讀者 #{user_id}（原始 ID: {orig}）")

# 借閱歷史
st.markdown("#### 借閱歷史 (最近 10 本)")
hist = get_history(splits, user_id, books, n=10)
if hist.empty:
    st.warning("此讀者在訓練集中沒有借閱紀錄")
else:
    hist_show = hist.copy()
    hist_show["ts"] = pd.to_datetime(hist_show["ts"]).dt.strftime("%Y-%m-%d")
    st.dataframe(hist_show.rename(columns={
        "ts": "日期", "title": "書名", "author": "作者", "category": "分類號"
    }), use_container_width=True, hide_index=True)

# 推薦比較
st.markdown(f"#### Top-{k} 推薦 (各模型並排)")

cols_to_show = []
if show_lgcn_si and lgcn_si is not None:
    cols_to_show.append(("LightGCN-SI", lgcn_si, "torch"))
if show_lgcn:
    cols_to_show.append(("LightGCN", lgcn, "torch"))
if show_bprmf and bprmf is not None:
    cols_to_show.append(("BPR-MF", bprmf, "torch"))
if show_popular:
    cols_to_show.append(("Popular", popular, "popular"))

if cols_to_show:
    cols = st.columns(len(cols_to_show))
    for col, (name, model, kind) in zip(cols, cols_to_show):
        with col:
            st.markdown(f"**{name}**")
            if kind == "torch":
                top, scores = recommend_torch(model, user_id, splits, k=k)
            else:
                top, scores = recommend_popular(model, user_id, splits, k=k)
            df = items_to_df(top, scores, splits, books)
            st.dataframe(df, use_container_width=True, hide_index=True, height=min(800, 36 + 35 * k))
else:
    st.info("請從左側選擇至少一個模型")

# 整體指標
with st.expander("各模型 Test Set 整體表現"):
    summary_path = PROJECT / "results" / "summary.csv"
    if summary_path.exists():
        summary = pd.read_csv(summary_path)
        st.dataframe(summary, use_container_width=True, hide_index=True)
    else:
        st.info("尚未產生 summary，先執行 python src\\visualize.py")

st.markdown("---")
st.caption("專題作者：[你的名字]　|　指導教授：[教授名字]　|　模型：LightGCN (SIGIR 2020)")
