"""
測試 FastAPI endpoints — 用 TestClient 直接呼叫，不需啟動 uvicorn。
若沒有 checkpoint 或 processed parquet 則 skip。
"""
from __future__ import annotations
import sys
from pathlib import Path

import pytest

PROJECT = Path(__file__).resolve().parent.parent
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))


@pytest.fixture(scope="module")
def client():
    """啟動 FastAPI TestClient（lifespan 會自動 trigger）"""
    proc = PROJECT / "data" / "processed"
    ckpt = PROJECT / "checkpoints" / "lightgcn_best.pt"
    if not (proc / "books.parquet").exists():
        pytest.skip("data/processed/books.parquet 不存在")
    if not ckpt.exists():
        pytest.skip("checkpoints/lightgcn_best.pt 不存在")

    try:
        from fastapi.testclient import TestClient
        from api.main import app
    except ImportError as e:
        pytest.skip(f"無法匯入 FastAPI 或 api.main: {e}")

    with TestClient(app) as c:
        yield c


# ============= 基本端點 =============

def test_health_endpoint(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "ok"
    assert data["n_items"] > 0


def test_root_serves_html(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "text/html" in r.headers.get("content-type", "")
    assert len(r.content) > 1000  # html 應該不少


def test_personas_list(client):
    r = client.get("/api/personas")
    assert r.status_code == 200
    personas = r.json()
    assert isinstance(personas, list)
    assert len(personas) >= 5  # 至少 5 個 persona
    # 每個 persona 應該有必要欄位
    for p in personas:
        assert "key" in p
        assert "name" in p
        assert "emoji" in p


def test_stats_endpoint(client):
    r = client.get("/api/stats")
    assert r.status_code == 200
    data = r.json()
    assert "best_model" in data
    assert "n_users" in data
    assert "n_items" in data


# ============= 搜尋 =============

def test_search_returns_results(client):
    """搜尋常見字應該有結果"""
    r = client.get("/api/search", params={"q": "東", "n": 5})
    assert r.status_code == 200
    results = r.json()
    assert isinstance(results, list)
    if len(results) == 0:
        pytest.skip("資料中沒有含『東』的書名")
    # 每個結果應該有完整 metadata
    for book in results:
        assert "book_id" in book
        assert "title" in book


def test_search_empty_query_rejected(client):
    """空查詢應該被擋（pydantic validation）"""
    r = client.get("/api/search", params={"q": "", "n": 5})
    # 422 = unprocessable entity（schema validation 失敗）
    assert r.status_code in (400, 422)


def test_search_n_limit(client):
    """n 參數限制結果數量"""
    r = client.get("/api/search", params={"q": "的", "n": 3})
    assert r.status_code == 200
    results = r.json()
    assert len(results) <= 3


# ============= 推薦 =============

def test_recommend_returns_no_seed_overlap(client):
    """推薦結果不應該包含 seed books"""
    # 先搜兩本書當 seed
    r = client.get("/api/search", params={"q": "推理", "n": 3})
    assert r.status_code == 200
    seeds = r.json()
    if len(seeds) < 2:
        pytest.skip("資料中找不到兩本含『推理』的書")
    seed_ids = [s["book_id"] for s in seeds[:2]]

    r = client.post(
        "/api/recommend",
        json={"book_ids": seed_ids, "k": 10},
    )
    assert r.status_code == 200
    data = r.json()
    assert "recommendations" in data
    rec_ids = [b["book_id"] for b in data["recommendations"]]
    # 推薦不應該包含 seed
    for sid in seed_ids:
        assert sid not in rec_ids, f"seed {sid} 出現在推薦中"


def test_recommend_k_respected(client):
    """k=5 應該回 5 本（或少於 5）"""
    r = client.get("/api/search", params={"q": "推理", "n": 1})
    assert r.status_code == 200
    seeds = r.json()
    if not seeds:
        pytest.skip("資料中沒有含『推理』的書")

    r = client.post("/api/recommend", json={"book_ids": [seeds[0]["book_id"]], "k": 5})
    assert r.status_code == 200
    data = r.json()
    assert len(data["recommendations"]) <= 5


def test_recommend_empty_book_ids_rejected(client):
    """空 seed 應該被擋"""
    r = client.post("/api/recommend", json={"book_ids": [], "k": 10})
    assert r.status_code in (400, 422)


def test_recommend_invalid_book_id(client):
    """不存在的 book_id 應該回 400 或空結果"""
    r = client.post(
        "/api/recommend",
        json={"book_ids": [9999999], "k": 10},
    )
    # 接受 400（找不到）或 200+空結果
    assert r.status_code in (200, 400, 404)


# ============= Persona 端點 =============

def test_persona_japanese_mystery(client):
    """日系推理 persona 應該回推薦"""
    r = client.get("/api/persona/japanese_mystery", params={"k": 5})
    assert r.status_code == 200
    data = r.json()
    assert "recommendations" in data
    assert "seeds" in data
    assert len(data["recommendations"]) <= 5


def test_persona_unknown_key_404(client):
    """未知 persona key 回 404"""
    r = client.get("/api/persona/this_persona_does_not_exist", params={"k": 5})
    assert r.status_code in (400, 404)


def test_persona_recommendation_excludes_seeds(client):
    """Persona 推薦不應該包含種子書"""
    r = client.get("/api/persona/japanese_mystery", params={"k": 10})
    assert r.status_code == 200
    data = r.json()
    seed_ids = {s["book_id"] for s in data["seeds"]}
    rec_ids = {b["book_id"] for b in data["recommendations"]}
    overlap = seed_ids & rec_ids
    assert len(overlap) == 0, f"推薦包含 seed: {overlap}"
