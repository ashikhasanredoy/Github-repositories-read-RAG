from fastapi.testclient import TestClient
from src.code_rag.api.main import app

def test_root_serves_html():
    client = TestClient(app)
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers.get("content-type", "")
    assert "Code RAG" in response.text

def test_static_assets():
    client = TestClient(app)
    css_res = client.get("/static/styles.css")
    assert css_res.status_code == 200

    js_res = client.get("/static/app.js")
    assert js_res.status_code == 200

def test_api_health():
    client = TestClient(app)
    res = client.get("/api/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "healthy"
