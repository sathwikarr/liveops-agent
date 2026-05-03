"""HTTP-level tests for the FastAPI website.

These use fastapi.testclient — same WSGI/ASGI plumbing as production but
in-process, so they're fast and don't need a live server.

We use a fresh app per test so SessionMiddleware's signing key is consistent
within the test, and DB state created by signup tests stays scoped.
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #

@pytest.fixture
def client(tmp_path, monkeypatch):
    """Spin up the FastAPI app with an isolated SQLite DB so signup/login
    tests don't pollute each other."""
    db_path = tmp_path / "liveops_test.db"
    monkeypatch.setenv("LIVEOPS_DB", str(db_path))
    monkeypatch.setenv("SESSION_SECRET", "test-secret-deterministic")
    monkeypatch.setenv("SLACK_WEBHOOK", "")
    monkeypatch.setenv("SMTP_HOST", "")

    # Reload agent.db so module-level DB_PATH picks up the env override.
    import importlib, agent.db as _db
    importlib.reload(_db)
    _db.init_db()

    # Import inside the fixture so env overrides are picked up.
    from web.server import create_app
    app = create_app()
    with TestClient(app) as c:
        yield c


# --------------------------------------------------------------------------- #
# Liveness + public pages
# --------------------------------------------------------------------------- #

def test_healthz(client):
    r = client.get("/healthz")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["service"] == "liveops-agent"


def test_version(client):
    r = client.get("/version")
    assert r.status_code == 200
    assert r.json()["app"] == "liveops-agent"


def test_landing_renders(client):
    r = client.get("/")
    assert r.status_code == 200
    # Sanity: page contains the brand and at least one CTA.
    assert "LiveOps Agent" in r.text
    assert "Try the demo" in r.text


def test_demo_renders_with_stats(client):
    r = client.get("/demo")
    assert r.status_code == 200
    # KPI labels present
    assert "Revenue" in r.text
    assert "Customers" in r.text


def test_workbench_lists_tools(client):
    r = client.get("/workbench")
    assert r.status_code == 200
    # Tool names present in the catalog sidebar
    assert "top_products" in r.text
    assert "churn_risk" in r.text


def test_evals_page_renders(client):
    r = client.get("/evals")
    assert r.status_code == 200
    assert "eval harness" in r.text.lower()


# --------------------------------------------------------------------------- #
# Auth-gated pages redirect to /login
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("path", ["/dashboard", "/run-agent"])
def test_auth_pages_redirect_when_anonymous(client, path):
    r = client.get(path, follow_redirects=False)
    assert r.status_code == 303
    assert "/login" in r.headers["location"]
    assert f"next={path}" in r.headers["location"]


# --------------------------------------------------------------------------- #
# Signup → login → dashboard flow
# --------------------------------------------------------------------------- #

def test_signup_login_dashboard_flow(client):
    # Signup form should render
    r = client.get("/signup")
    assert r.status_code == 200
    assert "Create account" in r.text

    # Submit signup — auto-logs in, redirects to dashboard
    r = client.post("/signup", data={
        "username": "alice", "password": "supersecret", "confirm": "supersecret",
        "next": "/dashboard",
    }, follow_redirects=False)
    assert r.status_code == 303
    assert r.headers["location"] == "/dashboard"

    # Following the redirect should now hit dashboard (auth cookie set)
    r = client.get("/dashboard")
    assert r.status_code == 200
    assert "alice" in r.text

    # Logout clears session, dashboard redirects to login again
    r = client.post("/logout", follow_redirects=False)
    assert r.status_code == 303
    assert r.headers["location"] == "/"
    r = client.get("/dashboard", follow_redirects=False)
    assert r.status_code == 303
    assert "/login" in r.headers["location"]


def test_signup_password_mismatch(client):
    r = client.post("/signup", data={
        "username": "bob", "password": "abcdefgh", "confirm": "different",
    }, follow_redirects=False)
    assert r.status_code == 303
    assert "/signup" in r.headers["location"]
    # Flash should mention the mismatch
    r2 = client.get("/signup")
    assert "don't match" in r2.text.lower() or "Passwords" in r2.text


def test_signup_short_password_rejected(client):
    r = client.post("/signup", data={
        "username": "carol", "password": "short", "confirm": "short",
    }, follow_redirects=False)
    assert r.status_code == 303
    assert "/signup" in r.headers["location"]


def test_login_with_wrong_password(client):
    # Create an account first
    client.post("/signup", data={
        "username": "dave", "password": "rightpass1", "confirm": "rightpass1",
    })
    # Log out
    client.post("/logout")
    # Wrong password rejected
    r = client.post("/login", data={
        "username": "dave", "password": "wrongpass1",
    }, follow_redirects=False)
    assert r.status_code == 303
    assert "/login" in r.headers["location"]


# --------------------------------------------------------------------------- #
# JSON APIs
# --------------------------------------------------------------------------- #

def test_api_agent_ask_returns_plan(client):
    r = client.post("/api/agent/ask", json={
        "question": "top 5 products by revenue", "backend": "heuristic",
    })
    assert r.status_code == 200
    body = r.json()
    assert "answer" in body
    assert body["plan"]["backend"] == "heuristic"
    tools = [s["tool"] for s in body["plan"]["steps"]]
    assert "top_products" in tools


def test_api_agent_ask_validates(client):
    r = client.post("/api/agent/ask", json={"question": ""})
    assert r.status_code == 400


def test_api_agent_ask_rejects_unknown_backend(client):
    r = client.post("/api/agent/ask", json={"question": "x", "backend": "rumour"})
    assert r.status_code == 400
    assert "backend must be" in r.json()["detail"]


def test_api_agent_ask_returns_backend_meta_no_fallback(client):
    """Heuristic-on-heuristic — no fallback, reason should be None."""
    r = client.post("/api/agent/ask",
                    json={"question": "top 5 products", "backend": "heuristic"})
    body = r.json()
    meta = body["backend_meta"]
    assert meta["requested"] == "heuristic"
    assert meta["actual"]    == "heuristic"
    assert meta["fallback"]  is False
    assert meta["info"]      is False
    assert meta["reason"]    is None


def test_api_agent_ask_backend_meta_signals_fallback(client, monkeypatch):
    """When LLM is requested but the planner returns None, response must
    flag fallback=True and include a user-readable reason string."""
    # Force _llm_plan to return None so the agent always falls back.
    import analyst.agent as agent_mod
    monkeypatch.setattr(agent_mod, "_llm_plan", lambda *a, **kw: None)
    # Pretend the LLM IS available so the reason discriminates "call failed"
    # from "key missing".
    monkeypatch.setattr(agent_mod, "_llm_available", lambda: True)

    r = client.post("/api/agent/ask",
                    json={"question": "top 5 products", "backend": "llm"})
    meta = r.json()["backend_meta"]
    assert meta["requested"] == "llm"
    assert meta["actual"]    == "heuristic"
    assert meta["fallback"]  is True
    assert "LLM call failed" in meta["reason"]


def test_api_agent_ask_backend_meta_when_key_missing(client, monkeypatch):
    """auto + llm-not-available -> info=True, reason cites missing key."""
    import analyst.agent as agent_mod
    monkeypatch.setattr(agent_mod, "_llm_plan",      lambda *a, **kw: None)
    monkeypatch.setattr(agent_mod, "_llm_available", lambda: False)

    r = client.post("/api/agent/ask",
                    json={"question": "top 5 products", "backend": "auto"})
    meta = r.json()["backend_meta"]
    assert meta["requested"]     == "auto"
    assert meta["actual"]        == "heuristic"
    assert meta["info"]          is True
    assert meta["fallback"]      is False
    assert meta["llm_available"] is False
    assert "GEMINI_API_KEY" in meta["reason"]


def test_api_evals_baseline(client):
    r = client.get("/api/evals/baseline")
    assert r.status_code == 200
    # Either present or null — both are valid responses.
    body = r.json()
    assert "baseline" in body


def test_api_evals_run_smoke(client):
    """Run a tiny tag-filtered subset so this test stays fast."""
    r = client.post("/api/evals/run", json={"backend": "heuristic", "tags": ["easy"]})
    assert r.status_code == 200
    report = r.json()
    assert "pass_rate" in report
    assert report["n_cases"] >= 1
    assert 0.0 <= report["pass_rate"] <= 1.0


def test_api_workbench_preview(client):
    r = client.get("/api/workbench/preview")
    assert r.status_code == 200
    body = r.json()
    assert body["n_rows"] > 0
    assert isinstance(body["columns"], list)
    assert isinstance(body["head"], list)


def test_api_workbench_tool_dispatch(client):
    r = client.post("/api/workbench/tool", json={
        "tool": "top_products", "args": {"n": 3},
    })
    assert r.status_code == 200
    body = r.json()
    assert body["tool"] == "top_products"
    assert isinstance(body["result"], list)
    assert len(body["result"]) == 3


def test_api_workbench_tool_unknown(client):
    r = client.post("/api/workbench/tool", json={"tool": "does_not_exist"})
    assert r.status_code == 400


# --------------------------------------------------------------------------- #
# Workbench upload + reset
# --------------------------------------------------------------------------- #

def _mini_orders_csv() -> bytes:
    """Smallest possible valid orders CSV — has the two required role keys
    (date + amount) and enough columns to be useful."""
    return (
        b"order_id,customer_id,product_id,order_date,amount,quantity,region,channel\n"
        b"O1,C1,SKU-001,2025-12-01,42.50,1,NA-East,web\n"
        b"O2,C2,SKU-002,2025-12-02,17.00,2,EMEA,mobile\n"
        b"O3,C1,SKU-001,2025-12-03,42.50,1,NA-East,web\n"
        b"O4,C3,SKU-003,2025-12-04,99.99,1,APAC,retail\n"
    )


def test_workbench_upload_happy_path(client, tmp_path, monkeypatch):
    monkeypatch.setenv("LIVEOPS_UPLOAD_DIR", str(tmp_path / "uploads"))
    # Recreate app so the env override is picked up.
    from web.server import create_app
    from fastapi.testclient import TestClient
    app = create_app()
    with TestClient(app) as c:
        r = c.post(
            "/api/workbench/upload",
            files={"file": ("my_orders.csv", _mini_orders_csv(), "text/csv")},
        )
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["ok"] is True
        assert body["dataset"]["source"] == "upload"
        assert body["dataset"]["n_rows"] == 4
        assert body["dataset"]["name"] == "my_orders.csv"
        assert "amount" in body["role_map"] and "date" in body["role_map"]

        # Subsequent /api/workbench/preview should return the uploaded data.
        r2 = c.get("/api/workbench/preview")
        assert r2.status_code == 200
        assert r2.json()["n_rows"] == 4
        assert r2.json()["dataset"]["source"] == "upload"

        # Agent ask now runs against the uploaded frame.
        r3 = c.post("/api/agent/ask",
                    json={"question": "top 5 products by revenue",
                          "backend": "heuristic"})
        assert r3.status_code == 200

        # Reset → back to bundled.
        r4 = c.post("/api/workbench/reset")
        assert r4.status_code == 200
        assert r4.json()["dataset"]["source"] == "bundled"
        r5 = c.get("/api/workbench/preview")
        assert r5.json()["dataset"]["source"] == "bundled"


def test_workbench_upload_rejects_bad_extension(client):
    r = client.post(
        "/api/workbench/upload",
        files={"file": ("notes.txt", b"hello world", "text/plain")},
    )
    assert r.status_code == 400
    assert "must be one of" in r.json()["detail"]


def test_workbench_upload_rejects_missing_required_column(client):
    # No date column -> should fail schema validation.
    body = b"customer_id,product_id,amount\nC1,SKU-001,42.50\n"
    r = client.post(
        "/api/workbench/upload",
        files={"file": ("bad.csv", body, "text/csv")},
    )
    assert r.status_code == 400
    detail = r.json()["detail"]
    assert "missing required column" in detail
    # The helpful hint should mention some accepted spellings.
    assert "order_date" in detail or "transaction_date" in detail


@pytest.mark.parametrize("body, expected_role_keys", [
    # Title-Case + spaces
    (b'"Order Date","Customer","Product","Sales","Qty"\n2025-12-01,C1,P1,42.50,1\n',
     {"date", "amount", "customer", "product", "quantity"}),
    # Synonyms a real customer CSV might use
    (b"order_id,client_id,sku_code,date_of_order,gross_amount\n1,C1,P1,2025-12-01,42.50\n",
     {"date", "amount", "customer", "product"}),
    # Mixed casing, parens, and 'Created At' for the date
    (b"OrderID,User Id,Product Code,Created At,Total Amount,Units\n1,U1,P1,2025-12-01,99.99,3\n",
     {"date", "amount", "customer", "product", "quantity"}),
    # camelCase headers
    (b"orderId,customerId,productId,orderDate,amount\n1,C1,P1,2025-12-01,42.50\n",
     {"date", "amount", "customer", "product"}),
])
def test_workbench_upload_accepts_real_world_headers(client, body, expected_role_keys):
    """The audit found real-world CSVs were rejected because the alias
    dictionary was too narrow.  These four variants are the kinds of headers
    a customer dump actually has — none should bounce."""
    r = client.post(
        "/api/workbench/upload",
        files={"file": ("real.csv", body, "text/csv")},
    )
    assert r.status_code == 200, r.text
    role_map = r.json()["role_map"]
    assert expected_role_keys.issubset(role_map.keys()), \
        f"missing roles: expected {expected_role_keys}, got {set(role_map)}"


def test_normalize_header_unit():
    """Sanity check on the header normalizer we rely on for upload sanity."""
    from web.server import _normalize_header
    assert _normalize_header("Order Date")        == "order_date"
    assert _normalize_header("Sales (USD)")       == "sales_usd"
    assert _normalize_header("  CustomerID  ")    == "customerid"
    assert _normalize_header("date_of_order")     == "date_of_order"
    assert _normalize_header("Order ID #")        == "order_id"
    assert _normalize_header("---weird---")       == "weird"


def test_workbench_upload_rejects_empty_file(client):
    r = client.post(
        "/api/workbench/upload",
        files={"file": ("empty.csv", b"", "text/csv")},
    )
    assert r.status_code == 400


def test_workbench_page_shows_dataset_chip(client):
    """The bundled-dataset chip should render server-side on first load."""
    r = client.get("/workbench")
    assert r.status_code == 200
    assert "bundled dataset" in r.text.lower() or "retail_orders.csv" in r.text
