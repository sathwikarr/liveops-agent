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


def test_demo_renders_charts(client):
    """The /demo page used to be just KPI tiles + a row table. After P1-5 it
    must include three chart canvases (revenue, products, churn) with their
    data injected as JSON so Chart.js can render client-side."""
    r = client.get("/demo")
    assert r.status_code == 200
    for needle in ("demo-chart-revenue", "demo-chart-churn",
                    "demo-chart-products", "demo-chart-data",
                    "Weekly revenue", "Customer status"):
        assert needle in r.text, f"missing: {needle}"


def test_build_demo_charts_unit():
    """Helper produces Chart.js-ready shapes from the bundled dataset."""
    from web.server import _build_demo_charts, _sample_df, _default_role_map
    df = _sample_df()
    rm = _default_role_map(df)
    charts = _build_demo_charts(df, rm)
    # All three should populate against the bundled retail dataset.
    for key in ("revenue", "top_products", "churn"):
        assert charts.get(key) is not None, f"chart '{key}' is None"
        assert "labels" in charts[key] and "values" in charts[key]
        assert len(charts[key]["labels"]) == len(charts[key]["values"])
        assert len(charts[key]["labels"]) > 0


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


def test_observation_hints_fire_on_tiny_dataset(client, tmp_path, monkeypatch):
    """A tiny upload (3 customers, 4 orders, 4 days, 3 SKUs) must produce
    actionable hints on every degenerate observation, not silent empties."""
    monkeypatch.setenv("LIVEOPS_UPLOAD_DIR", str(tmp_path / "uploads"))
    from web.server import create_app
    from fastapi.testclient import TestClient
    app = create_app()
    with TestClient(app) as c:
        body = (
            b"order_id,customer_id,product_id,order_date,amount\n"
            b"O1,C1,P1,2025-12-01,42.50\n"
            b"O2,C2,P2,2025-12-02,17.00\n"
            b"O3,C1,P1,2025-12-03,42.50\n"
            b"O4,C3,P3,2025-12-04,99.99\n"
        )
        c.post("/api/workbench/upload",
               files={"file": ("tiny.csv", body, "text/csv")})

        # Each question — the corresponding tool must get a hint.
        cases = [
            ("weekly revenue trend", "revenue_by_period", "too short"),
            ("top 5 products",       "top_products",      "only 3 unique SKUs"),
            ("rfm segments",         "segment_customers", "Only 3 customers"),
            ("product quadrants",    "product_quadrants", "Only 3 SKUs"),
            ("who is about to churn?","churn_risk",       "Active"),
        ]
        for q, tool, needle in cases:
            r = c.post("/api/agent/ask",
                       json={"question": q, "backend": "heuristic"}).json()
            obs = next((o for o in r["observations"] if o["tool"] == tool), None)
            assert obs is not None, f"tool {tool!r} not planned for question {q!r}"
            assert "hint" in obs, f"no hint on {tool} for {q!r}: {obs}"
            assert needle.lower() in obs["hint"].lower(), \
                f"hint for {tool} did not mention {needle!r}: {obs['hint']!r}"


def test_observation_hints_silent_on_healthy_dataset(client):
    """The bundled retail dataset has rich signal — most observations should
    NOT carry a hint, because hints are reserved for degenerate output."""
    questions_with_no_expected_hint = [
        "weekly revenue trend",
        "top 5 products by revenue",
        "rfm segments",
        "product quadrants",
        "who is about to churn?",
        "show me cohort retention",
    ]
    for q in questions_with_no_expected_hint:
        r = client.post("/api/agent/ask",
                        json={"question": q, "backend": "heuristic"}).json()
        for o in r["observations"]:
            assert "hint" not in o, \
                f"unexpected hint on healthy dataset for {q!r} → {o['tool']}: {o.get('hint')!r}"


def test_workbench_profile_endpoint_bundled(client):
    """Profile endpoint returns a populated payload for the bundled dataset
    with no signal-density tips (515 days, 200 customers, 30 SKUs)."""
    r = client.get("/api/workbench/profile")
    assert r.status_code == 200
    body = r.json()
    p = body["profile"]
    # Shape + KPIs are populated
    assert p["shape"]["n_rows"] > 0 and p["shape"]["n_cols"] > 0
    assert p["kpis"]["n_customers"] >= 50
    assert p["kpis"]["date_span"] >= 30
    assert p["kpis"]["n_orders"] >= 200
    assert p["kpis"]["n_products"] >= 8
    # Per-column profile populated for every column
    assert len(p["columns"]) == p["shape"]["n_cols"]
    for col in p["columns"]:
        assert {"name", "dtype", "null_pct", "n_unique", "sample", "role"} <= set(col)
    # Bundled retail dataset has rich signal — no tips should fire.
    assert p["tips"] == []


def test_workbench_upload_returns_profile_and_tips(client, tmp_path, monkeypatch):
    """Tiny upload (3 customers, 4 orders) should fire the segment_customers,
    revenue_by_period, price_elasticity, and product_quadrants tips."""
    monkeypatch.setenv("LIVEOPS_UPLOAD_DIR", str(tmp_path / "uploads"))
    from web.server import create_app
    from fastapi.testclient import TestClient
    app = create_app()
    with TestClient(app) as c:
        body = (
            b"order_id,customer_id,product_id,order_date,amount\n"
            b"O1,C1,P1,2025-12-01,42.50\n"
            b"O2,C2,P2,2025-12-02,17.00\n"
            b"O3,C1,P1,2025-12-03,42.50\n"
            b"O4,C3,P3,2025-12-04,99.99\n"
        )
        r = c.post("/api/workbench/upload",
                   files={"file": ("tiny.csv", body, "text/csv")})
        assert r.status_code == 200, r.text
        upload_body = r.json()
        # Profile rides on the upload response so user sees it instantly.
        assert "profile" in upload_body
        tips = upload_body["profile"]["tips"]
        tools_with_tips = {t["tool"] for t in tips}
        # All four expected tips must fire on this dataset.
        assert {"segment_customers", "revenue_by_period",
                "price_elasticity", "product_quadrants"}.issubset(tools_with_tips)


def test_logged_in_uploads_persist_across_logout(tmp_path, monkeypatch):
    """Logged-in uploads survive logout + re-login.  Anonymous uploads do
    not (they're session-only by design)."""
    db_path = tmp_path / "liveops_test.db"
    monkeypatch.setenv("LIVEOPS_DB", str(db_path))
    monkeypatch.setenv("LIVEOPS_UPLOAD_DIR", str(tmp_path / "uploads"))
    monkeypatch.setenv("SESSION_SECRET", "test-secret")
    import importlib, agent.db as _db; importlib.reload(_db); _db.init_db()
    from web.server import create_app
    from fastapi.testclient import TestClient

    body = (b"order_id,customer_id,product_id,order_date,amount\n"
            b"O1,C1,P1,2025-12-01,42.50\n")

    app = create_app()
    with TestClient(app) as c:
        c.post("/signup", data={"username": "alice", "password": "secret123",
                                "confirm": "secret123", "next": "/"})
        r = c.post("/api/workbench/upload",
                   files={"file": ("orders.csv", body, "text/csv")}).json()
        ds_id = r["dataset"]["id"]
        assert r["dataset"]["persisted"] is True

        # Listed.
        listed = c.get("/api/workbench/datasets").json()
        assert any(d["id"] == ds_id for d in listed["datasets"])
        assert listed["active_id"] == ds_id

        # Logout drops the session pointer; login restores access.
        c.post("/logout")
        c.post("/login", data={"username": "alice", "password": "secret123"})
        listed = c.get("/api/workbench/datasets").json()
        assert any(d["id"] == ds_id for d in listed["datasets"])

        # Select sets it active.
        sel = c.post("/api/workbench/select", json={"id": ds_id}).json()
        assert sel["dataset"]["id"] == ds_id

        # Delete removes from list and 404s on second delete.
        c.post("/api/workbench/delete", json={"id": ds_id})
        listed = c.get("/api/workbench/datasets").json()
        assert all(d["id"] != ds_id for d in listed["datasets"])
        r = c.post("/api/workbench/delete", json={"id": ds_id})
        assert r.status_code == 404


def test_user_dataset_isolation(tmp_path, monkeypatch):
    """User A cannot see, switch to, or delete user B's datasets."""
    db_path = tmp_path / "liveops_test.db"
    monkeypatch.setenv("LIVEOPS_DB", str(db_path))
    monkeypatch.setenv("LIVEOPS_UPLOAD_DIR", str(tmp_path / "uploads"))
    monkeypatch.setenv("SESSION_SECRET", "test-secret")
    import importlib, agent.db as _db; importlib.reload(_db); _db.init_db()
    from web.server import create_app
    from fastapi.testclient import TestClient

    body = (b"order_id,customer_id,product_id,order_date,amount\n"
            b"O1,C1,P1,2025-12-01,42.50\n")

    # Alice uploads
    app_a = create_app()
    with TestClient(app_a) as a:
        a.post("/signup", data={"username": "alice", "password": "secret123",
                                "confirm": "secret123", "next": "/"})
        r = a.post("/api/workbench/upload",
                   files={"file": ("orders.csv", body, "text/csv")}).json()
        alice_id = r["dataset"]["id"]

    # Bob lists -> empty; can't select or delete alice's
    app_b = create_app()
    with TestClient(app_b) as b:
        b.post("/signup", data={"username": "bob", "password": "secret123",
                                "confirm": "secret123", "next": "/"})
        listed = b.get("/api/workbench/datasets").json()
        assert listed["datasets"] == []
        sel = b.post("/api/workbench/select", json={"id": alice_id})
        assert sel.status_code == 404
        dele = b.post("/api/workbench/delete", json={"id": alice_id})
        assert dele.status_code == 404


def test_question_history_persists_with_pin(tmp_path, monkeypatch):
    """Logged-in questions are saved; pin survives a 'clear'; second login
    sees both pinned + recent unpinned."""
    db_path = tmp_path / "liveops_test.db"
    monkeypatch.setenv("LIVEOPS_DB", str(db_path))
    monkeypatch.setenv("LIVEOPS_UPLOAD_DIR", str(tmp_path / "uploads"))
    monkeypatch.setenv("SESSION_SECRET", "test-secret")
    import importlib, agent.db as _db; importlib.reload(_db); _db.init_db()
    from web.server import create_app
    from fastapi.testclient import TestClient

    app = create_app()
    with TestClient(app) as c:
        c.post("/signup", data={"username": "alice", "password": "secret123",
                                "confirm": "secret123", "next": "/"})
        for q in ["top 5 products by revenue",
                  "weekly revenue trend",
                  "who is about to churn?"]:
            r = c.post("/api/agent/ask",
                       json={"question": q, "backend": "heuristic"})
            assert r.status_code == 200
            assert "question_id" in r.json()

        h = c.get("/api/agent/history").json()["history"]
        assert len(h) == 3
        assert all("planned_tools" in row and isinstance(row["planned_tools"], list)
                   for row in h)
        # planned_tools is parsed JSON, not a string
        assert "top_products" in h[-1]["planned_tools"] or "top_products" in h[0]["planned_tools"]

        # Pin one, then clear; pinned survives.
        target_id = h[0]["id"]
        c.post("/api/agent/pin", json={"id": target_id, "pinned": True})
        cleared = c.post("/api/agent/history/clear", json={"keep_pinned": True}).json()
        assert cleared["deleted"] == 2
        h2 = c.get("/api/agent/history").json()["history"]
        assert len(h2) == 1 and h2[0]["id"] == target_id and h2[0]["pinned"] is True

        # Survives logout + re-login
        c.post("/logout")
        c.post("/login", data={"username": "alice", "password": "secret123"})
        h3 = c.get("/api/agent/history").json()["history"]
        assert len(h3) == 1 and h3[0]["pinned"] is True

        # Pin returns 404 for unknown id
        r = c.post("/api/agent/pin", json={"id": 99999, "pinned": True})
        assert r.status_code == 404


def test_connectors_save_list_delete_with_encryption(tmp_path, monkeypatch):
    """Save a Slack webhook, confirm encryption-at-rest, list it back,
    delete it.  Wrong URL shape returns 400."""
    db_path = tmp_path / "liveops_test.db"
    monkeypatch.setenv("LIVEOPS_DB", str(db_path))
    monkeypatch.setenv("SESSION_SECRET", "test-secret-deterministic")
    import importlib, agent.db as _db, agent.secret as _sec
    importlib.reload(_db); importlib.reload(_sec); _db.init_db()
    from web.server import create_app
    from fastapi.testclient import TestClient

    app = create_app()
    with TestClient(app) as c:
        c.post("/signup", data={"username": "alice", "password": "secret123",
                                "confirm": "secret123", "next": "/"})

        # Initial list = empty
        assert c.get("/api/connectors").json()["connectors"] == []

        # Bad URL shape → 400
        r = c.post("/api/connectors/save",
                   json={"kind": "slack_webhook",
                         "value": "http://example.com/bad"})
        assert r.status_code == 400

        # Unknown kind → 400
        r = c.post("/api/connectors/save",
                   json={"kind": "unknown_thing", "value": "x"})
        assert r.status_code == 400

        # Save real-shape webhook
        good = "https://hooks.slack.com/services/T12345/B12345/abcdef"
        r = c.post("/api/connectors/save",
                   json={"kind": "slack_webhook", "value": good})
        assert r.status_code == 200

        # Listed
        listed = c.get("/api/connectors").json()["connectors"]
        assert len(listed) == 1
        assert listed[0]["kind"] == "slack_webhook"
        assert "value" not in listed[0]                 # never expose raw

        # Encrypted at rest
        raw = _db.get_user_connector("alice", "slack_webhook")
        assert isinstance(raw, bytes)
        assert b"hooks.slack.com" not in raw            # ciphertext only
        assert _sec.decrypt(raw) == good                # decrypts back

        # Delete
        assert c.post("/api/connectors/delete",
                       json={"kind": "slack_webhook"}).status_code == 200
        assert c.get("/api/connectors").json()["connectors"] == []
        # Second delete → 404
        r = c.post("/api/connectors/delete", json={"kind": "slack_webhook"})
        assert r.status_code == 404


def test_connectors_anonymous_redirected(client):
    """All connector routes are auth-gated — anon hits 303 to /login."""
    for path in ("/settings",):
        r = client.get(path, follow_redirects=False)
        assert r.status_code == 303 and "/login" in r.headers["location"]
    for path in ("/api/connectors",):
        r = client.get(path, follow_redirects=False)
        assert r.status_code == 303
    r = client.post("/api/connectors/save",
                    json={"kind": "slack_webhook", "value": "x"},
                    follow_redirects=False)
    assert r.status_code == 303


def test_connectors_isolation(tmp_path, monkeypatch):
    """Bob can't see, save into, delete, or test alice's connectors —
    he just gets/sees his own."""
    db_path = tmp_path / "liveops_test.db"
    monkeypatch.setenv("LIVEOPS_DB", str(db_path))
    monkeypatch.setenv("SESSION_SECRET", "test-secret-deterministic")
    import importlib, agent.db as _db; importlib.reload(_db); _db.init_db()
    from web.server import create_app
    from fastapi.testclient import TestClient

    app_a = create_app()
    with TestClient(app_a) as a:
        a.post("/signup", data={"username": "alice", "password": "secret123",
                                "confirm": "secret123", "next": "/"})
        a.post("/api/connectors/save",
               json={"kind": "slack_webhook",
                     "value": "https://hooks.slack.com/services/T1/B1/x"})

    app_b = create_app()
    with TestClient(app_b) as b:
        b.post("/signup", data={"username": "bob", "password": "secret123",
                                "confirm": "secret123", "next": "/"})
        # Bob sees nothing
        assert b.get("/api/connectors").json()["connectors"] == []
        # Bob can't delete alice's
        r = b.post("/api/connectors/delete", json={"kind": "slack_webhook"})
        assert r.status_code == 404
        # Bob can't test against alice's webhook
        r = b.post("/api/connectors/test", json={"kind": "slack_webhook"})
        assert r.status_code == 404


def test_question_history_anonymous_silent(client):
    """Anon callers get an empty list (not 401), but pin/clear are 401."""
    h = client.get("/api/agent/history").json()
    assert h["history"] == []
    assert client.post("/api/agent/pin", json={"id": 1}).status_code == 401
    assert client.post("/api/agent/history/clear", json={}).status_code == 401


def test_question_history_isolation(tmp_path, monkeypatch):
    """Bob can't see, pin, or clear alice's questions."""
    db_path = tmp_path / "liveops_test.db"
    monkeypatch.setenv("LIVEOPS_DB", str(db_path))
    monkeypatch.setenv("LIVEOPS_UPLOAD_DIR", str(tmp_path / "uploads"))
    monkeypatch.setenv("SESSION_SECRET", "test-secret")
    import importlib, agent.db as _db; importlib.reload(_db); _db.init_db()
    from web.server import create_app
    from fastapi.testclient import TestClient

    app = create_app()
    with TestClient(app) as a:
        a.post("/signup", data={"username": "alice", "password": "secret123",
                                "confirm": "secret123", "next": "/"})
        a.post("/api/agent/ask", json={"question": "weekly revenue trend",
                                         "backend": "heuristic"})
        alice_qid = a.get("/api/agent/history").json()["history"][0]["id"]

    app2 = create_app()
    with TestClient(app2) as b:
        b.post("/signup", data={"username": "bob", "password": "secret123",
                                "confirm": "secret123", "next": "/"})
        assert b.get("/api/agent/history").json()["history"] == []
        # Bob trying to pin alice's question gets 404, not 200.
        assert b.post("/api/agent/pin",
                       json={"id": alice_qid, "pinned": True}).status_code == 404


def test_anonymous_upload_not_persisted(client):
    body = (b"order_id,customer_id,product_id,order_date,amount\n"
            b"O1,C1,P1,2025-12-01,42.50\n")
    r = client.post("/api/workbench/upload",
                    files={"file": ("anon.csv", body, "text/csv")}).json()
    assert r["dataset"].get("persisted") is False
    # Anonymous /datasets returns empty list, not 401.
    listed = client.get("/api/workbench/datasets").json()
    assert listed["datasets"] == [] and listed["active_id"] is None
    # Anonymous select/delete return 401.
    assert client.post("/api/workbench/select", json={"id": 1}).status_code == 401
    assert client.post("/api/workbench/delete", json={"id": 1}).status_code == 401


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
