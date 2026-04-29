"""FastAPI app — the public-facing website for LiveOps Agent.

All routes live here. The pattern:
  - GET  /<page>   → renders a Jinja template
  - POST /<page>   → form submit, redirects with a flash message
  - GET/POST /api/* → JSON endpoints called from the page JS

Sessions are stored in a signed cookie via SessionMiddleware (no server-side
state needed — fits a single-Fly-machine deploy fine, easy to scale to Redis
later if we ever need multi-region).

Anonymous users can hit: /, /demo, /workbench, /evals, /login, /signup.
Auth-gated pages: /dashboard, /run-agent.
"""
from __future__ import annotations

import json
import math
import os
import secrets
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from fastapi import (
    Body, Depends, FastAPI, Form, HTTPException, Request, status,
)
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.middleware.sessions import SessionMiddleware


# --------------------------------------------------------------------------- #
# Paths + globals
# --------------------------------------------------------------------------- #

REPO_ROOT = Path(__file__).resolve().parents[1]
TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"
STATIC_DIR = Path(__file__).resolve().parent / "static"
SAMPLE_CSV = REPO_ROOT / "analyst" / "sample_data" / "retail_orders.csv"
EVAL_BASELINE = REPO_ROOT / "tests" / "fixtures" / "eval_baseline.json"


# Cache the demo dataframe — reading 10k rows on every request is wasteful and
# every endpoint that needs it is read-only, so a single shared frame is safe.
_SAMPLE_DF: Optional[pd.DataFrame] = None


def _sample_df() -> pd.DataFrame:
    global _SAMPLE_DF
    if _SAMPLE_DF is None:
        if not SAMPLE_CSV.exists():
            raise HTTPException(503, f"Sample data not present at {SAMPLE_CSV}")
        _SAMPLE_DF = pd.read_csv(SAMPLE_CSV)
    return _SAMPLE_DF


# --------------------------------------------------------------------------- #
# App factory
# --------------------------------------------------------------------------- #

@asynccontextmanager
async def _lifespan(app: FastAPI):
    # Initialise DB tables once at boot — same migration pattern as the old
    # Streamlit app's lazy init, just centralised.
    try:
        from agent import db
        db.init_db()
    except Exception as e:  # pragma: no cover — db is optional in tests
        app.state.db_init_error = str(e)
    yield


def create_app() -> FastAPI:
    app = FastAPI(
        title="LiveOps Agent",
        description="AI co-pilot for retail ops + analytics.",
        version="0.4.0",
        lifespan=_lifespan,
    )

    # Sessions: prefer a stable secret from env so cookies survive deploys;
    # fall back to a random one in dev/tests so we never crash on import.
    secret = os.environ.get("SESSION_SECRET") or secrets.token_urlsafe(32)
    app.add_middleware(
        SessionMiddleware, secret_key=secret, session_cookie="liveops_session",
        https_only=False, same_site="lax",
    )

    if STATIC_DIR.exists():
        app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    templates = Jinja2Templates(directory=str(TEMPLATES_DIR))
    # Custom filters + globals available to every template
    templates.env.filters["money"] = lambda v: f"${float(v):,.0f}"
    templates.env.filters["pct"] = lambda v: f"{float(v) * 100:.1f}%"
    app.state.templates = templates

    _register_routes(app)
    return app


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

# Cache-buster string appended to /static/site.css so browsers always pick up
# the latest stylesheet on each deploy. Computed once at boot from the file's
# mtime — falls back to a random nonce if the file is missing.
def _static_version() -> str:
    css = STATIC_DIR / "site.css"
    try:
        return str(int(css.stat().st_mtime))
    except OSError:
        return secrets.token_hex(4)


_STATIC_VERSION = _static_version()


def _ctx(request: Request, **extra: Any) -> Dict[str, Any]:
    """Base template context — every render passes this."""
    return {
        "username": request.session.get("username"),
        "flash": request.session.pop("flash", None),
        "current_path": request.url.path,
        "static_version": _STATIC_VERSION,
        **extra,
    }


def _render(request: Request, template: str, **ctx: Any) -> HTMLResponse:
    templates: Jinja2Templates = request.app.state.templates
    # Modern Starlette signature: (request, name, context). Older positional
    # form (name, {"request": request, ...}) emits a DeprecationWarning and
    # breaks template caching.
    return templates.TemplateResponse(request, template, _ctx(request, **ctx))


def _flash(request: Request, message: str, kind: str = "info") -> None:
    request.session["flash"] = {"message": message, "kind": kind}


def _require_user(request: Request) -> str:
    user = request.session.get("username")
    if not user:
        raise HTTPException(
            status_code=status.HTTP_303_SEE_OTHER,
            headers={"Location": "/login?next=" + request.url.path},
        )
    return user


# --------------------------------------------------------------------------- #
# Routes
# --------------------------------------------------------------------------- #

def _register_routes(app: FastAPI) -> None:

    # ---- liveness ------------------------------------------------------- #
    @app.get("/healthz", response_class=JSONResponse)
    async def healthz():
        return {"status": "ok", "service": "liveops-agent"}

    @app.get("/version", response_class=JSONResponse)
    async def version():
        return {"app": "liveops-agent", "version": app.version}

    # ---- public marketing/demo pages ----------------------------------- #
    @app.get("/", response_class=HTMLResponse)
    async def landing(request: Request):
        return _render(request, "landing.html",
                       eval_baseline=_load_baseline_summary())

    @app.get("/demo", response_class=HTMLResponse)
    async def demo(request: Request):
        df = _sample_df()
        # Tiny preview — first 8 rows + headline numbers
        head = df.head(8).to_dict("records")
        cols = list(df.columns)
        from analyst import analysis as A
        role_map = _default_role_map(df)
        try:
            kpi_revenue = float(df[role_map["amount"]].sum())
            kpi_orders = int(len(df))
            kpi_customers = int(df[role_map["customer"]].nunique())
            kpi_products = int(df[role_map["product"]].nunique())
        except Exception:
            kpi_revenue = kpi_orders = kpi_customers = kpi_products = 0
        return _render(
            request, "demo.html",
            preview_rows=head, columns=cols,
            kpi_revenue=kpi_revenue, kpi_orders=kpi_orders,
            kpi_customers=kpi_customers, kpi_products=kpi_products,
        )

    @app.get("/workbench", response_class=HTMLResponse)
    async def workbench(request: Request):
        from analyst.agent import TOOLS
        return _render(request, "workbench.html",
                       tools=[t.to_dict() for t in TOOLS.values()])

    @app.get("/evals", response_class=HTMLResponse)
    async def evals_page(request: Request):
        from analyst.evals import ALL_CASES
        all_tags = sorted({t for c in ALL_CASES for t in c.tags})
        return _render(request, "evals.html",
                       n_cases=len(ALL_CASES), tags=all_tags,
                       baseline=_load_baseline_summary())

    # ---- auth ----------------------------------------------------------- #
    @app.get("/login", response_class=HTMLResponse)
    async def login_page(request: Request, next: str = "/dashboard"):
        if request.session.get("username"):
            return RedirectResponse(next or "/dashboard", status_code=303)
        return _render(request, "login.html", next=next, mode="login")

    @app.get("/signup", response_class=HTMLResponse)
    async def signup_page(request: Request, next: str = "/dashboard"):
        if request.session.get("username"):
            return RedirectResponse(next or "/dashboard", status_code=303)
        return _render(request, "login.html", next=next, mode="signup")

    @app.post("/login")
    async def login_submit(
        request: Request,
        username: str = Form(...), password: str = Form(...),
        next: str = Form("/dashboard"),
    ):
        from agent.auth import login
        ok, msg = login(username.strip(), password)
        if not ok:
            _flash(request, msg, "error")
            return RedirectResponse(f"/login?next={next}", status_code=303)
        request.session["username"] = username.strip()
        _flash(request, f"Welcome back, {username}.", "success")
        return RedirectResponse(next or "/dashboard", status_code=303)

    @app.post("/signup")
    async def signup_submit(
        request: Request,
        username: str = Form(...), password: str = Form(...),
        confirm: str = Form(""), next: str = Form("/dashboard"),
    ):
        if confirm and confirm != password:
            _flash(request, "Passwords don't match.", "error")
            return RedirectResponse(f"/signup?next={next}", status_code=303)
        from agent.auth import signup
        ok, msg = signup(username.strip(), password)
        if not ok:
            _flash(request, msg, "error")
            return RedirectResponse(f"/signup?next={next}", status_code=303)
        # Auto-login after signup — same UX as the old Streamlit page.
        request.session["username"] = username.strip()
        _flash(request, "Account created. You're logged in.", "success")
        return RedirectResponse(next or "/dashboard", status_code=303)

    @app.post("/logout")
    async def logout(request: Request):
        request.session.clear()
        return RedirectResponse("/", status_code=303)

    # ---- auth-gated pages ---------------------------------------------- #
    @app.get("/dashboard", response_class=HTMLResponse)
    async def dashboard(request: Request):
        user = _require_user(request)
        # Light-weight stats — pull from the demo dataset for a real preview.
        df = _sample_df()
        from analyst import analysis as A
        rm = _default_role_map(df)
        try:
            trend = A.revenue_trend(df, role_map=rm, freq="W").tail(8)
            trend_records = trend.to_dict("records")
        except Exception:
            trend_records = []
        return _render(request, "dashboard.html",
                       user=user, trend_records=trend_records)

    @app.get("/run-agent", response_class=HTMLResponse)
    async def run_agent_page(request: Request):
        user = _require_user(request)
        return _render(request, "run_agent.html", user=user)

    # ---- JSON APIs (called from page JS) -------------------------------- #
    @app.post("/api/agent/ask", response_class=JSONResponse)
    async def api_agent_ask(payload: Dict[str, Any] = Body(...)):
        question = (payload.get("question") or "").strip()
        backend = payload.get("backend") or "heuristic"
        if not question:
            raise HTTPException(400, "question is required")
        df = _sample_df()
        from analyst.agent import ask
        rm = _default_role_map(df)
        result = ask(question, df, role_map=rm, backend=backend)
        return _json_safe(result.to_dict())

    @app.post("/api/evals/run", response_class=JSONResponse)
    async def api_evals_run(payload: Dict[str, Any] = Body(default={})):
        backend = payload.get("backend") or "heuristic"
        tags = payload.get("tags") or None
        ids = payload.get("ids") or None
        from analyst.evals import load_cases, run_all
        df = _sample_df()
        cases = load_cases(tags=tags, ids=ids)
        if not cases:
            raise HTTPException(400, "no cases matched the filters")
        report = run_all(cases, df, backend=backend)
        return _json_safe(report.to_dict())

    @app.get("/api/evals/baseline", response_class=JSONResponse)
    async def api_evals_baseline():
        if not EVAL_BASELINE.exists():
            return JSONResponse({"baseline": None})
        return JSONResponse({"baseline": json.loads(EVAL_BASELINE.read_text())})

    @app.get("/api/workbench/preview", response_class=JSONResponse)
    async def api_workbench_preview():
        df = _sample_df()
        return _json_safe({
            "n_rows": int(len(df)),
            "columns": list(df.columns),
            "head": df.head(20).to_dict("records"),
        })

    @app.post("/api/workbench/tool", response_class=JSONResponse)
    async def api_workbench_tool(payload: Dict[str, Any] = Body(...)):
        tool = payload.get("tool")
        args = payload.get("args") or {}
        from analyst.agent import TOOLS
        spec = TOOLS.get(tool)
        if spec is None:
            raise HTTPException(400, f"unknown tool: {tool}")
        df = _sample_df()
        rm = _default_role_map(df)
        try:
            result = spec.fn(df, rm, **args)
        except TypeError:
            # Be forgiving — drop kwargs the tool doesn't take.
            import inspect
            sig = inspect.signature(spec.fn)
            valid = {k: v for k, v in args.items() if k in sig.parameters}
            result = spec.fn(df, rm, **valid)
        return _json_safe({"tool": tool, "args": args, "result": result})


# --------------------------------------------------------------------------- #
# Misc helpers reused from the old Streamlit code path
# --------------------------------------------------------------------------- #

def _default_role_map(df: pd.DataFrame) -> Dict[str, str]:
    """Best-effort column inference. Mirrors analyst.evals.runner._default_role_map_for
    but avoids importing it so /api/workbench/* keeps working even if evals
    package is restructured."""
    cols = {c.lower(): c for c in df.columns}
    pick = lambda *names: next((cols[n] for n in names if n in cols), None)
    out = {
        "customer": pick("customer_id", "customer", "user_id"),
        "date":     pick("order_date", "date", "timestamp", "created_at"),
        "amount":   pick("revenue", "amount", "total", "price_total"),
        "product":  pick("product_id", "product", "sku", "item"),
        "quantity": pick("quantity", "qty", "units"),
        "price":    pick("price", "unit_price"),
        "region":   pick("region", "country", "market"),
    }
    return {k: v for k, v in out.items() if v}


def _json_safe(obj: Any) -> Any:
    """Recursively replace NaN/+Inf/-Inf with None so JSON encoding never
    chokes. Pandas + numpy happily emit these for empty groupbys, log() of
    zeros, etc.; the stdlib json encoder rejects them."""
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    return obj


def _load_baseline_summary() -> Optional[Dict[str, Any]]:
    if not EVAL_BASELINE.exists():
        return None
    try:
        return json.loads(EVAL_BASELINE.read_text())
    except Exception:
        return None


# Convenient module-level app for `uvicorn web.server:app`
app = create_app()
