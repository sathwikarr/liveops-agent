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

import io
import json
import math
import os
import re
import secrets
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from fastapi import (
    Body, Depends, FastAPI, File, Form, HTTPException, Request, UploadFile, status,
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

# Per-session uploaded datasets. Lives on local disk; on Render's free tier
# this dies with the dyno (acceptable for a session-scoped feature).
UPLOAD_DIR = Path(os.environ.get("LIVEOPS_UPLOAD_DIR")
                  or REPO_ROOT / "user_data" / "workbench_uploads")
UPLOAD_MAX_BYTES = 5 * 1024 * 1024     # 5 MB
UPLOAD_TTL_SECONDS = 6 * 3600          # 6 h, swept on each upload
UPLOAD_REQUIRED = ("date", "amount")   # role-keys we must be able to infer


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


def _active_df(request: Request) -> pd.DataFrame:
    """Return the session's uploaded frame if it exists, else the bundled one.

    The session cookie holds a relative filename (not a full path) so a leaked
    cookie can't escape UPLOAD_DIR. We re-validate the path on every read.
    """
    name = request.session.get("workbench_dataset")
    if name:
        path = (UPLOAD_DIR / name).resolve()
        if path.parent == UPLOAD_DIR.resolve() and path.exists():
            try:
                return pd.read_csv(path)
            except Exception:
                # Corrupt file — drop the session pointer and fall back.
                request.session.pop("workbench_dataset", None)
                request.session.pop("workbench_dataset_meta", None)
    return _sample_df()


def _active_meta(request: Request) -> Dict[str, Any]:
    """Lightweight description of the active dataset for the UI chip."""
    meta = request.session.get("workbench_dataset_meta")
    if meta and (UPLOAD_DIR / meta.get("file", "")).exists():
        return {**meta, "source": "upload"}
    df = _sample_df()
    return {
        "source": "bundled",
        "name": SAMPLE_CSV.name,
        "n_rows": int(len(df)),
        "n_cols": int(len(df.columns)),
    }


def _sweep_old_uploads() -> None:
    """Best-effort cleanup so /user_data doesn't grow forever. Anything older
    than UPLOAD_TTL_SECONDS gets unlinked. Errors are swallowed — this is
    advisory, not load-bearing."""
    if not UPLOAD_DIR.exists():
        return
    cutoff = time.time() - UPLOAD_TTL_SECONDS
    for p in UPLOAD_DIR.iterdir():
        try:
            if p.is_file() and p.stat().st_mtime < cutoff:
                p.unlink()
        except OSError:
            pass


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
                       tools=[t.to_dict() for t in TOOLS.values()],
                       dataset=_active_meta(request))

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
    async def api_agent_ask(request: Request, payload: Dict[str, Any] = Body(...)):
        question = (payload.get("question") or "").strip()
        backend = payload.get("backend") or "heuristic"
        if not question:
            raise HTTPException(400, "question is required")
        df = _active_df(request)
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
    async def api_workbench_preview(request: Request):
        df = _active_df(request)
        return _json_safe({
            "n_rows": int(len(df)),
            "columns": list(df.columns),
            "head": df.head(20).to_dict("records"),
            "dataset": _active_meta(request),
        })

    @app.post("/api/workbench/tool", response_class=JSONResponse)
    async def api_workbench_tool(request: Request, payload: Dict[str, Any] = Body(...)):
        tool = payload.get("tool")
        args = payload.get("args") or {}
        from analyst.agent import TOOLS
        spec = TOOLS.get(tool)
        if spec is None:
            raise HTTPException(400, f"unknown tool: {tool}")
        df = _active_df(request)
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

    @app.post("/api/workbench/upload", response_class=JSONResponse)
    async def api_workbench_upload(request: Request,
                                    file: UploadFile = File(...)):
        """Accept a CSV (or .csv.gz / .parquet) and pin it to this session.

        Validates: extension, size cap, parseability, presence of at least a
        date column and an amount column under any of the alias spellings the
        agent's role-map already recognises.
        """
        # Extension gate (cheap reject before reading bytes)
        name = (file.filename or "upload.csv").strip()
        ext = "".join(Path(name).suffixes).lower()
        allowed = (".csv", ".csv.gz", ".parquet")
        if not any(ext.endswith(e) for e in allowed):
            raise HTTPException(400, f"file must be one of {allowed} (got {ext!r})")

        # Read with a hard size cap; UploadFile.read() honours the limit.
        raw = await file.read(UPLOAD_MAX_BYTES + 1)
        if len(raw) > UPLOAD_MAX_BYTES:
            raise HTTPException(413, f"file exceeds {UPLOAD_MAX_BYTES // 1024 // 1024} MB cap")
        if not raw:
            raise HTTPException(400, "file is empty")

        # Parse into a DataFrame — fail loudly with a useful message.
        try:
            if ext.endswith(".parquet"):
                df = pd.read_parquet(io.BytesIO(raw))
            elif ext.endswith(".csv.gz"):
                df = pd.read_csv(io.BytesIO(raw), compression="gzip")
            else:
                df = pd.read_csv(io.BytesIO(raw))
        except Exception as e:
            raise HTTPException(400, f"could not parse file: {e}")

        if len(df) == 0:
            raise HTTPException(400, "file parsed but contains 0 rows")

        # Schema validation: must be able to infer at minimum date + amount.
        rm = _default_role_map(df)
        missing = [k for k in UPLOAD_REQUIRED if k not in rm]
        if missing:
            raise HTTPException(
                400,
                f"missing required column(s) for: {missing}. "
                f"detected columns: {list(df.columns)}"
            )

        # Persist to disk, atomic-ish (write to .tmp then rename).
        UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        _sweep_old_uploads()
        slug = re.sub(r"[^A-Za-z0-9._-]+", "_", Path(name).stem)[:40] or "upload"
        token = uuid.uuid4().hex[:12]
        out_name = f"{slug}-{token}.csv"
        out_path = UPLOAD_DIR / out_name
        tmp_path = out_path.with_suffix(".csv.tmp")
        df.to_csv(tmp_path, index=False)
        tmp_path.replace(out_path)

        # Drop any previous upload for this session.
        prev = request.session.get("workbench_dataset")
        if prev and prev != out_name:
            try: (UPLOAD_DIR / prev).unlink()
            except OSError: pass

        meta = {
            "file":   out_name,
            "name":   name,
            "n_rows": int(len(df)),
            "n_cols": int(len(df.columns)),
            "uploaded_at": int(time.time()),
        }
        request.session["workbench_dataset"] = out_name
        request.session["workbench_dataset_meta"] = meta

        return _json_safe({
            "ok": True,
            "dataset": {**meta, "source": "upload"},
            "role_map": rm,
            "head": df.head(8).to_dict("records"),
            "columns": list(df.columns),
        })

    @app.post("/api/workbench/reset", response_class=JSONResponse)
    async def api_workbench_reset(request: Request):
        """Drop the upload pointer and any persisted file, return to bundled."""
        prev = request.session.pop("workbench_dataset", None)
        request.session.pop("workbench_dataset_meta", None)
        if prev:
            try: (UPLOAD_DIR / prev).unlink()
            except OSError: pass
        return {"ok": True, "dataset": _active_meta(request)}


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
