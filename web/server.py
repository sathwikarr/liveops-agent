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

# Eval baselines: the main 55-case corpus + the 14-case holdout split that the
# heuristic was NOT tuned on. We surface ALL THREE numbers in the UI — pinning
# only the 100% main number was the credibility hole the audit flagged.
EVAL_FIXTURES = REPO_ROOT / "tests" / "fixtures"
EVAL_BASELINE              = EVAL_FIXTURES / "eval_baseline.json"
EVAL_HOLDOUT_HEURISTIC     = EVAL_FIXTURES / "eval_holdout_baseline_heuristic.json"
EVAL_HOLDOUT_LLM           = EVAL_FIXTURES / "eval_holdout_baseline_llm.json"

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


def _user_upload_dir(username: str) -> Path:
    """Per-user directory for persisted uploads. Slugged for filesystem safety."""
    safe = re.sub(r"[^A-Za-z0-9_-]+", "_", username)[:48] or "anon"
    return UPLOAD_DIR / "users" / safe


def _active_dataset_path(request: Request) -> Optional[Path]:
    """Resolve the session's pinned dataset to a real file on disk.

    Two backends:
      1. Logged-in user with a pinned dataset id → look up via DB
      2. Anonymous (or pre-login) session with a file pointer → resolve under UPLOAD_DIR

    Returns None if no upload is pinned, or if the pointer is stale.
    Always re-validates the resolved path against the allowed roots so a
    leaked cookie can't escape into another user's uploads.
    """
    user = request.session.get("username")
    # ---- 1. Persistent (logged-in) ---- #
    ds_id = request.session.get("workbench_dataset_id") if user else None
    if user and ds_id:
        try:
            from agent import db
            row = db.get_user_dataset(user, int(ds_id))
        except Exception:
            row = None
        if row:
            p = (_user_upload_dir(user) / row["file"]).resolve()
            if p.parent == _user_upload_dir(user).resolve() and p.exists():
                return p
        # Pointer is stale (file deleted, user changed, etc.) — drop it.
        request.session.pop("workbench_dataset_id", None)
        request.session.pop("workbench_dataset_meta", None)

    # ---- 2. Session-scoped (anonymous) ---- #
    name = request.session.get("workbench_dataset")
    if name:
        p = (UPLOAD_DIR / name).resolve()
        if p.parent == UPLOAD_DIR.resolve() and p.exists():
            return p
        request.session.pop("workbench_dataset", None)
        request.session.pop("workbench_dataset_meta", None)
    return None


def _active_df(request: Request) -> pd.DataFrame:
    """Return the active uploaded frame, falling back to the bundled CSV."""
    p = _active_dataset_path(request)
    if p:
        try:
            return pd.read_csv(p)
        except Exception:
            # Corrupt file — clear pointers and fall back.
            request.session.pop("workbench_dataset", None)
            request.session.pop("workbench_dataset_id", None)
            request.session.pop("workbench_dataset_meta", None)
    return _sample_df()


def _active_meta(request: Request) -> Dict[str, Any]:
    """Lightweight description of the active dataset for the UI chip."""
    if _active_dataset_path(request):
        meta = request.session.get("workbench_dataset_meta") or {}
        return {**meta, "source": "upload",
                "persisted": bool(request.session.get("workbench_dataset_id"))}
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
                       eval_baseline=_load_baseline_summary(),
                       eval_baselines=_load_eval_baselines())

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
            # Date range subtitle for the KPI tiles ("Nov 2024 – Apr 2025")
            dates = pd.to_datetime(df[role_map["date"]], errors="coerce").dropna()
            date_range = (f"{dates.min().strftime('%b %Y')} – "
                          f"{dates.max().strftime('%b %Y')}") if len(dates) else ""
        except Exception:
            kpi_revenue = kpi_orders = kpi_customers = kpi_products = 0
            date_range = ""

        # Chart data — server-rendered, ready for Chart.js to consume.
        charts = _build_demo_charts(df, role_map)

        return _render(
            request, "demo.html",
            preview_rows=head, columns=cols,
            kpi_revenue=kpi_revenue, kpi_orders=kpi_orders,
            kpi_customers=kpi_customers, kpi_products=kpi_products,
            date_range=date_range,
            charts=charts,
        )

    @app.get("/workbench", response_class=HTMLResponse)
    async def workbench(request: Request):
        from analyst.agent import TOOLS
        return _render(request, "workbench.html",
                       tools=[t.to_dict() for t in TOOLS.values()],
                       dataset=_active_meta(request))

    @app.get("/evals", response_class=HTMLResponse)
    async def evals_page(request: Request):
        from analyst.evals import ALL_CASES, HOLDOUT_CASES
        all_tags = sorted({t for c in ALL_CASES for t in c.tags})
        return _render(request, "evals.html",
                       n_cases=len(ALL_CASES),
                       n_holdout_cases=len(HOLDOUT_CASES),
                       tags=all_tags,
                       baseline=_load_baseline_summary(),
                       baselines=_load_eval_baselines())

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

    @app.get("/settings", response_class=HTMLResponse)
    async def settings_page(request: Request):
        user = _require_user(request)
        from agent import db
        connectors = db.list_user_connectors(user)
        return _render(request, "settings.html",
                       user=user, connectors=connectors)

    @app.get("/api/connectors", response_class=JSONResponse)
    async def api_connectors_list(request: Request):
        user = _require_user(request)
        from agent import db
        return {"connectors": db.list_user_connectors(user)}

    @app.post("/api/connectors/save", response_class=JSONResponse)
    async def api_connectors_save(request: Request,
                                   payload: Dict[str, Any] = Body(...)):
        user = _require_user(request)
        kind  = (payload.get("kind") or "").strip()
        value = (payload.get("value") or "").strip()
        if kind not in ("slack_webhook",):
            raise HTTPException(400, f"unsupported kind: {kind!r}")
        if not value:
            raise HTTPException(400, "value is required")
        if kind == "slack_webhook" and not value.startswith("https://hooks.slack.com/"):
            raise HTTPException(400,
                "slack_webhook must start with https://hooks.slack.com/")
        from agent import db
        from agent.secret import encrypt
        db.upsert_user_connector(username=user, kind=kind,
                                  value_encrypted=encrypt(value))
        return {"ok": True, "kind": kind}

    @app.post("/api/connectors/delete", response_class=JSONResponse)
    async def api_connectors_delete(request: Request,
                                     payload: Dict[str, Any] = Body(...)):
        user = _require_user(request)
        kind = (payload.get("kind") or "").strip()
        if not kind:
            raise HTTPException(400, "kind is required")
        from agent import db
        if not db.delete_user_connector(user, kind):
            raise HTTPException(404, "connector not configured")
        return {"ok": True, "kind": kind}

    @app.post("/api/connectors/test", response_class=JSONResponse)
    async def api_connectors_test(request: Request,
                                   payload: Dict[str, Any] = Body(default={})):
        """Send a real test ping to the user's configured Slack webhook."""
        user = _require_user(request)
        kind = (payload.get("kind") or "slack_webhook").strip()
        from agent import db
        from agent.secret import decrypt
        token = db.get_user_connector(user, kind)
        if not token:
            raise HTTPException(404, "connector not configured")
        value = decrypt(token)
        if not value:
            raise HTTPException(500, "stored value could not be decrypted "
                                "(LIVEOPS_FERNET_KEY may have changed)")
        if kind == "slack_webhook":
            import urllib.request as _u, json as _j
            text = (f"✅ LiveOps Agent test ping from @{user} — your Slack "
                    f"connector is wired up correctly.")
            req = _u.Request(value, method="POST",
                              headers={"Content-Type": "application/json"},
                              data=_j.dumps({"text": text}).encode("utf-8"))
            try:
                with _u.urlopen(req, timeout=10) as resp:
                    body_text = resp.read(2000).decode("utf-8", "replace")
                    if resp.status >= 300:
                        raise HTTPException(502,
                            f"slack returned {resp.status}: {body_text[:200]}")
                    return {"ok": True, "status": resp.status, "body": body_text}
            except HTTPException:
                raise
            except Exception as e:
                raise HTTPException(502, f"slack ping failed: {type(e).__name__}: {e}")
        raise HTTPException(400, f"test not implemented for kind: {kind!r}")

    # ---- JSON APIs (called from page JS) -------------------------------- #
    @app.post("/api/agent/ask", response_class=JSONResponse)
    async def api_agent_ask(request: Request, payload: Dict[str, Any] = Body(...)):
        question = (payload.get("question") or "").strip()
        backend  = payload.get("backend") or "heuristic"
        if not question:
            raise HTTPException(400, "question is required")
        if backend not in ("heuristic", "llm", "auto"):
            raise HTTPException(400, f"backend must be one of heuristic|llm|auto (got {backend!r})")
        df = _active_df(request)
        from analyst.agent import ask, _llm_available
        rm = _default_role_map(df)
        result = ask(question, df, role_map=rm, backend=backend)
        body = result.to_dict()
        body["observations"] = _attach_hints(body.get("observations", []), df, rm)
        actual_backend = body.get("plan", {}).get("backend", "heuristic")
        body["backend_meta"] = _backend_meta(
            requested=backend, actual=actual_backend,
            llm_available=_llm_available(),
        )
        # Persist the question for logged-in users.  We store the plan and
        # routing metadata, NOT the answer (could be huge).  Errors are best-
        # effort — never let DB hiccups break the answer pipeline.
        user = request.session.get("username")
        if user:
            try:
                from agent import db
                planned = [s.get("tool", "") for s in body.get("plan", {}).get("steps", [])]
                ds_meta = _active_meta(request)
                qid = db.insert_user_question(
                    username=user, question=question, backend=backend,
                    actual_backend=actual_backend, planned_tools=planned,
                    dataset_name=ds_meta.get("name"),
                )
                body["question_id"] = qid
            except Exception:
                pass
        return _json_safe(body)

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

        # Normalize headers BEFORE validation so 'Order Date', 'Sales (USD)'
        # etc. resolve to canonical snake_case and the alias dictionary works.
        original_columns = list(df.columns)
        df = _normalize_columns(df)

        # Schema validation: must be able to infer at minimum date + amount.
        rm = _default_role_map(df)
        missing = [k for k in UPLOAD_REQUIRED if k not in rm]
        if missing:
            raise HTTPException(
                400,
                f"missing required column(s) for: {missing}. "
                f"got columns: {original_columns}. "
                f"hint: a column for each of {missing} must exist (we accept many "
                f"common spellings — e.g. 'order_date'/'date'/'transaction_date' "
                f"for date, 'amount'/'sales'/'gross_revenue'/'order_total' for amount)."
            )

        # Persist to disk, atomic-ish (write to .tmp then rename).
        slug = re.sub(r"[^A-Za-z0-9._-]+", "_", Path(name).stem)[:40] or "upload"
        token = uuid.uuid4().hex[:12]
        out_name = f"{slug}-{token}.csv"
        user = request.session.get("username")

        if user:
            # Logged-in: persist under per-user dir + DB row.
            from agent import db
            target_dir = _user_upload_dir(user)
            target_dir.mkdir(parents=True, exist_ok=True)
            out_path = target_dir / out_name
            tmp_path = out_path.with_suffix(".csv.tmp")
            df.to_csv(tmp_path, index=False)
            tmp_path.replace(out_path)
            ds_id = db.insert_user_dataset(
                username=user, file=out_name, original_name=name,
                n_rows=len(df), n_cols=len(df.columns),
            )
            meta = {
                "id":     ds_id,
                "file":   out_name,
                "name":   name,
                "n_rows": int(len(df)),
                "n_cols": int(len(df.columns)),
                "uploaded_at": int(time.time()),
                "persisted":   True,
            }
            # Drop any previous session-scoped pointer; we use the DB id now.
            request.session.pop("workbench_dataset", None)
            request.session["workbench_dataset_id"]   = ds_id
            request.session["workbench_dataset_meta"] = meta
        else:
            # Anonymous: session-scoped, no DB row.
            UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
            _sweep_old_uploads()
            out_path = UPLOAD_DIR / out_name
            tmp_path = out_path.with_suffix(".csv.tmp")
            df.to_csv(tmp_path, index=False)
            tmp_path.replace(out_path)
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
                "persisted":   False,
            }
            request.session.pop("workbench_dataset_id", None)
            request.session["workbench_dataset"]      = out_name
            request.session["workbench_dataset_meta"] = meta

        return _json_safe({
            "ok": True,
            "dataset": {**meta, "source": "upload"},
            "role_map": rm,
            "head": df.head(8).to_dict("records"),
            "columns": list(df.columns),
            "profile": _dataset_profile(df, rm),
        })

    @app.post("/api/workbench/reset", response_class=JSONResponse)
    async def api_workbench_reset(request: Request):
        """Stop pointing at any upload; return to the bundled dataset.

        For anonymous sessions this also unlinks the uploaded file (it's
        ephemeral by design).  For logged-in users we keep the file on disk —
        they can still re-select it from /api/workbench/datasets later.
        """
        prev_anon = request.session.pop("workbench_dataset", None)
        request.session.pop("workbench_dataset_id", None)
        request.session.pop("workbench_dataset_meta", None)
        if prev_anon:
            try: (UPLOAD_DIR / prev_anon).unlink()
            except OSError: pass
        return {"ok": True, "dataset": _active_meta(request)}

    @app.get("/api/workbench/profile", response_class=JSONResponse)
    async def api_workbench_profile(request: Request):
        """Return the active dataset's profile + signal-density tips."""
        df = _active_df(request)
        rm = _default_role_map(df)
        return _json_safe({
            "dataset": _active_meta(request),
            "profile": _dataset_profile(df, rm),
        })

    @app.get("/api/workbench/datasets", response_class=JSONResponse)
    async def api_workbench_datasets(request: Request):
        """List the logged-in user's persisted datasets (newest first).
        Anonymous callers get an empty list — they don't have an account."""
        user = request.session.get("username")
        if not user:
            return {"datasets": [], "active_id": None}
        from agent import db
        rows = db.list_user_datasets(user)
        active_id = request.session.get("workbench_dataset_id")
        return {"datasets": rows, "active_id": active_id}

    @app.post("/api/workbench/select", response_class=JSONResponse)
    async def api_workbench_select(request: Request,
                                    payload: Dict[str, Any] = Body(...)):
        """Switch the active dataset to a previously-saved one (logged-in)."""
        user = request.session.get("username")
        if not user:
            raise HTTPException(401, "log in to use saved datasets")
        ds_id = payload.get("id")
        if not ds_id:
            raise HTTPException(400, "id is required")
        from agent import db
        row = db.get_user_dataset(user, int(ds_id))
        if not row:
            raise HTTPException(404, "dataset not found")
        meta = {
            "id":     row["id"],
            "file":   row["file"],
            "name":   row["original_name"],
            "n_rows": int(row["n_rows"]),
            "n_cols": int(row["n_cols"]),
            "uploaded_at": row["uploaded_at"],
            "persisted":   True,
        }
        request.session.pop("workbench_dataset", None)
        request.session["workbench_dataset_id"]   = int(row["id"])
        request.session["workbench_dataset_meta"] = meta
        return {"ok": True, "dataset": {**meta, "source": "upload"}}

    @app.get("/api/agent/history", response_class=JSONResponse)
    async def api_agent_history(request: Request, limit: int = 50):
        """Per-user question history. Pinned first, then newest. Anon users
        get an empty list (not 401) so the UI can omit the panel cleanly."""
        user = request.session.get("username")
        if not user:
            return {"history": []}
        from agent import db
        return {"history": db.list_user_questions(user, limit=int(limit))}

    @app.post("/api/agent/pin", response_class=JSONResponse)
    async def api_agent_pin(request: Request,
                             payload: Dict[str, Any] = Body(...)):
        """Toggle pinned on a saved question. Pinned rows survive history clears."""
        user = request.session.get("username")
        if not user:
            raise HTTPException(401, "log in to pin questions")
        qid = payload.get("id")
        pinned = bool(payload.get("pinned", True))
        if not qid:
            raise HTTPException(400, "id is required")
        from agent import db
        if not db.set_user_question_pinned(user, int(qid), pinned):
            raise HTTPException(404, "question not found")
        return {"ok": True, "id": int(qid), "pinned": pinned}

    @app.post("/api/agent/history/clear", response_class=JSONResponse)
    async def api_agent_history_clear(request: Request,
                                       payload: Dict[str, Any] = Body(default={})):
        """Wipe history. Pinned rows survive unless `keep_pinned: false`."""
        user = request.session.get("username")
        if not user:
            raise HTTPException(401, "log in to clear history")
        from agent import db
        n = db.clear_user_questions(user, keep_pinned=bool(payload.get("keep_pinned", True)))
        return {"ok": True, "deleted": int(n)}

    @app.post("/api/workbench/delete", response_class=JSONResponse)
    async def api_workbench_delete(request: Request,
                                    payload: Dict[str, Any] = Body(...)):
        """Delete a saved dataset (logged-in only).  If it was active, the
        session falls back to the bundled dataset on the next read."""
        user = request.session.get("username")
        if not user:
            raise HTTPException(401, "log in to manage saved datasets")
        ds_id = payload.get("id")
        if not ds_id:
            raise HTTPException(400, "id is required")
        from agent import db
        file_basename = db.delete_user_dataset(user, int(ds_id))
        if file_basename is None:
            raise HTTPException(404, "dataset not found")
        # Unlink the on-disk file (best effort).
        try:
            (_user_upload_dir(user) / file_basename).unlink()
        except OSError:
            pass
        # If the deleted dataset was active, drop the pointer.
        if request.session.get("workbench_dataset_id") == int(ds_id):
            request.session.pop("workbench_dataset_id", None)
            request.session.pop("workbench_dataset_meta", None)
        return {"ok": True, "deleted_id": int(ds_id)}


# --------------------------------------------------------------------------- #
# Misc helpers reused from the old Streamlit code path
# --------------------------------------------------------------------------- #

# Column-alias dictionary used to infer the role of each uploaded column.
# Keep these EXHAUSTIVE — every entry here is a header we promise to recognise
# without requiring the user to rename anything.  Comparison is done after
# `_normalize_header` (lowercase + alphanumeric+underscore only), so add the
# normalized form, e.g. "ordered quantity" → "ordered_quantity".
_ROLE_ALIASES: Dict[str, tuple[str, ...]] = {
    "customer": (
        "customer_id", "customer", "customerid", "user_id", "user", "userid",
        "client_id", "client", "clientid", "account_id", "account",
        "buyer_id", "buyer", "member_id", "member", "shopper_id", "shopper",
    ),
    "date": (
        "order_date", "orderdate", "date", "timestamp", "created_at", "createdat",
        "date_of_order", "transaction_date", "txn_date", "txndate",
        "purchase_date", "order_dt", "order_datetime", "order_time",
        "occurred_at", "event_date", "event_time", "ts",
    ),
    "amount": (
        "revenue", "amount", "total", "price_total", "sales", "sales_amount",
        "gross_amount", "gross_revenue", "gross", "net_amount", "net",
        "total_amount", "value", "subtotal", "line_total", "order_value",
        "order_total", "transaction_amount", "spend",
    ),
    "product": (
        "product_id", "productid", "product", "sku", "skus", "item", "item_id",
        "itemid", "product_code", "productcode", "sku_code", "skucode",
        "item_code", "itemcode", "product_name", "productname", "prod_id",
    ),
    "quantity": (
        "quantity", "qty", "units", "count", "num_items", "item_count",
        "ordered_quantity", "order_qty", "qty_ordered",
    ),
    "price": (
        "price", "unit_price", "unitprice", "list_price", "price_each",
        "item_price", "itemprice", "unit_cost",
    ),
    "region": (
        "region", "country", "market", "area", "geo", "zone", "territory",
        "locale", "state", "country_code",
    ),
}


def _normalize_header(name: str) -> str:
    """Map a raw column name to its canonical form for alias lookup.

    'Order Date' → 'order_date'; 'Sales (USD)' → 'sales_usd'.  Lowercases,
    replaces every non-alphanumeric run with a single underscore, strips
    leading/trailing underscores."""
    return re.sub(r"_+", "_", re.sub(r"[^a-z0-9]+", "_", name.strip().lower())).strip("_")


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of `df` with column names normalized.  Original headers
    are dropped — the renamed frame is what gets persisted on upload."""
    new_cols, seen = [], set()
    for c in df.columns:
        n = _normalize_header(str(c)) or "col"
        # Disambiguate duplicates after normalization, e.g. two "Sales" cols.
        candidate, i = n, 1
        while candidate in seen:
            i += 1
            candidate = f"{n}_{i}"
        seen.add(candidate)
        new_cols.append(candidate)
    out = df.copy()
    out.columns = new_cols
    return out


def _default_role_map(df: pd.DataFrame) -> Dict[str, str]:
    """Infer which uploaded column plays each analytic role (date, amount,
    customer, product, quantity, price, region).

    Two-pass match: exact alias on the normalized header, then fall back to
    'normalized header contains the alias as a whole token' so 'order_date_utc'
    still resolves to date even though it's not an exact alias.
    """
    # Build {normalized_header: original_header} so we can return originals.
    norm_to_orig = {_normalize_header(str(c)): c for c in df.columns}

    def _pick(role_key: str) -> Optional[str]:
        aliases = _ROLE_ALIASES[role_key]
        # Pass 1 — exact match
        for a in aliases:
            if a in norm_to_orig:
                return norm_to_orig[a]
        # Pass 2 — whole-token containment (split norm header on '_' and check)
        for norm, orig in norm_to_orig.items():
            tokens = norm.split("_")
            for a in aliases:
                a_tokens = a.split("_")
                # Require all alias tokens to appear as substrings of the
                # normalized header in order — keeps 'shopper_id' from
                # accidentally matching 'product_id'.
                if all(t in tokens for t in a_tokens):
                    return orig
        return None

    out = {role: _pick(role) for role in _ROLE_ALIASES}
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


# Signal-density floors per analytic tool. If the dataset doesn't clear the
# threshold, the workbench surfaces a tip telling the user the tool's output
# will be unreliable — no point asking "show me cohort retention" when there
# are 4 customers in the file.
_SIGNAL_FLOORS = (
    # (key, message)
    ("customers_for_segments",
     50, "fewer than 50 customers — RFM segments will be noisy or collapse to one bucket"),
    ("days_for_trend",
     30, "fewer than 30 days of order history — weekly trends will be too short to read"),
    ("orders_for_elasticity",
     200, "fewer than 200 orders — price-elasticity needs more price/quantity pairs to fit reliably"),
    ("skus_for_quadrants",
     8,  "fewer than 8 SKUs — BCG quadrants won't separate cleanly"),
    ("repeat_for_cohorts",
     0.20, "less than 20% of customers have a second purchase — cohort retention curves will collapse to zero"),
)


def _dataset_profile(df: pd.DataFrame, role_map: Dict[str, str]) -> Dict[str, Any]:
    """A one-page summary of an uploaded dataset so users can judge whether
    the tools have anything to chew on before they ask questions.

    Returns a dict with three top-level keys:
      - "shape":   row/col counts, KPI numbers, date range
      - "columns": one row per column (dtype, null %, sample, distinct count)
      - "tips":    signal-density warnings keyed to specific tools
    """
    profile: Dict[str, Any] = {
        "shape":   {"n_rows": int(len(df)), "n_cols": int(len(df.columns))},
        "columns": [],
        "tips":    [],
        "kpis":    {},
        "role_map": role_map,
    }

    # ---- Per-column profile --------------------------------------------- #
    n = max(1, len(df))
    for col in df.columns:
        s = df[col]
        dtype = str(s.dtype)
        nulls = int(s.isna().sum())
        # cap distinct count work — for high-cardinality columns we just need
        # an upper bound, not an exact number.
        try:
            n_unique = int(s.nunique(dropna=True))
        except Exception:
            n_unique = -1
        sample = next((str(v) for v in s.dropna().head(3).tolist()), "")
        profile["columns"].append({
            "name":        col,
            "dtype":       dtype,
            "null_pct":    round(nulls / n * 100, 1),
            "n_unique":    n_unique,
            "sample":      sample[:40],
            "role":        next((k for k, v in role_map.items() if v == col), None),
        })

    # ---- KPIs (only when role_map allows) ------------------------------- #
    try:
        if "amount" in role_map:
            profile["kpis"]["total_revenue"] = float(df[role_map["amount"]].sum())
        if "customer" in role_map:
            profile["kpis"]["n_customers"] = int(df[role_map["customer"]].nunique())
        if "product" in role_map:
            profile["kpis"]["n_products"] = int(df[role_map["product"]].nunique())
        if "date" in role_map:
            dates = pd.to_datetime(df[role_map["date"]], errors="coerce").dropna()
            if len(dates):
                profile["kpis"]["date_min"]  = dates.min().strftime("%Y-%m-%d")
                profile["kpis"]["date_max"]  = dates.max().strftime("%Y-%m-%d")
                profile["kpis"]["date_span"] = int((dates.max() - dates.min()).days)
        profile["kpis"]["n_orders"] = int(len(df))
    except Exception:
        pass

    # ---- Signal-density tips -------------------------------------------- #
    k = profile["kpis"]
    tips = profile["tips"]
    if "customer" in role_map and k.get("n_customers", 9999) < 50:
        tips.append({
            "tool": "segment_customers",
            "severity": "warn",
            "message": f"Only {k['n_customers']} unique customers — RFM segments need ≥50 to form meaningful buckets.",
        })
    if "date" in role_map and k.get("date_span", 9999) < 30:
        tips.append({
            "tool": "revenue_by_period",
            "severity": "warn",
            "message": f"Only {k.get('date_span', 0)} days of order history — weekly trend will be too short to interpret.",
        })
    if k.get("n_orders", 9999) < 200:
        tips.append({
            "tool": "price_elasticity",
            "severity": "warn",
            "message": f"Only {k['n_orders']} orders — price-elasticity needs ≥200 price/quantity pairs to fit a reliable slope.",
        })
    if "product" in role_map and k.get("n_products", 9999) < 8:
        tips.append({
            "tool": "product_quadrants",
            "severity": "info",
            "message": f"Only {k['n_products']} SKUs — BCG quadrants will look sparse with this few products.",
        })
    if "customer" in role_map and k.get("n_customers", 0) >= 1:
        try:
            counts = df[role_map["customer"]].value_counts()
            repeat_pct = float((counts > 1).mean())
            if repeat_pct < 0.20:
                tips.append({
                    "tool": "cohort_retention",
                    "severity": "warn",
                    "message": f"Only {repeat_pct*100:.0f}% of customers have a second order — retention curves will collapse to near-zero.",
                })
        except Exception:
            pass

    return profile


def _observation_hint(tool: str, result: Any,
                      df: pd.DataFrame,
                      role_map: Dict[str, str]) -> Optional[str]:
    """Return a one-sentence explanation when a tool's result looks degenerate.

    Real-world test: an analyst uploads 10 rows of orders, asks "show me
    cohort retention", gets back `{}`, and walks away thinking the tool is
    broken. With a hint, they see "Your dataset spans 4 days — cohort
    retention curves need at least one full month per cohort."
    """
    # Pass through tool errors untouched; they already contain the message.
    if isinstance(result, dict) and "error" in result:
        return None

    n_orders    = len(df)
    n_customers = int(df[role_map["customer"]].nunique()) if "customer" in role_map else 0
    n_products  = int(df[role_map["product"]].nunique())  if "product"  in role_map else 0
    date_span = 0
    if "date" in role_map:
        try:
            d = pd.to_datetime(df[role_map["date"]], errors="coerce").dropna()
            date_span = int((d.max() - d.min()).days) if len(d) else 0
        except Exception:
            pass

    # ---- per-tool degenerate-output detection --------------------------- #

    if tool == "revenue_by_period":
        if isinstance(result, list):
            if not result:
                return ("No periods produced — the date column may not be "
                        "parseable or the amount column is all zero.")
            if len(result) < 4:
                return (f"Only {len(result)} periods in the result — "
                        "the dataset's date span is too short to read a trend.")
        return None

    if tool == "top_products":
        if isinstance(result, list):
            if not result:
                return "No products to rank — check that product and amount columns parse."
            if len(result) == 1:
                return "Only one SKU appears in the data — there's nothing to rank."
            if n_products and len(result) >= n_products:
                # Asked for top-N but the dataset only has N or fewer SKUs.
                if n_products < 5:
                    return (f"Your dataset has only {n_products} unique SKUs, "
                            "so the ranking is the full catalog.")
        return None

    if tool == "top_customers":
        if isinstance(result, list):
            if not result:
                return "No customers to rank — check that customer and amount columns parse."
            if len(result) == 1:
                return "Only one customer appears in the data — there's nothing to rank."
        return None

    if tool == "segment_customers":
        if isinstance(result, dict):
            if not result:
                return "No segments formed — the RFM scorer needs more customers and orders."
            if len(result) == 1:
                only = next(iter(result))
                return (f"Every customer landed in the same segment ('{only}') — "
                        "there's not enough variance in recency/frequency/spend to split them.")
            total = sum(result.values()) if all(isinstance(v, (int, float)) for v in result.values()) else 0
            if total and total < 50:
                return (f"Only {total} customers segmented — buckets are too small "
                        "to act on individually.")
        return None

    if tool == "product_quadrants":
        if isinstance(result, dict):
            if not result:
                return "No quadrants assigned — too few SKUs or no revenue/volume signal."
            if len(result) == 1:
                only = next(iter(result))
                return (f"All SKUs landed in '{only}' — there's no separation "
                        "in revenue × volume to fill the other quadrants.")
            if n_products and n_products < 8:
                return (f"Only {n_products} SKUs in the catalog — quadrants will look "
                        "sparse with this few products.")
        return None

    if tool == "co_purchases":
        if isinstance(result, list) and not result:
            return ("No co-purchase pairs found — most orders contain only a "
                    "single product, or order_id is unique per row instead of "
                    "shared across line items.")
        return None

    if tool == "price_elasticity":
        if isinstance(result, dict):
            if not result:
                return ("Couldn't fit elasticity — need at least a few price points "
                        "for the same SKU. Verify your price/quantity columns.")
            # The tool may emit per-SKU rows; if none have a meaningful slope, hint.
            slopes = [abs(v.get("elasticity", 0)) for v in result.values()
                      if isinstance(v, dict)]
            if slopes and max(slopes) < 0.05:
                return ("Elasticity slopes are all near zero — your dataset "
                        "doesn't show much price variation per SKU.")
        return None

    if tool == "churn_risk":
        if isinstance(result, dict) and result:
            total = sum(result.values()) if all(isinstance(v, (int, float))
                                                 for v in result.values()) else 0
            if total:
                active = result.get("Active", 0)
                churned = result.get("Churned", 0)
                if active / total >= 0.99:
                    return ("Everyone is Active — every customer has ordered "
                            "within the recency window. Try a longer lookback "
                            "or a dataset with older customers.")
                if churned / total >= 0.80:
                    return ("Most customers are Churned — the dataset's most "
                            "recent order is far in the past, so the recency "
                            "window catches almost no one.")
        return None

    if tool == "cohort_retention":
        if isinstance(result, dict):
            if not result:
                if date_span < 60:
                    return (f"Only {date_span} days of order history — cohort "
                            "retention needs at least 2 months to compute a curve.")
                return ("No cohorts produced — verify the customer and date "
                        "columns parse correctly.")
            # All retention values zero across all periods -> dead curves.
            try:
                all_zero = all(
                    all(float(v or 0) == 0 for v in (period or {}).values())
                    for period in result.values() if isinstance(period, dict)
                )
                if all_zero:
                    return ("All cohort retention values are 0 — no customers "
                            "have a second purchase in any month.")
            except Exception:
                pass
        return None

    return None


def _attach_hints(observations: List[Dict[str, Any]],
                  df: pd.DataFrame,
                  role_map: Dict[str, str]) -> List[Dict[str, Any]]:
    """Return a copy of `observations` with a `hint` field added per row when
    the result is degenerate. Hints are intentionally optional — the absence
    of a hint means "the result is meaningful, no caveat needed."""
    out = []
    for o in observations:
        h = _observation_hint(o.get("tool", ""), o.get("result"), df, role_map)
        out.append({**o, **({"hint": h} if h else {})})
    return out


def _build_demo_charts(df: pd.DataFrame,
                       role_map: Dict[str, str]) -> Dict[str, Any]:
    """Pre-compute the three /demo charts server-side so the page renders
    instantly without a JS round-trip.  Returns shapes Chart.js consumes
    directly (label arrays + numeric arrays) — never raises; on failure each
    chart is silently set to None and the template hides it.

    The analyst tools live in `analyst.agent.TOOLS` (not `analyst.analysis`),
    so we go through the registry to keep behaviour identical to /workbench.
    """
    from analyst.agent import TOOLS
    out: Dict[str, Any] = {"revenue": None, "top_products": None, "churn": None}

    def _run(tool: str, **kw):
        spec = TOOLS.get(tool)
        return spec.fn(df, role_map, **kw) if spec else None

    # ---- Weekly revenue trend ---------------------------------------- #
    try:
        rev = _run("revenue_by_period", freq="W")
        # Tool returns list[{period, revenue, ...}].
        if isinstance(rev, list) and rev:
            out["revenue"] = {
                "labels": [str(r.get("period", "")) for r in rev],
                "values": [float(r.get("revenue", 0)) for r in rev],
            }
    except Exception:
        pass

    # ---- Top 5 products ---------------------------------------------- #
    try:
        tp = _run("top_products", n=5)
        if isinstance(tp, list) and tp:
            out["top_products"] = {
                "labels": [str(r.get("product") or r.get("product_id") or "") for r in tp],
                "values": [float(r.get("revenue", 0)) for r in tp],
            }
    except Exception:
        pass

    # ---- Churn-tier breakdown ---------------------------------------- #
    try:
        ch = _run("churn_risk")
        # Returns {tier: count}
        if isinstance(ch, dict) and ch:
            order = ["Active", "Cooling", "At-Risk", "Churned"]
            keys = [k for k in order if k in ch] + [k for k in ch if k not in order]
            out["churn"] = {
                "labels": keys,
                "values": [int(ch[k]) for k in keys],
            }
    except Exception:
        pass

    return out


def _backend_meta(*, requested: str, actual: str,
                  llm_available: bool) -> Dict[str, Any]:
    """Describe the requested-vs-actual backend pair so the workbench UI can
    surface a banner instead of silently downgrading the user's choice.

    The "reason" string is a short, user-facing explanation — not a stack
    trace. We only need to distinguish three cases for the UI:
      1. user got what they asked for → no banner
      2. user asked for llm but heuristic ran → why? (key missing vs. call failed)
      3. user picked auto and got heuristic → informational, not a warning
    """
    fallback = (requested == "llm" and actual != "llm")
    info     = (requested == "auto" and actual != "llm")  # expected, but worth noting

    if fallback or info:
        if not llm_available:
            reason = ("LLM backend isn't configured on this server "
                      "(no GEMINI_API_KEY). Heuristic ran instead.")
        else:
            reason = ("LLM call failed (network, quota, or response-parse "
                      "error). Heuristic ran instead.")
    else:
        reason = None

    return {
        "requested":     requested,
        "actual":        actual,
        "llm_available": bool(llm_available),
        "fallback":      bool(fallback),
        "info":          bool(info),
        "reason":        reason,
    }


def _load_baseline_summary() -> Optional[Dict[str, Any]]:
    if not EVAL_BASELINE.exists():
        return None
    try:
        return json.loads(EVAL_BASELINE.read_text())
    except Exception:
        return None


def _load_eval_baselines() -> Dict[str, Optional[Dict[str, Any]]]:
    """Return all three pinned baselines so the UI can tell the honest dual-
    baseline story (main 100% / holdout heuristic 7.1% / holdout llm 28.6%)."""
    def _read(p: Path) -> Optional[Dict[str, Any]]:
        if not p.exists():
            return None
        try:
            return json.loads(p.read_text())
        except Exception:
            return None
    return {
        "main":              _read(EVAL_BASELINE),
        "holdout_heuristic": _read(EVAL_HOLDOUT_HEURISTIC),
        "holdout_llm":       _read(EVAL_HOLDOUT_LLM),
    }


# Convenient module-level app for `uvicorn web.server:app`
app = create_app()
