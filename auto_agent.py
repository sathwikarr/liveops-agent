"""LiveOps autonomous loop.

Polls every N seconds and runs the pipeline for every user that has uploaded
a CSV under user_data/. Designed to run alongside the FastAPI website:

    python auto_agent.py                # default 30s interval
    python auto_agent.py --interval 10  # poll every 10s
    python auto_agent.py --once         # run a single pass and exit
    python auto_agent.py --user alice   # only run for one user

Graceful shutdown on SIGINT / SIGTERM.
"""
from __future__ import annotations

import argparse
import logging
import signal
import sys
import time
from pathlib import Path

from dotenv import find_dotenv, load_dotenv

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

# Load .env before importing modules that read env vars
load_dotenv(find_dotenv(), override=False)

from agent import db  # noqa: E402
from agent.pipeline import USER_DATA_DIR as USER_DATA  # noqa: E402
from agent.pipeline import run_pipeline  # noqa: E402

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    level=logging.INFO,
    datefmt="%H:%M:%S",
)
log = logging.getLogger("auto_agent")

_stop = False


def _on_signal(signum, _frame):
    global _stop
    log.info("received signal %s — shutting down after current tick", signum)
    _stop = True


def _users_with_csv(only: str | None) -> list[str]:
    if only:
        return [only] if (USER_DATA / f"{only}.csv").exists() else []
    if not USER_DATA.exists():
        return []
    return sorted(p.stem for p in USER_DATA.glob("*.csv"))


def tick(threshold: float, top_k: int, only: str | None) -> int:
    users = _users_with_csv(only)
    if not users:
        log.info("no users with CSVs in %s — skipping tick", USER_DATA)
        return 0
    total = 0
    for u in users:
        try:
            total += run_pipeline(u, threshold=threshold, top_k=top_k, log=log.info)
        except Exception as e:
            log.exception("pipeline failed for user %s: %s", u, e)
    return total


def main() -> int:
    parser = argparse.ArgumentParser(description="LiveOps autonomous loop")
    parser.add_argument("--interval", type=float, default=30.0,
                        help="seconds between ticks (default: 30)")
    parser.add_argument("--threshold", type=float, default=2.0,
                        help="z-score threshold for anomalies (default: 2.0)")
    parser.add_argument("--top-k", type=int, default=20,
                        help="max anomalies acted on per user per tick (default: 20)")
    parser.add_argument("--user", type=str, default=None,
                        help="only run for this username")
    parser.add_argument("--once", action="store_true",
                        help="run a single pass and exit")
    args = parser.parse_args()

    signal.signal(signal.SIGINT, _on_signal)
    signal.signal(signal.SIGTERM, _on_signal)

    db.init_db()
    db.migrate_csv_if_needed()

    log.info("LiveOps auto-agent starting (interval=%.1fs, threshold=%.2f, top_k=%d, user=%s)",
             args.interval, args.threshold, args.top_k, args.user or "ALL")

    if args.once:
        n = tick(args.threshold, args.top_k, args.user)
        log.info("one-shot tick complete — %d anomalies acted on", n)
        return 0

    while not _stop:
        n = tick(args.threshold, args.top_k, args.user)
        log.info("tick complete — %d anomalies acted on; sleeping %.1fs", n, args.interval)
        # Sleep in small chunks so SIGINT is responsive
        slept = 0.0
        while slept < args.interval and not _stop:
            time.sleep(min(0.5, args.interval - slept))
            slept += 0.5

    log.info("auto-agent stopped cleanly")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
