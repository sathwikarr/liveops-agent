# agent/utils.py
import time
from typing import Any, Callable


def with_backoff(call: Callable[[], Any], max_retries: int = 2) -> Any:
    """Run `call()` with simple backoff. Designed for Gemini 429s.

    - Sleeps ~25s when the exception message mentions a rate limit / 429.
    - Retries up to `max_retries` times.
    - Returns the call() return value, or None if all retries failed.
    """
    for attempt in range(max_retries + 1):
        try:
            return call()
        except Exception as e:
            msg = str(e)
            if "429" in msg or "rate limit" in msg.lower():
                time.sleep(25)
                continue
            if attempt == max_retries:
                raise
    return None
