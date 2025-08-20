# agent/utils.py
import time
from typing import Callable, Any

def with_backoff(call: Callable[[], Any], max_retries: int = 2) -> Any:
    """
    Run `call()` with simple backoff. Designed for Gemini 429s.
    - Sleeps ~25s when the exception message mentions a retry delay.
    - Retries up to `max_retries` times.
    - Returns the call() return value or None if all retries failed.
    """
    for attempt in range(max_retries + 1):
        try:
            return call()
        except Exception as e:
            msg = str(e)
            # Gemini 429 strings often include "retry_delay"
            if "429" in msg or "rate limit" in msg.lower():
                # crude parse; pick a safe sleep
                time.sleep(25)
                continue
            if attempt == max_retries:
                raise
    return None
# agent/utils.py
import time
from typing import Callable, Any

def with_backoff(call: Callable[[], Any], max_retries: int = 2) -> Any:
    """
    Run `call()` with simple backoff. Designed for Gemini 429s.
    - Sleeps ~25s when the exception message mentions a retry delay.
    - Retries up to `max_retries` times.
    - Returns the call() return value or None if all retries failed.
    """
    for attempt in range(max_retries + 1):
        try:
            return call()
        except Exception as e:
            msg = str(e)
            # Gemini 429 strings often include "retry_delay"
            if "429" in msg or "rate limit" in msg.lower():
                # crude parse; pick a safe sleep
                time.sleep(25)
                continue
            if attempt == max_retries:
                raise
    return None
