"""Action calendar — turn a list of recommendations into a monthly schedule.

Public API:
- `build_calendar(recs, weeks=8, start=None)` -> DataFrame
    Columns: week_start (date), action, category, audience, confidence,
             bandit_arm, score.
    The placement strategy slots one action per category per week, prioritising
    by score, so the calendar reads like an actual marketing/ops plan.
"""
from __future__ import annotations

from datetime import date, timedelta
from typing import Optional

import pandas as pd


def build_calendar(recs: list, weeks: int = 8, start: Optional[date] = None) -> pd.DataFrame:
    """Greedy week-by-week placement: one item per category per week.

    Inputs are `Recommendation` objects (from analyst.recommend). The output is
    a DataFrame the UI can render as a Gantt-ish list.
    """
    if not recs:
        return pd.DataFrame(columns=["week_start", "action", "category",
                                     "audience", "confidence", "bandit_arm", "score"])

    start_dt = start or date.today()
    # Snap to the upcoming Monday so weeks line up cleanly.
    days_to_mon = (7 - start_dt.weekday()) % 7
    if days_to_mon == 0:
        days_to_mon = 7
    start_monday = start_dt + timedelta(days=days_to_mon)

    pool = sorted(recs, key=lambda r: r.score, reverse=True)
    rows = []
    for w in range(weeks):
        week_start = start_monday + timedelta(weeks=w)
        used_categories: set[str] = set()
        for r in list(pool):
            if r.category in used_categories:
                continue
            rows.append({
                "week_start": week_start,
                "action": r.action,
                "category": r.category,
                "audience": r.audience,
                "confidence": r.confidence,
                "bandit_arm": r.bandit_arm,
                "score": round(r.score, 3),
            })
            used_categories.add(r.category)
            pool.remove(r)
            if len(used_categories) >= 4:  # cap weekly load
                break
        if not pool:
            break
    return pd.DataFrame(rows)


__all__ = ["build_calendar"]
