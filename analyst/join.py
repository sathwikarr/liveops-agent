"""Multi-dataset join — find shared keys between two DataFrames automatically.

Public API:
- `suggest_keys(left, right)` -> list[JoinSuggestion]
    Score every column pair by name + content overlap, return ranked
    suggestions.

- `auto_join(left, right, how='inner')` -> (joined_df, key_used)
    Pick the highest-scoring single-column key, perform the merge.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd


@dataclass
class JoinSuggestion:
    left_col: str
    right_col: str
    name_score: float       # 0..1 from name similarity
    overlap_score: float    # 0..1 from value-set overlap
    score: float            # combined

    def to_dict(self) -> dict:
        return self.__dict__


def _name_similarity(a: str, b: str) -> float:
    a, b = a.lower().strip(), b.lower().strip()
    if a == b: return 1.0
    if a in b or b in a: return 0.8
    if a.replace("_", "") == b.replace("_", ""): return 0.9
    # Common ID suffixes
    suffixes = ("_id", "id", "_no", "_number")
    for s in suffixes:
        if a.endswith(s) and b.endswith(s) and a[:-len(s)] == b[:-len(s)]:
            return 0.85
    return 0.0


def _overlap(left: pd.Series, right: pd.Series) -> float:
    a = set(left.dropna().astype(str).head(2000))
    b = set(right.dropna().astype(str).head(2000))
    if not a or not b: return 0.0
    inter = len(a & b)
    return inter / min(len(a), len(b))


def suggest_keys(left: pd.DataFrame, right: pd.DataFrame, top_n: int = 5) -> list[JoinSuggestion]:
    suggestions: list[JoinSuggestion] = []
    for lc in left.columns:
        for rc in right.columns:
            ns = _name_similarity(lc, rc)
            os_ = _overlap(left[lc], right[rc])
            if ns < 0.4 and os_ < 0.1:
                continue
            score = 0.5 * ns + 0.5 * os_
            suggestions.append(JoinSuggestion(lc, rc, round(ns, 3), round(os_, 3), round(score, 3)))
    suggestions.sort(key=lambda s: s.score, reverse=True)
    return suggestions[:top_n]


def auto_join(left: pd.DataFrame, right: pd.DataFrame, how: str = "inner") -> tuple[pd.DataFrame, Optional[JoinSuggestion]]:
    sugg = suggest_keys(left, right, top_n=1)
    if not sugg or sugg[0].score < 0.3:
        return pd.DataFrame(), None
    key = sugg[0]
    joined = left.merge(right, left_on=key.left_col, right_on=key.right_col, how=how, suffixes=("_l", "_r"))
    return joined, key


__all__ = ["suggest_keys", "auto_join", "JoinSuggestion"]
