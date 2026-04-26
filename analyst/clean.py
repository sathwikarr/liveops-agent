"""Cleaning with an audit log.

Public API:
- `propose(df, schema, kind)` -> CleaningPlan
    Build a list of `CleaningStep`s without applying them. The UI shows the
    list and lets the user toggle each step before commit.

- `apply(df, plan)` -> (cleaned_df, audit_log)
    Apply the (filtered) plan in order, recording before/after counts.

Each step is small, self-contained, and reversible at the plan level — if the
user un-checks "fill missing prices with median", we just don't run it.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd

from analyst.ingest import Schema, ColumnInfo

# --------------------------------------------------------------------------- #
# Plan + audit dataclasses
# --------------------------------------------------------------------------- #

@dataclass
class CleaningStep:
    id: str
    label: str            # short title
    description: str      # what + why, plain English
    column: Optional[str] = None
    rationale: str = ""
    applied: bool = True  # default-on; user can untick in UI
    payload: dict = field(default_factory=dict)
    apply_fn: Optional[Callable[[pd.DataFrame, dict], pd.DataFrame]] = None

    def to_dict(self) -> dict:
        return {"id": self.id, "label": self.label, "description": self.description,
                "column": self.column, "rationale": self.rationale,
                "applied": self.applied, "payload": self.payload}


@dataclass
class CleaningPlan:
    steps: list[CleaningStep] = field(default_factory=list)

    def enabled(self) -> list[CleaningStep]:
        return [s for s in self.steps if s.applied]

    def to_dict(self) -> dict:
        return {"steps": [s.to_dict() for s in self.steps]}


@dataclass
class AuditEntry:
    step_id: str
    label: str
    rows_before: int
    rows_after: int
    notes: str

    def to_dict(self) -> dict: return self.__dict__


# --------------------------------------------------------------------------- #
# Step builders
# --------------------------------------------------------------------------- #

def _drop_full_dupes(df: pd.DataFrame, _payload: dict) -> pd.DataFrame:
    return df.drop_duplicates(keep="first").reset_index(drop=True)


def _impute_median(df: pd.DataFrame, payload: dict) -> pd.DataFrame:
    col = payload["column"]
    grp = payload.get("group_by")
    out = df.copy()
    s = pd.to_numeric(out[col], errors="coerce")
    if grp and grp in out.columns:
        out[col] = s.fillna(s.groupby(out[grp]).transform("median"))
        # backfill any group that was entirely NaN
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(s.median())
    else:
        out[col] = s.fillna(s.median())
    return out


def _impute_mode(df: pd.DataFrame, payload: dict) -> pd.DataFrame:
    col = payload["column"]
    out = df.copy()
    s = out[col]
    mode = s.mode(dropna=True)
    if not mode.empty:
        out[col] = s.fillna(mode.iloc[0])
    return out


def _strip_whitespace(df: pd.DataFrame, payload: dict) -> pd.DataFrame:
    col = payload["column"]
    out = df.copy()
    out[col] = out[col].astype("string").str.strip()
    return out


def _drop_constant(df: pd.DataFrame, payload: dict) -> pd.DataFrame:
    col = payload["column"]
    return df.drop(columns=[col])


def _drop_id_subset_dupes(df: pd.DataFrame, payload: dict) -> pd.DataFrame:
    cols = payload["columns"]
    return df.drop_duplicates(subset=cols, keep="first").reset_index(drop=True)


def _coerce_numeric_clip_negative(df: pd.DataFrame, payload: dict) -> pd.DataFrame:
    col = payload["column"]
    floor = payload.get("floor", 0.0)
    out = df.copy()
    out[col] = pd.to_numeric(out[col], errors="coerce").clip(lower=floor)
    return out


# --------------------------------------------------------------------------- #
# Plan builder — looks at schema + dataset kind, proposes steps
# --------------------------------------------------------------------------- #

def propose(df: pd.DataFrame, schema: Schema, kind: Optional[str] = None,
            role_map: Optional[dict] = None) -> CleaningPlan:
    role_map = role_map or {}
    plan = CleaningPlan()

    # 1. Full-row duplicates
    n_dup = int(df.duplicated().sum())
    if n_dup:
        plan.steps.append(CleaningStep(
            id="drop_full_dupes",
            label=f"Drop {n_dup} fully-duplicate row(s)",
            description="Identical rows usually represent ingestion errors.",
            rationale=f"{n_dup} of {len(df):,} rows are exact duplicates.",
            apply_fn=_drop_full_dupes,
        ))

    # 2. ID-subset duplicates (if dataset has an obvious ID column)
    id_cols = schema.id_cols()
    if id_cols:
        primary = id_cols[0]
        n_id_dup = int(df.duplicated(subset=[primary]).sum())
        if n_id_dup and n_id_dup > n_dup:
            plan.steps.append(CleaningStep(
                id=f"dedupe_by_{primary}",
                label=f"Keep first row per {primary}",
                description=f"Multiple rows share the same {primary}.",
                rationale=f"{n_id_dup} duplicate {primary} value(s) found.",
                column=primary,
                payload={"columns": [primary]},
                apply_fn=_drop_id_subset_dupes,
            ))

    # 3. Whitespace strip on text/categorical columns that have ragged values
    for c in schema.columns:
        if c.inferred_type in {"text", "categorical"}:
            s = df[c.name].astype("string").dropna()
            if s.empty: continue
            if (s.str.strip() != s).any():
                plan.steps.append(CleaningStep(
                    id=f"strip_{c.name}",
                    label=f"Trim whitespace in {c.name!r}",
                    description="Some values have leading/trailing spaces — they collapse otherwise-equal categories.",
                    rationale="Detected ragged whitespace in text values.",
                    column=c.name,
                    payload={"column": c.name},
                    apply_fn=_strip_whitespace,
                ))

    # 4. Median imputation for numeric columns with <30% nulls
    amount = role_map.get("amount")
    product = role_map.get("product")
    region = role_map.get("region")
    grp_candidate = product or region

    for c in schema.columns:
        if c.inferred_type not in {"numeric", "currency", "percent"}:
            continue
        if c.null_pct == 0 or c.null_pct > 0.3:
            continue
        n_missing = int(df[c.name].isna().sum())
        grp = grp_candidate if c.name == amount and grp_candidate else None
        rationale = f"{n_missing:,} missing value(s) ({c.null_pct:.1%})."
        if grp:
            rationale += f" Imputing per {grp} median preserves segment differences."
        plan.steps.append(CleaningStep(
            id=f"impute_{c.name}",
            label=f"Fill {n_missing:,} missing {c.name!r} value(s) with " +
                  (f"{grp}-grouped median" if grp else "column median"),
            description="Median is robust to outliers; grouped imputation keeps per-segment scale.",
            rationale=rationale,
            column=c.name,
            payload={"column": c.name, "group_by": grp},
            apply_fn=_impute_median,
        ))

    # 5. Mode imputation for low-null categorical
    for c in schema.columns:
        if c.inferred_type != "categorical":
            continue
        if c.null_pct == 0 or c.null_pct > 0.3:
            continue
        n_missing = int(df[c.name].isna().sum())
        plan.steps.append(CleaningStep(
            id=f"mode_{c.name}",
            label=f"Fill {n_missing:,} missing {c.name!r} with most-common value",
            description="Mode imputation for low-cardinality categoricals.",
            rationale=f"{n_missing:,} missing ({c.null_pct:.1%}).",
            column=c.name,
            payload={"column": c.name},
            apply_fn=_impute_mode,
        ))

    # 6. Drop constant columns — they carry no information
    for c in schema.columns:
        s = df[c.name].dropna()
        if not s.empty and s.nunique() == 1:
            plan.steps.append(CleaningStep(
                id=f"drop_const_{c.name}",
                label=f"Drop constant column {c.name!r}",
                description="Column has a single value — useless for analysis.",
                rationale=f"Only value: {s.iloc[0]!r}.",
                column=c.name,
                payload={"column": c.name},
                apply_fn=_drop_constant,
            ))

    # 7. Clip negative values on amount-like columns
    if amount and amount in df.columns:
        s = pd.to_numeric(df[amount], errors="coerce")
        n_neg = int((s < 0).sum())
        if n_neg:
            plan.steps.append(CleaningStep(
                id=f"clip_neg_{amount}",
                label=f"Floor {n_neg} negative {amount!r} values at 0",
                description="Revenue/amount columns shouldn't be negative — usually a refund encoding issue.",
                rationale=f"{n_neg} negative values detected.",
                column=amount,
                payload={"column": amount, "floor": 0.0},
                apply_fn=_coerce_numeric_clip_negative,
            ))

    return plan


# --------------------------------------------------------------------------- #
# Apply — runs enabled steps in order, returns audit log
# --------------------------------------------------------------------------- #

def apply(df: pd.DataFrame, plan: CleaningPlan) -> tuple[pd.DataFrame, list[AuditEntry]]:
    out = df.copy()
    audit: list[AuditEntry] = []
    for step in plan.enabled():
        if step.apply_fn is None:
            continue
        before = len(out)
        try:
            new = step.apply_fn(out, step.payload)
        except Exception as e:
            audit.append(AuditEntry(step.id, step.label, before, before,
                                    f"FAILED: {type(e).__name__}: {e}"))
            continue
        after = len(new)
        notes = step.rationale
        if before != after:
            notes = f"Rows {before:,} → {after:,}. " + notes
        audit.append(AuditEntry(step.id, step.label, before, after, notes))
        out = new
    return out, audit


__all__ = ["propose", "apply", "CleaningPlan", "CleaningStep", "AuditEntry"]
