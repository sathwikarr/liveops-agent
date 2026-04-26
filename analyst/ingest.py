"""Schema-agnostic ingestion.

Public API:
- `read_any(path_or_buf)` -> pd.DataFrame
    Read CSV / TSV / Excel / JSON / JSONL into a DataFrame, auto-detecting
    encoding, separator, and currency columns.

- `infer_schema(df)` -> Schema
    Classify each column as datetime / numeric / categorical / text / id /
    boolean. Coerces a clean copy of the DataFrame in the process.

- `classify_dataset(df, schema)` -> DatasetKind
    Sales / inventory / customers / transactions / events / generic, picked
    by column-name + content heuristics.

- `IngestResult` bundles all three so `pages/analyst.py` can show a single
  "here's what I see" panel.

Design notes:
    Type inference is conservative and explainable — every column comes back
    with the *reason* we picked the type, so the UI can show "we treated
    `price_usd` as numeric because 100% of values parsed as floats after
    stripping `$,`." That transparency is the point of this layer.

    Nothing here is destructive — `read_any` returns the parsed-as-strings
    DataFrame; `infer_schema` returns a coerced *copy* alongside the original.
"""
from __future__ import annotations

import io
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional, Union

import numpy as np
import pandas as pd

PathLike = Union[str, Path, io.IOBase, bytes]

# --------------------------------------------------------------------------- #
# Format detection + reading
# --------------------------------------------------------------------------- #

_CURRENCY_RE = re.compile(r"^\s*[\$€£¥₹]?\s*-?[\d,]+(?:\.\d+)?\s*[%]?\s*$")
_PERCENT_RE = re.compile(r"^\s*-?\d+(?:\.\d+)?\s*%\s*$")
_NUMBER_LIKE_RE = re.compile(r"^\s*-?[\d,]+(?:\.\d+)?\s*$")


def _detect_encoding(buf: bytes) -> str:
    """Cheap encoding sniff. Try UTF-8 first, then latin-1."""
    for enc in ("utf-8", "utf-8-sig", "latin-1", "cp1252"):
        try:
            buf.decode(enc)
            return enc
        except UnicodeDecodeError:
            continue
    return "utf-8"  # fall back; pandas will raise if it really can't


def _sniff_delimiter(sample: str) -> str:
    """Count tabs/semis/commas in the first non-empty line."""
    for line in sample.splitlines():
        if line.strip():
            counts = {",": line.count(","), "\t": line.count("\t"),
                      ";": line.count(";"), "|": line.count("|")}
            return max(counts, key=counts.get)
    return ","


def read_any(path_or_buf: PathLike, max_rows: Optional[int] = None) -> pd.DataFrame:
    """Read CSV / TSV / Excel / JSON / JSONL. Returns DataFrame of strings/objects."""
    if isinstance(path_or_buf, (str, Path)):
        p = Path(path_or_buf)
        suffix = p.suffix.lower()
        raw = p.read_bytes()
    elif isinstance(path_or_buf, bytes):
        suffix = ""
        raw = path_or_buf
    else:
        # file-like
        raw = path_or_buf.read()
        suffix = ""
        if isinstance(raw, str):
            raw = raw.encode("utf-8")

    # Excel — magic bytes first
    if raw[:4] == b"PK\x03\x04" or suffix in {".xlsx", ".xlsm", ".xls"}:
        return pd.read_excel(io.BytesIO(raw), nrows=max_rows)

    enc = _detect_encoding(raw)
    text = raw.decode(enc, errors="replace")

    stripped = text.lstrip()
    if stripped.startswith("[") or stripped.startswith("{"):
        # Try strict JSON first, then JSONL
        try:
            data = json.loads(text)
            if isinstance(data, dict):
                # nested object — assume the first list-of-dicts value is the table
                for v in data.values():
                    if isinstance(v, list) and v and isinstance(v[0], dict):
                        return pd.DataFrame(v)[:max_rows] if max_rows else pd.DataFrame(v)
                return pd.json_normalize(data)
            return pd.DataFrame(data)[:max_rows] if max_rows else pd.DataFrame(data)
        except json.JSONDecodeError:
            # JSONL
            rows = [json.loads(ln) for ln in text.splitlines() if ln.strip()]
            return pd.DataFrame(rows)[:max_rows] if max_rows else pd.DataFrame(rows)

    # CSV-ish
    sep = _sniff_delimiter(text[:4096])
    return pd.read_csv(
        io.StringIO(text),
        sep=sep,
        engine="python",
        on_bad_lines="skip",
        nrows=max_rows,
        dtype=str,
        keep_default_na=False,
        na_values=["", "NA", "N/A", "null", "NULL", "None", "nan"],
    )


# --------------------------------------------------------------------------- #
# Type inference
# --------------------------------------------------------------------------- #

@dataclass
class ColumnInfo:
    name: str
    inferred_type: str  # one of: datetime, numeric, percent, currency, boolean, categorical, text, id
    reason: str
    null_pct: float
    unique_pct: float
    sample_values: list[Any] = field(default_factory=list)
    coerced: pd.Series | None = None  # cleaned series; None when read-only

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "inferred_type": self.inferred_type,
            "reason": self.reason,
            "null_pct": round(self.null_pct, 3),
            "unique_pct": round(self.unique_pct, 3),
            "sample": self.sample_values[:5],
        }


@dataclass
class Schema:
    columns: list[ColumnInfo]
    coerced: pd.DataFrame  # same shape as input but with parsed dtypes

    def by_type(self, t: str) -> list[str]:
        return [c.name for c in self.columns if c.inferred_type == t]

    def datetime_cols(self) -> list[str]: return self.by_type("datetime")
    def numeric_cols(self) -> list[str]:
        return [c.name for c in self.columns if c.inferred_type in {"numeric", "currency", "percent"}]
    def categorical_cols(self) -> list[str]: return self.by_type("categorical")
    def id_cols(self) -> list[str]: return self.by_type("id")

    def to_dict(self) -> dict:
        return {"columns": [c.to_dict() for c in self.columns]}


def _try_datetime(s: pd.Series) -> tuple[Optional[pd.Series], float]:
    """Return (parsed, hit_rate). hit_rate is fraction of non-null parsed."""
    non_null = s.dropna()
    if non_null.empty:
        return None, 0.0
    parsed = pd.to_datetime(non_null, errors="coerce", utc=False)
    hits = parsed.notna().sum()
    rate = hits / len(non_null)
    if rate < 0.85:
        return None, rate
    out = pd.to_datetime(s, errors="coerce", utc=False)
    return out, rate


def _strip_currency(s: pd.Series) -> pd.Series:
    return (s.astype(str)
              .str.replace(r"[\$€£¥₹]", "", regex=True)
              .str.replace(",", "", regex=False)
              .str.replace("%", "", regex=False)
              .str.strip())


def _try_numeric(s: pd.Series) -> tuple[Optional[pd.Series], float, bool, bool]:
    """Return (parsed, hit_rate, was_currency, was_percent)."""
    non_null = s.dropna().astype(str)
    if non_null.empty:
        return None, 0.0, False, False
    has_currency = non_null.str.contains(r"[\$€£¥₹]", regex=True).any()
    has_percent = non_null.str.contains("%").any()
    cleaned = _strip_currency(s)
    parsed = pd.to_numeric(cleaned, errors="coerce")
    hits = parsed.notna().sum()
    rate = hits / max(non_null.shape[0], 1)
    if rate < 0.9:
        return None, rate, has_currency, has_percent
    return parsed, rate, has_currency, has_percent


_BOOL_TRUE = {"true", "yes", "y", "1", "t"}
_BOOL_FALSE = {"false", "no", "n", "0", "f"}


def _try_boolean(s: pd.Series) -> Optional[pd.Series]:
    non_null = s.dropna().astype(str).str.strip().str.lower()
    if non_null.empty:
        return None
    vals = set(non_null.unique())
    if vals.issubset(_BOOL_TRUE | _BOOL_FALSE) and len(vals) <= 4:
        out = s.astype(str).str.strip().str.lower().map(
            lambda v: True if v in _BOOL_TRUE else (False if v in _BOOL_FALSE else None)
        )
        return out
    return None


def _id_like(name: str, unique_pct: float) -> bool:
    n = name.lower()
    if unique_pct >= 0.95 and (n.endswith("_id") or n == "id" or "uuid" in n or n.endswith("number")):
        return True
    return False


def infer_schema(df: pd.DataFrame) -> Schema:
    """Classify every column. Returns a Schema with a coerced DataFrame copy."""
    # Treat empty strings + common null-sentinels as NaN before inference
    # so per-column rate calculations don't get diluted.
    df = df.copy()
    obj_cols = df.select_dtypes(include=["object", "string"]).columns
    if len(obj_cols):
        df[obj_cols] = df[obj_cols].replace(
            to_replace=[r"^\s*$"], value=np.nan, regex=True
        )
        df[obj_cols] = df[obj_cols].replace(
            {"NA": np.nan, "N/A": np.nan, "null": np.nan,
             "NULL": np.nan, "None": np.nan, "nan": np.nan, "NaN": np.nan}
        )
    out = df.copy()
    cols: list[ColumnInfo] = []
    n = max(len(df), 1)

    for col in df.columns:
        s = df[col]
        nn = s.dropna()
        null_pct = 1 - len(nn) / n
        unique_pct = nn.nunique() / max(len(nn), 1)
        sample = list(nn.head(5).astype(str)) if not nn.empty else []

        # ID columns — high cardinality + name pattern
        if _id_like(str(col), unique_pct):
            cols.append(ColumnInfo(col, "id",
                                   f"Unique={unique_pct:.0%} and name matches ID pattern",
                                   null_pct, unique_pct, sample))
            continue

        # Boolean
        b = _try_boolean(s)
        if b is not None:
            out[col] = b
            cols.append(ColumnInfo(col, "boolean",
                                   "Values are exclusively true/false-like",
                                   null_pct, unique_pct, sample, b))
            continue

        # Datetime
        dt, dt_rate = _try_datetime(s)
        if dt is not None:
            out[col] = dt
            cols.append(ColumnInfo(col, "datetime",
                                   f"{dt_rate:.0%} of values parsed as datetimes",
                                   null_pct, unique_pct, sample, dt))
            continue

        # Numeric / currency / percent
        num, num_rate, was_cur, was_pct = _try_numeric(s)
        if num is not None:
            t = "currency" if was_cur else ("percent" if was_pct else "numeric")
            reason = f"{num_rate:.0%} of values parsed as numbers"
            if was_cur: reason += " (currency symbols stripped)"
            if was_pct: reason += " (percent signs stripped)"
            out[col] = num
            cols.append(ColumnInfo(col, t, reason, null_pct, unique_pct, sample, num))
            continue

        # Categorical vs free text — split on uniqueness
        if unique_pct <= 0.5 and nn.nunique() <= max(50, int(0.05 * n)):
            cols.append(ColumnInfo(col, "categorical",
                                   f"{nn.nunique()} distinct values across {len(nn)} rows",
                                   null_pct, unique_pct, sample))
        else:
            cols.append(ColumnInfo(col, "text",
                                   "High-cardinality string column",
                                   null_pct, unique_pct, sample))

    return Schema(columns=cols, coerced=out)


# --------------------------------------------------------------------------- #
# Dataset classification
# --------------------------------------------------------------------------- #

@dataclass
class DatasetKind:
    kind: str        # "sales" | "inventory" | "customers" | "transactions" | "events" | "generic"
    confidence: float
    reasons: list[str]
    role_map: dict[str, str]  # role -> column name (e.g. {"date": "order_date", "amount": "revenue"})

    def to_dict(self) -> dict:
        return {"kind": self.kind, "confidence": round(self.confidence, 2),
                "reasons": self.reasons, "role_map": self.role_map}


_ROLE_PATTERNS: dict[str, list[str]] = {
    "date": ["date", "timestamp", "ts", "datetime", "time", "created", "ordered"],
    "amount": ["revenue", "sales", "amount", "total", "price", "value", "gmv"],
    "quantity": ["qty", "quantity", "units", "orders", "count"],
    "customer": ["customer", "user", "client", "buyer", "account", "email"],
    "product": ["product", "sku", "item", "asin"],
    "region": ["region", "country", "state", "city", "zone", "market"],
    "inventory": ["inventory", "stock", "on_hand", "available"],
    "channel": ["channel", "source", "campaign", "medium"],
    "event": ["event", "action", "click", "view"],
}


def _match_role(name: str) -> Optional[str]:
    n = name.lower()
    for role, kws in _ROLE_PATTERNS.items():
        if any(kw in n for kw in kws):
            return role
    return None


def classify_dataset(df: pd.DataFrame, schema: Schema) -> DatasetKind:
    role_map: dict[str, str] = {}
    for c in schema.columns:
        role = _match_role(c.name)
        if role and role not in role_map:
            role_map[role] = c.name

    has = lambda *roles: all(r in role_map for r in roles)

    reasons: list[str] = []

    if has("date", "amount") and ("product" in role_map or "customer" in role_map):
        reasons.append(f"Has date ({role_map['date']}) + amount ({role_map['amount']}) + entity")
        return DatasetKind("sales", 0.9, reasons, role_map)

    if has("inventory", "product"):
        reasons.append(f"Has inventory ({role_map['inventory']}) + product ({role_map['product']})")
        return DatasetKind("inventory", 0.85, reasons, role_map)

    if has("customer") and "amount" not in role_map and "event" in role_map:
        reasons.append(f"Has customer + event ({role_map['event']}) — looks like activity log")
        return DatasetKind("events", 0.7, reasons, role_map)

    if has("customer") and not has("amount"):
        reasons.append(f"Has customer ({role_map['customer']}) and no amount — customer master")
        return DatasetKind("customers", 0.7, reasons, role_map)

    if has("date", "amount"):
        reasons.append("Has date + amount but no product/customer — generic transactions")
        return DatasetKind("transactions", 0.6, reasons, role_map)

    reasons.append("No strong role signals — falling back to generic")
    return DatasetKind("generic", 0.3, reasons, role_map)


# --------------------------------------------------------------------------- #
# Public bundle
# --------------------------------------------------------------------------- #

@dataclass
class IngestResult:
    raw: pd.DataFrame
    schema: Schema
    kind: DatasetKind
    n_rows: int
    n_cols: int
    issues: list[str]

    @property
    def df(self) -> pd.DataFrame:
        """Coerced, parsed DataFrame ready for analysis."""
        return self.schema.coerced

    def summary(self) -> dict:
        return {
            "rows": self.n_rows, "cols": self.n_cols,
            "kind": self.kind.to_dict(),
            "schema": self.schema.to_dict(),
            "issues": self.issues,
        }


def _scan_issues(df: pd.DataFrame, schema: Schema) -> list[str]:
    issues: list[str] = []
    n = len(df)
    if n == 0:
        issues.append("Dataset is empty.")
        return issues
    for c in schema.columns:
        if c.null_pct > 0.5:
            issues.append(f"{c.name!r}: {c.null_pct:.0%} of rows are missing values.")
    if df.duplicated().any():
        dups = int(df.duplicated().sum())
        issues.append(f"Found {dups} fully-duplicate row(s).")
    return issues


def ingest(path_or_buf: PathLike) -> IngestResult:
    """One-call pipeline: read → infer → classify → scan."""
    raw = read_any(path_or_buf)
    schema = infer_schema(raw)
    kind = classify_dataset(raw, schema)
    issues = _scan_issues(raw, schema)
    return IngestResult(raw=raw, schema=schema, kind=kind,
                        n_rows=len(raw), n_cols=raw.shape[1], issues=issues)


__all__ = ["read_any", "infer_schema", "classify_dataset", "ingest",
           "Schema", "ColumnInfo", "DatasetKind", "IngestResult"]
