"""GoogleSheetsConnector — fetch a tab from a Google Sheet.

Authentication options (in priority order):
1. `service_account_json`: paste of the SA key JSON (treated as secret)
2. `service_account_file`: path to a SA key JSON on disk
3. Public sheet (no auth) — works only when the sheet is shared publicly
   AND the connector falls back to the CSV export URL.

Identification:
- `sheet_url` OR `sheet_id` (canonical)
- `worksheet`: tab name or 0-based index (default 0)
"""
from __future__ import annotations

import io
import json
import re
from typing import Optional

import pandas as pd

from analyst.connectors.base import (
    Connector, ConnectionError, ConnectionResult,
)


_SHEET_ID_RE = re.compile(r"/d/([A-Za-z0-9_-]{20,})")


class GoogleSheetsConnector(Connector):
    kind = "gsheets"
    param_schema = {
        "sheet_url": "url", "sheet_id": "text",
        "worksheet": "text",
        "service_account_json": "secret",
        "service_account_file": "text",
    }

    def _resolve_sheet_id(self) -> str:
        sid = (self.params.get("sheet_id") or "").strip()
        if sid:
            return sid
        url = (self.params.get("sheet_url") or "").strip()
        if not url:
            raise ConnectionError("Provide sheet_url or sheet_id")
        m = _SHEET_ID_RE.search(url)
        if not m:
            raise ConnectionError(f"Could not parse sheet id from URL: {url}")
        return m.group(1)

    def _fetch_via_gspread(self, sheet_id: str, worksheet: str | int):
        try:
            import gspread
            from google.oauth2.service_account import Credentials
        except ImportError as e:
            raise ConnectionError(
                "gspread + google-auth are required for authenticated "
                "Google Sheets access — pip install gspread google-auth"
            ) from e

        creds = None
        sa_json = self.params.get("service_account_json")
        sa_file = self.params.get("service_account_file")
        scopes = ["https://www.googleapis.com/auth/spreadsheets.readonly",
                  "https://www.googleapis.com/auth/drive.readonly"]
        if sa_json:
            try:
                info = json.loads(sa_json) if isinstance(sa_json, str) else sa_json
                creds = Credentials.from_service_account_info(info, scopes=scopes)
            except Exception as e:
                raise ConnectionError(f"Invalid service_account_json: {e}") from e
        elif sa_file:
            try:
                creds = Credentials.from_service_account_file(sa_file, scopes=scopes)
            except Exception as e:
                raise ConnectionError(f"Invalid service_account_file: {e}") from e

        if creds is None:
            raise ConnectionError("No service-account credentials supplied")

        try:
            gc = gspread.authorize(creds)
            sh = gc.open_by_key(sheet_id)
            ws = (sh.worksheet(worksheet) if isinstance(worksheet, str)
                  else sh.get_worksheet(int(worksheet)))
            data = ws.get_all_records()
            return pd.DataFrame(data)
        except Exception as e:
            raise ConnectionError(f"Google Sheets fetch failed: {e}") from e

    def _fetch_via_csv_export(self, sheet_id: str, worksheet: str | int) -> pd.DataFrame:
        """Fallback for publicly shared sheets — no auth required."""
        gid = "0" if isinstance(worksheet, str) else str(int(worksheet))
        url = (f"https://docs.google.com/spreadsheets/d/{sheet_id}"
               f"/export?format=csv&gid={gid}")
        try:
            return pd.read_csv(url)
        except Exception as e:
            raise ConnectionError(
                f"CSV export fetch failed (sheet may not be shared publicly): {e}"
            ) from e

    def fetch(self) -> ConnectionResult:
        sheet_id = self._resolve_sheet_id()
        worksheet = self.params.get("worksheet", 0)
        if isinstance(worksheet, str) and worksheet.isdigit():
            worksheet = int(worksheet)

        # Try authenticated path first if credentials are present
        if self.params.get("service_account_json") or self.params.get("service_account_file"):
            df = self._fetch_via_gspread(sheet_id, worksheet)
        else:
            df = self._fetch_via_csv_export(sheet_id, worksheet)

        return ConnectionResult.from_df(
            df, source=f"gsheet:{sheet_id}/{worksheet}",
            sheet_id=sheet_id, worksheet=str(worksheet),
        )


__all__ = ["GoogleSheetsConnector"]
