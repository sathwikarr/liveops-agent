"""S3Connector — fetch a CSV/Parquet object from S3 (or any S3-compatible
endpoint like MinIO/R2/B2). Auth via:
1. Explicit AWS access keys in params (treated as secrets)
2. Default boto3 credential chain (env vars, instance role, ~/.aws/credentials)
"""
from __future__ import annotations

import io
from typing import Optional
from urllib.parse import urlparse

import pandas as pd

from analyst.connectors.base import (
    Connector, ConnectionError, ConnectionResult,
)


def _parse_s3_uri(uri: str) -> tuple[str, str]:
    if not uri.startswith("s3://"):
        raise ConnectionError(f"Not an S3 URI: {uri}")
    parsed = urlparse(uri)
    bucket = parsed.netloc
    key = parsed.path.lstrip("/")
    if not bucket or not key:
        raise ConnectionError(f"Could not parse bucket/key from: {uri}")
    return bucket, key


class S3Connector(Connector):
    kind = "s3"
    param_schema = {
        "uri": "url",                      # s3://bucket/path/to/object.csv
        "endpoint_url": "url",             # for MinIO/R2 — optional
        "aws_access_key_id": "secret",
        "aws_secret_access_key": "secret",
        "region": "text",
        "format": "text",                  # auto | csv | parquet | json
    }

    def _detect_format(self, key: str) -> str:
        fmt = (self.params.get("format") or "auto").lower()
        if fmt != "auto":
            return fmt
        kl = key.lower()
        if kl.endswith(".parquet") or kl.endswith(".pq"):
            return "parquet"
        if kl.endswith(".json") or kl.endswith(".jsonl"):
            return "json"
        return "csv"

    def fetch(self) -> ConnectionResult:
        try:
            import boto3
        except ImportError as e:
            raise ConnectionError(
                "boto3 is required for the s3 connector — pip install boto3"
            ) from e

        uri = (self.params.get("uri") or "").strip()
        if not uri:
            raise ConnectionError("Missing 'uri' param")
        bucket, key = _parse_s3_uri(uri)

        client_kwargs = {}
        if self.params.get("region"):
            client_kwargs["region_name"] = self.params["region"]
        if self.params.get("endpoint_url"):
            client_kwargs["endpoint_url"] = self.params["endpoint_url"]
        if self.params.get("aws_access_key_id"):
            client_kwargs["aws_access_key_id"] = self.params["aws_access_key_id"]
            client_kwargs["aws_secret_access_key"] = self.params.get(
                "aws_secret_access_key", ""
            )

        try:
            s3 = boto3.client("s3", **client_kwargs)
            obj = s3.get_object(Bucket=bucket, Key=key)
            body = obj["Body"].read()
        except Exception as e:
            raise ConnectionError(f"S3 fetch failed: {e}") from e

        fmt = self._detect_format(key)
        try:
            if fmt == "parquet":
                df = pd.read_parquet(io.BytesIO(body))
            elif fmt == "json":
                # Try line-delimited first, fall back to a single JSON array
                try:
                    df = pd.read_json(io.BytesIO(body), lines=True)
                except ValueError:
                    df = pd.read_json(io.BytesIO(body))
            else:
                df = pd.read_csv(io.BytesIO(body))
        except Exception as e:
            raise ConnectionError(f"Failed to parse {fmt}: {e}") from e

        return ConnectionResult.from_df(
            df, source=uri, format=fmt, bytes=len(body),
        )


__all__ = ["S3Connector"]
