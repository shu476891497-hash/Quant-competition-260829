from __future__ import annotations

import hashlib
import json
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


def sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_bytes(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def retry_session() -> requests.Session:
    retry = Retry(
        total=2,
        connect=2,
        read=2,
        backoff_factor=0.6,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset({"GET"}),
    )
    session = requests.Session()
    session.mount("https://", HTTPAdapter(max_retries=retry))
    return session


def latest_completed_us_date(now: datetime | None = None) -> pd.Timestamp:
    current = now or datetime.now(ZoneInfo("America/New_York"))
    if current.tzinfo is None:
        current = current.replace(tzinfo=ZoneInfo("America/New_York"))
    else:
        current = current.astimezone(ZoneInfo("America/New_York"))
    cutoff = current.date() if current.hour >= 18 else current.date() - timedelta(days=1)
    return pd.Timestamp(cutoff)


def retrieval_record(path: Path, url: str, rows: int | None = None) -> dict[str, object]:
    return {
        "path": str(path),
        "url": url,
        "retrieved_at_utc": datetime.now(ZoneInfo("UTC")).isoformat(),
        "sha256": sha256_file(path),
        "rows": rows,
    }
