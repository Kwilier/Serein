"""Small HTTP helpers: retries, byte ranges, polite downloads."""

from __future__ import annotations

import time

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

_session: requests.Session | None = None


def session() -> requests.Session:
    global _session
    if _session is None:
        s = requests.Session()
        retry = Retry(
            total=5, backoff_factor=1.0,
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=("GET", "HEAD"),
        )
        adapter = HTTPAdapter(max_retries=retry, pool_connections=32, pool_maxsize=32)
        s.mount("https://", adapter)
        s.mount("http://", adapter)
        s.headers["User-Agent"] = "serein-research/0.1 (student science project)"
        _session = s
    return _session


def get(url: str, *, headers: dict | None = None, params: dict | None = None,
        timeout: float = 60, allow_404: bool = False) -> requests.Response | None:
    for attempt in range(3):
        try:
            r = session().get(url, headers=headers, params=params, timeout=timeout)
        except requests.RequestException:
            if attempt == 2:
                raise
            time.sleep(2 ** attempt)
            continue
        if r.status_code == 404 and allow_404:
            return None
        r.raise_for_status()
        return r
    return None


def get_range(url: str, start: int, end: int | None) -> bytes:
    """Download bytes [start, end] inclusive (end=None means to the end of file)."""
    rng = f"bytes={start}-" if end is None else f"bytes={start}-{end}"
    r = get(url, headers={"Range": rng}, timeout=120)
    return r.content
