"""Small HTTP helpers: retries, byte ranges, polite downloads."""

from __future__ import annotations

import re
import time
from numbers import Integral

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
    if (isinstance(start, bool) or not isinstance(start, Integral) or start < 0
            or (end is not None and (isinstance(end, bool)
                                     or not isinstance(end, Integral) or end < start))):
        raise ValueError("invalid byte range")
    rng = f"bytes={start}-" if end is None else f"bytes={start}-{end}"
    r = get(url, headers={"Range": rng}, timeout=120)
    match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+|\*)", r.headers.get("Content-Range", ""))
    if r.status_code != 206 or match is None:
        raise requests.RequestException("server did not return a byte range", response=r)
    first, last = int(match[1]), int(match[2])
    total = None if match[3] == "*" else int(match[3])
    expected_last = end
    if total is not None:
        expected_last = total - 1 if end is None else min(end, total - 1)
    if (first != start or last < first or len(r.content) != last - first + 1
            or (end is None and total is None)
            or (expected_last is not None and last != expected_last)):
        raise requests.RequestException("response does not match requested byte range", response=r)
    return r.content
