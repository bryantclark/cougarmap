"""HTTP with retries plus a simple on-disk cache keyed by request."""

from __future__ import annotations

import hashlib
import json
import os
import pickle
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import requests

from .config import CACHE_DIR, USER_AGENT

_local = threading.local()


def session() -> requests.Session:
    """One HTTP session (connection pooling) per thread: downloads run in parallel, and requests.Session is not
    guaranteed thread-safe."""
    s: requests.Session | None = getattr(_local, "session", None)
    if s is None:
        s = requests.Session()
        s.headers["User-Agent"] = USER_AGENT
        _local.session = s
    return s


def get(
    url: str,
    *,
    params: dict[str, Any] | None = None,
    data: dict[str, Any] | None = None,
    timeout: float = 180,
    retries: int = 4,
) -> requests.Response:
    last: Exception | None = None
    for attempt in range(retries):
        try:
            if data is not None:
                r = session().post(url, data=data, timeout=timeout)
            else:
                r = session().get(url, params=params, timeout=timeout)
            if r.status_code in (429, 500, 502, 503, 504):
                raise requests.HTTPError(f"{r.status_code} from {url}")
            r.raise_for_status()
            return r
        except (requests.RequestException, requests.HTTPError) as e:
            last = e
            time.sleep(2 * (attempt + 1) ** 2)
    raise RuntimeError(f"request failed after {retries} tries: {url}: {last}")


def get_json(
    url: str,
    *,
    params: dict[str, Any] | None = None,
    data: dict[str, Any] | None = None,
    timeout: float = 180,
    retries: int = 4,
) -> Any:
    return get(url, params=params, data=data, timeout=timeout, retries=retries).json()


def _key(namespace: str, parts: Any, suffix: str = ".pkl") -> Path:
    h = hashlib.sha1(json.dumps(parts, sort_keys=True, default=str).encode()).hexdigest()[:20]
    d = CACHE_DIR / namespace
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{h}{suffix}"


def cache_path(namespace: str, parts: Any, suffix: str) -> Path:
    """Where a cached file for (namespace, parts) lives (for entries kept in their own format, e.g. a GeoTIFF)."""
    return _key(namespace, parts, suffix)


_locks: dict[Path, threading.Lock] = {}
_locks_guard = threading.Lock()


def _lock_for(p: Path) -> threading.Lock:
    with _locks_guard:
        return _locks.setdefault(p, threading.Lock())


def cache_lock(p: Path) -> threading.Lock:
    """The lock parallel callers share for one cache file (see cached)."""
    return _lock_for(p)


def cached(namespace: str, parts: Any, fn: Callable[[], Any]) -> Any:
    """Return fn() cached on disk under (namespace, parts). Parallel callers asking for the same entry wait for
    one download instead of each doing it."""
    p = _key(namespace, parts)
    with _lock_for(p):
        return _cached(p, fn)


def _cached(p: Path, fn: Callable[[], Any]) -> Any:
    if p.exists():
        try:
            with p.open("rb") as f:
                return pickle.load(f)
        except Exception:
            p.unlink(missing_ok=True)
    val = fn()
    tmp = p.with_suffix(f".{os.getpid()}.{threading.get_ident()}.tmp")  # parallel downloads never share a temp
    with tmp.open("wb") as f:
        pickle.dump(val, f, protocol=pickle.HIGHEST_PROTOCOL)
    tmp.replace(p)
    return val
