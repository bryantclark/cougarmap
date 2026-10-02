"""The saved-state file format: a pickled tree of plain objects whose big numpy arrays are stored as compressed
chunks, (de)compressed on several threads (zlib and numpy copies release the GIL).

Layout (version 2):

    MAGIC | array chunks ... | index (pickle) | index offset, index length (2 x uint64 LE) | END

The index holds the object tree, with every large array swapped for an `_ArrayRef` to its chunks. Each chunk is
a run of whole elements, byte-shuffled (all first bytes, then all second bytes, ...: smooth float layers then
compress ~20% smaller and faster) and zlib-compressed. Keeping the index at the end lets `update` swap small
entries (such as the run options) by rewriting only the tail of a copy, so a re-pick never recompresses hundreds
of MB of unchanged arrays. `load` also reads the older formats: a gzip-compressed pickle (states saved before
version 2) and a plain pickle.
"""

from __future__ import annotations

import gzip
import mmap
import os
import pickle
import shutil
import struct
import threading
import time
import zlib
from collections import deque
from collections.abc import Callable, Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO

import numpy as np

from .config import THREADS

MAGIC = b"CGMSTATE\x02\n"
END = b"CGMEND\n"
_TRAILER = struct.Struct("<QQ")
CHUNK = 1 << 20  # bytes of raw array data per chunk (small enough that even a small area keeps every thread busy)
MIN_ARRAY = 64 << 10  # smaller arrays are pickled into the index as-is
LEVEL = 1  # zlib level: byte-shuffled layers compress within ~5% of level 3, ~15% faster


@dataclass(frozen=True)
class _ArrayRef:
    """Where one array's chunks are: (offset, compressed length) per chunk, in order. Chunk k holds raw bytes
    [k * step, (k + 1) * step) of the array (step is a whole number of elements)."""

    dtype: str
    shape: tuple[int, ...]
    step: int
    chunks: tuple[tuple[int, int], ...]


def _step(itemsize: int) -> int:
    return max(1, CHUNK // itemsize) * itemsize


@dataclass(frozen=True)
class _Slot:
    i: int


def _walk(tree: Any, leaf: Callable[[Any], Any]) -> Any:
    """Copy of tree (nested dicts/lists/tuples) with leaf() applied to everything else."""
    if isinstance(tree, dict):
        return {k: _walk(v, leaf) for k, v in tree.items()}
    if isinstance(tree, list):
        return [_walk(v, leaf) for v in tree]
    if isinstance(tree, tuple):
        return tuple(_walk(v, leaf) for v in tree)
    return leaf(tree)


def _shuffle(b: np.ndarray, itemsize: int) -> np.ndarray:
    return b if itemsize == 1 else np.ascontiguousarray(b.reshape(-1, itemsize).T)


def _unshuffle(b: np.ndarray, itemsize: int) -> np.ndarray:
    return b if itemsize == 1 else np.ascontiguousarray(b.reshape(itemsize, -1).T).reshape(-1)


def _pack(b: np.ndarray, itemsize: int, level: int) -> bytes:
    return zlib.compress(_shuffle(b, itemsize), level)


def _chunks(a: np.ndarray) -> Iterator[np.ndarray]:
    """Raw bytes of a (as uint8 views), CHUNK at a time, cut at element boundaries."""
    raw = np.ascontiguousarray(a).reshape(-1).view(np.uint8)
    step = _step(a.itemsize)
    for i in range(0, raw.size, step):
        yield raw[i : i + step]


def _write_index(f: BinaryIO, blob: bytes) -> None:
    """Write a pickled index at the current position, then the trailer pointing at it."""
    off = f.tell()
    f.write(blob)
    f.write(_TRAILER.pack(off, len(blob)))
    f.write(END)


def save(tree: Any, path: str | Path, level: int = LEVEL) -> None:
    """Write tree (nested dicts/lists/tuples of picklable objects and numpy arrays) to path, atomically."""
    path = Path(path)
    arrays: list[np.ndarray] = []

    def take(v: Any) -> Any:
        if isinstance(v, np.ndarray) and v.nbytes >= MIN_ARRAY and v.dtype.kind in "biuf":
            arrays.append(v)
            return _Slot(len(arrays) - 1)
        return v

    skeleton = _walk(tree, take)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    try:
        with tmp.open("wb") as f, ThreadPoolExecutor(THREADS) as ex:
            f.write(MAGIC)
            spans: list[list[tuple[int, int]]] = [[] for _ in arrays]
            # all chunks of all arrays stream through the pool in order, a bounded number in flight
            pending: deque[tuple[int, Future[bytes]]] = deque()

            def drain(keep: int) -> None:
                while len(pending) > keep:
                    i, fut = pending.popleft()
                    comp = fut.result()
                    spans[i].append((f.tell(), len(comp)))
                    f.write(comp)

            for i, a in enumerate(arrays):
                for b in _chunks(a):
                    pending.append((i, ex.submit(_pack, b, a.itemsize, level)))
                    drain(4 * THREADS)
            drain(0)
            refs = [
                _ArrayRef(a.dtype.str, tuple(a.shape), _step(a.itemsize), tuple(s))
                for a, s in zip(arrays, spans, strict=True)
            ]
            index = _walk(skeleton, lambda v: refs[v.i] if isinstance(v, _Slot) else v)
            _write_index(f, pickle.dumps(index, protocol=pickle.HIGHEST_PROTOCOL))
        _swap(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def _read_index(f: BinaryIO) -> tuple[Any, int]:
    """(index tree, index offset) of a version-2 file."""
    f.seek(-(_TRAILER.size + len(END)), os.SEEK_END)
    tail = f.read()
    if tail[-len(END) :] != END:
        raise ValueError("truncated or corrupt state file (no end marker)")
    off, n = _TRAILER.unpack(tail[: _TRAILER.size])
    f.seek(off)
    return pickle.loads(f.read(n)), off


def is_v2(path: str | Path) -> bool:
    with Path(path).open("rb") as f:
        return f.read(len(MAGIC)) == MAGIC


def load(path: str | Path) -> Any:
    """Read a state file in any format (version 2, gzip pickle, plain pickle)."""
    path = Path(path)
    with path.open("rb") as f:
        head = f.read(len(MAGIC))
        if head != MAGIC:
            f.seek(0)
            if head[:2] == b"\x1f\x8b":
                with gzip.open(f, "rb") as g:
                    return pickle.load(g)
            return pickle.load(f)
        index, _ = _read_index(f)
        with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm, ThreadPoolExecutor(THREADS) as ex:
            jobs: list[Future[None]] = []

            def unpack(dst: np.ndarray, itemsize: int, off: int, n: int) -> None:
                with memoryview(mm)[off : off + n] as src:
                    b = np.frombuffer(zlib.decompress(src), np.uint8)
                dst[:] = _unshuffle(b, itemsize)

            def materialize(v: Any) -> Any:
                if not isinstance(v, _ArrayRef):
                    return v
                arr = np.empty(v.shape, np.dtype(v.dtype))
                raw, step = arr.reshape(-1).view(np.uint8), v.step
                for k, (off, n) in enumerate(v.chunks):
                    jobs.append(ex.submit(unpack, raw[k * step : (k + 1) * step], arr.itemsize, off, n))
                return arr

            out = _walk(index, materialize)
            for j in jobs:
                j.result()
            return out


SWAP_WAIT_S = 10.0  # how long a swap waits for readers to close the file (Windows only blocks)


def _swap(tmp: Path, path: Path, wait_s: float = SWAP_WAIT_S) -> None:
    """Atomically replace path with tmp. Windows can't replace a file another process has open (an MCP server or
    job loading the same state), so a refused swap is retried until the readers let go, then raised."""
    deadline = time.monotonic() + wait_s
    while True:
        try:
            tmp.replace(path)
            return
        except PermissionError:
            if time.monotonic() >= deadline:
                raise
            time.sleep(0.05)


def update(path: str | Path, fn: Callable[[Any], Any]) -> None:
    """Replace a version-2 file's object tree with fn(tree), rewriting only the index at the end of the file.
    fn sees arrays as opaque references: change small entries, keep the references where they are.

    Atomic, like save: the file is copied (a kernel-side copy, ~0.1 s for a 600 MB state on an SSD), the copy's
    index is replaced, and the copy is swapped in. A failure part way leaves the old file intact,
    and a reader in another process sees the old file or the new one, never a half-written index."""
    path = Path(path)
    with path.open("rb") as f:
        index, off = _read_index(f)
    blob = pickle.dumps(fn(index), protocol=pickle.HIGHEST_PROTOCOL)  # fails here, before any file is touched
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    try:
        shutil.copyfile(path, tmp)
        with tmp.open("r+b") as f:
            f.seek(off)
            f.truncate()
            _write_index(f, blob)
            f.flush()
            os.fsync(f.fileno())
        _swap(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)
