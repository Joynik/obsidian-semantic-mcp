#!/usr/bin/env python3
"""MCP server for Obsidian vault search, file tools, and live indexing."""

from __future__ import annotations

import asyncio
import contextlib
import contextvars
import fcntl
import hashlib
import logging
import math
import os
import re
import secrets
import signal
import stat
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import anyio
import psycopg2
import psycopg2.pool
import requests
import yaml
from dotenv import load_dotenv
from mcp.server import Server
from mcp.shared.message import SessionMessage
from mcp.types import (
    CallToolResult,
    ListToolsResult,
    TextContent,
    Tool,
    jsonrpc_message_adapter,
)
from psycopg2 import sql
from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer
from watchdog.observers.api import BaseObserver
from watchdog.observers.polling import PollingObserver

# Source checkouts may use their gitignored .env. Installed entry points load
# owner-only settings in src.launcher before this module is imported.
_checkout_env = Path(__file__).resolve().parent.parent
if (_checkout_env / "pyproject.toml").is_file():
    load_dotenv(_checkout_env / ".env", override=False)

try:
    from .config import (  # installed as a package (uv tool / pip install)
        DEFAULT_EMBED_MODEL,
        REINDEX_LOCK_KEY,
        REQUIRED_FRONTMATTER_DEFAULTS,
        build_dsn,
        search_language,
    )
    from .embedding_profiles import embedding_input
    from .postgres import PostgresStore
    from .qdrant import QdrantStore, local_url
except ImportError:
    from config import (  # fallback: run directly from src/ during dev
        DEFAULT_EMBED_MODEL,
        REINDEX_LOCK_KEY,
        REQUIRED_FRONTMATTER_DEFAULTS,
        build_dsn,
        search_language,
    )
    from embedding_profiles import embedding_input
    from postgres import PostgresStore
    from qdrant import QdrantStore, local_url


# ─────────────────────────────────── Config ─────────────────────────────────


def _parse_vault_paths() -> list[str]:
    """Return vault path list from OBSIDIAN_VAULTS (comma-separated) or OBSIDIAN_VAULT."""
    multi = os.environ.get("OBSIDIAN_VAULTS", "")
    if multi:
        return [v.strip() for v in multi.split(",") if v.strip()]
    single = os.environ.get("OBSIDIAN_VAULT", "")
    return [single] if single else []


VAULT_PATHS: list[str] = _parse_vault_paths()
VAULT_PATH: str = VAULT_PATHS[0] if VAULT_PATHS else ""  # primary vault (backward compat)
# Snapshot of VAULT_PATHS used by path helpers. Patchable by tests via
# monkeypatch.setattr(server, "_VAULT_LIST", [...]).
_VAULT_LIST: list[str] = list(VAULT_PATHS)
OLLAMA_URL = os.environ.get("OLLAMA_URL", "")
EMBED_MODEL = os.environ.get("EMBEDDING_MODEL", DEFAULT_EMBED_MODEL)
EMBED_TIMEOUT = int(os.environ.get("EMBED_TIMEOUT", "30"))
EMBED_WORKERS = int(
    os.environ.get("EMBED_WORKERS", "1")
)  # parallel embedding threads — match Ollama's serving slots (default 1, CPU-only)
EMBED_BATCH_SIZE = int(
    os.environ.get("EMBED_BATCH_SIZE", "16")
)  # texts per /api/embed call (Ollama 0.4+)
RERANK_MODEL = os.environ.get("RERANK_MODEL", "")  # cross-encoder model; empty = disabled
RERANK_CANDIDATES = int(
    os.environ.get("RERANK_CANDIDATES", "20")
)  # candidate pool size before re-ranking
# Polling watcher: "true" forces PollingObserver for all vaults (required for
# network filesystems like NFS/SMB where OS-level inotify/ReadDirectoryChangesW
# does not fire for writes from remote clients).  "auto" (default) uses a
# heuristic — UNC paths and mapped network drives get polling, local paths get
# native events.  "false" always uses native events.
VAULT_WATCH_POLLING = os.environ.get("VAULT_WATCH_POLLING", "auto").lower()
VAULT_POLL_INTERVAL = int(os.environ.get("VAULT_POLL_INTERVAL", "10"))  # seconds between polls

DATABASE_URL = build_dsn()
DATABASE_BACKEND = os.environ.get("OSM_DATABASE_BACKEND", "postgresql")


def _qstore() -> QdrantStore:
    return QdrantStore(local_url(), EMBED_MODEL)


def _pgstore() -> PostgresStore:
    return PostgresStore(db_conn, EMBED_MODEL)


MAX_EMBED_CHARS = 2000  # Bound per-note input size and embedding cost.
CHUNK_CHARS = 1200
CHUNK_OVERLAP = 200
_TIMESTAMP_FMT = "%Y-%m-%d %H:%M"
_DEBOUNCE_SECS = 0.5  # collapse rapid saves from Obsidian autosave

# Set during background_init so search_vault can return a useful message
# instead of the misleading "No indexed notes found. Try running reindex_vault."
# threading.Event is used rather than a bare bool to avoid any cross-thread
# visibility issues without relying on the GIL.
#
# Kept as a fast in-process short-circuit even after persistent state made
# index_state (Postgres) the source of truth for indexing status — a DB
# round-trip on every search would be wasteful when this process itself is
# the one indexing. It is NOT cross-process visible: a search hitting a
# different process while this one indexes will not see it set. That
# case is covered by index_state, which get_last_rebuild_failures() and
# /api/stats read instead.
_INDEXING_IN_PROGRESS = threading.Event()
_STOPPING = threading.Event()
_readiness_lock = threading.Lock()
_readiness_state = "waiting_ollama"
_readiness_error = ""


def _set_readiness(state: str, error: str = "") -> None:
    global _readiness_state, _readiness_error
    with _readiness_lock:
        _readiness_state, _readiness_error = state, error


def _get_readiness() -> tuple[str, str]:
    with _readiness_lock:
        return _readiness_state, _readiness_error


def _search_readiness_message(mode: str) -> str | None:
    state, error = _get_readiness()
    if state == "waiting_ollama" and mode != "keyword":
        return "Semantic search is waiting for Ollama and the embedding model. Try again shortly."
    if state == "error":
        return f"OSM initialization failed: {error}. Check the server log and run osm status."
    return None


def _empty_search_message(vault_ids: list[str] | None) -> str:
    """Explain an empty result using persistent per-vault index state."""
    try:
        states = get_index_state()
    except Exception as exc:
        log.warning("Cannot read index readiness: %s", exc)
        state, _ = _get_readiness()
        if state in ("indexing", "partial_error"):
            return "Index status is unavailable. Check osm status and the server log."
        return f"Search index unavailable: {type(exc).__name__}. Run osm status."
    selected = [row for row in states if vault_ids is None or row["vault_id"] in vault_ids]
    failures = [row for row in selected if row["status"] in ("failed", "dimension_mismatch")]
    if failures:
        details = "; ".join(
            f"{Path(row['vault_id']).name}: {row.get('error') or row['status']}" for row in failures
        )
        return f"Index error: {details}. Check osm status and the server log."
    if _get_readiness()[0] == "indexing" or any(
        row["status"] in ("indexing", "pending") for row in selected
    ):
        return "Vault indexing is in progress; no matching indexed notes yet. Try again shortly."
    if not selected:
        return "Index has no status for this vault. Run osm status."
    return "No matching indexed notes found."


_OLLAMA_RETRY_SECONDS = 10
_OLLAMA_NOTIFY_AFTER_SECONDS = 120

# Wikilink targets include short names and paths relative to a vault root.
_link_index: dict[str, set[str]] = {}
_link_index_lock = threading.Lock()

# Matches [[note]], [[note|alias]], [[note#heading]], [[folder/note]]
_WIKILINK_RE = re.compile(r"\[\[([^\]|#]+)(?:[|#][^\]]*)?\]\]")


def extract_wikilinks(content: str) -> list[str]:
    """Return all wikilink targets found in content (deduplicated, order-preserved)."""
    seen: set[str] = set()
    out: list[str] = []
    for m in _WIKILINK_RE.finditer(content):
        name = m.group(1).strip()
        if name and name not in seen:
            seen.add(name)
            out.append(name)
    return out


def _link_keys(path: Path, vault: str) -> set[str]:
    relative = path.relative_to(vault).with_suffix("")
    return {relative.as_posix().casefold(), path.stem.casefold()}


def _add_link(index: dict[str, set[str]], path: Path, vault: str) -> None:
    for key in _link_keys(path, vault):
        index.setdefault(key, set()).add(str(path))


def _remove_link(index: dict[str, set[str]], path: str) -> None:
    for key, paths in list(index.items()):
        paths.discard(path)
        if not paths:
            del index[key]


def _merge_links(target: dict[str, set[str]], source: dict[str, set[str]]) -> None:
    for key, paths in source.items():
        target.setdefault(key, set()).update(paths)


def _build_link_index(vault: str) -> dict[str, set[str]]:
    """Index short names and vault-relative paths, retaining duplicates."""
    index: dict[str, set[str]] = {}
    for f in Path(vault).rglob("*.md"):
        if not _should_skip_path(f):
            _add_link(index, f, vault)
    return index


def _resolve_links(names: list[str], index: dict[str, set[str]]) -> dict[str, str | None]:
    """Resolve only unique wikilinks; path-style links use their full relative path."""
    result: dict[str, str | None] = {}
    for name in names:
        key = name.removesuffix(".md").replace("\\", "/").casefold()
        paths = index.get(key, set())
        result[name] = next(iter(paths)) if len(paths) == 1 else None
    return result


logging.basicConfig(
    stream=sys.stderr,
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

_DEFAULT_IGNORED_PATH_SEGMENTS = {"archive"}
_ALWAYS_SKIPPED_PATH_SEGMENTS = {".obsidian", ".trash", ".git"}


# ──────────────────────────────── Database ───────────────────────────────────


class _SearchCacheCompatibility:
    """Keep the old invalidation hook available to existing integrations."""

    def invalidate(self) -> None:
        pass


_search_cache = _SearchCacheCompatibility()

_pool: psycopg2.pool.ThreadedConnectionPool | None = None
_pool_lock = threading.Lock()

# Watcher observers — one per vault; held here so the shutdown handler can stop them.
_observers: list[BaseObserver] = []


def _get_pool() -> psycopg2.pool.ThreadedConnectionPool:
    """Return the shared connection pool, initialising it on first call."""
    global _pool
    if _pool is None:
        with _pool_lock:
            if _pool is None:
                _pool = psycopg2.pool.ThreadedConnectionPool(1, 5, DATABASE_URL, connect_timeout=5)
    return _pool


def require_active_model(cur, model: str) -> None:
    """Hold a read lock on the model identity until the caller commits."""
    PostgresStore(db_conn, model).require_model(cur)


@contextlib.contextmanager
def db_conn():
    """Acquire a connection from the pool and return it on exit.

    On exception the connection is discarded (close=True) so any open
    transaction is rolled back and the pool gets a fresh connection next time.
    """
    pool = _get_pool()
    conn = pool.getconn()
    try:
        yield conn
    except Exception:
        # Return the connection as broken so the pool replaces it rather than
        # recycling a connection that may have an aborted transaction.
        pool.putconn(conn, close=True)
        raise
    else:
        pool.putconn(conn)


_SEARCH_METRIC_CONTEXT: contextvars.ContextVar[dict[str, Any] | None] = contextvars.ContextVar(
    "search_metric_context", default=None
)
_SEARCH_METRIC_BACKENDS = {"local"}
_SEARCH_METRIC_MODES = {"hybrid", "semantic", "keyword"}
_SEARCH_METRIC_OUTCOMES = {"success", "error"}


def _init_search_metrics():
    """Create an additive aggregate table; metrics availability is optional."""
    if DATABASE_BACKEND == "qdrant":
        return
    try:
        _pgstore().init_search_metrics()
    except Exception as exc:
        log.warning("search metrics initialization failed type=%s", type(exc).__name__)


def _record_search_metric(state, duration_ms, now=None):
    """Upsert only allowlisted dimensions and numeric aggregates, retaining 30 days."""
    if DATABASE_BACKEND == "qdrant":
        return
    try:
        if (
            state["backend"] not in _SEARCH_METRIC_BACKENDS
            or state["mode"] not in _SEARCH_METRIC_MODES
            or state["outcome"] not in _SEARCH_METRIC_OUTCOMES
        ):
            raise ValueError("invalid metric dimensions")
        if type(state["results"]) is not int or state["results"] < 0:
            raise ValueError("invalid result count")
        if (
            not isinstance(duration_ms, (int, float))
            or not math.isfinite(duration_ms)
            or duration_ms < 0
        ):
            raise ValueError("invalid metric duration")
        hour = (now or datetime.now(UTC)).astimezone(UTC).replace(minute=0, second=0, microsecond=0)
        _pgstore().record_search_metric(hour, hour - timedelta(hours=719), state, duration_ms)
    except Exception as exc:
        log.warning("search metrics write failed type=%s", type(exc).__name__)


def _search_metric_success(result_count):
    state = _SEARCH_METRIC_CONTEXT.get()
    if state is not None:
        state.update(outcome="success", results=result_count)


@contextlib.contextmanager
def reindex_lock():
    """Postgres advisory lock guarding re-index mutual exclusion.

    A process-local lock would allow the dashboard and MCP server to re-index
    concurrently.

    Session-level (`pg_try_advisory_lock` / `pg_advisory_unlock`), not
    `pg_advisory_xact_lock`: the lock must outlive the short transaction
    that acquires it and span the whole re-index pass, which can run for
    hours on a CPU-only Ollama.

    Held on a DEDICATED connection checked out from the pool for the whole
    critical section — never a connection returned to the pool while the
    lock is still held. Session-level advisory locks are bound to the
    session (the connection): returning the connection to the pool mid-hold
    would let some unrelated caller borrow that same connection and would
    silently release the lock as a side effect of that connection's next
    use, not of an explicit unlock call.

    Non-blocking: `pg_try_advisory_lock` returns immediately rather than
    waiting. Callers that get `False` must NOT proceed with the re-index —
    another holder already has it (dashboard's 409 / the MCP tool's "busy"
    message both key off this).

    Yields `False` (rather than raising) when no DB connection can be
    obtained at all, so a caller can treat "DB unreachable" and "lock held
    by someone else" the same way — neither should be started as if a lock
    signal were missing entirely.

    The connection pool is `ThreadedConnectionPool(1, 5)`. Holding one
    connection for the duration of a re-index leaves 4 free — acceptable,
    but do not raise the pool minimum without understanding this: a
    minconn bump would eagerly open connections this lock's hold pattern
    doesn't need released early.

    Usable both as `with reindex_lock() as acquired:` for a single
    acquire-do-release scope, and via manual `.__enter__()` / `.__exit__()`
    when the acquiring thread (an HTTP request handler) and the releasing
    thread (a background re-index worker) are different — the generator-
    based context manager object has no thread affinity of its own.
    """
    if DATABASE_BACKEND == "qdrant":
        lock_path = Path(f"/private/tmp/osm-qdrant-{os.geteuid()}.lock")
        fd = os.open(lock_path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
        try:
            details = os.fstat(fd)
            if (
                not stat.S_ISREG(details.st_mode)
                or details.st_uid != os.geteuid()
                or stat.S_IMODE(details.st_mode) != 0o600
                or details.st_nlink != 1
            ):
                raise ValueError("unsafe Qdrant reindex lock file")
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                yield False
                return
            try:
                yield True
            finally:
                fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)
        return
    try:
        pool = _get_pool()
        conn = pool.getconn()
    except Exception as e:
        log.warning("reindex_lock: could not obtain a DB connection: %s", e)
        yield False
        return

    acquired = False
    try:
        acquired = _pgstore().acquire_reindex_lock(conn, REINDEX_LOCK_KEY)
        conn.commit()
        yield acquired
    finally:
        if acquired:
            try:
                _pgstore().release_reindex_lock(conn, REINDEX_LOCK_KEY)
                conn.commit()
                pool.putconn(conn)
            except Exception as e:
                log.warning("reindex_lock: failed to release advisory lock: %s", e)
                pool.putconn(conn, close=True)
        else:
            pool.putconn(conn)


def init_db(embed_dim: int) -> None:
    if DATABASE_BACKEND == "qdrant":
        _qstore().ensure(embed_dim)
        return
    lists, existing_dim = _pgstore().initialize(embed_dim, VAULT_PATH)
    log.info("Database initialised (IVFFlat lists=%d, embed_dim=%d)", lists, embed_dim)
    if existing_dim is not None:
        log.warning(
            "Embedding dimension mismatch: DB has vector(%d) but %s produces %d. "
            "Run `osm migrate --embedding-model MODEL` before changing models.",
            existing_dim,
            EMBED_MODEL,
            embed_dim,
        )
        mismatch_error = (
            f"DB has vector({existing_dim}) but {EMBED_MODEL} produces "
            f"vector({embed_dim}). Run `osm migrate --embedding-model MODEL` "
            "to migrate without data loss."
        )
        for vault in VAULT_PATHS:
            set_index_state(vault, "dimension_mismatch", error=mismatch_error)


_DIMENSION_MISMATCH_RE = re.compile(r"expected \d+ dimensions?, not \d+")

# Guards the read-merge-write in _record_watcher_dimension_failure. This is an
# in-process lock, not a cross-process one like reindex_lock: the watcher
# (VaultEventHandler debounce timers) only ever runs inside this one
# mcp-server process, so a plain threading.Lock is the correctly-scoped fix
# for the intra-process lost-update race (two debounced files in the same
# vault failing at the same time) — no advisory lock needed.
_dimension_failure_lock = threading.Lock()


def _record_watcher_dimension_failure(vault_id: str, path: str, error: str) -> None:
    """Merge a watcher-detected column-dimension-mismatch write failure into
    index_state, so it's dashboard-visible instead of only a log line.

    The boot-time check in init_db() only runs once at startup; an edit made
    afterward that fails with the same pgvector column-dimension error was
    previously silent beyond a per-file log.warning in _handle_upsert. This
    does a read-merge-write against the existing failed_paths list (deduping)
    rather than overwriting the vault's row wholesale, so it doesn't clobber
    an unrelated "idle"/"indexing" status recorded by a concurrent boot pass.

    The merge is serialized by _dimension_failure_lock so two watcher threads
    recording failures for the same vault_id at the same time can't lose one
    path to a lost-update race (each read-then-write must complete before the
    next one starts).

    Never raises — same contract as set_index_state (observability must not
    break the caller it's called from).
    """
    try:
        with _dimension_failure_lock:
            current = get_index_state(vault_id)
            failed = list(current["failed_paths"]) if current else []
            if path not in failed:
                failed.append(path)
            set_index_state(vault_id, "dimension_mismatch", failed_paths=failed, error=error)
    except Exception as e:
        log.warning(
            "_record_watcher_dimension_failure failed for vault_id=%s path=%s: %s",
            vault_id,
            path,
            e,
        )


def _row_to_index_state_dict(row: tuple) -> dict:
    vault_id, status, started_at, finished_at, failed_paths, error = row
    return {
        "vault_id": vault_id,
        "status": status,
        "started_at": started_at,
        "finished_at": finished_at,
        "failed_paths": list(failed_paths or []),
        "error": error,
    }


def set_index_state(
    vault_id: str,
    status: str,
    *,
    failed_paths: list[str] | None = None,
    error: str | None = None,
) -> None:
    """Upsert this vault's row in index_state.

    Called at the start ("indexing"), end ("idle" or "failed"), and
    exception path of background_init/index_vault so indexing progress and
    failures are visible across process boundaries — the
    module-global list this replaces was only ever readable by the process
    that set it, which is why the dashboard's rebuild-failure panel was
    structurally always empty (it runs in a separate process).

    Never raises — a failure writing observability data must not abort an
    in-flight index pass. Logs a warning and returns on any DB error.
    """
    if DATABASE_BACKEND == "qdrant":
        try:
            profile = _qstore().profile()
            if profile:
                _qstore().set_state(
                    vault_id, status, failed_paths or [], error, int(profile["dimension"])
                )
        except Exception as exc:
            log.warning("Qdrant index state write failed: %s", exc)
        return
    try:
        _pgstore().set_index_state(vault_id, status, failed_paths or [], error)
    except Exception as e:
        log.warning("set_index_state failed for vault_id=%s status=%s: %s", vault_id, status, e)


def get_index_state(vault_id: str | None = None):
    """Return index_state row(s).

    With `vault_id`: a single dict, or None if that vault has no row yet.
    Without: a list of dicts, one per vault, ordered by vault_id.

    Unlike set_index_state, this DOES raise on a DB error — callers that
    must degrade gracefully (get_last_rebuild_failures, the /api/stats
    path) catch it explicitly rather than have failure silently disguised
    as "no vaults indexed yet."
    """
    if DATABASE_BACKEND == "qdrant":
        states = sorted(_qstore().states(), key=lambda row: row["vault_id"])
        if vault_id is not None:
            return next((state for state in states if state["vault_id"] == vault_id), None)
        return states
    rows = _pgstore().index_state(vault_id)
    if vault_id is not None:
        return _row_to_index_state_dict(rows[0]) if rows else None
    return [_row_to_index_state_dict(row) for row in rows]


def get_last_rebuild_failures() -> list[str]:
    """Union of failed_paths across every vault's index_state row.

    Falls back to an empty list on any DB error rather than raising —
    observability (the dashboard's rebuild-failure panel) must never break
    the /api/stats endpoint that surfaces it.
    """
    try:
        rows = get_index_state()
    except Exception as e:
        log.warning("get_last_rebuild_failures: could not read index_state: %s", e)
        return []
    failed: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for path in row["failed_paths"]:
            if path not in seen:
                seen.add(path)
                failed.append(path)
    return failed


# ──────────────────────────────── Embeddings ─────────────────────────────────


def _vec_to_str(vec: list[float]) -> str:
    """Format a float list as a pgvector literal, e.g. '[0.1,0.2,...]'."""
    if not vec:
        raise ValueError("Cannot convert empty list to vector literal")
    return "[" + ",".join(str(v) for v in vec) + "]"


def _embedding_input(text: str, task: str, model: str | None = None) -> str:
    return embedding_input(text, task, model or EMBED_MODEL)


def embed(text: str, task: str = "document", model: str | None = None) -> list[float]:
    """Embed text with Ollama, truncating input to MAX_EMBED_CHARS.

    Retries up to 3 times with exponential backoff (1s → 2s → 4s) on transient errors.
    """
    if task not in ("document", "query"):
        raise ValueError("invalid embedding task")
    text = _embedding_input(text[:MAX_EMBED_CHARS], task, model)
    for attempt in range(3):
        try:
            resp = requests.post(
                f"{OLLAMA_URL}/api/embed",
                json={"model": model or EMBED_MODEL, "input": text},
                timeout=EMBED_TIMEOUT,
            )
            resp.raise_for_status()
            vectors = resp.json().get("embeddings", [])
            vec = vectors[0] if vectors else []
            if not vec:
                raise ValueError(f"Empty embedding returned by Ollama for text: {text[:50]!r}")
            return vec
        except (requests.RequestException, ValueError) as e:
            if attempt == 2:
                raise
            wait = 2**attempt
            log.warning("embed attempt %d failed: %s — retrying in %ds", attempt + 1, e, wait)
            time.sleep(wait)
    raise RuntimeError("embed: exhausted retries without raising — should not reach here")


def embed_batch(texts: list[str]) -> list[list[float]]:
    """Embed multiple texts in a single Ollama /api/embed call (Ollama 0.4+).

    Returns embeddings in the same order as the input list. Truncates each
    text to MAX_EMBED_CHARS. Raises on transient errors after 3 attempts —
    the caller is responsible for falling back to per-item embed() if the
    batch endpoint isn't available (older Ollama) or returns an empty
    embedding for any item.
    """
    if not texts:
        return []
    truncated = [_embedding_input(t[:MAX_EMBED_CHARS], "document") for t in texts]
    for attempt in range(3):
        try:
            resp = requests.post(
                f"{OLLAMA_URL}/api/embed",
                json={"model": EMBED_MODEL, "input": truncated},
                timeout=EMBED_TIMEOUT,
            )
            resp.raise_for_status()
            data = resp.json()
            vecs = data.get("embeddings", [])
            if len(vecs) != len(truncated):
                raise ValueError(
                    f"Ollama batch returned {len(vecs)} embeddings for {len(truncated)} inputs"
                )
            return vecs
        except (requests.RequestException, ValueError) as e:
            if attempt == 2:
                raise
            wait = 2**attempt
            log.warning(
                "embed_batch attempt %d failed (%d items): %s — retrying in %ds",
                attempt + 1,
                len(truncated),
                e,
                wait,
            )
            time.sleep(wait)
    raise RuntimeError("embed_batch: exhausted retries — should not reach here")


def get_embed_dim() -> int:
    """Probe the configured model; never guess a vector dimension."""
    return len(embed("dimension probe"))


# ───────────────────────────────── Indexing ──────────────────────────────────


def file_hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def split_note(content: str) -> list[tuple[int, int, str]]:
    """Return ordered, overlapping passages with offsets into the full note."""
    if not content:
        return [(0, 0, "")]
    chunks = []
    start = 0
    while start < len(content):
        end = min(start + CHUNK_CHARS, len(content))
        if end < len(content):
            boundary = max(content.rfind("\n", end - 160, end), content.rfind(" ", end - 160, end))
            if boundary > start + CHUNK_CHARS // 2:
                end = boundary + 1
        chunks.append((start, end, content[start:end]))
        if end == len(content):
            break
        start = max(start + 1, end - CHUNK_OVERLAP)
    return chunks


def embed_note_chunks(content: str) -> tuple[list[tuple[int, int, str]], list[list[float]]]:
    """Embed every passage; retry individual passages if a batch fails."""
    chunks = split_note(content)
    vectors = []
    for offset in range(0, len(chunks), EMBED_BATCH_SIZE):
        texts = [text or " " for _, _, text in chunks[offset : offset + EMBED_BATCH_SIZE]]
        try:
            vectors.extend(embed_batch(texts))
        except requests.RequestException, ValueError:
            vectors.extend(embed(text) for text in texts)
    return chunks, vectors


def _rerank_score(query: str, doc: str) -> float:
    """Call RERANK_MODEL to score (query, doc) relevance. Returns 0.0 on any failure."""
    prompt = (
        "Score how relevant the document is to the query. "
        "Output only a single decimal number between 0.0 and 1.0. Nothing else.\n\n"
        f"Query: {query}\n\nDocument: {doc[:400]}\n\nScore:"
    )
    try:
        resp = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={"model": RERANK_MODEL, "prompt": prompt, "stream": False},
            timeout=EMBED_TIMEOUT,
        )
        resp.raise_for_status()
        text = resp.json().get("response", "0").strip()
        for token in text.split():
            try:
                return max(0.0, min(1.0, float(token)))
            except ValueError:
                continue
        return 0.0
    except Exception as e:
        log.warning("rerank_score failed: %s", e)
        return 0.0


def _rerank(query: str, rows: list[tuple], limit: int) -> list[tuple]:
    """Re-rank candidate rows with RERANK_MODEL cross-encoder, return top `limit`.

    Runs re-scoring in parallel (up to EMBED_WORKERS threads). Falls back to
    the original order if RERANK_MODEL is not configured.
    """
    if not RERANK_MODEL or not rows:
        return rows[:limit]

    scores: dict[str, float] = {}
    with ThreadPoolExecutor(max_workers=min(EMBED_WORKERS, len(rows))) as pool:
        futures = {pool.submit(_rerank_score, query, content): path for path, content, _ in rows}
        for fut in as_completed(futures):
            path = futures[fut]
            try:
                scores[path] = fut.result()
            except Exception:
                scores[path] = 0.0

    reranked = sorted(rows, key=lambda r: scores.get(r[0], 0.0), reverse=True)
    log.info("reranked %d candidates → top %d", len(rows), limit)
    return reranked[:limit]


def search_chunk_rows(
    query: str,
    vector: list[float] | None,
    mode: str,
    limit: int,
    min_similarity: float = 0.0,
    vault_ids: list[str] | None = None,
) -> list[tuple[str, str, float]]:
    """Find the strongest passage per note and include its immediate neighbors."""
    if DATABASE_BACKEND == "qdrant":
        store = _qstore()
        results: list[tuple[str, str, float]] = []
        seen: set[str] = set()
        for hit in store.search(vector, max(limit * 8, 40), vault_ids, query=query, mode=mode):
            payload = hit["payload"]
            path = payload["path"]
            if path in seen:
                continue
            note = store.note(path)
            if not note or note.get("hash") != payload.get("hash"):
                continue
            score = float(hit["score"])
            if mode == "keyword":
                score = score / (1 + score)
            if score < min_similarity:
                continue
            if mode == "keyword":
                results.append((path, note["content"], score))
                seen.add(path)
                if len(results) >= limit:
                    break
                continue
            chunks = split_note(note["content"])
            index = int(payload["index"])
            start = chunks[max(0, index - 1)][0]
            end = chunks[min(len(chunks) - 1, index + 1)][1]
            results.append((path, note["content"][start:end], score))
            seen.add(path)
            if len(results) >= limit:
                break
        return results
    if vector is None:
        raise ValueError("PostgreSQL passage search requires an embedding")
    return _pgstore().search_chunks(query, vector, mode, limit, min_similarity, vault_ids)


def _bulk_load_hashes(paths: list[str]) -> dict[str, str]:
    """Fetch existing path→hash pairs in one DB query."""
    if not paths:
        return {}
    if DATABASE_BACKEND == "qdrant":
        wanted = set(paths)
        return {
            row["path"]: row["hash"]
            for row in _qstore().notes(fields=["path", "hash"])
            if row["path"] in wanted
        }
    return _pgstore().hashes(paths)


def _upsert_note(
    path: str,
    content: str,
    h: str,
    vec: list[float],
    vault_id: str = "",
    chunks: list[tuple[int, int, str]] | None = None,
    chunk_vectors: list[list[float]] | None = None,
) -> None:
    """Upsert a single note row given a precomputed embedding vector. Retries
    on serialization deadlocks (40P01). Also extracts and stores wikilinks."""
    links = extract_wikilinks(content)
    with _link_index_lock:
        idx_snapshot = {key: paths.copy() for key, paths in _link_index.items()}
    resolved = _resolve_links(links, idx_snapshot) if links else {}
    if chunks is None or chunk_vectors is None:
        chunks = [(0, min(len(content), CHUNK_CHARS), content[:CHUNK_CHARS])]
        chunk_vectors = [vec]
    if len(chunks) != len(chunk_vectors) or not chunk_vectors:
        raise ValueError("missing note chunk embeddings")

    if DATABASE_BACKEND == "qdrant":
        _qstore().write_note(path, content, h, vault_id, chunks, chunk_vectors, resolved)
        return

    _pgstore().upsert_note(path, content, h, vault_id, chunks, chunk_vectors, resolved)


def _embed_and_upsert(path: str, content: str, h: str, vault_id: str = "") -> None:
    """Single-item embed + upsert. Kept for backwards compatibility and as
    the per-item fallback when batch embedding fails for any reason."""
    chunks, vectors = embed_note_chunks(content)
    _upsert_note(path, content, h, vectors[0], vault_id, chunks, vectors)
    log.info("Indexed: %s", path)


def _embed_and_upsert_batch(items: list[tuple[str, str, str]], vault_id: str = "") -> list[str]:
    """Batch passages across notes and upsert each note atomically."""
    if not items:
        return []
    passages = [split_note(content) for _, content, _ in items]
    texts = [passage or " " for note_passages in passages for _, _, passage in note_passages]
    vectors: list[list[float]] = []
    try:
        for offset in range(0, len(texts), EMBED_BATCH_SIZE):
            vectors.extend(embed_batch(texts[offset : offset + EMBED_BATCH_SIZE]))
        if len(vectors) != len(texts) or any(not vector for vector in vectors):
            raise ValueError("Ollama returned incomplete passage embeddings")
    except (requests.RequestException, ValueError) as exc:
        log.warning("Passage batch embedding failed (%s); retrying notes separately", exc)
        failed = []
        for path, content, content_hash in items:
            try:
                _embed_and_upsert(path, content, content_hash, vault_id)
            except Exception as note_exc:
                log.warning("Failed to index %s: %s", path, note_exc)
                failed.append(path)
        return failed

    failed = []
    position = 0
    for (path, content, content_hash), note_passages in zip(items, passages, strict=True):
        note_vectors = vectors[position : position + len(note_passages)]
        position += len(note_passages)
        try:
            _upsert_note(
                path, content, content_hash, note_vectors[0], vault_id, note_passages, note_vectors
            )
            log.info("Indexed: %s", path)
        except Exception as exc:
            log.warning("Failed to index %s: %s", path, exc)
            failed.append(path)
    return failed


def _ignored_path_segments() -> set[str]:
    """Return ignored vault path segments.

    OBSIDIAN_IGNORE_PATHS replaces the default archive exclusion when set.
    Set it to an empty string to allow archive/ content to be indexed.
    """
    raw = os.environ.get("OBSIDIAN_IGNORE_PATHS")
    if raw is None:
        if os.environ.get("OBSIDIAN_INDEX_ARCHIVE") == "yes":
            return set()
        return set(_DEFAULT_IGNORED_PATH_SEGMENTS)
    return {segment.strip() for segment in raw.split(",") if segment.strip()}


def _should_skip_path(path: Path) -> bool:
    """Skip hidden/system directories and archive/ relative to any vault root.

    Falls back to VAULT_PATH when VAULT_PATHS is empty (test environments and
    single-vault setups that patch VAULT_PATH directly).
    """
    # _VAULT_LIST is computed at import time; fall back to current VAULT_PATH so
    # tests that monkey-patch VAULT_PATH after import still work correctly.
    vaults = _VAULT_LIST or ([VAULT_PATH] if VAULT_PATH else [])
    ignored_segments = _ignored_path_segments()
    for vp in vaults:
        try:
            rel = path.relative_to(Path(vp))
            return any(
                part.startswith(".")
                or part in _ALWAYS_SKIPPED_PATH_SEGMENTS
                or part in ignored_segments
                for part in rel.parts
            )
        except ValueError:
            continue
    return True  # not under any known vault — skip


# Backward-compatible alias used by existing tests
_is_system_path = _should_skip_path


def index_note(path: str, content: str, vault_id: str = "") -> None:
    """Embed a single note and upsert into the database. Skips unchanged files.

    The hash check uses a short-lived DB connection that is released before
    embedding — embedding can block for EMBED_TIMEOUT seconds and must never
    hold a pool slot.
    """
    h = file_hash(content)
    if DATABASE_BACKEND == "qdrant":
        note = _qstore().note(path)
        if note and note["hash"] == h:
            return
        _embed_and_upsert(path, content, h, vault_id)
        return
    if _pgstore().note_is_current(path, h):
        return
    # DB connection released above before any network call.
    # _embed_and_upsert handles embed + write with its own connection and retry logic.
    _embed_and_upsert(path, content, h, vault_id)


def delete_note(path: str) -> None:
    if DATABASE_BACKEND == "qdrant":
        _qstore().delete_path(path)
        with _link_index_lock:
            _remove_link(_link_index, path)
        log.info("Removed: %s", path)
        return
    _pgstore().delete_note(path)
    with _link_index_lock:
        _remove_link(_link_index, path)
    log.info("Removed: %s", path)


def _safe_delete_note(path: str) -> None:
    """delete_note(), guarded so a DB outage can never kill the watchdog
    observer thread. watchdog invokes handler callbacks directly on its
    observer thread with no supervising try/except of its own — an
    unguarded exception here (e.g. psycopg2.OperationalError while Postgres
    is unreachable) propagates out of the callback and silently ends all
    further event dispatch on that observer, which looks like indexing has
    just stopped rather than like a crash."""
    try:
        delete_note(path)
    except Exception as e:
        log.warning("Watcher: delete_note failed for %s: %s", path, e)


def _run_embed_pass(batch: list[tuple[str, str, str]], vault: str) -> list[str]:
    """Embed + upsert by chunking into EMBED_BATCH_SIZE-sized batches and
    submitting each batch to a worker pool. Each batch is one /api/embed
    call — saves one HTTP round-trip per note. Returns paths that failed
    even after the single-embed fallback inside _embed_and_upsert_batch."""
    if not batch:
        return []
    chunks = [batch[i : i + EMBED_BATCH_SIZE] for i in range(0, len(batch), EMBED_BATCH_SIZE)]
    failed: list[str] = []
    with ThreadPoolExecutor(max_workers=min(EMBED_WORKERS, len(chunks))) as pool:
        futures = [pool.submit(_embed_and_upsert_batch, chunk, vault) for chunk in chunks]
        for fut in as_completed(futures):
            try:
                failed.extend(fut.result())
            except Exception as e:
                log.warning("Batch worker crashed: %s", e)
    return failed


def prune_orphans() -> int:
    """Delete DB rows whose `path` no longer exists on disk. Returns the
    number of rows deleted. Resyncs `indexed_count` with `vault_file_count`
    after files are deleted, the vault path changes, or OBSIDIAN_IGNORE_PATHS
    is updated — the slow drift the watcher can't catch."""
    if not VAULT_PATHS or any(not Path(vault).is_dir() for vault in VAULT_PATHS):
        raise RuntimeError("Cannot prune: a configured vault is unavailable")
    if DATABASE_BACKEND == "qdrant":
        roots = [Path(vault) for vault in VAULT_PATHS]
        notes = _qstore().notes(fields=["path"])
        for root in roots:
            indexed_here = any(Path(row["path"]).is_relative_to(root) for row in notes)
            visible_here = any(not _should_skip_path(path) for path in root.rglob("*.md"))
            if indexed_here and not visible_here:
                raise RuntimeError(f"Cannot prune: vault appears empty or unavailable: {root}")
        missing = [
            row["path"]
            for row in notes
            if any(Path(row["path"]).is_relative_to(root) for root in roots)
            and not Path(row["path"]).exists()
        ]
        for path in missing:
            _qstore().delete_path(path)
        return len(missing)
    paths = _pgstore().note_paths()
    roots = [Path(vault) for vault in VAULT_PATHS]
    for root in roots:
        indexed_here = any(Path(path).is_relative_to(root) for path in paths)
        visible_here = any(not _should_skip_path(path) for path in root.rglob("*.md"))
        if indexed_here and not visible_here:
            raise RuntimeError(f"Cannot prune: vault appears empty or unavailable: {root}")
    missing = [
        path
        for path in paths
        if any(Path(path).is_relative_to(root) for root in roots) and not Path(path).exists()
    ]
    _pgstore().delete_notes(missing)
    log.info("Pruned %d orphan rows from DB", len(missing))
    return len(missing)


def rebuild_index_safely(vaults: list[str]) -> int:
    """Build a complete index beside the active one, then swap tables atomically.

    The caller holds reindex_lock(). Old tables are archived after a verified
    rebuild, so a failed build never removes the searchable index.
    """
    if DATABASE_BACKEND == "qdrant":
        return _rebuild_qdrant(vaults)
    if not vaults or any(not Path(vault).is_dir() for vault in vaults):
        raise RuntimeError("Cannot rebuild: a configured vault is unavailable")
    dimension = get_embed_dim()
    suffix = secrets.token_hex(6)
    stage_notes = f"osm_notes_stage_{suffix}"
    stage_links = f"osm_links_stage_{suffix}"
    stage_chunks = f"osm_chunks_stage_{suffix}"
    stage_embedding_idx = f"osm_embedding_stage_{suffix}"
    stage_tsv_idx = f"osm_tsv_stage_{suffix}"
    stage_vault_idx = f"osm_vault_stage_{suffix}"
    stage_links_idx = f"osm_links_target_stage_{suffix}"
    stage_chunks_idx = f"osm_chunks_model_stage_{suffix}"
    with db_conn() as conn:
        with conn:
            with conn.cursor() as cur:
                require_active_model(cur, EMBED_MODEL)
                cur.execute("SET LOCAL lock_timeout = '5s'")
                cur.execute(
                    sql.SQL("""
                    CREATE TABLE {} (
                        id BIGSERIAL PRIMARY KEY,
                        path TEXT UNIQUE NOT NULL,
                        content TEXT NOT NULL,
                        hash TEXT NOT NULL,
                        embedding vector({}),
                        content_tsv tsvector,
                        vault_id TEXT,
                        indexed_at TIMESTAMP DEFAULT NOW()
                    )
                """).format(sql.Identifier(stage_notes), sql.SQL(str(dimension)))
                )
                cur.execute(
                    sql.SQL("""
                    CREATE TABLE {} (
                        source_path TEXT NOT NULL,
                        target_name TEXT NOT NULL,
                        target_path TEXT,
                        PRIMARY KEY (source_path, target_name)
                    )
                """).format(sql.Identifier(stage_links))
                )
                cur.execute(
                    sql.SQL("""
                    CREATE TABLE {} (
                        path TEXT NOT NULL REFERENCES {}(path) ON DELETE CASCADE,
                        model TEXT NOT NULL,
                        note_hash TEXT NOT NULL,
                        chunk_index INTEGER NOT NULL,
                        start_offset INTEGER NOT NULL,
                        end_offset INTEGER NOT NULL,
                        content TEXT NOT NULL,
                        embedding vector NOT NULL,
                        content_tsv tsvector,
                        PRIMARY KEY (path, model, chunk_index)
                    )
                """).format(sql.Identifier(stage_chunks), sql.Identifier(stage_notes))
                )

    link_index: dict[str, set[str]] = {}
    for vault in vaults:
        _merge_links(link_index, _build_link_index(vault))

    expected: dict[str, str] = {}
    batch: list[tuple[str, str, str, str]] = []

    def write_batch() -> None:
        if not batch:
            return
        passages = [split_note(content) for _, content, _, _ in batch]
        texts = [passage or " " for note_passages in passages for _, _, passage in note_passages]
        vectors = []
        for offset in range(0, len(texts), EMBED_BATCH_SIZE):
            vectors.extend(embed_batch(texts[offset : offset + EMBED_BATCH_SIZE]))
        if any(len(vector) != dimension for vector in vectors):
            raise ValueError("Ollama returned an unexpected embedding dimension")
        with db_conn() as conn:
            with conn:
                with conn.cursor() as cur:
                    vector_offset = 0
                    for (path, content, content_hash, vault), note_passages in zip(
                        batch, passages, strict=True
                    ):
                        note_vectors = vectors[vector_offset : vector_offset + len(note_passages)]
                        vector_offset += len(note_passages)
                        cur.execute(
                            sql.SQL("""
                            INSERT INTO {}
                                (path, content, hash, embedding, content_tsv, vault_id)
                            VALUES (%s, %s, %s, %s::vector,
                                    setweight(to_tsvector(%s::regconfig, %s), 'A') ||
                                    setweight(to_tsvector('simple', %s), 'B'), %s)
                        """).format(sql.Identifier(stage_notes)),
                            (
                                path,
                                content,
                                content_hash,
                                _vec_to_str(note_vectors[0]),
                                search_language(),
                                content,
                                content,
                                vault,
                            ),
                        )
                        chunk_insert = sql.SQL("""
                            INSERT INTO {} (path, model, note_hash, chunk_index,
                                            start_offset, end_offset, content,
                                            embedding, content_tsv)
                            VALUES (%s, %s, %s, %s, %s, %s, %s, %s::vector,
                                    setweight(to_tsvector(%s::regconfig, %s), 'A') ||
                                    setweight(to_tsvector('simple', %s), 'B'))
                        """).format(sql.Identifier(stage_chunks))
                        for index, ((start, end, passage), chunk_vector) in enumerate(
                            zip(note_passages, note_vectors, strict=True)
                        ):
                            cur.execute(
                                chunk_insert,
                                (
                                    path,
                                    EMBED_MODEL,
                                    content_hash,
                                    index,
                                    start,
                                    end,
                                    passage,
                                    _vec_to_str(chunk_vector),
                                    search_language(),
                                    passage,
                                    passage,
                                ),
                            )
                        links = _resolve_links(extract_wikilinks(content), link_index)
                        if links:
                            link_insert = sql.SQL("""
                                INSERT INTO {} (source_path, target_name, target_path)
                                VALUES (%s, %s, %s)
                            """).format(sql.Identifier(stage_links))
                            for name, target in links.items():
                                cur.execute(link_insert, (path, name, target))
        batch.clear()

    for vault in vaults:
        for path in Path(vault).rglob("*.md"):
            if _should_skip_path(path):
                continue
            content = path.read_text(encoding="utf-8", errors="ignore")
            content_hash = file_hash(content)
            expected[str(path)] = content_hash
            batch.append((str(path), content, content_hash, vault))
            if len(batch) >= EMBED_BATCH_SIZE:
                write_batch()
    write_batch()

    current: dict[str, str] = {}
    for vault in vaults:
        if not Path(vault).is_dir():
            raise RuntimeError(f"Vault became unavailable during rebuild: {vault}")
        for path in Path(vault).rglob("*.md"):
            if not _should_skip_path(path):
                current[str(path)] = file_hash(path.read_text(encoding="utf-8", errors="ignore"))
    if current != expected:
        raise RuntimeError("Vault files changed during rebuild; active index was left untouched")

    lists = max(10, min(len(expected) // 50, 500)) if expected else 100
    with db_conn() as conn:
        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    sql.SQL(
                        "CREATE INDEX {} ON {} USING ivfflat (embedding vector_cosine_ops) WITH (lists = {})"
                    ).format(
                        sql.Identifier(stage_embedding_idx),
                        sql.Identifier(stage_notes),
                        sql.SQL(str(lists)),
                    )
                )
                cur.execute(
                    sql.SQL("CREATE INDEX {} ON {} USING GIN (content_tsv)").format(
                        sql.Identifier(stage_tsv_idx), sql.Identifier(stage_notes)
                    )
                )
                cur.execute(
                    sql.SQL("CREATE INDEX {} ON {} (vault_id)").format(
                        sql.Identifier(stage_vault_idx), sql.Identifier(stage_notes)
                    )
                )
                cur.execute(
                    sql.SQL("CREATE INDEX {} ON {} (target_path)").format(
                        sql.Identifier(stage_links_idx), sql.Identifier(stage_links)
                    )
                )
                cur.execute(
                    sql.SQL("CREATE INDEX {} ON {} (model, path)").format(
                        sql.Identifier(stage_chunks_idx), sql.Identifier(stage_chunks)
                    )
                )

    with db_conn() as conn:
        with conn:
            with conn.cursor() as cur:
                cur.execute("SET LOCAL lock_timeout = '5s'")
                require_active_model(cur, EMBED_MODEL)
                cur.execute("LOCK TABLE notes, note_links, note_chunks IN ACCESS EXCLUSIVE MODE")
                cur.execute(sql.SQL("SELECT COUNT(*) FROM {}").format(sql.Identifier(stage_notes)))
                staged_count = cur.fetchone()[0]
                if staged_count != len(expected):
                    raise RuntimeError("Staged index is incomplete")
                cur.execute("SELECT COUNT(*) FROM notes")
                if cur.fetchone()[0] and not staged_count:
                    raise RuntimeError(
                        "Refusing to replace a nonempty index with an empty vault scan"
                    )
                cur.execute(
                    "CREATE TABLE IF NOT EXISTS index_rebuild_archives (notes_table TEXT PRIMARY KEY, links_table TEXT NOT NULL, chunks_table TEXT, archived_at TIMESTAMPTZ DEFAULT NOW())"
                )
                cur.execute(
                    "ALTER TABLE index_rebuild_archives ADD COLUMN IF NOT EXISTS chunks_table TEXT"
                )
                for old, new in (
                    ("notes_embedding_idx", f"notes_embedding_archive_{suffix}"),
                    ("notes_tsv_idx", f"notes_tsv_archive_{suffix}"),
                    ("notes_vault_idx", f"notes_vault_archive_{suffix}"),
                    ("note_links_target_idx", f"note_links_target_archive_{suffix}"),
                    ("note_chunks_model_idx", f"note_chunks_model_archive_{suffix}"),
                ):
                    cur.execute(
                        sql.SQL("ALTER INDEX {} RENAME TO {}").format(
                            sql.Identifier(old), sql.Identifier(new)
                        )
                    )
                archive_notes = f"notes_archive_{suffix}"
                archive_links = f"note_links_archive_{suffix}"
                archive_chunks = f"note_chunks_archive_{suffix}"
                cur.execute(
                    sql.SQL("ALTER TABLE notes RENAME TO {}").format(sql.Identifier(archive_notes))
                )
                cur.execute(
                    sql.SQL("ALTER TABLE note_links RENAME TO {}").format(
                        sql.Identifier(archive_links)
                    )
                )
                cur.execute(
                    sql.SQL("ALTER TABLE note_chunks RENAME TO {}").format(
                        sql.Identifier(archive_chunks)
                    )
                )
                cur.execute(
                    sql.SQL("ALTER TABLE {} RENAME TO notes").format(sql.Identifier(stage_notes))
                )
                cur.execute(
                    sql.SQL("ALTER TABLE {} RENAME TO note_links").format(
                        sql.Identifier(stage_links)
                    )
                )
                cur.execute(
                    sql.SQL("ALTER TABLE {} RENAME TO note_chunks").format(
                        sql.Identifier(stage_chunks)
                    )
                )
                for old, new in (
                    (stage_embedding_idx, "notes_embedding_idx"),
                    (stage_tsv_idx, "notes_tsv_idx"),
                    (stage_vault_idx, "notes_vault_idx"),
                    (stage_links_idx, "note_links_target_idx"),
                    (stage_chunks_idx, "note_chunks_model_idx"),
                ):
                    cur.execute(
                        sql.SQL("ALTER INDEX {} RENAME TO {}").format(
                            sql.Identifier(old), sql.Identifier(new)
                        )
                    )
                cur.execute(
                    "INSERT INTO index_rebuild_archives (notes_table, links_table, chunks_table) VALUES (%s, %s, %s)",
                    (archive_notes, archive_links, archive_chunks),
                )
    for vault in vaults:
        index_vault(vault)
    return len(expected)


def _rebuild_qdrant(vaults: list[str]) -> int:
    """Build a separate collection and switch its alias after a stable vault scan."""
    if not vaults or any(not Path(vault).is_dir() for vault in vaults):
        raise RuntimeError("Cannot rebuild: a configured vault is unavailable")
    store = _qstore()
    dimension = get_embed_dim()
    previous = store.alias_target()
    stage = (
        f"osm_notes_{hashlib.sha256(EMBED_MODEL.encode()).hexdigest()[:10]}_{secrets.token_hex(4)}"
    )
    store.create_collection(stage, dimension)
    links: dict[str, set[str]] = {}
    for vault in vaults:
        _merge_links(links, _build_link_index(vault))
    expected: dict[str, str] = {}
    for vault in vaults:
        for path in Path(vault).rglob("*.md"):
            if _should_skip_path(path):
                continue
            content = path.read_text(encoding="utf-8", errors="ignore")
            content_hash = file_hash(content)
            expected[str(path)] = content_hash
            passages = split_note(content)
            vectors: list[list[float]] = []
            for offset in range(0, len(passages), EMBED_BATCH_SIZE):
                vectors.extend(
                    embed_batch(
                        [
                            passage or " "
                            for _, _, passage in passages[offset : offset + EMBED_BATCH_SIZE]
                        ]
                    )
                )
            if len(vectors) != len(passages) or any(len(vec) != dimension for vec in vectors):
                raise RuntimeError(f"Incomplete Qdrant embedding for {path}")
            resolved = _resolve_links(extract_wikilinks(content), links)
            store.write_note(
                str(path), content, content_hash, vault, passages, vectors, resolved, stage
            )
    current: dict[str, str] = {}
    for vault in vaults:
        if not Path(vault).is_dir():
            raise RuntimeError(f"Vault became unavailable during rebuild: {vault}")
        for path in Path(vault).rglob("*.md"):
            if not _should_skip_path(path):
                current[str(path)] = file_hash(path.read_text(encoding="utf-8", errors="ignore"))
    if current != expected:
        raise RuntimeError("Vault changed during Qdrant rebuild; active index preserved")
    staged = {row["path"]: row["hash"] for row in store.notes(stage)}
    if staged != expected:
        raise RuntimeError("Qdrant staged index is incomplete; active index preserved")
    if previous and store.notes(previous) and not staged:
        raise RuntimeError("Refusing to replace a nonempty Qdrant index with an empty scan")
    store.swap(stage)
    for vault in vaults:
        index_vault(vault)
    return len(expected)


def index_vault(vault: str) -> None:
    """Walk the vault and index every markdown file (parallel, hash-skipping)."""
    set_index_state(vault, "indexing")
    try:
        root = Path(vault)
        # rglob on a nonexistent directory yields nothing rather than raising,
        # so without this check a vault that vanished (unmounted NAS, deleted
        # folder, typo'd OBSIDIAN_VAULT) indexes zero notes and reports
        # SUCCESS — the exact silent-failure mode index_state exists to make
        # visible. Fail loudly instead so the except branch records it.
        if not root.is_dir():
            raise FileNotFoundError(f"Vault path does not exist or is not a directory: {vault}")
        md_files = [f for f in root.rglob("*.md") if not _should_skip_path(f)]
        log.info("Indexing %d notes in %s…", len(md_files), vault)

        # Build link index before embedding so _upsert_note can resolve wikilinks.
        new_index = _build_link_index(vault)
        with _link_index_lock:
            for key, paths in list(_link_index.items()):
                paths.difference_update({p for p in paths if Path(p).is_relative_to(root)})
                if not paths:
                    del _link_index[key]
            _merge_links(_link_index, new_index)
        log.info("Link index built: %d entries for %s", len(new_index), vault)

        # Read all contents and compute hashes in the main thread (fast, no DB)
        file_data: list[tuple[str, str, str]] = []  # (path_str, content, hash)
        for f in md_files:
            try:
                content = f.read_text(encoding="utf-8", errors="ignore")
                file_data.append((str(f), content, file_hash(content)))
            except Exception as e:
                log.warning("Skipped reading %s: %s", f, e)

        # Single DB query to fetch all existing hashes
        paths = [item[0] for item in file_data]
        existing = _bulk_load_hashes(paths)

        # Filter to only files that are new or changed
        changed = [(p, c, h) for p, c, h in file_data if existing.get(p) != h]
        skipped = len(file_data) - len(changed)
        log.info("Changed: %d, Skipped (unchanged): %d", len(changed), skipped)

        # Parallel embed + upsert. Track failed paths so we can retry once —
        # Ollama tends to wedge under heavy concurrent load, and a single retry
        # after the first pass usually catches transient timeouts. Without this,
        # a wedged Ollama silently drops notes from a full rebuild.
        by_path = {p: (p, c, h) for p, c, h in changed}
        failed: list[str] = _run_embed_pass(list(by_path.values()), vault)

        if failed:
            log.warning("First pass had %d failures — retrying once", len(failed))
            retry_batch = [by_path[p] for p in failed if p in by_path]
            failed = _run_embed_pass(retry_batch, vault)

        if failed:
            log.warning("Indexing finished with %d persistent failures", len(failed))
            set_index_state(
                vault,
                "failed",
                failed_paths=failed,
                error=f"{len(failed)} note(s) failed to index after retry",
            )
        else:
            set_index_state(vault, "idle", failed_paths=[])

        # Rebuild IVFFlat index now that data exists — an index built on an empty
        # table has no list centroids and returns zero results.
        if DATABASE_BACKEND == "postgresql":
            try:
                _pgstore().rebuild_vector_index()
                log.info("Rebuilt IVFFlat index")
            except Exception as e:
                log.warning("Index rebuild skipped: %s", e)

        log.info("Vault indexing complete")
    except Exception as e:
        set_index_state(vault, "failed", error=str(e))
        raise


# ─────────────────────────────── File Watcher ────────────────────────────────


class VaultEventHandler(FileSystemEventHandler):
    def __init__(self, vault_id: str = ""):
        super().__init__()
        self._vault_id = vault_id
        self._timers: dict[str, threading.Timer] = {}
        self._lock = threading.Lock()

    def _schedule(self, path: str):
        """Debounce rapid events for the same path (e.g. Obsidian autosave)."""
        if not path.endswith(".md") or _should_skip_path(Path(path)):
            return
        with self._lock:
            existing = self._timers.pop(path, None)
            if existing:
                existing.cancel()
            t = threading.Timer(_DEBOUNCE_SECS, self._handle_upsert, args=(path,))
            self._timers[path] = t
            t.start()

    def _handle_upsert(self, path: str):
        with self._lock:
            self._timers.pop(path, None)
        try:
            content = Path(path).read_text(encoding="utf-8", errors="ignore")
            # Keep link index current so wikilink resolution stays accurate
            with _link_index_lock:
                _remove_link(_link_index, path)
                _add_link(_link_index, Path(path), self._vault_id)
            index_note(path, content, self._vault_id)
        except FileNotFoundError:
            # File vanished between the debounce firing and this read (e.g.
            # a rapid create+delete, or an editor's atomic-save temp-file
            # dance). Recover via the guarded delete so a DB outage during
            # *this* recovery path cannot escape and kill the observer
            # thread — the same defect class this iteration fixes below in
            # on_deleted/on_moved, just reached via a different trigger.
            _safe_delete_note(path)
        except psycopg2.DataError as e:
            if _DIMENSION_MISMATCH_RE.search(str(e)):
                log.error(
                    "Watcher: column-dimension mismatch writing %s — recording in index_state: %s",
                    path,
                    e,
                )
                _record_watcher_dimension_failure(self._vault_id, path, str(e))
            else:
                log.warning("Watcher: skipped %s: %s", path, e)
        except Exception as e:
            log.warning("Watcher: skipped %s: %s", path, e)

    def on_created(self, event):
        if not event.is_directory:
            self._schedule(event.src_path)

    def on_modified(self, event):
        if not event.is_directory:
            self._schedule(event.src_path)

    def on_deleted(self, event):
        if (
            not event.is_directory
            and event.src_path.endswith(".md")
            and not _should_skip_path(Path(event.src_path))
        ):
            _safe_delete_note(event.src_path)

    def on_moved(self, event):
        if not event.is_directory:
            if event.src_path.endswith(".md") and not _should_skip_path(Path(event.src_path)):
                _safe_delete_note(event.src_path)
            self._schedule(event.dest_path)


def _needs_polling(vault: str) -> bool:
    """Heuristic: return True if the vault path looks like a network filesystem.

    Native filesystem events can miss writes made by remote clients. Inspect
    the mounted filesystem on macOS and retain the Windows/Linux checks.
    """
    vp = Path(vault)
    # UNC paths are always network (\\server\share on Windows, //server/share on POSIX)
    vault_s = str(vp)
    if vault_s[:2] in ("\\\\", "//"):
        return True
    if sys.platform == "darwin":
        try:
            import subprocess

            result = subprocess.run(["mount"], capture_output=True, text=True, timeout=5)
            if result.returncode != 0:
                return False
            resolved = vp.resolve()
            best_mount: Path | None = None
            best_options = ""
            for line in result.stdout.splitlines():
                if " on " not in line or " (" not in line:
                    continue
                location, options = line.rsplit(" (", 1)
                mount_path = Path(location.rsplit(" on ", 1)[1])
                if resolved.is_relative_to(mount_path) and (
                    best_mount is None or len(mount_path.parts) > len(best_mount.parts)
                ):
                    best_mount = mount_path
                    best_options = options.removesuffix(")")
            if best_mount is None:
                return False
            flags = {part.strip().lower() for part in best_options.split(",")}
            return "local" not in flags
        except OSError, ValueError, subprocess.TimeoutExpired:
            return False
    if sys.platform == "win32":
        try:
            import ctypes
            import subprocess

            drive_letter = str(vp.resolve()).split(":")[0]
            drive_root = drive_letter + ":\\"
            # Check 1: GetDriveTypeW (catches SMB/CIFS mapped drives)
            DRIVE_REMOTE = 4
            if ctypes.windll.kernel32.GetDriveTypeW(drive_root) == DRIVE_REMOTE:
                return True
            # Check 2: `net use` output (catches NFS and other mounts that
            # Windows NFS Client registers outside the standard redirector)
            result = subprocess.run(
                ["net", "use", drive_letter + ":"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0 and (
                "remote" in result.stdout.lower()
                or "\\\\" in result.stdout
                or "nfs" in result.stdout.lower()
            ):
                return True
            # Check 3: WMI via wmic (last resort — catches all network drives)
            result = subprocess.run(
                [
                    "wmic",
                    "logicaldisk",
                    "where",
                    f"DeviceID='{drive_letter}:'",
                    "get",
                    "DriveType",
                    "/value",
                ],
                capture_output=True,
                text=True,
                timeout=5,
            )
            # DriveType=4 is network drive in WMI
            if "DriveType=4" in result.stdout:
                return True
            return False
        except Exception:
            return False
    else:
        # Linux: check /proc/mounts for nfs/cifs
        try:
            mount_point = str(vp.resolve())
            with open("/proc/mounts") as f:
                for line in f:
                    parts = line.split()
                    if len(parts) >= 3 and mount_point.startswith(parts[1]):
                        if parts[2] in ("nfs", "nfs4", "cifs", "smb", "smbfs", "9p", "fuse.sshfs"):
                            return True
        except OSError, IndexError:
            pass
        return False


def start_watcher(vault: str) -> BaseObserver:
    """Start a filesystem watcher for the given vault.

    Uses PollingObserver when VAULT_WATCH_POLLING is "true" or when auto-
    detection identifies the vault as a network mount.  Falls back to the
    native Observer otherwise.
    """
    use_polling = VAULT_WATCH_POLLING == "true" or (
        VAULT_WATCH_POLLING == "auto" and _needs_polling(vault)
    )
    native_started = not use_polling
    if use_polling:
        obs = PollingObserver(timeout=VAULT_POLL_INTERVAL)
        log.info("Watching vault (polling, %ds interval): %s", VAULT_POLL_INTERVAL, vault)
    else:
        obs = Observer()
        log.info("Watching vault (native events): %s", vault)
    obs.schedule(VaultEventHandler(vault), vault, recursive=True)
    try:
        obs.start()
    except Exception:
        if use_polling or VAULT_WATCH_POLLING == "false":
            raise
        log.warning("Native watcher could not start; falling back to polling: %s", vault)
        if obs.is_alive():
            obs.stop()
            obs.join(timeout=2)
        obs = PollingObserver(timeout=VAULT_POLL_INTERVAL)
        obs.schedule(VaultEventHandler(vault), vault, recursive=True)
        obs.start()
        native_started = False
    if native_started and VAULT_WATCH_POLLING == "auto":
        time.sleep(0.2)
        if not obs.is_alive() or any(not emitter.is_alive() for emitter in obs.emitters):
            log.warning("Native watcher stopped unexpectedly; falling back to polling: %s", vault)
            obs.stop()
            obs.join(timeout=2)
            obs = PollingObserver(timeout=VAULT_POLL_INTERVAL)
            obs.schedule(VaultEventHandler(vault), vault, recursive=True)
            obs.start()
    _observers.append(obs)
    return obs


# ──────────────────────────── Background Init ────────────────────────────────


def _notify_ollama_unavailable(ready: threading.Event) -> None:
    """Show one macOS notification if the startup probe is still waiting."""
    if sys.platform != "darwin" or ready.is_set() or _STOPPING.is_set():
        return
    try:
        result = subprocess.run(
            [
                "osascript",
                "-e",
                'display notification "Ollama has not responded for over 2 minutes. '
                'OSM will resume automatically." with title "Obsidian Semantic MCP"',
            ],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if result.returncode != 0:
            log.warning("macOS Ollama notification failed (exit code %d)", result.returncode)
    except (OSError, subprocess.TimeoutExpired) as exc:
        log.warning("macOS Ollama notification failed (%s)", type(exc).__name__)


def background_init(vaults: list[str]):
    """Wait for Ollama, then initialize the DB, index, and watch vaults."""
    if _STOPPING.wait(1):
        return
    _INDEXING_IN_PROGRESS.set()
    _set_readiness("waiting_ollama")
    ollama_ready = threading.Event()
    notification = threading.Timer(
        _OLLAMA_NOTIFY_AFTER_SECONDS, _notify_ollama_unavailable, args=(ollama_ready,)
    )
    notification.daemon = True
    notification.start()
    try:
        while not _STOPPING.is_set():
            try:
                embed_dim = get_embed_dim()
                ollama_ready.set()
                log.info("Embedding model %s is ready (dimension %d)", EMBED_MODEL, embed_dim)
                break
            except (requests.RequestException, ValueError) as exc:
                detail = type(exc).__name__
                if isinstance(exc, requests.HTTPError) and exc.response is not None:
                    detail = f"HTTP {exc.response.status_code}"
                log.warning(
                    "Ollama model %s is unavailable (%s); retrying in %ds",
                    EMBED_MODEL,
                    detail,
                    _OLLAMA_RETRY_SECONDS,
                )
                _STOPPING.wait(_OLLAMA_RETRY_SECONDS)
        else:
            return
        if _STOPPING.is_set():
            return
        _init_search_metrics()
        init_db(embed_dim)
        _set_readiness("indexing")
        failed_vaults = []
        for vault in vaults:
            if _STOPPING.is_set():
                return
            try:
                # The shared advisory lock prevents concurrent full indexes.
                with reindex_lock() as acquired:
                    if acquired:
                        index_vault(vault)
                        state = get_index_state(vault)
                        if state and state["status"] in ("failed", "dimension_mismatch"):
                            failed_vaults.append(vault)
                    else:
                        log.warning("Boot index for %s skipped; re-index already active", vault)
                start_watcher(vault)
            except Exception as exc:
                log.error("Vault startup failed for %s: %s", vault, exc)
                set_index_state(vault, "failed", error=str(exc))
                failed_vaults.append(vault)
        _set_readiness("partial_error" if failed_vaults else "ready")
    except Exception as e:
        log.error("Background init failed: %s", e)
        _set_readiness("error", str(e))
    finally:
        notification.cancel()
        _INDEXING_IN_PROGRESS.clear()


# ─────────────────────────── Shutdown Handler ────────────────────────────────

_STARTED_AT: float | None = None


def _log_external_kill_diagnostic(signame: str) -> None:
    """Log a hint when killed early — likely an MCP host Stop-hook regression.

    In Claude Code, the `Stop` event fires at the end of every assistant turn,
    not at session exit. If a Stop hook pkills MCP children, this server gets
    killed between turns. Run `verify-mcp-stop-hook` to detect the regression
    in the Claude Code settings file.
    """
    import time as _time

    if _STARTED_AT is None:
        return
    uptime = _time.monotonic() - _STARTED_AT
    log.warning(
        "obsidian-semantic-mcp received %s after %.1fs of uptime — exiting",
        signame,
        uptime,
    )
    if uptime < 60:
        log.warning(
            "Killed within 60s of startup. If this also fires every assistant turn, "
            "your MCP host is killing child processes via a Stop hook. Run "
            "`verify-mcp-stop-hook` to detect a regression in the Stop hook entry. "
            "The `Stop` event fires per-turn, not per-session."
        )


def _shutdown(stop_requested: asyncio.Event, signame: str = "signal"):
    """Stop the watcher and close the DB pool, then request task cancellation.

    Called via loop.add_signal_handler() so it runs on the event loop thread,
    making it safe to call asyncio-adjacent code without deadlocking.
    Blocking operations (observer.join) are intentionally absent — the daemon
    thread will be killed when the process exits.
    """
    _log_external_kill_diagnostic(signame)
    log.info("Shutting down…")
    _STOPPING.set()
    for obs in _observers:
        obs.stop()
    if _pool is not None:
        _pool.closeall()
    stop_requested.set()


# ─────────────────────────── Wikilink Graph Expansion ───────────────────────


def expand_via_links(paths: list[str], hops: int = 1) -> list[tuple[str, str, str]]:
    """Return notes reachable within `hops` link-steps from `paths`.

    Traverses both outgoing ([[linked to]]) and incoming (linked from) edges.
    Returns (path, content, via_path) tuples where via_path is the note that
    bridged the connection. Excludes nodes already in `paths`.
    """
    if DATABASE_BACKEND == "qdrant":
        notes = {row["path"]: row for row in _qstore().notes()}
        seen = set(paths)
        frontier = set(paths)
        expansions: list[tuple[str, str, str]] = []
        for _ in range(hops):
            next_frontier: set[str] = set()
            for path in frontier:
                outgoing = set(notes.get(path, {}).get("links", {}).values()) - {None}
                incoming = {
                    source for source, row in notes.items() if path in row.get("links", {}).values()
                }
                for neighbor in outgoing | incoming:
                    if neighbor in notes and neighbor not in seen:
                        expansions.append((neighbor, notes[neighbor]["content"], path))
                        seen.add(neighbor)
                        next_frontier.add(neighbor)
            frontier = next_frontier
        return expansions
    return _pgstore().linked_notes(paths, hops)


# ──────────────────────────── Vault Filesystem Helpers ───────────────────────


def _vault_root() -> Path:
    return Path(VAULT_PATH)


def _resolve_vault_path(relpath: str) -> Path:
    """Resolve a vault-relative path safely (no escaping the vault)."""
    root = _vault_root()
    relative = Path(relpath)
    if len(VAULT_PATHS) > 1 and relative.parts:
        matching = [Path(vault) for vault in VAULT_PATHS if Path(vault).name == relative.parts[0]]
        if len(matching) > 1:
            raise ValueError("Vault name is ambiguous; use a unique vault name")
        if matching:
            root = matching[0]
            relative = Path(*relative.parts[1:])
    resolved = (root / relative).resolve()  # resolve symlinks
    vault_resolved = root.resolve()
    if not resolved.is_relative_to(vault_resolved):
        raise ValueError(f"Path escapes vault: {relpath}")
    return resolved


def _ensure_frontmatter(content: str) -> str:
    """Guarantee every note written via write_file carries the mandatory
    frontmatter keys (see REQUIRED_FRONTMATTER_DEFAULTS + created/updated).

    `created` is set once and never overwritten. `updated` always reflects
    this write. Every other required key is added only if missing -- an
    already-present value (e.g. a caller-supplied `category`) is never
    silently replaced. Any extra keys the caller already set (project-
    specific fields) are preserved untouched.
    """
    today = datetime.now().date()

    body = content
    frontmatter: dict[str, object] = {}
    if content.startswith("---\n"):
        end = content.find("\n---", 4)
        if end != -1:
            raw_fm = content[4:end]
            body = content[end + 4 :].lstrip("\n")
            frontmatter = yaml.safe_load(raw_fm) or {}

    frontmatter.setdefault("created", today)
    frontmatter["updated"] = today
    for key, default in REQUIRED_FRONTMATTER_DEFAULTS.items():
        frontmatter.setdefault(key, default)

    dumped = yaml.safe_dump(
        frontmatter, sort_keys=False, default_flow_style=False, allow_unicode=True
    )
    return f"---\n{dumped}---\n\n{body}"


def _relative(abspath: Path) -> str:
    """Return vault-relative path string. With multiple vaults, prefixes with vault basename."""
    vaults = _VAULT_LIST or ([VAULT_PATH] if VAULT_PATH else [])
    for vp in vaults:
        try:
            rel = abspath.relative_to(Path(vp))
            if len(vaults) > 1:
                return f"{Path(vp).name}/{rel}"
            return str(rel)
        except ValueError:
            continue
    return str(abspath)


def _render_neighbor_parts(neighbors: list[tuple[str, str, str]]) -> list[str]:
    """Format expand_via_links() results; shared by both retrieval backends so
    the neighbor output shape cannot drift between them."""
    parts = ["\n**Wikilink neighbors** _(connected notes not in top results)_\n"]
    for npath, ncontent, via in neighbors:
        nrel = _relative(Path(npath))
        via_rel = _relative(Path(via))
        npreview = ncontent[:300].strip()
        while "\n\n\n" in npreview:
            npreview = npreview.replace("\n\n\n", "\n\n")
        parts.append(f"**{nrel}** _(linked via {via_rel})_\n\n{npreview}\n")
    return parts


# ───────────────────────────────── MCP Server ────────────────────────────────


def _make_tool(*, name: str, description: str, inputSchema: dict[str, Any]) -> Tool:
    """Validate MCP's camel-case schema alias without losing static checks."""
    return Tool.model_validate(
        {
            "name": name,
            "description": description,
            "inputSchema": inputSchema,
        }
    )


async def list_tools():
    tools = [
        _make_tool(
            name="search_vault",
            description=(
                "Search across your Obsidian vault(s). "
                "Three modes: 'hybrid' (default) combines semantic meaning with keyword matching for best results; "
                "'semantic' searches by meaning only; 'keyword' matches exact words using full-text search. "
                "Semantic results include the best passage and its neighboring passages. "
                "Use get_file with the returned path to read the complete note."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural language search query",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Number of results to return (default: 5, max: 20)",
                        "default": 5,
                    },
                    "min_similarity": {
                        "type": "number",
                        "description": "Minimum similarity score (0.0–1.0). Results below this threshold are excluded. Default: 0.0",
                        "default": 0.0,
                    },
                    "mode": {
                        "type": "string",
                        "enum": ["hybrid", "semantic", "keyword"],
                        "description": "Search mode: 'hybrid' (default) combines semantic + keyword; 'semantic' uses vector similarity only; 'keyword' uses full-text search only.",
                        "default": "hybrid",
                    },
                    "vault": {
                        "type": "string",
                        "description": "Filter results to a specific vault by its name (basename of vault path). Omit to search all vaults.",
                    },
                    "graph_expand": {
                        "type": "boolean",
                        "description": "Follow wikilinks from top results to surface connected notes that didn't rank semantically. Useful for discovering missed connections.",
                        "default": False,
                    },
                },
                "required": ["query"],
            },
        ),
        _make_tool(
            name="list_indexed_notes",
            description="List all notes that have been indexed, with their last indexed timestamp.",
            inputSchema={"type": "object", "properties": {}},
        ),
        _make_tool(
            name="reindex_vault",
            description=(
                "Force a full re-index of all notes in the vault. "
                "Runs in the background — use list_indexed_notes to check progress."
            ),
            inputSchema={"type": "object", "properties": {}},
        ),
        # ── Vault CRUD tools ─────────────────────────────────────────────────
        _make_tool(
            name="list_files",
            description="List all files and directories in a vault directory. Defaults to vault root.",
            inputSchema={
                "type": "object",
                "properties": {
                    "dirpath": {
                        "type": "string",
                        "description": "Directory path relative to vault root (default: root)",
                        "default": "",
                    },
                },
            },
        ),
        _make_tool(
            name="get_file",
            description="Read the full content of a file in the vault.",
            inputSchema={
                "type": "object",
                "properties": {
                    "filepath": {
                        "type": "string",
                        "description": "File path relative to vault root",
                    },
                },
                "required": ["filepath"],
            },
        ),
        _make_tool(
            name="get_files_batch",
            description="Read the contents of multiple files at once.",
            inputSchema={
                "type": "object",
                "properties": {
                    "filepaths": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of file paths relative to vault root",
                    },
                },
                "required": ["filepaths"],
            },
        ),
        _make_tool(
            name="append_content",
            description="Append content to the end of a file. Creates the file if it doesn't exist.",
            inputSchema={
                "type": "object",
                "properties": {
                    "filepath": {
                        "type": "string",
                        "description": "File path relative to vault root",
                    },
                    "content": {
                        "type": "string",
                        "description": "Content to append",
                    },
                },
                "required": ["filepath", "content"],
            },
        ),
        _make_tool(
            name="write_file",
            description=(
                "Write or overwrite a file in the vault. Creates parent directories if needed. "
                "WARNING: overwrites existing content without confirmation — use append_content "
                "if you want to add to an existing file without replacing it."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "filepath": {
                        "type": "string",
                        "description": "File path relative to vault root",
                    },
                    "content": {
                        "type": "string",
                        "description": "Full content to write",
                    },
                },
                "required": ["filepath", "content"],
            },
        ),
        _make_tool(
            name="simple_search",
            description=(
                "Text/keyword search across vault files. "
                "Use search_vault for semantic/meaning-based search, "
                "use this for exact text matching."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Text to search for (case-insensitive)",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Max results (default: 10)",
                        "default": 10,
                    },
                    "context_length": {
                        "type": "integer",
                        "description": "Characters of context around each match (default: 100)",
                        "default": 100,
                    },
                },
                "required": ["query"],
            },
        ),
        _make_tool(
            name="get_note_connections",
            description=(
                "Return all notes connected to a given note via wikilinks — "
                "both notes it links to (outgoing) and notes that link back to it (incoming). "
                "Useful for exploring your knowledge graph and finding related notes."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "filepath": {
                        "type": "string",
                        "description": "Vault-relative path to the note (e.g. 'notes/concepts/resilience.md')",
                    },
                    "hops": {
                        "type": "integer",
                        "description": "How many link-hops to traverse (default: 1, max: 2)",
                        "default": 1,
                    },
                },
                "required": ["filepath"],
            },
        ),
        _make_tool(
            name="recent_changes",
            description="Get recently modified files in the vault.",
            inputSchema={
                "type": "object",
                "properties": {
                    "limit": {
                        "type": "integer",
                        "description": "Max files to return (default: 10)",
                        "default": 10,
                    },
                    "days": {
                        "type": "integer",
                        "description": "Only files modified within this many days (default: 30)",
                        "default": 30,
                    },
                },
            },
        ),
    ]
    if os.environ.get("OSM_ALLOW_WRITE") != "1":
        tools = [tool for tool in tools if tool.name not in {"append_content", "write_file"}]
    return tools


# SECURITY: MCP protocol has no built-in auth. Access control relies on
# the transport layer (stdio). Do not expose this server over network without auth proxy.
async def call_tool(name: str, arguments: dict):
    if name in {"append_content", "write_file"} and os.environ.get("OSM_ALLOW_WRITE") != "1":
        return [TextContent(type="text", text="Write access is disabled for this MCP client.")]
    if name != "search_vault" or _SEARCH_METRIC_CONTEXT.get() is not None:
        return await _call_tool_impl(name, arguments)
    mode = arguments.get("mode", "hybrid")
    if not isinstance(mode, str) or mode not in _SEARCH_METRIC_MODES:
        mode = "hybrid"
    state = {
        "backend": "local",
        "mode": mode,
        "outcome": "error",
        "results": 0,
    }
    token = _SEARCH_METRIC_CONTEXT.set(state)
    started = time.monotonic()
    cancelled = False
    try:
        return await _call_tool_impl(name, arguments)
    except asyncio.CancelledError:
        cancelled = True
        raise
    finally:
        _SEARCH_METRIC_CONTEXT.reset(token)
        if not cancelled:
            duration_ms = max(0.0, (time.monotonic() - started) * 1000)
            try:
                await asyncio.get_running_loop().run_in_executor(
                    None, _record_search_metric, state, duration_ms
                )
            except Exception as exc:
                log.warning("search metrics recording failed type=%s", type(exc).__name__)


async def _call_tool_impl(name: str, arguments: dict):

    # ── search_vault ──────────────────────────────────────────────────────────
    if name == "search_vault":
        query = arguments.get("query", "").strip()
        limit = max(1, min(int(arguments.get("limit", 5)), 20))
        min_similarity = float(arguments.get("min_similarity", 0.0))
        mode = arguments.get("mode", "hybrid")
        vault_filter = arguments.get("vault", "").strip()
        graph_expand = bool(arguments.get("graph_expand", False))
        if mode not in ("hybrid", "semantic", "keyword"):
            mode = "hybrid"

        if not query:
            return [TextContent(type="text", text="Please provide a search query.")]

        readiness_message = _search_readiness_message(mode)
        if readiness_message:
            return [TextContent(type="text", text=readiness_message)]

        # Resolve vault filter: match by name (basename) or full path
        vault_ids: list[str] | None = None
        if vault_filter:
            vault_ids = [
                v for v in VAULT_PATHS if v == vault_filter or os.path.basename(v) == vault_filter
            ]
            if not vault_ids:
                return [
                    TextContent(
                        type="text",
                        text=f"No vault matching '{vault_filter}' found. "
                        f"Available: {', '.join(os.path.basename(v) for v in VAULT_PATHS)}",
                    )
                ]

        try:
            _t0 = time.monotonic()
            loop = asyncio.get_running_loop()

            # When re-ranking is enabled, fetch a wider candidate pool first
            fetch_limit = max(limit, RERANK_CANDIDATES) if RERANK_MODEL else limit

            if mode == "keyword" and DATABASE_BACKEND == "qdrant":
                profile = _qstore().profile()
                if profile and profile.get("search_profile") == "bm25-russian-v1":
                    rows = await loop.run_in_executor(
                        None,
                        search_chunk_rows,
                        query,
                        None,
                        mode,
                        fetch_limit,
                        min_similarity,
                        vault_ids,
                    )
                else:
                    terms = query.casefold().split()
                    candidates = (
                        row
                        for row in _qstore().notes()
                        if vault_ids is None or row.get("vault_id") in vault_ids
                    )
                    rows = sorted(
                        (
                            (
                                row["path"],
                                row["content"],
                                sum(row["content"].casefold().count(term) for term in terms)
                                / max(len(row["content"].split()), 1),
                            )
                            for row in candidates
                            if any(term in row["content"].casefold() for term in terms)
                        ),
                        key=lambda item: item[2],
                        reverse=True,
                    )[:fetch_limit]
            elif mode == "keyword":
                rows = _pgstore().keyword_search(query, fetch_limit, vault_ids)
            else:
                vec = await loop.run_in_executor(None, lambda: embed(query, task="query"))
                rows = await loop.run_in_executor(
                    None,
                    search_chunk_rows,
                    query,
                    vec,
                    mode,
                    fetch_limit,
                    min_similarity,
                    vault_ids,
                )

            # Optional cross-encoder re-ranking (runs only when RERANK_MODEL is set)
            rows = await loop.run_in_executor(None, _rerank, query, list(rows), limit)

            # Apply similarity threshold filter
            results = [r for r in rows if r[2] >= min_similarity]

            if not results:
                _search_metric_success(0)
                return [
                    TextContent(
                        type="text",
                        text=_empty_search_message(vault_ids),
                    )
                ]

            parts = []
            for path, content, sim in results:
                rel = _relative(Path(path))
                preview = (content if mode != "keyword" else content[:600]).strip()
                while "\n\n\n" in preview:
                    preview = preview.replace("\n\n\n", "\n\n")
                parts.append(f"**{rel}** _(similarity: {sim:.2f})_\n\n{preview}\n")

            if graph_expand and results:
                result_paths = [r[0] for r in results]
                neighbors = await loop.run_in_executor(None, expand_via_links, result_paths, 1)
                if neighbors:
                    parts.extend(_render_neighbor_parts(neighbors))
                    log.info("graph_expand added %d neighbor(s)", len(neighbors))

            result = [TextContent(type="text", text="\n---\n".join(parts))]

            _duration_ms = int((time.monotonic() - _t0) * 1000)
            _query_hash = hashlib.sha256(query.encode()).hexdigest()[:8]
            log.info(
                "search mode=%s query_hash=%s limit=%d found=%d duration_ms=%d",
                mode,
                _query_hash,
                limit,
                len(results),
                _duration_ms,
            )

            _search_metric_success(len(results))
            return result

        except Exception as e:
            log.error("search_vault error: %s", e)
            return [TextContent(type="text", text=f"Search error: {e}")]

    # ── list_indexed_notes ────────────────────────────────────────────────────
    elif name == "list_indexed_notes":
        try:
            if DATABASE_BACKEND == "qdrant":
                rows = [
                    (row["path"], datetime.fromisoformat(row["indexed_at"]))
                    for row in _qstore().notes()
                ]
                rows.sort(key=lambda item: item[1], reverse=True)
            else:
                rows = _pgstore().indexed_notes()

            if not rows:
                return [
                    TextContent(
                        type="text",
                        text="No notes indexed yet. Run reindex_vault to start.",
                    )
                ]

            lines = [f"**{len(rows)} notes indexed**\n"]
            for path, ts in rows:
                rel = _relative(Path(path))
                lines.append(f"- {rel}  _(indexed {ts.strftime(_TIMESTAMP_FMT)})_")

            return [TextContent(type="text", text="\n".join(lines))]

        except Exception as e:
            log.error("list_indexed_notes error: %s", e)
            return [TextContent(type="text", text=f"Error: {e}")]

    # ── reindex_vault ─────────────────────────────────────────────────────────
    elif name == "reindex_vault":
        if not VAULT_PATHS:
            return [
                TextContent(
                    type="text",
                    text="No vault configured. Set OBSIDIAN_VAULTS or OBSIDIAN_VAULT.",
                )
            ]

        # Acquired here (synchronously, on the tool-call coroutine) so we can
        # report busy immediately; released inside the background thread once
        # indexing finishes — the acquiring and releasing call sites are
        # deliberately on different threads, see reindex_lock()'s docstring.
        lock_cm = reindex_lock()
        acquired = lock_cm.__enter__()
        if not acquired:
            lock_cm.__exit__(None, None, None)
            return [
                TextContent(
                    type="text",
                    text="Re-index already in progress (held by another process). "
                    "Use list_indexed_notes to check its progress.",
                )
            ]

        def _reindex_all():
            try:
                failures = []
                _set_readiness("indexing")
                for vp in VAULT_PATHS:
                    try:
                        index_vault(vp)
                        state = get_index_state(vp)
                        if state and state["status"] in ("failed", "dimension_mismatch"):
                            failures.append(vp)
                    except Exception as exc:
                        log.error("Re-index failed for %s: %s", vp, exc)
                        set_index_state(vp, "failed", error=str(exc))
                        failures.append(vp)
                _set_readiness("partial_error" if failures else "ready")
            finally:
                lock_cm.__exit__(None, None, None)

        threading.Thread(target=_reindex_all, daemon=True).start()

        vault_list = ", ".join(VAULT_PATHS)
        return [
            TextContent(
                type="text",
                text=(
                    f"Re-indexing started in background for: {vault_list}\n"
                    "Use list_indexed_notes to check progress."
                ),
            )
        ]

    # ── list_files ─────────────────────────────────────────────────────────────
    elif name == "list_files":
        try:
            dirpath = arguments.get("dirpath", "")
            target = _resolve_vault_path(dirpath) if dirpath else _vault_root()
            if not target.is_dir():
                return [TextContent(type="text", text=f"Not a directory: {dirpath}")]

            entries = sorted(target.iterdir())
            lines = []
            for e in entries:
                if e.name.startswith("."):
                    continue
                rel = _relative(e)
                prefix = "📁 " if e.is_dir() else "📄 "
                lines.append(f"{prefix}{rel}")

            return [TextContent(type="text", text="\n".join(lines) or "Empty directory")]
        except Exception as e:
            return [TextContent(type="text", text=f"Error: {e}")]

    # ── get_file ──────────────────────────────────────────────────────────────
    elif name == "get_file":
        try:
            filepath = arguments.get("filepath", "")
            target = _resolve_vault_path(filepath)
            if not target.is_file():
                return [TextContent(type="text", text=f"File not found: {filepath}")]
            content = target.read_text(encoding="utf-8", errors="ignore")
            return [TextContent(type="text", text=content)]
        except Exception as e:
            return [TextContent(type="text", text=f"Error: {e}")]

    # ── get_files_batch ───────────────────────────────────────────────────────
    elif name == "get_files_batch":
        try:
            filepaths = arguments.get("filepaths", [])
            parts = []
            for fp in filepaths:
                target = _resolve_vault_path(fp)
                if target.is_file():
                    content = target.read_text(encoding="utf-8", errors="ignore")
                    parts.append(f"--- {fp} ---\n{content}")
                else:
                    parts.append(f"--- {fp} ---\n[File not found]")
            return [TextContent(type="text", text="\n\n".join(parts))]
        except Exception as e:
            return [TextContent(type="text", text=f"Error: {e}")]

    # ── append_content ────────────────────────────────────────────────────────
    elif name == "append_content":
        try:
            filepath = arguments.get("filepath", "")
            content = arguments.get("content", "")
            target = _resolve_vault_path(filepath)
            target.parent.mkdir(parents=True, exist_ok=True)
            with open(target, "a", encoding="utf-8") as f:
                f.write(content)
            log.info("Appended to: %s", filepath)
            return [TextContent(type="text", text=f"Appended to {filepath}")]
        except Exception as e:
            return [TextContent(type="text", text=f"Error: {e}")]

    # ── write_file ────────────────────────────────────────────────────────────
    elif name == "write_file":
        try:
            filepath = arguments.get("filepath", "")
            content = arguments.get("content", "")
            content = _ensure_frontmatter(content)
            target = _resolve_vault_path(filepath)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
            log.info("Wrote: %s", filepath)
            return [TextContent(type="text", text=f"Wrote {filepath} ({len(content)} chars)")]
        except Exception as e:
            return [TextContent(type="text", text=f"Error: {e}")]

    # ── simple_search ─────────────────────────────────────────────────────────
    elif name == "simple_search":
        try:
            query = arguments.get("query", "").strip()
            limit = max(1, min(int(arguments.get("limit", 10)), 50))
            ctx_len = max(1, int(arguments.get("context_length", 100)))
            if not query:
                return [TextContent(type="text", text="Please provide a search query.")]

            query_lower = query.lower()
            results = []
            root = _vault_root()
            for f in root.rglob("*.md"):
                if _should_skip_path(f):
                    continue
                try:
                    text = f.read_text(encoding="utf-8", errors="ignore")
                except Exception:
                    continue
                text_lower = text.lower()
                idx = text_lower.find(query_lower)
                if idx == -1:
                    continue
                # Collect match contexts
                matches = []
                search_from = 0
                while len(matches) < 3:
                    idx = text_lower.find(query_lower, search_from)
                    if idx == -1:
                        break
                    start = max(0, idx - ctx_len)
                    end = min(len(text), idx + len(query) + ctx_len)
                    matches.append(text[start:end].strip())
                    search_from = idx + len(query)

                results.append((_relative(f), matches))
                if len(results) >= limit:
                    break

            if not results:
                return [TextContent(type="text", text=f"No matches for: {query}")]

            parts = []
            for rel, matches in results:
                match_text = "\n".join(f"  ...{m}..." for m in matches)
                parts.append(f"**{rel}**\n{match_text}")
            return [TextContent(type="text", text="\n\n".join(parts))]
        except Exception as e:
            return [TextContent(type="text", text=f"Error: {e}")]

    # ── get_note_connections ──────────────────────────────────────────────────
    elif name == "get_note_connections":
        try:
            filepath = arguments.get("filepath", "").strip()
            hops = max(1, min(int(arguments.get("hops", 1)), 2))
            if not filepath:
                return [TextContent(type="text", text="Please provide a filepath.")]

            target = _resolve_vault_path(filepath)
            abs_path = str(target)

            neighbors = await asyncio.get_running_loop().run_in_executor(
                None, expand_via_links, [abs_path], hops
            )

            if not neighbors:
                return [
                    TextContent(
                        type="text",
                        text=f"No linked notes found for {filepath}. "
                        "The note may have no wikilinks, or linked notes are not yet indexed.",
                    )
                ]

            lines = [f"**{len(neighbors)} connected note(s)** for `{filepath}` (hops={hops})\n"]
            for npath, ncontent, via in neighbors:
                nrel = _relative(Path(npath))
                via_rel = _relative(Path(via))
                preview = ncontent[:200].strip().split("\n")[0]
                lines.append(f"- **{nrel}** _(via {via_rel})_ — {preview}")

            return [TextContent(type="text", text="\n".join(lines))]
        except Exception as e:
            return [TextContent(type="text", text=f"Error: {e}")]

    # ── recent_changes ────────────────────────────────────────────────────────
    elif name == "recent_changes":
        try:
            limit = min(int(arguments.get("limit", 10)), 100)
            days = int(arguments.get("days", 30))
            cutoff = time.time() - (days * 86400)
            root = _vault_root()

            files = []
            for f in root.rglob("*.md"):
                if _should_skip_path(f):
                    continue
                try:
                    mtime = f.stat().st_mtime
                    if mtime >= cutoff:
                        files.append((mtime, f))
                except Exception:
                    continue

            files.sort(key=lambda x: x[0], reverse=True)
            files = files[:limit]

            if not files:
                return [
                    TextContent(type="text", text=f"No files modified in the last {days} days.")
                ]

            lines = [f"**{len(files)} recently modified files** (last {days} days)\n"]
            for mtime, f in files:
                dt = datetime.fromtimestamp(mtime, tz=UTC)
                lines.append(f"- {_relative(f)}  _{dt.strftime(_TIMESTAMP_FMT)}_")

            return [TextContent(type="text", text="\n".join(lines))]
        except Exception as e:
            return [TextContent(type="text", text=f"Error: {e}")]

    return [TextContent(type="text", text=f"Unknown tool: {name}")]


# mcp 2.0.0 replaced decorator-based handler registration (@server.list_tools(),
# @server.call_tool()) with constructor callbacks taking (ctx, params). These
# adapters keep list_tools/call_tool's simpler list[Tool]/list[TextContent]
# signatures unchanged and just translate at the boundary.
async def _on_list_tools(ctx, params):
    return ListToolsResult(tools=await list_tools())


async def _on_call_tool(ctx, params):
    content = await call_tool(params.name, params.arguments or {})
    return CallToolResult(content=content)


server = Server(
    "obsidian-semantic",
    on_list_tools=_on_list_tools,
    on_call_tool=_on_call_tool,
)


# ──────────────────────────────── Entry Point ────────────────────────────────


async def main():
    if not VAULT_PATHS:
        log.error("No vault configured. Set OBSIDIAN_VAULTS or OBSIDIAN_VAULT.")
        sys.exit(1)

    log.info("Vaults: %s", ", ".join(VAULT_PATHS))

    import time as _time

    global _STARTED_AT
    _STARTED_AT = _time.monotonic()

    loop = asyncio.get_event_loop()
    stop_requested = asyncio.Event()
    if sys.platform != "win32":
        loop.add_signal_handler(signal.SIGTERM, lambda: _shutdown(stop_requested, "SIGTERM"))
        loop.add_signal_handler(signal.SIGINT, lambda: _shutdown(stop_requested, "SIGINT"))
        try:
            loop.add_signal_handler(signal.SIGHUP, lambda: _shutdown(stop_requested, "SIGHUP"))
        except AttributeError, NotImplementedError:
            pass
    else:
        # Windows: add_signal_handler is not implemented on ProactorEventLoop.
        # signal.signal() on the main thread is the fallback.
        signal.signal(signal.SIGINT, lambda *_: _shutdown(stop_requested, "SIGINT"))
        signal.signal(signal.SIGTERM, lambda *_: _shutdown(stop_requested, "SIGTERM"))

    # Full index + watchers start in background — server is immediately ready
    threading.Thread(
        target=background_init,
        args=(VAULT_PATHS,),
        daemon=True,
    ).start()

    # Raw stdin/stdout transport — avoids two known bugs:
    #   1. anyio.wrap_file() EOF death (May 7 2026): anyio's async-for over
    #      stdin exits when the client closes stdin between cycles, taking
    #      the server down. Solved by using a blocking readline that
    #      waits forever and treats EOF as "idle, retry."
    #   2. Blocking-loop event-loop freeze (May 8 2026): a synchronous
    #      `for line in sys.stdin.buffer:` in an async coroutine blocks
    #      the entire asyncio event loop, so `_stdout_writer` cannot
    #      schedule and the response is never written. Symptom: Claude
    #      Code times out at 30s during initialize over a pipe that
    #      stays open. Solved by offloading the blocking readline to a
    #      worker thread via anyio.to_thread.run_sync, which keeps the
    #      event loop free between reads.
    read_writer, read_stream = anyio.create_memory_object_stream(0)
    write_stream, write_reader = anyio.create_memory_object_stream(0)

    async def _stdin_reader():
        async with read_writer:
            while True:
                # Thread-offload the blocking readline so the event loop
                # stays free for _stdout_writer and server.run.
                line = await anyio.to_thread.run_sync(
                    sys.stdin.buffer.readline, abandon_on_cancel=True
                )
                if not line:
                    # EOF: stdin closed. Don't exit — Claude Desktop
                    # closes stdin between cycles. Sleep briefly and
                    # retry; the thread offload will block in-thread
                    # until new data arrives or stdin is fully gone.
                    await anyio.sleep(0.1)
                    continue
                line_str = line.decode("utf-8").strip()
                if not line_str:
                    continue
                try:
                    # mcp 2.0.0 made JSONRPCMessage a bare Union type alias (no
                    # .model_validate_json()); jsonrpc_message_adapter is the
                    # TypeAdapter mcp's own stdio transport uses to parse it.
                    message = jsonrpc_message_adapter.validate_json(line_str, by_name=False)
                    await read_writer.send(SessionMessage(message))
                except Exception as exc:
                    await read_writer.send(exc)

    async def _stdout_writer():
        async with write_reader:
            async for session_message in write_reader:
                json_str = session_message.message.model_dump_json(by_alias=True, exclude_none=True)
                sys.stdout.write(json_str + "\n")
                sys.stdout.flush()

    async def _cancel_on_shutdown(scope: anyio.CancelScope):
        await stop_requested.wait()
        scope.cancel()

    async with anyio.create_task_group() as tg:
        tg.start_soon(_stdin_reader)
        tg.start_soon(_stdout_writer)
        tg.start_soon(_cancel_on_shutdown, tg.cancel_scope)
        try:
            await server.run(
                read_stream,
                write_stream,
                server.create_initialization_options(),
            )
        finally:
            tg.cancel_scope.cancel()


def run_server():
    """Run the MCP server on standard input and output."""
    asyncio.run(main())


if __name__ == "__main__":
    asyncio.run(main())
