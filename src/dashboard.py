#!/usr/bin/env python3
"""
Monitoring dashboard for obsidian-semantic-mcp.

Started by `osm dashboard` as an installed module.
"""

from __future__ import annotations

import hmac
import http.server
import json
import logging
import math
import os
import socket
import stat
import subprocess
import threading
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import psycopg2
import requests

from .config import DEFAULT_EMBED_MODEL, build_dsn, resolve_dashboard_token
from .launcher import _run_directory
from .qdrant import QdrantStore, local_url
from .server import (
    VAULT_PATHS,
    _pgstore,
    _relative,
    _should_skip_path,
    embed,
    get_index_state,
    get_last_rebuild_failures,
    index_vault,
    prune_orphans,
    rebuild_index_safely,
    reindex_lock,
    search_chunk_rows,
)

# Importing server.py above already calls logging.basicConfig() — reuse the
# same handler/format rather than configuring a second one.
log = logging.getLogger(__name__)

VAULT_PATH = VAULT_PATHS[0] if VAULT_PATHS else ""
OLLAMA_URL = os.environ.get("OLLAMA_URL", "")
EMBED_MODEL = os.environ.get("EMBEDDING_MODEL", DEFAULT_EMBED_MODEL)
DASH_PORT = int(os.environ["DASHBOARD_PORT"])
DASHBOARD_BIND = "127.0.0.1"

DATABASE_URL = build_dsn()
DATABASE_BACKEND = os.environ.get("OSM_DATABASE_BACKEND", "postgresql")


def _read_version() -> str:
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version("obsidian-semantic-mcp")
    except PackageNotFoundError:
        pass
    try:
        toml = (Path(__file__).parent.parent / "pyproject.toml").read_text()
        for line in toml.splitlines():
            if line.startswith("version"):
                return line.split("=", 1)[1].strip().strip('"')
    except Exception:
        pass
    return "dev"


APP_VERSION = _read_version()

# Bearer token guarding the mutating endpoints (/api/reindex, /api/reindex/full,
# /api/prune, /api/ollama/start). Resolved once at import time — see
# config.resolve_dashboard_token() for the DASHBOARD_TOKEN-env / config-file /
# generate-and-persist precedence.
DASHBOARD_TOKEN: str | None = None


def get_dashboard_token() -> str:
    """Resolve the bearer token on first use, caching it in DASHBOARD_TOKEN.

    Deliberately NOT resolved at import time. resolve_dashboard_token()
    generates and persists a token file on first call, so an import-time
    call would make merely importing this module write a secret to
    ~/.config/obsidian-semantic-mcp/ — which anything that imports for
    reasons of its own (test collection, a linter, an IDE indexer, a doc
    generator) would then trigger as a side effect. Resolve when a request
    actually needs the token instead.

    DASHBOARD_TOKEN stays a module attribute rather than a private cache so
    tests can monkeypatch it directly to a known value; a non-None value
    here short-circuits resolution entirely.
    """
    global DASHBOARD_TOKEN
    if DASHBOARD_TOKEN is None:
        DASHBOARD_TOKEN = resolve_dashboard_token()
    return DASHBOARD_TOKEN


# Ollama health cache: (result_dict, expiry_timestamp)
_ollama_cache: tuple[dict, float] | None = None
_ollama_cache_lock = threading.Lock()
_OLLAMA_CACHE_TTL = 10.0  # seconds

# Orphan check cache — counting missing files requires O(n) filesystem calls;
# cache for 5 minutes to avoid blocking the stats endpoint on every refresh.
_orphan_cache: tuple[int, float] | None = None
_orphan_cache_lock = threading.Lock()
_ORPHAN_CACHE_TTL = 300.0  # seconds

# Vault file count cache — rglob across the vault is slow on networked mounts
# (NAS, sshfs); cache to keep /api/stats fast under the dashboard auto-refresh.
_vault_count_cache: tuple[tuple, int, float] | None = None  # (key, total, expiry)
_vault_count_cache_lock = threading.Lock()
_VAULT_COUNT_CACHE_TTL = 30.0  # seconds


def search_notes(
    query: str,
    limit: int = 5,
    min_similarity: float = 0.0,
    mode: str = "hybrid",
    vault: str | None = None,
) -> list[dict]:
    """Return the best passage per note, with adjacent context."""
    if mode not in ("hybrid", "semantic", "keyword"):
        mode = "hybrid"
    vault_ids = None
    if vault:
        vault_ids = [
            path for path in VAULT_PATHS if path == vault or os.path.basename(path) == vault
        ]
        if not vault_ids:
            return []
    if mode == "keyword":
        if DATABASE_BACKEND == "qdrant":
            terms = query.casefold().split()
            notes = QdrantStore(local_url(), EMBED_MODEL).notes()
            rows = sorted(
                (
                    (
                        row["path"],
                        row["content"],
                        sum(row["content"].casefold().count(term) for term in terms)
                        / max(len(row["content"].split()), 1),
                    )
                    for row in notes
                    if (vault_ids is None or row.get("vault_id") in vault_ids)
                    and any(term in row["content"].casefold() for term in terms)
                ),
                key=lambda item: item[2],
                reverse=True,
            )[:limit]
            return [
                {
                    "path": _relative(Path(path)),
                    "content": content,
                    "preview": content[:3600].strip(),
                    "similarity": round(score, 3),
                }
                for path, content, score in rows
                if score >= min_similarity
            ]
        rows = _pgstore().keyword_search(query, limit, vault_ids)
    else:
        rows = search_chunk_rows(
            query, embed(query, task="query"), mode, limit, min_similarity, vault_ids
        )

    return [
        {
            "path": _relative(Path(path)),
            "content": content or "",
            "preview": (content or "")[:3600].strip(),
            "similarity": round(float(similarity), 3),
        }
        for path, content, similarity in rows
        if similarity >= min_similarity
    ]


def _get_db_stats(stats: dict) -> None:
    if DATABASE_BACKEND == "qdrant":
        store = QdrantStore(local_url(), EMBED_MODEL)
        store.collections()
        stats["db_ok"] = True
        stats["db_label"] = "Qdrant"
        stats["vector_label"] = "Embedding model"
        profile = store.profile() if store.alias_target() else None
        if profile and profile.get("model") != EMBED_MODEL:
            raise RuntimeError("Qdrant index model differs from configured model")
        notes = store.notes() if profile else []
        stats["indexed_count"] = len(notes)
        dates = [row["indexed_at"] for row in notes]
        stats["last_indexed"] = max(dates, default=None)
        stats["oldest_indexed"] = min(dates, default=None)
        stats["pg_version"] = "Qdrant"
        stats["pgvector_version"] = EMBED_MODEL
        stats["recent_notes"] = [
            {
                "path": _relative(Path(row["path"])),
                "indexed_at": datetime.fromisoformat(row["indexed_at"]).strftime("%Y-%m-%d %H:%M"),
            }
            for row in sorted(notes, key=lambda row: row["indexed_at"], reverse=True)[:10]
        ]
        stats["orphaned_embeddings"] = sum(not Path(row["path"]).exists() for row in notes)
        return
    snapshot = _pgstore().dashboard_snapshot()
    stats["db_ok"] = True
    stats["pg_version"] = snapshot["pg_version"]
    stats["pgvector_version"] = snapshot["pgvector_version"]
    stats["indexed_count"] = snapshot["indexed_count"]
    last, oldest = snapshot["last_indexed"], snapshot["oldest_indexed"]
    stats["last_indexed"] = last.isoformat() if last else None
    stats["oldest_indexed"] = oldest.isoformat() if oldest else None
    size = snapshot["db_size_bytes"]
    stats["db_size_bytes"] = size
    if size < 1024:
        stats["db_size_human"] = f"{size} B"
    elif size < 1024 * 1024:
        stats["db_size_human"] = f"{size / 1024:.1f} KB"
    else:
        stats["db_size_human"] = f"{size / (1024 * 1024):.1f} MB"
    for path, indexed_at in snapshot["recent_notes"]:
        stats["recent_notes"].append(
            {
                "path": _relative(Path(path)),
                "indexed_at": indexed_at.strftime("%Y-%m-%d %H:%M"),
            }
        )
    all_paths = snapshot["paths"]

    # O(n) filesystem check — cached for _ORPHAN_CACHE_TTL to avoid blocking
    # the stats endpoint on every 30-second dashboard refresh.
    global _orphan_cache
    now = time.monotonic()
    orphaned: int
    with _orphan_cache_lock:
        if _orphan_cache is not None and now < _orphan_cache[1]:
            orphaned = _orphan_cache[0]
        else:
            orphaned = sum(1 for p in all_paths if not os.path.exists(p))
            _orphan_cache = (orphaned, now + _ORPHAN_CACHE_TTL)
    stats["orphaned_embeddings"] = orphaned


def _get_vault_stats(stats: dict) -> None:
    global _vault_count_cache
    vaults = VAULT_PATHS if VAULT_PATHS else ([VAULT_PATH] if VAULT_PATH else [])
    if not vaults:
        return

    # Key the cache on the vault set + the ignore-paths env so any config
    # change invalidates immediately (and unit tests that swap vaults between
    # cases don't see stale counts).
    cache_key = (tuple(vaults), os.environ.get("OBSIDIAN_IGNORE_PATHS", "<unset>"))
    now = time.monotonic()
    with _vault_count_cache_lock:
        if _vault_count_cache is not None:
            cached_key, cached_total, expiry = _vault_count_cache
            if cached_key == cache_key and now < expiry:
                stats["vault_file_count"] = cached_total
                stats["unindexed_count"] = max(0, cached_total - stats["indexed_count"])
                return

    total = sum(1 for vp in vaults for f in Path(vp).rglob("*.md") if not _should_skip_path(f))
    with _vault_count_cache_lock:
        _vault_count_cache = (cache_key, total, now + _VAULT_COUNT_CACHE_TTL)
    stats["vault_file_count"] = total
    stats["unindexed_count"] = max(0, total - stats["indexed_count"])


def _get_ollama_stats(stats: dict) -> None:
    global _ollama_cache
    now = time.monotonic()

    with _ollama_cache_lock:
        if _ollama_cache is not None:
            cached_result, expiry = _ollama_cache
            if now < expiry:
                stats.update(cached_result)
                return

    # Fetch fresh result outside the lock to avoid blocking other threads
    result: dict = {}
    try:
        resp = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
        resp.raise_for_status()
        data = resp.json()
        models = [m["name"] for m in data.get("models", [])]
        result["ollama_ok"] = True
        result["model_loaded"] = any(EMBED_MODEL in m for m in models)
    except Exception as e:
        result["ollama_ok"] = False
        result["model_loaded"] = False
        result["ollama_error"] = str(e)

    with _ollama_cache_lock:
        _ollama_cache = (result, now + _OLLAMA_CACHE_TTL)

    stats.update(result)


def read_search_metrics(now: datetime | None = None) -> dict:
    """Read aggregate-only current UTC hour and the previous 23 buckets."""
    hour = (now or datetime.now(UTC)).astimezone(UTC).replace(minute=0, second=0, microsecond=0)
    start, end = hour - timedelta(hours=23), hour + timedelta(hours=1)
    metrics: dict[str, Any] = {
        "window": "current_utc_hour_and_previous_23",
        "window_start": start.isoformat(),
        "window_end": end.isoformat(),
        "available": False,
        "requests": 0,
        "results": 0,
        "duration_ms": 0.0,
        "errors": 0,
        "series": [],
    }
    if DATABASE_BACKEND == "qdrant":
        return metrics
    try:
        rows = _pgstore().search_metric_rows(start, end)
        for backend, mode, outcome, requests_count, results, duration in rows:
            if (
                backend != "local"
                or mode not in ("hybrid", "semantic", "keyword")
                or outcome not in ("success", "error")
            ):
                raise ValueError("invalid search metric dimensions")
            counts = [requests_count, results]
            if any(isinstance(value, bool) or value < 0 or int(value) != value for value in counts):
                raise ValueError("invalid search metric counts")
            duration = float(duration)
            if not math.isfinite(duration) or duration < 0:
                raise ValueError("invalid search metric duration")
            row: dict[str, Any] = dict(zip(("requests", "results"), map(int, counts), strict=True))
            row.update(
                backend=backend,
                mode=mode,
                outcome=outcome,
                duration_ms=duration,
                errors=int(requests_count) if outcome == "error" else 0,
            )
            metrics["series"].append(row)
            for name in ("requests", "results", "duration_ms", "errors"):
                metrics[name] += row[name]
        if not math.isfinite(metrics["duration_ms"]):
            raise ValueError("invalid search metric duration total")
        metrics["available"] = True
    except (psycopg2.Error, ValueError, TypeError, OverflowError, RuntimeError, OSError) as exc:
        log.warning("dashboard search metrics unavailable type=%s", type(exc).__name__)
        metrics.update(requests=0, results=0, duration_ms=0.0, errors=0, series=[])
    return metrics


def _prometheus_label(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


def render_search_metrics(metrics: dict) -> str:
    """Window aggregates are gauges because expiring buckets decrease values."""
    lines = []
    for name, field, divisor in (
        ("requests", "requests", 1),
        ("duration_seconds", "duration_ms", 1000),
        ("results", "results", 1),
        ("errors", "errors", 1),
    ):
        metric_name = f"osm_search_{name}"
        lines.extend(
            (
                f"# HELP {metric_name} Current UTC hour and previous 23 buckets.",
                f"# TYPE {metric_name} gauge",
            )
        )
        if not metrics["series"]:
            lines.append(f"{metric_name} 0")
        for row in metrics["series"]:
            labels = ",".join(
                f'{key}="{_prometheus_label(row[key])}"' for key in ("backend", "mode", "outcome")
            )
            value = str(row[field]) if divisor == 1 else repr(row[field] / divisor)
            lines.append(f"{metric_name}{{{labels}}} {value}")
    lines.extend(
        (
            "# HELP osm_search_metrics_available Whether aggregate data was read successfully.",
            "# TYPE osm_search_metrics_available gauge",
            f"osm_search_metrics_available {int(metrics['available'])}",
        )
    )
    return "\n".join(lines) + "\n"


def gather_stats() -> dict:
    stats = {
        "timestamp": datetime.now(UTC).isoformat(),
        "db_ok": False,
        "indexed_count": 0,
        "last_indexed": None,
        "oldest_indexed": None,
        "db_size_bytes": 0,
        "db_size_human": "—",
        "vault_file_count": 0,
        "unindexed_count": 0,
        "orphaned_embeddings": 0,
        "ollama_ok": False,
        "model_loaded": False,
        # The Start button is available only for a local Ollama URL.
        "can_start_ollama": "localhost" in OLLAMA_URL,
        "reindex_busy": False,
        "last_rebuild_failed_count": 0,
        "last_rebuild_failed_sample": [],
        "dimension_mismatch": False,
        "dimension_mismatch_message": None,
        "recent_notes": [],
        "pg_version": "—",
        "pgvector_version": "—",
        "db_label": "PostgreSQL",
        "vector_label": "pgvector",
        "embedding_model": EMBED_MODEL,
    }

    try:
        _get_db_stats(stats)
    except Exception as e:
        stats["db_error"] = str(e)

    try:
        _get_vault_stats(stats)
    except Exception:
        pass

    try:
        _get_ollama_stats(stats)
    except Exception as e:
        stats["ollama_error"] = str(e)

    # Probe the shared Postgres advisory lock rather than a process-local
    # threading.Lock — see reindex_lock()'s docstring. A probe acquire+
    # release in one `with` correctly reports busy=True whenever ANOTHER
    # another process holds the lock, not just another thread in this one.
    with reindex_lock() as acquired:
        pass
    stats["reindex_busy"] = not acquired

    try:
        failed = get_last_rebuild_failures()
        stats["last_rebuild_failed_count"] = len(failed)
        stats["last_rebuild_failed_sample"] = failed[:5]
    except Exception:
        pass

    try:
        for row in get_index_state():
            if row["status"] == "dimension_mismatch":
                stats["dimension_mismatch"] = True
                stats["dimension_mismatch_message"] = row["error"]
                break
    except Exception:
        pass

    stats["search_metrics"] = read_search_metrics()
    return stats


HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Obsidian Semantic MCP — Monitor</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Newsreader:ital,wght@0,400;0,500;0,700;1,400&family=Hanken+Grotesk:wght@400;500;600;700&family=DM+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
  /* Cobalt Grid palette, dashboard variant — cool blue-grey paper + navy-blue
     ink (see ../semblar/design_references/max_graphic_design_mood/cobalt_grid_01.json,
     recolored here from the source's warm-neutral paper for this UI) */
  :root {
    --paper: #E4E9F2; --paper-2: #D3DCEC; --ink: #1F3868; --ink-soft: #4F6FA8;
    --grid: rgba(31,56,104,.10); --ink-faint: rgba(31,56,104,.18);
    --serif: 'Newsreader', Georgia, serif;
    --sans: 'Hanken Grotesk', -apple-system, sans-serif;
    --mono: 'DM Mono', ui-monospace, monospace;
  }
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body {
    font-family: var(--sans);
    background: var(--paper); color: var(--ink); padding: 24px;
    min-height: 100vh;
  }
  h1 { font-family: var(--serif); font-weight: 500; font-size: 1.6rem; color: var(--ink); margin-bottom: 8px; }
  .subtitle { font-size: 0.85rem; color: var(--ink-soft); margin-bottom: 24px; font-style: italic; }
  .grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 16px; margin-bottom: 24px;
  }
  .card {
    background: var(--paper-2); border-radius: 4px; padding: 20px;
  }
  .card-label { font-family: var(--sans); font-size: 0.75rem; text-transform: uppercase;
    letter-spacing: 0.08em; color: var(--ink-soft); font-weight: 600; margin-bottom: 8px; }
  .card-value { font-family: var(--mono); font-size: 1.8rem; font-weight: 500; }
  .card-detail { font-size: 0.8rem; color: var(--ink-soft); margin-top: 4px; }
  .status-row {
    display: flex; gap: 24px; margin-bottom: 24px; flex-wrap: wrap;
  }
  .status {
    display: flex; align-items: center; gap: 8px;
    font-size: 0.9rem; background: var(--paper-2); padding: 10px 16px;
    border-radius: 4px;
  }
  .dot {
    width: 10px; height: 10px; border-radius: 50%;
    display: inline-block; flex-shrink: 0;
  }
  .dot.green  { background: #22c55e; box-shadow: 0 0 6px #22c55e80; }
  .dot.red    { background: #ef4444; box-shadow: 0 0 6px #ef444480; }
  .dot.yellow { background: #eab308; box-shadow: 0 0 6px #eab30880; }
  .dot.grey   { background: #9a9488; }
  .recent { background: var(--paper-2); border-radius: 4px; padding: 20px;
    margin-bottom: 24px; }
  .recent h2 { font-family: var(--serif); font-weight: 500; font-size: 1.15rem; color: var(--ink); margin-bottom: 12px;
    cursor: pointer; user-select: none; display: flex; align-items: center; gap: 8px; }
  .recent h2 #recent-arrow { font-size: 0.8em; transition: transform 0.15s; display: inline-block; }
  .recent h2.collapsed #recent-arrow { transform: rotate(-90deg); }
  .recent-item {
    display: flex; justify-content: space-between; padding: 6px 0;
    border-bottom: 1px solid var(--ink-faint); font-size: 0.85rem;
  }
  .recent-item:last-child { border-bottom: none; }
  .recent-path { font-family: var(--mono); color: var(--ink); overflow: hidden; text-overflow: ellipsis;
    white-space: nowrap; max-width: 70%; }
  .recent-time { color: var(--ink-soft); white-space: nowrap; }
  .footer {
    text-align: center; margin-top: 24px; font-size: 0.75rem; color: var(--ink-soft);
  }
  .error-msg { color: #ef4444; font-size: 0.8rem; margin-top: 4px; }
  .btn {
    background: var(--ink); color: var(--paper); border: none; padding: 6px 14px;
    border-radius: 4px; font-size: 0.8rem; font-weight: 600;
    cursor: pointer; margin-left: 8px; transition: opacity 0.2s;
  }
  .btn:hover { opacity: 0.85; }
  .btn:disabled { opacity: 0.4; cursor: not-allowed; }
  .btn-danger { background: #b3402f; color: #fff; }
  .btn-standalone { margin-left: 0; }
  .actions-row {
    display: flex; gap: 10px; margin-bottom: 24px; flex-wrap: wrap;
  }
  .hidden { display: none !important; }
  .indexing-banner {
    background: var(--paper-2); border: 1.5px solid var(--ink-soft); border-radius: 4px;
    padding: 10px 16px; margin-bottom: 24px; font-size: 0.85rem; color: var(--ink);
    display: flex; align-items: center; gap: 10px;
  }
  .spinner {
    width: 14px; height: 14px; border: 2px solid var(--ink-soft);
    border-top-color: transparent; border-radius: 50%;
    animation: spin 0.8s linear infinite; flex-shrink: 0;
  }
  @keyframes spin { to { transform: rotate(360deg); } }
  .search-panel {
    background: var(--paper-2); border-radius: 4px; padding: 20px;
    margin-bottom: 24px;
  }
  .search-panel h2 { font-family: var(--serif); font-weight: 500; font-size: 1.15rem; color: var(--ink); margin-bottom: 14px; }
  .search-controls {
    display: flex; gap: 10px; flex-wrap: wrap; align-items: flex-end;
    margin-bottom: 14px;
  }
  .search-controls label {
    display: flex; flex-direction: column; gap: 4px;
    font-size: 0.75rem; color: var(--ink-soft); text-transform: uppercase;
    letter-spacing: 0.08em; font-weight: 600;
  }
  .search-controls input[type="text"] {
    background: var(--paper); border: 1.5px solid var(--ink-faint); border-radius: 4px;
    color: var(--ink); padding: 7px 10px; font-size: 0.85rem; width: 320px;
    outline: none; font-family: var(--sans);
  }
  .search-controls input[type="text"]:focus { border-color: var(--ink); }
  .search-controls input[type="number"] {
    background: var(--paper); border: 1.5px solid var(--ink-faint); border-radius: 4px;
    color: var(--ink); padding: 7px 10px; font-size: 0.85rem; width: 80px;
    outline: none; font-family: var(--mono);
  }
  .search-controls input[type="number"]:focus { border-color: var(--ink); }
  .search-result {
    border: 1.5px solid var(--ink-faint); border-radius: 4px; padding: 12px 14px;
    margin-bottom: 10px; background: var(--paper);
  }
  .search-result:last-child { margin-bottom: 0; }
  .search-result p { font-size: 0.82rem; color: var(--ink-soft); margin-top: 6px;
    line-height: 1.5; }
  .search-result code { font-family: var(--mono); font-size: 0.8rem; color: var(--ink); }
  @media (max-width: 600px) {
    .grid { grid-template-columns: 1fr 1fr; }
    .search-controls input[type="text"] { width: 100%; }
  }
</style>
</head>
<body>

<h1>Obsidian Semantic MCP</h1>
<p class="subtitle">Monitoring Dashboard {{VERSION}} — auto-refreshes every 30s</p>

<div class="status-row" id="statuses">
  <div class="status"><span class="dot grey" id="dot-db"></span><span id="lbl-db">PostgreSQL</span><span class="error-msg" id="err-db"></span></div>
  <div class="status"><span class="dot grey" id="dot-ollama"></span><span id="lbl-ollama">Ollama</span><button class="btn hidden" id="btn-ollama" onclick="startOllama()">Start</button></div>
  <div class="status"><span class="dot grey" id="dot-model"></span><span id="lbl-model">Embedding Model</span></div>
</div>

<div class="actions-row">
  <button class="btn btn-standalone" id="btn-reindex" onclick="triggerReindex(false)">Re-index</button>
  <button class="btn btn-standalone" id="btn-rebuild" onclick="triggerReindex(true)">Safe Rebuild</button>
</div>

<div class="indexing-banner hidden" id="indexing-banner">
  <div class="spinner"></div>
  <span>Indexing in progress — stats will update on completion</span>
</div>

<div class="grid">
  <div class="card">
    <div class="card-label">Indexed Notes</div>
    <div class="card-value" id="v-indexed">—</div>
    <div class="card-detail" id="d-indexed"></div>
  </div>
  <div class="card">
    <div class="card-label">Vault Files</div>
    <div class="card-value" id="v-vault">—</div>
    <div class="card-detail" id="d-vault"></div>
  </div>
  <div class="card">
    <div class="card-label">Unindexed</div>
    <div class="card-value" id="v-gap">—</div>
    <div class="card-detail">files not yet embedded</div>
  </div>
  <div class="card">
    <div class="card-label">Orphaned Embeddings</div>
    <div class="card-value" id="v-orphaned">—</div>
    <div class="card-detail">in DB but not on disk</div>
  </div>
  <div class="card">
    <div class="card-label">DB Size</div>
    <div class="card-value" id="v-dbsize">—</div>
    <div class="card-detail" id="d-dbsize"></div>
  </div>
  <div class="card">
    <div class="card-label">Last Indexed</div>
    <div class="card-value" id="v-last">—</div>
    <div class="card-detail" id="d-last"></div>
  </div>
  <div class="card">
    <div class="card-label" id="lbl-vector">pgvector</div>
    <div class="card-value" id="v-pgvec">—</div>
    <div class="card-detail" id="d-pgver"></div>
  </div>
</div>

<div class="grid">
  <div class="card">
    <div class="card-label">Searches / Recherches</div>
    <div class="card-value" id="v-search-requests">—</div>
    <div class="card-detail">Current UTC hour + previous 23 / Heure UTC actuelle + 23 précédentes</div>
  </div>
  <div class="card">
    <div class="card-label">Search duration / Durée des recherches</div>
    <div class="card-value" id="v-search-duration">—</div>
    <div class="card-detail" id="d-search-outcomes"></div>
  </div>
  <div class="card">
    <div class="card-label">Effective backend / Backend effectif</div>
    <div class="card-detail" id="d-search-backends">—</div>
  </div>
</div>

<div class="recent">
  <h2 onclick="toggleRecent()" id="recent-toggle">Recently Indexed <span id="recent-arrow">&#9662;</span></h2>
  <div id="recent-list"><div class="recent-item"><span class="recent-path">Loading...</span></div></div>
</div>

<div class="search-panel">
  <h2>Test Search</h2>
  <div class="search-controls">
    <label>
      Query
      <input type="text" id="search-query" placeholder="Enter search query..." />
    </label>
    <label>
      Mode
      <select id="search-mode">
        <option value="hybrid" selected>Hybrid</option>
        <option value="semantic">Semantic</option>
        <option value="keyword">Keyword</option>
      </select>
    </label>
    <label id="vault-label" style="display:none">
      Vault
      <select id="search-vault"><option value="">All vaults</option></select>
    </label>
    <label>
      Limit
      <input type="number" id="search-limit" value="5" min="1" max="20" />
    </label>
    <label>
      Min Similarity
      <input type="number" id="search-min-sim" value="0.0" min="0.0" max="1.0" step="0.1" />
    </label>
    <button class="btn btn-standalone" onclick="testSearch()">Search</button>
  </div>
  <div id="search-results"></div>
</div>

<p class="footer" id="footer">Fetching...</p>

<script>
const DASHBOARD_TOKEN = "{{TOKEN}}";

function timeAgo(iso) {
  if (!iso) return '—';
  const diff = (Date.now() - new Date(iso).getTime()) / 1000;
  if (diff < 60) return Math.floor(diff) + 's ago';
  if (diff < 3600) return Math.floor(diff / 60) + 'm ago';
  if (diff < 86400) return Math.floor(diff / 3600) + 'h ago';
  return Math.floor(diff / 86400) + 'd ago';
}

function dot(el, ok) {
  el.className = 'dot ' + (ok ? 'green' : 'red');
}

function toggleRecent() {
  document.getElementById('recent-list').classList.toggle('hidden');
  document.getElementById('recent-toggle').classList.toggle('collapsed');
}

async function fetchStats() {
  try {
    const ctrl = new AbortController();
    const timer = setTimeout(() => ctrl.abort(), 15000);
    const r = await fetch('/api/stats', { signal: ctrl.signal });
    clearTimeout(timer);
    const s = await r.json();
    const metrics = s.search_metrics;
    if (metrics) {
      document.getElementById('v-search-requests').textContent = metrics.available ? metrics.requests : '—';
      document.getElementById('v-search-duration').textContent = metrics.available ? (metrics.duration_ms / 1000).toFixed(2) + ' s' : '—';
      document.getElementById('d-search-outcomes').textContent =
        'Results / Résultats: ' + metrics.results + ' · Errors / Erreurs: ' + metrics.errors;
      const backends = {};
      metrics.series.forEach(row => { backends[row.backend] = (backends[row.backend] || 0) + row.requests; });
      document.getElementById('d-search-backends').textContent = metrics.available
        ? (Object.entries(backends).map(([name, count]) => name + ': ' + count).join(' · ') || '0')
        : 'Unavailable / Indisponible';
    }

    dot(document.getElementById('dot-db'), s.db_ok);
    dot(document.getElementById('dot-ollama'), s.ollama_ok);
    dot(document.getElementById('dot-model'), s.model_loaded);

    document.getElementById('lbl-db').textContent =
      s.db_ok ? s.db_label : s.db_label + ' — DOWN';
    document.getElementById('err-db').textContent =
      (!s.db_ok && s.db_error) ? s.db_error.split('\\n')[0] : '';
    document.getElementById('lbl-ollama').textContent =
      s.ollama_ok ? 'Ollama' : 'Ollama — DOWN';
    document.getElementById('btn-ollama').classList.toggle('hidden', s.ollama_ok || !s.can_start_ollama);
    document.getElementById('lbl-model').textContent =
      s.model_loaded ? s.embedding_model : 'Model — NOT LOADED';
    document.getElementById('lbl-vector').textContent = s.vector_label;

    document.getElementById('v-indexed').textContent = s.indexed_count;
    document.getElementById('v-vault').textContent = s.vault_file_count;
    document.getElementById('v-gap').textContent = s.unindexed_count;
    document.getElementById('v-orphaned').textContent =
      s.orphaned_embeddings !== undefined ? s.orphaned_embeddings : '—';
    document.getElementById('v-dbsize').textContent = s.db_size_human;
    document.getElementById('v-last').textContent = timeAgo(s.last_indexed);
    document.getElementById('d-last').textContent = s.last_indexed
      ? new Date(s.last_indexed).toLocaleString() : '';
    document.getElementById('v-pgvec').textContent =
      s.db_label === 'Qdrant' ? s.pgvector_version : 'v' + s.pgvector_version;
    document.getElementById('d-pgver').textContent = s.pg_version;

    const coverage = s.vault_file_count > 0
      ? Math.round(s.indexed_count / s.vault_file_count * 100) : 0;
    document.getElementById('d-indexed').textContent = coverage + '% coverage';
    document.getElementById('d-vault').textContent = '.md files in vault';

    const list = document.getElementById('recent-list');
    if (s.recent_notes.length === 0) {
      list.innerHTML = '<div class="recent-item"><span class="recent-path">No notes indexed yet</span></div>';
    } else {
      list.innerHTML = '';
      s.recent_notes.forEach(n => {
        const row = document.createElement('div');
        row.className = 'recent-item';
        const pathEl = document.createElement('span');
        pathEl.className = 'recent-path';
        pathEl.textContent = n.path;
        const timeEl = document.createElement('span');
        timeEl.className = 'recent-time';
        timeEl.textContent = n.indexed_at;
        row.appendChild(pathEl);
        row.appendChild(timeEl);
        list.appendChild(row);
      });
    }

    // Keep banner in sync with actual reindex state — prevents it getting
    // stuck visible after a server restart or page reload.
    if (!s.reindex_busy) {
      document.getElementById('indexing-banner').classList.add('hidden');
    }

    document.getElementById('footer').textContent =
      'Last refresh: ' + new Date().toLocaleTimeString() + ' — auto-refresh 30s';

  } catch (e) {
    const msg = e.name === 'AbortError' ? 'timed out' : e.message;
    document.getElementById('footer').textContent =
      'Service unreachable (' + msg + ') — run: osm status';
    ['dot-db', 'dot-ollama', 'dot-model'].forEach(id => {
      document.getElementById(id).className = 'dot grey';
    });
  }
}

function pollReindexDone(id, label) {
  fetch('/api/reindex/status').then(r => r.json()).then(d => {
    if (!d.busy) {
      const btn = document.getElementById(id);
      btn.disabled = false;
      btn.textContent = label;
      document.getElementById('indexing-banner').classList.add('hidden');
      fetchStats();
    } else {
      setTimeout(() => pollReindexDone(id, label), 3000);
    }
  }).catch(() => {
    setTimeout(() => pollReindexDone(id, label), 5000);
  });
}

async function triggerReindex(full) {
  if (full && !confirm('Build a new index, then switch to it? The previous index will be archived.')) return;
  const id = full ? 'btn-rebuild' : 'btn-reindex';
  const label = full ? 'Safe Rebuild' : 'Re-index';
  const btn = document.getElementById(id);
  btn.disabled = true;
  btn.textContent = 'Starting\u2026';
  try {
    const r = await fetch(full ? '/api/reindex/full' : '/api/reindex', {
      method: 'POST',
      headers: { 'Authorization': 'Bearer ' + DASHBOARD_TOKEN },
    });
    const d = await r.json();
    if (d.ok) {
      btn.textContent = 'Running\u2026';
      document.getElementById('indexing-banner').classList.remove('hidden');
      setTimeout(() => pollReindexDone(id, label), 3000);
    } else {
      btn.textContent = 'Failed: ' + (d.message || '');
      setTimeout(() => { btn.disabled = false; btn.textContent = label; }, 5000);
    }
  } catch (e) {
    btn.textContent = 'Error';
    setTimeout(() => { btn.disabled = false; btn.textContent = label; }, 5000);
  }
}

async function startOllama() {
  const btn = document.getElementById('btn-ollama');
  btn.disabled = true;
  btn.textContent = 'Starting...';
  try {
    const r = await fetch('/api/ollama/start', {
      method: 'POST',
      headers: { 'Authorization': 'Bearer ' + DASHBOARD_TOKEN },
    });
    const d = await r.json();
    btn.textContent = d.ok ? 'Started' : 'Failed';
    setTimeout(fetchStats, 3000);
  } catch (e) {
    btn.textContent = 'Error';
  }
  setTimeout(() => { btn.disabled = false; btn.textContent = 'Start'; }, 5000);
}

async function testSearch() {
  const q = document.getElementById('search-query').value.trim();
  if (!q) return;
  const limit = document.getElementById('search-limit').value;
  const minSim = document.getElementById('search-min-sim').value;
  const mode = document.getElementById('search-mode').value;
  const vault = document.getElementById('search-vault').value;
  let url = `/api/search?q=${encodeURIComponent(q)}&limit=${limit}&min_similarity=${minSim}&mode=${mode}`;
  if (vault) url += `&vault=${encodeURIComponent(vault)}`;
  const res = await fetch(url);
  const data = await res.json();
  const div = document.getElementById('search-results');
  if (!data.results || data.results.length === 0) {
    div.innerHTML = '<p>No results.</p>';
    return;
  }
  div.innerHTML = data.results.map(r => `
    <div class="search-result">
      <strong>${(r.similarity * 100).toFixed(1)}%</strong> \u2014 <code>${r.path}</code>
      <p>${r.content.substring(0, 200)}...</p>
    </div>
  `).join('');
}

async function fetchVaults() {
  try {
    const r = await fetch('/api/vaults');
    const d = await r.json();
    if (!d.vaults || d.vaults.length <= 1) return;
    const sel = document.getElementById('search-vault');
    d.vaults.forEach(v => {
      const opt = document.createElement('option');
      opt.value = v.name;
      opt.textContent = v.name;
      sel.appendChild(opt);
    });
    document.getElementById('vault-label').style.display = '';
  } catch (e) { /* single-vault — keep hidden */ }
}

fetchStats();
fetchVaults();
setInterval(fetchStats, 30000);
</script>
</body>
</html>"""

HTML_PAGE = HTML_PAGE.replace("{{VERSION}}", f"v{APP_VERSION}")
# {{TOKEN}} is deliberately NOT substituted here — see get_dashboard_token().
# Import-time substitution would force token resolution (and its file write)
# at import. The placeholder is filled per-request in _render_page().


def _render_page() -> bytes:
    """Return the dashboard HTML with the live bearer token injected."""
    return HTML_PAGE.replace("{{TOKEN}}", get_dashboard_token()).encode()


class DashboardHandler(http.server.BaseHTTPRequestHandler):
    def _json_response(self, code: int, data: dict) -> None:
        body = json.dumps(data).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _require_auth(self) -> bool:
        """Gate mutating endpoints behind a bearer token.

        Returns True and does nothing further when the request carries a
        valid `Authorization: Bearer <token>` header. On any mismatch (header
        missing, malformed, or wrong token) writes a 401 in the same
        `{"ok": ..., "message": ...}` shape every other branch uses and
        returns False — callers must `return` immediately when this is False.

        GET endpoints remain read-only and are bound to loopback by default.
        """
        header = self.headers.get("Authorization", "")
        prefix = "Bearer "
        supplied = header[len(prefix) :] if header.startswith(prefix) else ""
        if not supplied or not hmac.compare_digest(supplied, get_dashboard_token()):
            log.warning(
                "dashboard: unauthorized %s %s from %s",
                self.command,
                self.path,
                self.client_address[0],
            )
            self._json_response(401, {"ok": False, "message": "unauthorized"})
            return False
        return True

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/api/search":
            qs = parse_qs(parsed.query)
            query = qs.get("q", [""])[0].strip()
            limit = min(int(qs.get("limit", ["5"])[0]), 20)
            min_similarity = float(qs.get("min_similarity", ["0.0"])[0])
            mode = qs.get("mode", ["hybrid"])[0]
            vault = qs.get("vault", [""])[0].strip() or None
            if not query:
                self._json_response(400, {"error": "missing ?q="})
                return
            try:
                results = search_notes(query, limit, min_similarity, mode, vault)
                self._json_response(
                    200, {"query": query, "mode": mode, "vault": vault, "results": results}
                )
            except Exception as e:
                log.error("dashboard: /api/search failed for query %r: %s", query, e)
                self._json_response(500, {"error": str(e)})
        elif parsed.path == "/api/vaults":
            vaults = [{"name": os.path.basename(v), "path": v} for v in VAULT_PATHS]
            self._json_response(200, {"vaults": vaults})
        elif parsed.path == "/api/reindex/status":
            with reindex_lock() as acquired:
                pass
            self._json_response(200, {"busy": not acquired})
        elif parsed.path == "/api/stats":
            self._json_response(200, gather_stats())
        elif parsed.path == "/metrics":
            body = render_search_metrics(read_search_metrics()).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; version=0.0.4; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            body = _render_page()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    def do_POST(self) -> None:
        if not self._require_auth():
            return
        path = urlparse(self.path).path
        if path == "/api/ollama/start":
            self._handle_ollama_start()
        elif path in ("/api/reindex", "/api/reindex/full"):
            self._handle_reindex(full=path == "/api/reindex/full")
        elif path == "/api/prune":
            self._handle_prune()
        else:
            self._json_response(404, {"error": "not found"})

    def _handle_ollama_start(self) -> None:
        try:
            subprocess.Popen(
                ["ollama", "serve"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
            log.info("dashboard: ollama serve started via /api/ollama/start")
            self._json_response(200, {"ok": True, "message": "ollama serve started"})
        except Exception as e:
            log.error("dashboard: /api/ollama/start failed: %s", e)
            self._json_response(500, {"ok": False, "message": str(e)})

    def _busy_response(self) -> None:
        self._json_response(409, {"ok": False, "message": "Re-index already in progress"})

    def _handle_reindex(self, *, full: bool) -> None:
        if not VAULT_PATHS:
            self._json_response(400, {"ok": False, "message": "No vault configured"})
            return

        # Acquired here, on the request-handling thread, so we can respond
        # 409 synchronously; released inside the background worker thread
        # once indexing finishes — see reindex_lock()'s docstring for why
        # acquire and release deliberately happen on different threads.
        lock_cm = reindex_lock()
        acquired = lock_cm.__enter__()
        if not acquired:
            lock_cm.__exit__(None, None, None)
            log.info("dashboard: reindex (full=%s) rejected — already in progress", full)
            self._busy_response()
            return

        log.info("dashboard: reindex (full=%s) started for %d vault(s)", full, len(VAULT_PATHS))

        def _run():
            try:
                if full:
                    rebuilt = rebuild_index_safely(VAULT_PATHS)
                    log.info("dashboard: safely rebuilt %d notes", rebuilt)
                else:
                    for vp in VAULT_PATHS:
                        index_vault(vp)
                log.info("dashboard: reindex (full=%s) finished", full)
            except Exception:
                log.exception("dashboard: reindex (full=%s) failed", full)
            finally:
                lock_cm.__exit__(None, None, None)

        threading.Thread(target=_run, daemon=True).start()
        self._json_response(200, {"ok": True, "message": "started"})

    def _handle_prune(self) -> None:
        try:
            n = prune_orphans()
            log.info("dashboard: /api/prune deleted %d orphaned note(s)", n)
            self._json_response(200, {"ok": True, "deleted": n})
        except Exception as e:
            log.error("dashboard: /api/prune failed: %s", e)
            self._json_response(500, {"ok": False, "message": str(e)})

    def log_message(self, format, *args):
        # Suppress default request logging
        pass


class DashboardServer(http.server.ThreadingHTTPServer):
    """Serve independent dashboard clients without one slow request blocking all."""

    daemon_threads = True


def serve_dashboard() -> None:
    """Serve the dashboard until its private local control socket is stopped."""
    server = DashboardServer((DASHBOARD_BIND, DASH_PORT), DashboardHandler)
    control_path = _run_directory() / "osm-dashboard.sock"
    if control_path.exists():
        details = control_path.lstat()
        if not stat.S_ISSOCK(details.st_mode) or details.st_uid != os.geteuid():
            raise ValueError("unsafe dashboard control socket")
        try:
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as existing:
                existing.settimeout(1)
                existing.connect(str(control_path))
                raise RuntimeError("dashboard is already running")
        except ConnectionRefusedError, FileNotFoundError:
            pass
        control_path.unlink()
    control = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        control.bind(str(control_path))
        control_path.chmod(0o600)
        control.listen(1)
        control.settimeout(0.5)

        def control_loop() -> None:
            while True:
                try:
                    connection, _ = control.accept()
                except TimeoutError:
                    continue
                except OSError:
                    return
                with connection:
                    command = connection.recv(32)
                    if command == b"OSM_DASHBOARD_PING\n":
                        connection.sendall(f"PONG {DASH_PORT}\n".encode("ascii"))
                    elif command == b"OSM_DASHBOARD_STOP\n":
                        connection.sendall(b"OK\n")
                        server.shutdown()
                        return

        threading.Thread(target=control_loop, daemon=True, name="osm-dashboard-stop").start()
        print(f"Dashboard running at http://{DASHBOARD_BIND}:{DASH_PORT}", flush=True)
        server.serve_forever()
    finally:
        control.close()
        control_path.unlink(missing_ok=True)
        server.server_close()


if __name__ == "__main__":
    serve_dashboard()
