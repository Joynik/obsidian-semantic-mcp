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
import secrets
import socket
import stat
import subprocess
import threading
import time
from datetime import UTC, datetime, timedelta
from html import escape
from importlib.resources import files
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
DASHBOARD_INSTANCE = secrets.token_hex(8)

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
    ~/.config/osm/ — which anything that imports for
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
        "db_label": "Qdrant" if DATABASE_BACKEND == "qdrant" else "PostgreSQL",
        "vector_label": "Embedding model" if DATABASE_BACKEND == "qdrant" else "pgvector",
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


HTML_PAGE = files("dashboard_ui").joinpath("index.html").read_text(encoding="utf-8")
DASHBOARD_CSS = files("dashboard_ui").joinpath("dashboard.css").read_bytes()
DASHBOARD_JS = files("dashboard_ui").joinpath("dashboard.js").read_bytes()
DASHBOARD_I18N = files("dashboard_ui").joinpath("i18n.js").read_bytes()
HTML_PAGE = HTML_PAGE.replace("{{VERSION}}", f"v{APP_VERSION}")


def _render_page() -> bytes:
    """Return the dashboard HTML with the live bearer token injected."""
    return HTML_PAGE.replace("{{TOKEN}}", escape(get_dashboard_token(), quote=True)).encode()


class DashboardHandler(http.server.BaseHTTPRequestHandler):
    def _asset_response(self, body: bytes, content_type: str) -> None:
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _json_response(self, code: int, data: dict) -> None:
        body = json.dumps(data).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Cache-Control", "no-store")
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
        if parsed.path == "/dashboard.css":
            self._asset_response(DASHBOARD_CSS, "text/css; charset=utf-8")
        elif parsed.path == "/dashboard.js":
            self._asset_response(DASHBOARD_JS, "text/javascript; charset=utf-8")
        elif parsed.path == "/i18n.js":
            self._asset_response(DASHBOARD_I18N, "text/javascript; charset=utf-8")
        elif parsed.path == "/api/health":
            self._json_response(200, {"ok": True, "instance": DASHBOARD_INSTANCE})
        elif parsed.path == "/api/search":
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
            self.send_header("Cache-Control", "no-store")
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
    bound_port = server.server_port
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
                        connection.sendall(f"PONG {bound_port}\n".encode("ascii"))
                    elif command == b"OSM_DASHBOARD_STOP\n":
                        connection.sendall(b"OK\n")
                        server.shutdown()
                        return

        threading.Thread(target=control_loop, daemon=True, name="osm-dashboard-stop").start()
        print(f"Dashboard running at http://{DASHBOARD_BIND}:{bound_port}", flush=True)
        server.serve_forever()
    finally:
        control.close()
        control_path.unlink(missing_ok=True)
        server.server_close()


if __name__ == "__main__":
    serve_dashboard()
