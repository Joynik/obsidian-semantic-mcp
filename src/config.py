"""
config.py — shared runtime configuration for server.py and dashboard.py.

Centralises DSN construction so both modules read from the same source of truth
and cannot silently diverge.
"""

import os
import re
import secrets
from pathlib import Path

DEFAULT_EMBED_MODEL = "qwen3-embedding:0.6b"
SEARCH_LANGUAGES = ("russian", "english", "simple")


def search_language() -> str:
    """Return the configured PostgreSQL text-search language."""
    language = os.environ.get("OSM_SEARCH_LANGUAGE", "russian").strip().lower()
    if language not in SEARCH_LANGUAGES:
        raise ValueError(f"OSM_SEARCH_LANGUAGE must be one of: {', '.join(SEARCH_LANGUAGES)}")
    return language


# Mandatory frontmatter keys every note written via write_file must carry.
# `None` means "no default value" -- the key still gets added, just empty
# (list types) or left for the caller to fill in (string types). `created`
# and `updated` are computed at write time (see server.py's
# _ensure_frontmatter), not listed here.
REQUIRED_FRONTMATTER_DEFAULTS: dict[str, object] = {
    "aliases": [],
    "tags": [],
    "category": "",
    "session": "",
    "nas-path": "",
    "related": [],
}


def build_dsn() -> str:
    """Build a psycopg2-compatible connection string from environment variables.

    Prefers DATABASE_URL when set (native installs).  Falls back to individual
    POSTGRES_* vars in libpq keyword format so no credential URL ever appears
    in a committed source file.
    """
    if os.environ.get("OSM_DATABASE_BACKEND", "postgresql") == "qdrant":
        return ""
    if url := os.environ.get("DATABASE_URL"):
        return url
    host = os.environ.get("POSTGRES_HOST", "")
    port = os.environ.get("POSTGRES_PORT", "")
    db = os.environ.get("POSTGRES_DB", "")
    user = os.environ.get("POSTGRES_USER", "")
    pw = os.environ.get("POSTGRES_PASSWORD", "")
    if not all((host, port, db, user, pw)):
        raise RuntimeError(
            "Set DATABASE_URL or all POSTGRES_HOST, POSTGRES_PORT, POSTGRES_DB, POSTGRES_USER and POSTGRES_PASSWORD variables"
        )
    return f"host={host} port={port} dbname={db} user={user} password={pw}"


def _redact_dsn(dsn: str) -> str:
    """Return a DSN with only host and dbname visible — never user or password.

    build_dsn() can return either shape:
      - a DATABASE_URL:        postgresql://user:pass@host:port/dbname
      - libpq keyword format:  host=... port=... dbname=... user=... password=...

    Used for startup banners and logs so a copy-pasted terminal output never
    leaks the Postgres credential.
    """
    url_match = re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*://[^@/]*@?([^/:@]+)(?::\d+)?/([^?\s]*)", dsn)
    if url_match:
        host, dbname = url_match.group(1), url_match.group(2)
        return f"host={host} dbname={dbname}"

    host_match = re.search(r"\bhost=(\S+)", dsn)
    db_match = re.search(r"\bdbname=(\S+)", dsn)
    host = host_match.group(1) if host_match else "?"
    dbname = db_match.group(1) if db_match else "?"
    return f"host={host} dbname={dbname}"


# Advisory-lock key shared by src/server.py and src/dashboard.py so re-index
# mutual exclusion resolves to one Postgres advisory lock no
# matter which module takes it. Any fixed 64-bit int works — it is
# namespaced by the database, not global to the Postgres instance.
REINDEX_LOCK_KEY = 8474927


# Module-level (not function-local) so tests can monkeypatch it to a
# tmp_path and guarantee no test ever reads or writes the real config dir.
# Read at CALL time inside resolve_dashboard_token(), never cached in a
# module-level constant computed once at import — that would make the
# monkeypatch a no-op for any test that runs after the first import.
OSM_CONFIG_DIR = Path.home() / ".config" / "osm"
DASHBOARD_TOKEN_FILE_NAME = "dashboard_token"


def resolve_dashboard_token() -> str:
    """Resolve the bearer token that guards the dashboard's mutating endpoints.

    Precedence:
      1. DASHBOARD_TOKEN env var, if set — always wins, nothing is written.
         This option is useful for a process with an explicit token override.
      2. ~/.config/osm/dashboard_token, if it exists —
         read and reused so the token survives restarts.
      3. Otherwise generate one with secrets.token_urlsafe(32), persist it
         at that path with mode 0600, and return it.

    Never written under the repo root; persistent config lives in
    ~/.config/osm/.
    """
    if env_token := os.environ.get("DASHBOARD_TOKEN"):
        return env_token

    token_file = OSM_CONFIG_DIR / DASHBOARD_TOKEN_FILE_NAME
    if token_file.exists():
        try:
            existing = token_file.read_text(encoding="utf-8").strip()
            if existing:
                return existing
        except OSError:
            pass  # unreadable (permissions, race) — fall through and regenerate

    legacy_file = OSM_CONFIG_DIR.parent / "obsidian-semantic-mcp" / DASHBOARD_TOKEN_FILE_NAME
    if legacy_file.exists():
        try:
            existing = legacy_file.read_text(encoding="utf-8").strip()
            if existing:
                OSM_CONFIG_DIR.mkdir(mode=0o700, parents=True, exist_ok=True)
                token_file.write_text(existing, encoding="utf-8")
                token_file.chmod(0o600)
                return existing
        except OSError:
            pass

    token = secrets.token_urlsafe(32)
    try:
        OSM_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        token_file.write_text(token, encoding="utf-8")
        token_file.chmod(0o600)
    except OSError:
        pass  # can't persist — still return a usable in-memory token this run
    return token
