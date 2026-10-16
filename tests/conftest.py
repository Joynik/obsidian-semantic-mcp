"""Shared test fixtures, including an opt-in isolated PostgreSQL database."""

import os
import re
import sys
from pathlib import Path

import pytest

# Keep imports from loading a developer's checkout .env or contacting live data.
# Live PostgreSQL tests opt in separately through PYTEST_DATABASE_URL.
os.environ["OSM_DATABASE_BACKEND"] = "postgresql"
os.environ["DATABASE_URL"] = "postgresql://127.0.0.1:1/osm_pytest_test"
os.environ["OBSIDIAN_VAULT"] = "/nonexistent/osm-pytest-vault"
os.environ["DASHBOARD_PORT"] = "8484"

sys.path.insert(0, str(Path(__file__).parent.parent))

import osm_cli

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from src import config

try:
    from src import dashboard
except Exception:  # pragma: no cover - dashboard needs DB env vars to import
    dashboard = None


@pytest.fixture(autouse=True)
def _isolate_codex_config(monkeypatch, tmp_path):
    """All registration and removal tests use disposable Codex configuration."""
    monkeypatch.setenv("CODEX_HOME", str(tmp_path / "codex"))


@pytest.fixture(autouse=True)
def _isolate_data_dir(monkeypatch, tmp_path):
    """Keep tests isolated from the installed stack's data directory."""
    monkeypatch.setenv("OSM_DATA_DIR", str(tmp_path / "osm-data"))


@pytest.fixture(autouse=True)
def _isolate_pi_agent(monkeypatch):
    """register_pi_agent writes ~/.pi/agent/mcp.json on hosts where ``pi`` is
    installed; stub it so the suite never mutates the developer's real pi config
    and stays deterministic regardless of host state."""
    monkeypatch.setattr(osm_cli, "register_pi_agent", lambda *a, **kw: None)


@pytest.fixture(autouse=True)
def _isolate_osm_config_dir(monkeypatch, tmp_path):
    """Keep runtime settings and dashboard tokens out of the user's config."""
    fake_config = tmp_path / "osm-config"
    fake_config.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(config, "OSM_CONFIG_DIR", fake_config, raising=False)
    monkeypatch.setattr(osm_cli, "OSM_CONFIG_DIR", fake_config, raising=False)
    from src import launcher

    monkeypatch.setattr(launcher, "OSM_CONFIG_DIR", fake_config)
    # Patch paths computed at import time as well as their parent directory.
    monkeypatch.setattr(osm_cli, "PROJECT_ROOT_FILE", fake_config / "project_root", raising=False)
    monkeypatch.setattr(osm_cli, "_OSM_CONFIG_DIR", fake_config)
    monkeypatch.setattr(osm_cli, "_DASHBOARD_TOKEN_FILE", fake_config / "dashboard_token")

    # Force re-resolution against the fake dir rather than reusing whatever a
    # previous test cached.
    if dashboard is not None:
        monkeypatch.setattr(dashboard, "DASHBOARD_TOKEN", None, raising=False)


def _reset():
    """Reset global mutable state in osm_cli between tests."""
    osm_cli.DRY_RUN = False
    osm_cli._DRY_ACTIONS.clear()
    osm_cli._PARAMS.clear()


# ─────────────────────────── `pg` marker fixture ────────────────────────────
#
# Tests marked @pytest.mark.pg need a live PostgreSQL connection. This
# fixture is the ONLY sanctioned way to get one — it is the hard boundary
# between the pg-marked suite and any real database, enforced by
# _require_test_database_name() below.


def pg_dsn() -> str:
    """Resolve the DSN a pg test should connect to — PYTEST_DATABASE_URL ONLY.

    DATABASE_URL is deliberately not consulted, not even as a fallback.
    It is the *production* pointer: on a developer machine it addresses the
    real vault index, and during a test run it is whatever synthetic value
    some other test module last exported. Falling back to it made the pg
    fixture hard-fail on a bare `pytest` (the pre-commit hook's invocation)
    because it picked up another module's placeholder DSN, and would have
    connected to real data on any machine whose database happened to be
    named with a `_test` suffix.

    Opting a database in is therefore explicit and single-purpose: set
    PYTEST_DATABASE_URL. Unset means skip, never guess.
    """
    return os.environ.get("PYTEST_DATABASE_URL", "")


def _dbname_from_dsn(dsn: str) -> str:
    """Extract the target database name from either DSN shape
    config.build_dsn() can produce: a URL
    (postgresql://user:pass@host:port/dbname) or libpq keyword format
    (host=... dbname=... user=... password=...)."""
    if "://" in dsn:
        match = re.search(r"/([^/?]+)(?:\?.*)?$", dsn)
        return match.group(1) if match else ""
    match = re.search(r"\bdbname=(\S+)", dsn)
    return match.group(1) if match else ""


def _require_test_database_name(dsn: str) -> None:
    """Hard-fail unless the target database name ends in `_test`.

    This is the safety guard the plan requires be written and passing
    BEFORE any other pg test runs (test_pg_fixture_refuses_non_test_database
    in tests/test_advisory_lock.py calls this function directly, with no
    real connection attempt, so the guard itself is provable without a DB).
    """
    dbname = _dbname_from_dsn(dsn)
    if not dbname.endswith("_test"):
        raise RuntimeError(
            f"pg fixture refuses to run against database {dbname!r} — the "
            "target database name must end in '_test'. Point "
            "PYTEST_DATABASE_URL at a *_test database before running "
            "`pytest -m pg`."
        )


@pytest.fixture
def pg():
    """Yield a live psycopg2 connection to a `*_test` database.

    Skips (does not fail) when neither PYTEST_DATABASE_URL nor DATABASE_URL
    is set, so `pytest -m "not pg"` — and a bare `pytest` on a machine with
    no database configured at all — both work without this fixture ever
    being instantiated. Raises (does not skip) when a DSN IS configured but
    fails the *_test naming guard: that is a misconfiguration, not "no
    database available," and must be loud.
    """
    import psycopg2  # imported lazily so collection never requires it

    dsn = pg_dsn()
    if not dsn:
        pytest.skip("no PYTEST_DATABASE_URL/DATABASE_URL configured for pg tests")
    _require_test_database_name(dsn)
    conn = psycopg2.connect(dsn)
    # Autocommit so a test that only SELECTs does not sit `idle in
    # transaction` holding ACCESS SHARE on `notes`. Without this, any test
    # whose code-under-test opens its own connection (db_conn(), which is a
    # DIFFERENT connection) and issues ALTER TABLE deadlocks against this
    # fixture's own open read transaction — the suite hung for 10 minutes
    # twice on 2026-07-20 before this was understood.
    #
    # Tests needing explicit transaction control set conn.autocommit = False
    # locally; the `with conn:` blocks in migrations.py manage their own.
    conn.autocommit = True
    try:
        yield conn
    finally:
        try:
            conn.rollback()  # discard anything a test left open
        except Exception:
            pass
        conn.close()
