"""tests/test_dashboard_security.py — Security & Correctness plan, iterations 1-2.

Iteration 1: the dashboard binds 127.0.0.1 by default and never prints the
Postgres password to stdout.
Iteration 2: every mutating dashboard endpoint requires a bearer token.

Side-effect fence: no test in this file may read or write the real
`~/.config/osm/` directory. Tests that exercise file-based token persistence
monkeypatch `config.OSM_CONFIG_DIR` to a `tmp_path` and explicitly unset
`DASHBOARD_TOKEN` for the duration of that test only.
"""

from __future__ import annotations

import http.server
import importlib
import os
import select
import re
import signal
import shutil
import subprocess
import sys
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import requests

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Minimal env so server.py / dashboard.py import without crashing.
os.environ.setdefault("OBSIDIAN_VAULT", "/tmp/test_vault")
os.environ.setdefault("DATABASE_URL", "postgresql://localhost/test")
os.environ.setdefault("OLLAMA_URL", "http://localhost:11434")
os.environ.setdefault("DASHBOARD_TOKEN", "test-fixture-token-not-a-secret")

from src import config  # noqa: E402
from src import dashboard  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
DASHBOARD_SRC = REPO_ROOT / "src" / "dashboard.py"
ENV_EXAMPLE = REPO_ROOT / ".env.example"


def _reload_dashboard():
    """Reload dashboard.py so its module-level constants re-read the
    current env (DASHBOARD_BIND, DASHBOARD_TOKEN, DATABASE_URL, ...)."""
    return importlib.reload(dashboard)


def _startup_error(proc: subprocess.Popen[str]) -> str:
    if proc.poll() is None:
        proc.send_signal(signal.SIGABRT)
    output, _ = proc.communicate(timeout=5)
    return f"process exit={proc.returncode}; output={output[-6000:]}"


@pytest.fixture(autouse=True)
def _restore_dashboard_token_env(monkeypatch):
    """Every test in this file gets the safe default token env, even if a
    prior test in the same session deleted it — belt-and-suspenders on top
    of the module-level setdefault, since monkeypatch.delenv in one test
    only reverts at that test's teardown, not before the next test runs."""
    monkeypatch.setenv("DASHBOARD_TOKEN", "test-fixture-token-not-a-secret")


# ─────────────────────────────── Iteration 1 ────────────────────────────────


def test_smoke_dashboard_module_imports_cleanly():
    """Smoke: importing dashboard with DATABASE_URL set does not raise."""
    mod = _reload_dashboard()
    assert mod is dashboard


def test_dashboard_assets_load_from_homebrew_install_layout(tmp_path):
    """Homebrew copies src into libexec; dashboard assets must travel with it."""
    libexec = tmp_path / "libexec"
    shutil.copytree(
        REPO_ROOT / "src", libexec / "src", ignore=shutil.ignore_patterns("__pycache__")
    )
    env = {
        **os.environ,
        "PYTHONPATH": str(libexec),
        "HOME": str(tmp_path),
        "OBSIDIAN_VAULT": str(tmp_path),
        "DASHBOARD_PORT": "0",
        "DASHBOARD_TOKEN": "test-fixture-token-not-a-secret",
    }
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from src.dashboard import HTML_PAGE; assert 'Index health' in HTML_PAGE",
        ],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_default_bind_is_loopback(monkeypatch):
    monkeypatch.delenv("DASHBOARD_BIND", raising=False)
    mod = _reload_dashboard()
    assert mod.DASHBOARD_BIND == "127.0.0.1"


def test_bind_cannot_be_overridden_to_wildcard(monkeypatch):
    monkeypatch.setenv("DASHBOARD_BIND", "0.0.0.0")
    mod = _reload_dashboard()
    assert mod.DASHBOARD_BIND == "127.0.0.1"


def test_startup_banner_does_not_leak_password(tmp_path):
    """The startup banner must redact the PostgreSQL password."""
    secret = "s3cr3t-p4ssw0rd-marker"
    env = {
        **os.environ,
        "HOME": str(tmp_path),
        "DATABASE_URL": f"postgresql://obsidian:{secret}@localhost:5432/obsidian_brain",
        "OBSIDIAN_VAULT": "/tmp/test_vault",
        "DASHBOARD_PORT": "0",  # OS picks a free ephemeral port — never binds a fixed one
        "DASHBOARD_TOKEN": "test-fixture-token-not-a-secret",
        "PYTHONUNBUFFERED": "1",
        "PYTHONFAULTHANDLER": "1",
    }
    proc = subprocess.Popen(
        [sys.executable, "-u", "-m", "src.dashboard"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
    )
    try:
        assert proc.stdout is not None
        ready, _, _ = select.select([proc.stdout], [], [], 20)
        assert ready, _startup_error(proc) if not ready else ""
        banner = proc.stdout.readline()
        assert secret not in banner, f"password leaked in startup banner: {banner!r}"
        assert "127.0.0.1" in banner, f"expected loopback bind in banner: {banner!r}"
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


@pytest.mark.parametrize("module_args", [["src.dashboard"], ["src.workers", "dashboard"]])
def test_dashboard_stop_closes_listening_port(tmp_path, module_args):
    env = {
        **os.environ,
        "HOME": str(tmp_path),
        "OBSIDIAN_VAULT": "/tmp/test_vault",
        "DASHBOARD_PORT": "0",
        "DASHBOARD_TOKEN": "test-fixture-token-not-a-secret",
        "PYTHONUNBUFFERED": "1",
        "PYTHONFAULTHANDLER": "1",
    }
    proc = subprocess.Popen(
        [sys.executable, "-u", "-m", *module_args],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
    )
    try:
        assert proc.stdout is not None
        ready, _, _ = select.select([proc.stdout], [], [], 20)
        assert ready, _startup_error(proc) if not ready else ""
        banner = proc.stdout.readline()
        assert "Dashboard running at http://127.0.0.1:" in banner, banner
        port = int(banner.rsplit(":", 1)[1])
        page = requests.get(f"http://127.0.0.1:{port}/", timeout=2)
        assert page.status_code == 200
        assert page.headers["Cache-Control"] == "no-store"
        assert '<script src="/dashboard.js"></script>' in page.text
        assert '<script src="/i18n.js"></script>' in page.text
        assert '<link rel="stylesheet" href="/dashboard.css">' in page.text
        for asset, content_type in (
            ("dashboard.css", "text/css"),
            ("dashboard.js", "text/javascript"),
            ("i18n.js", "text/javascript"),
        ):
            response = requests.get(f"http://127.0.0.1:{port}/{asset}", timeout=2)
            assert response.status_code == 200
            assert response.headers["Content-Type"].startswith(content_type)
            assert response.headers["Cache-Control"] == "no-store"
            assert "{{TOKEN}}" not in response.text
        health = requests.get(f"http://127.0.0.1:{port}/api/health", timeout=2)
        assert health.json()["ok"] is True
        assert len(health.json()["instance"]) == 16
        assert health.headers["Cache-Control"] == "no-store"

        stopped = subprocess.run(
            [sys.executable, "-m", "osm_cli", "dashboard", "stop"],
            env=env,
            capture_output=True,
            text=True,
            timeout=8,
            check=False,
        )
        assert stopped.returncode == 0, stopped.stdout + stopped.stderr
        assert f"port {port} is closed" in stopped.stdout
        proc.wait(timeout=2)
        with pytest.raises(requests.ConnectionError):
            requests.get(f"http://127.0.0.1:{port}/", timeout=1)
    finally:
        if proc.poll() is None:
            proc.terminate()
            proc.wait(timeout=5)


# ─────────────────────────────── Iteration 2 ────────────────────────────────
#
# A real DashboardHandler bound to an ephemeral 127.0.0.1 port, per-test. This
# is a genuine HTTP round trip (not a mocked handler) so a broken auth gate
# would actually be exercised, not assumed. VAULT_PATHS is left empty by
# default so /api/reindex's authenticated-but-unconfigured path returns 400
# without ever touching a DB connection; individual tests override it.

TEST_TOKEN = "iteration2-test-token-not-a-secret"


class _FakeFreeLockCM:
    """Let HTTP authentication tests run without a database lock."""

    def __enter__(self):
        return True

    def __exit__(self, *exc_info):
        return False


@contextmanager
def _running_dashboard(monkeypatch, token: str = TEST_TOKEN):
    monkeypatch.setattr(dashboard, "DASHBOARD_TOKEN", token)
    monkeypatch.setattr(dashboard, "reindex_lock", lambda: _FakeFreeLockCM())
    monkeypatch.setattr(dashboard, "VAULT_PATHS", [])
    httpd = http.server.HTTPServer(("127.0.0.1", 0), dashboard.DashboardHandler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{httpd.server_port}"
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=5)


def test_slow_request_does_not_block_a_second_dashboard_request(monkeypatch):
    """The dashboard must serve health/UI traffic while another client is slow."""
    monkeypatch.setattr(dashboard, "DASHBOARD_TOKEN", TEST_TOKEN)
    started = threading.Event()
    release = threading.Event()

    class SlowClientHandler(dashboard.DashboardHandler):
        def do_GET(self):
            if self.path == "/slow-client":
                started.set()
                assert release.wait(timeout=5)
                self.send_response(200)
                self.end_headers()
                return
            super().do_GET()

    httpd = dashboard.DashboardServer(("127.0.0.1", 0), SlowClientHandler)
    server_thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    server_thread.start()
    base_url = f"http://127.0.0.1:{httpd.server_port}"
    slow_client = threading.Thread(
        target=requests.get, args=(f"{base_url}/slow-client",), kwargs={"timeout": 6}, daemon=True
    )
    slow_client.start()
    try:
        assert started.wait(timeout=1), "slow request never reached the server"
        response = requests.get(f"{base_url}/", timeout=1)
        assert response.status_code == 200
    finally:
        release.set()
        slow_client.join(timeout=5)
        httpd.shutdown()
        httpd.server_close()
        server_thread.join(timeout=5)


def test_post_reindex_without_token_returns_401(monkeypatch):
    with _running_dashboard(monkeypatch) as base_url:
        r = requests.post(f"{base_url}/api/reindex")
        assert r.status_code == 401
        assert r.json() == {"ok": False, "message": "unauthorized"}


def test_unauthorized_request_is_logged(monkeypatch, caplog):
    """Regression for the arch-audit's observability finding: dashboard.py
    had zero structured logging, so a failed auth attempt left no trace.
    An unauthorized request must now produce a log record — this is a
    security-relevant event, not just an HTTP status code."""
    with caplog.at_level("WARNING", logger="dashboard"):
        with _running_dashboard(monkeypatch) as base_url:
            r = requests.post(f"{base_url}/api/reindex")
            assert r.status_code == 401
    assert any("unauthorized" in rec.message for rec in caplog.records)


def test_post_reindex_full_without_token_returns_401(monkeypatch):
    with _running_dashboard(monkeypatch) as base_url:
        r = requests.post(f"{base_url}/api/reindex/full")
        assert r.status_code == 401


def test_post_prune_without_token_returns_401(monkeypatch):
    with _running_dashboard(monkeypatch) as base_url:
        r = requests.post(f"{base_url}/api/prune")
        assert r.status_code == 401


def test_post_ollama_start_without_token_returns_401(monkeypatch):
    with _running_dashboard(monkeypatch) as base_url:
        r = requests.post(f"{base_url}/api/ollama/start")
        assert r.status_code == 401


def test_post_with_valid_token_is_accepted(monkeypatch):
    """A valid token must reach the handler and get the endpoint's
    pre-existing status code (400 here, because VAULT_PATHS is empty) —
    not 401. Proves the gate lets a correct token *through*, which a test
    that only checks "not 401" for garbage input would not prove."""
    with _running_dashboard(monkeypatch) as base_url:
        r = requests.post(
            f"{base_url}/api/reindex",
            headers={"Authorization": f"Bearer {TEST_TOKEN}"},
        )
        assert r.status_code == 400
        assert r.json() == {"ok": False, "message": "No vault configured"}


def test_post_with_wrong_token_returns_401(monkeypatch):
    with _running_dashboard(monkeypatch) as base_url:
        r = requests.post(
            f"{base_url}/api/reindex",
            headers={"Authorization": "Bearer wrong-token-entirely"},
        )
        assert r.status_code == 401


def test_token_comparison_uses_compare_digest():
    """Guards against a later `token == DASHBOARD_TOKEN` regression, which
    would reintroduce a timing side-channel."""
    src = DASHBOARD_SRC.read_text()
    assert "compare_digest" in src


def test_get_endpoints_remain_unauthenticated(monkeypatch):
    """Read-only GET endpoints remain accessible without a bearer token."""
    with _running_dashboard(monkeypatch) as base_url:
        r = requests.get(f"{base_url}/api/reindex/status")
        assert r.status_code == 200
        r = requests.get(f"{base_url}/api/stats")
        assert r.status_code == 200


# ── Token file persistence — config.resolve_dashboard_token() ───────────────
# Every test below monkeypatches config.OSM_CONFIG_DIR to a tmp_path and
# explicitly deletes DASHBOARD_TOKEN from the env for the duration of the
# test. Neither the real ~/.config/osm/ directory nor any
# path under the repo root is ever touched.


def test_token_file_created_with_0600_when_absent(tmp_path, monkeypatch):
    fake_config_dir = tmp_path / ".config" / "osm"
    monkeypatch.setattr(config, "OSM_CONFIG_DIR", fake_config_dir)
    monkeypatch.delenv("DASHBOARD_TOKEN", raising=False)

    token = config.resolve_dashboard_token()

    token_file = fake_config_dir / "dashboard_token"
    assert token_file.exists()
    assert token_file.read_text(encoding="utf-8").strip() == token
    mode = token_file.stat().st_mode & 0o777
    assert mode == 0o600, f"expected mode 0600, got {oct(mode)}"


def test_token_file_is_reused_across_restarts(tmp_path, monkeypatch):
    fake_config_dir = tmp_path / ".config" / "osm"
    monkeypatch.setattr(config, "OSM_CONFIG_DIR", fake_config_dir)
    monkeypatch.delenv("DASHBOARD_TOKEN", raising=False)

    first = config.resolve_dashboard_token()
    second = config.resolve_dashboard_token()
    assert first == second


def test_legacy_dashboard_token_is_copied(tmp_path, monkeypatch):
    fake_config_dir = tmp_path / ".config" / "osm"
    legacy_dir = fake_config_dir.parent / "obsidian-semantic-mcp"
    legacy_dir.mkdir(parents=True)
    legacy_file = legacy_dir / "dashboard_token"
    legacy_file.write_text("existing-token", encoding="utf-8")
    monkeypatch.setattr(config, "OSM_CONFIG_DIR", fake_config_dir)
    monkeypatch.delenv("DASHBOARD_TOKEN", raising=False)

    assert config.resolve_dashboard_token() == "existing-token"
    current_file = fake_config_dir / "dashboard_token"
    assert current_file.read_text() == "existing-token"
    assert current_file.stat().st_mode & 0o777 == 0o600
    assert legacy_file.read_text() == "existing-token"


def test_env_token_overrides_file(tmp_path, monkeypatch):
    fake_config_dir = tmp_path / ".config" / "osm"
    monkeypatch.setattr(config, "OSM_CONFIG_DIR", fake_config_dir)
    monkeypatch.setenv("DASHBOARD_TOKEN", "the-env-token")

    token = config.resolve_dashboard_token()

    assert token == "the-env-token"
    assert not fake_config_dir.exists(), "env override must write nothing to disk"


def test_token_is_never_written_to_repo_root(tmp_path, monkeypatch):
    """Guards the decoupling guardrail directly: no `.env`, `dashboard_token`,
    or similar new file appears under the repo root after token resolution."""
    fake_config_dir = tmp_path / ".config" / "osm"
    monkeypatch.setattr(config, "OSM_CONFIG_DIR", fake_config_dir)
    monkeypatch.delenv("DASHBOARD_TOKEN", raising=False)

    before = set(REPO_ROOT.iterdir())
    config.resolve_dashboard_token()
    after = set(REPO_ROOT.iterdir())

    assert after == before, f"repo root gained files: {after - before}"


def test_token_resolution_precedence(tmp_path, monkeypatch):
    """State machine: env-set -> env wins; env-unset + file-present -> file
    wins; env-unset + file-absent -> generate and persist."""
    fake_config_dir = tmp_path / ".config" / "osm"
    monkeypatch.setattr(config, "OSM_CONFIG_DIR", fake_config_dir)

    # env-unset + file-absent -> generate and persist
    monkeypatch.delenv("DASHBOARD_TOKEN", raising=False)
    generated = config.resolve_dashboard_token()
    token_file = fake_config_dir / "dashboard_token"
    assert token_file.exists()

    # env-unset + file-present -> file wins (same value, no regeneration)
    reread = config.resolve_dashboard_token()
    assert reread == generated

    # env-set -> env wins, even though a file already exists
    monkeypatch.setenv("DASHBOARD_TOKEN", "env-wins-token")
    assert config.resolve_dashboard_token() == "env-wins-token"


def test_unreadable_token_file_falls_back_to_generated(tmp_path, monkeypatch):
    """Chaos: a config dir that exists but whose token file cannot be read
    (permissions, race) must not raise — fall back to a freshly generated
    token rather than crashing the dashboard at import time."""
    fake_config_dir = tmp_path / ".config" / "osm"
    fake_config_dir.mkdir(parents=True)
    token_file = fake_config_dir / "dashboard_token"
    token_file.write_text("unreadable-content", encoding="utf-8")
    token_file.chmod(0o000)
    monkeypatch.setattr(config, "OSM_CONFIG_DIR", fake_config_dir)
    monkeypatch.delenv("DASHBOARD_TOKEN", raising=False)

    try:
        token = config.resolve_dashboard_token()
        assert token  # did not raise, returned something usable
    finally:
        token_file.chmod(0o600)  # restore so tmp_path cleanup can delete it


# ── Integration / contract / regression / chaos / e2e ───────────────────────


def test_ui_script_sends_auth_header_on_every_post():
    """Every `method: 'POST'` fetch in dashboard JS must carry an
    Authorization header nearby in the same fetch() call."""
    mod = _reload_dashboard()
    script = mod.DASHBOARD_JS.decode()
    post_markers = [m.start() for m in re.finditer(r"method:\s*'POST'", script)]
    assert post_markers, "no POST fetch calls found in dashboard JS"
    for pos in post_markers:
        window = script[max(0, pos - 250) : pos + 100]
        assert "Authorization" in window, (
            f"POST fetch near offset {pos} has no Authorization header nearby: {window!r}"
        )


def test_dashboard_token_is_escaped_in_html_and_absent_from_static_js(monkeypatch):
    monkeypatch.setattr(dashboard, "DASHBOARD_TOKEN", 'token"><script>bad()</script>')
    page = dashboard._render_page().decode()
    assert 'content="token&quot;&gt;&lt;script&gt;bad()&lt;/script&gt;"' in page
    assert "<script>bad()</script>" not in page
    assert b'token"><script>bad()' not in dashboard.DASHBOARD_JS


def test_401_body_matches_existing_error_shape(monkeypatch):
    """Contract: the 401 body must carry the same {"ok", "message"} keys the
    other do_POST branches (400/409/500) already use."""
    with _running_dashboard(monkeypatch) as base_url:
        r = requests.post(f"{base_url}/api/reindex")
        assert r.status_code == 401
        body = r.json()
        assert set(body.keys()) == {"ok", "message"}
        assert body["ok"] is False


def test_cross_origin_simple_post_is_rejected(monkeypatch):
    """Reproduces the audit's CSRF scenario directly: a "simple request"
    cross-origin POST (no Authorization header, Content-Type: text/plain —
    the shape a no-preflight cross-origin fetch/form-submit produces) must
    be rejected."""
    with _running_dashboard(monkeypatch) as base_url:
        r = requests.post(
            f"{base_url}/api/reindex",
            headers={"Content-Type": "text/plain"},
            data="whatever a forged cross-origin request sends",
        )
        assert r.status_code == 401


@pytest.mark.parametrize(
    "header_value",
    ["Bearer", "Bearer ", "Basic xyz", "Bearer " + ("x" * 10_000)],
)
def test_malformed_authorization_header_returns_401(monkeypatch, header_value):
    with _running_dashboard(monkeypatch) as base_url:
        r = requests.post(
            f"{base_url}/api/reindex",
            headers={"Authorization": header_value},
        )
        assert r.status_code == 401


def test_reindex_roundtrip_with_token(monkeypatch):
    """E2E: a valid-token POST to /api/reindex with a configured (stubbed)
    vault returns 200 and actually invokes index_vault exactly once."""
    stub_index_vault = MagicMock()
    monkeypatch.setattr(dashboard, "index_vault", stub_index_vault)

    with _running_dashboard(monkeypatch) as base_url:
        # _running_dashboard defaults VAULT_PATHS to [] — override for this test.
        monkeypatch.setattr(dashboard, "VAULT_PATHS", ["/fake/vault"])
        r = requests.post(
            f"{base_url}/api/reindex",
            headers={"Authorization": f"Bearer {TEST_TOKEN}"},
        )
        assert r.status_code == 200
        assert r.json() == {"ok": True, "message": "started"}

        deadline = time.time() + 5
        while time.time() < deadline and stub_index_vault.call_count == 0:
            time.sleep(0.05)
        assert stub_index_vault.call_count == 1
        stub_index_vault.assert_called_once_with("/fake/vault")
