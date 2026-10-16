"""Verify MCP initialize responds while the client keeps stdin open."""

from __future__ import annotations

import json
import os
import select
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent


def _initialize_request() -> bytes:
    return (
        json.dumps(
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2025-06-18",
                    "capabilities": {},
                    "clientInfo": {"name": "test", "version": "1.0"},
                },
            }
        )
        + "\n"
    ).encode()


def _have_postgres() -> bool:
    """Quick check that postgres is reachable. Skip the test if not."""
    if os.environ.get("DATABASE_URL"):
        return True
    if os.environ.get("POSTGRES_PASSWORD"):
        return True
    return False


@pytest.mark.pg
def test_initialize_responds_over_anonymous_pipe(tmp_path):
    """Reproduce the May 8 stdin hang: spawn server with pipe stdin/stdout,
    write `initialize`, do NOT close stdin, expect response within 20s.

    This test is the canonical RED case for the bug. It will fail until
    the stdin reader is converted to a non-blocking thread offload.

    Marked `pg`: spawns a real server subprocess that opens a live Postgres
    connection pool. The `_have_postgres()` liveness probe itself makes a
    TCP connection, so it must never run during collection or during a
    `-m "not pg"` run — it is called here, inside the test body, only when
    this test is actually selected to execute.
    """
    if not _have_postgres():
        pytest.skip("postgres not reachable; integration test requires a DB")

    vault = tmp_path / "vault"
    vault.mkdir()
    (vault / "hello.md").write_text("# hello\n")

    env = os.environ.copy()
    env["OBSIDIAN_VAULT"] = str(vault)
    env.setdefault("POSTGRES_PASSWORD", "obsidian_brain")
    env.setdefault(
        "DATABASE_URL",
        "postgresql://obsidian_brain:obsidian_brain@127.0.0.1:5433/obsidian_brain",
    )
    # The child resolves its install pointer and .env from HOME.
    home = tmp_path / "home"
    home.mkdir()
    env["HOME"] = str(home)
    env["PYTHONUNBUFFERED"] = "1"

    proc = subprocess.Popen(
        [sys.executable, "-m", "src.launcher"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=0,
        env=env,
        cwd=REPO_ROOT,
    )
    try:
        assert proc.stdin is not None and proc.stdout is not None and proc.stderr is not None
        proc.stdin.write(_initialize_request())
        proc.stdin.flush()
        # CRITICAL: do NOT close stdin. Claude Code keeps the pipe open
        # for the lifetime of the connection; the bug only manifests
        # when stdin stays open.

        ready, _, _ = select.select([proc.stdout], [], [], 20.0)
        if not ready:
            stderr_ready, _, _ = select.select([proc.stderr], [], [], 0)
            stderr_tail = (os.read(proc.stderr.fileno(), 4000) if stderr_ready else b"").decode(
                errors="replace"
            )
            pytest.fail(
                "no response on stdout within 20s — stdin hang regression. "
                f"stderr tail:\n{stderr_tail[-1500:]}"
            )

        line = proc.stdout.readline()
        assert line, "stdout closed without responding"

        response = json.loads(line)
        assert response.get("jsonrpc") == "2.0"
        assert response.get("id") == 1
        assert "result" in response, f"expected result, got {response}"
        result = response["result"]
        assert "serverInfo" in result
        assert "obsidian-semantic" in result["serverInfo"]["name"].lower()
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
