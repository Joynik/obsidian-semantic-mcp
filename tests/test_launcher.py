"""Native launcher configuration and startup tests."""

import json
import os
import select
import subprocess
import sys
import threading
import time
from pathlib import Path
from unittest.mock import patch

import pytest

from src import launcher


@pytest.fixture(autouse=True)
def isolated_settings(monkeypatch, tmp_path):
    monkeypatch.setattr(launcher, "OSM_CONFIG_DIR", tmp_path / "config")
    for name in (
        "OBSIDIAN_VAULT",
        "OBSIDIAN_VAULTS",
        "DATABASE_URL",
        "POSTGRES_PASSWORD",
        "OLLAMA_URL",
        "EMBEDDING_MODEL",
    ):
        monkeypatch.delenv(name, raising=False)


def test_native_settings_start_server(tmp_path):
    launcher.write_native_runtime(
        [str(tmp_path / "vault")],
        "postgresql://localhost/osm",
        ollama_url="http://localhost:11434",
        embedding_model="nomic-embed-text-v2-moe",
    )
    with patch.object(launcher, "_run_server") as server:
        launcher.main()
    server.assert_called_once()
    assert os.environ["EMBEDDING_MODEL"] == "nomic-embed-text-v2-moe"
    assert os.environ["OLLAMA_URL"] == "http://localhost:11434"
    settings = launcher.OSM_CONFIG_DIR / "runtime.json"
    assert settings.stat().st_mode & 0o777 == 0o600
    assert "OSM_DOCKER" not in json.loads(settings.read_text())["env"]


def test_legacy_runtime_is_copied_without_changing_original(tmp_path):
    legacy_dir = tmp_path / "obsidian-semantic-mcp"
    launcher.write_native_runtime(
        [str(tmp_path / "vault")],
        "postgresql://localhost/osm",
        config_dir=legacy_dir,
        ollama_url="http://localhost:11434",
    )
    legacy = legacy_dir / "native_runtime.json"
    (legacy_dir / "runtime.json").rename(legacy)
    original = legacy.read_bytes()

    loaded = launcher.load_native_runtime()

    current = launcher.OSM_CONFIG_DIR / "runtime.json"
    assert loaded["OBSIDIAN_VAULT"] == str(tmp_path / "vault")
    assert current.stat().st_mode & 0o777 == 0o600
    assert legacy.read_bytes() == original
    assert json.loads(current.read_text())["env"] == json.loads(original)["env"]


def test_current_runtime_takes_precedence_over_legacy(tmp_path):
    legacy_dir = tmp_path / "obsidian-semantic-mcp"
    launcher.write_native_runtime(
        [str(tmp_path / "old")], "postgresql://localhost/old", config_dir=legacy_dir
    )
    (legacy_dir / "runtime.json").rename(legacy_dir / "native_runtime.json")
    launcher.write_native_runtime([str(tmp_path / "new")], "postgresql://localhost/new")

    loaded = launcher.load_native_runtime()

    assert loaded["OBSIDIAN_VAULT"] == str(tmp_path / "new")
    assert (
        json.loads((launcher.OSM_CONFIG_DIR / "runtime.json").read_text())["env"]["DATABASE_URL"]
        == "postgresql://localhost/new"
    )


def test_obsolete_runtime_key_is_rejected(tmp_path):
    launcher.write_native_runtime([str(tmp_path / "vault")], "postgresql://localhost/osm")
    path = launcher.OSM_CONFIG_DIR / "runtime.json"
    saved = json.loads(path.read_text())
    saved["env"]["OSM_DOCKER"] = "0"
    path.write_text(json.dumps(saved))
    with pytest.raises(ValueError, match="invalid native runtime variables"):
        launcher._read_native_runtime(path)


def test_archive_choice_survives_runtime_reload(tmp_path):
    vault = tmp_path / "vault"
    vault.mkdir()
    launcher.write_native_runtime([str(vault)], "postgresql://localhost/osm", index_archive=True)
    saved = launcher._read_native_runtime(launcher.OSM_CONFIG_DIR / "runtime.json")
    assert saved["OBSIDIAN_INDEX_ARCHIVE"] == "yes"


def test_archive_folder_survives_runtime_reload(tmp_path):
    vault = tmp_path / "vault"
    vault.mkdir()
    launcher.write_native_runtime(
        [str(vault)], "postgresql://localhost/osm", index_archive=False, archive_folder="Old Notes"
    )
    saved = launcher._read_native_runtime(launcher.OSM_CONFIG_DIR / "runtime.json")
    assert saved["OBSIDIAN_ARCHIVE_FOLDER"] == "Old Notes"


@pytest.mark.parametrize("name", ["", ".", "..", "Parent/Archive", "Parent\\Archive"])
def test_invalid_archive_folder_is_rejected(tmp_path, name):
    with pytest.raises(ValueError):
        launcher.write_native_runtime(
            [str(tmp_path / "vault")], "postgresql://localhost/osm", archive_folder=name
        )


def test_missing_configuration_exits(monkeypatch):
    monkeypatch.setattr(launcher, "_load_checkout_env", lambda: None)
    with pytest.raises(SystemExit) as error:
        launcher.main()
    assert error.value.code == 1


def test_invalid_model_setting_rejected(tmp_path):
    with pytest.raises(ValueError):
        launcher.write_native_runtime(
            [str(tmp_path / "vault")], "postgresql://localhost/osm", embedding_model="bad\nvalue"
        )


def test_runtime_removal_keeps_other_files(tmp_path):
    launcher.write_native_runtime([str(tmp_path / "vault")], "postgresql://localhost/osm")
    unrelated = launcher.OSM_CONFIG_DIR / "unrelated.json"
    unrelated.write_text("{}")
    launcher.remove_native_runtime()
    assert unrelated.exists()
    assert not (launcher.OSM_CONFIG_DIR / "runtime.json").exists()


def test_stop_runs_signals_only_registered_server(monkeypatch):
    signals = []
    signaled = threading.Event()

    def fake_kill(pid, sig):
        signals.append((pid, sig))
        signaled.set()

    monkeypatch.setattr(launcher.os, "kill", fake_kill)
    listener, path = launcher._register_run()
    try:
        assert launcher.stop_runs() == (1, 0)
        assert signaled.wait(1)
        assert len(signals) == 1
    finally:
        listener.close()
        path.unlink(missing_ok=True)


def test_stop_runs_exits_server_without_event_loop_traceback(monkeypatch, tmp_path):
    monkeypatch.setattr(launcher, "OSM_CONFIG_DIR", tmp_path / ".config" / "osm")
    source = """
import asyncio
import os
from pathlib import Path
from src import server
server.VAULT_PATHS = ['/tmp/osm-stop-test']
server.background_init = lambda *_: None
async def wait_forever(*_):
    Path(os.environ['OSM_STOP_READY']).touch()
    await asyncio.Event().wait()
server.server.run = wait_forever
from src.launcher import main
main()
"""
    ready = tmp_path / "server-ready"
    env = os.environ.copy()
    env.update(
        HOME=str(tmp_path),
        OBSIDIAN_VAULT="/tmp/osm-stop-test",
        OSM_DATABASE_BACKEND="qdrant",
        QDRANT_URL="http://127.0.0.1:6333",
        OSM_STOP_READY=str(ready),
    )
    process = subprocess.Popen(
        [sys.executable, "-c", source],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
    )
    try:
        deadline = time.monotonic() + 5
        while not ready.exists() and process.poll() is None:
            if time.monotonic() >= deadline:
                break
            time.sleep(0.05)
        assert ready.exists()
        assert launcher.stop_runs()[0] == 1
        _, stderr = process.communicate(timeout=5)
        assert process.returncode == 0, stderr.decode()
        assert b"Event loop stopped before Future completed" not in stderr
    finally:
        if process.poll() is None:
            process.kill()
            process.communicate()


def test_stop_runs_exits_real_server_with_open_stdin(monkeypatch, tmp_path):
    config_dir = tmp_path / ".config" / "osm"
    monkeypatch.setattr(launcher, "OSM_CONFIG_DIR", config_dir)
    vault = tmp_path / "vault"
    vault.mkdir()
    (vault / "note.md").write_text("# Note\n")
    env = os.environ.copy()
    env.update(
        HOME=str(tmp_path),
        OBSIDIAN_VAULT=str(vault),
        OSM_DATABASE_BACKEND="qdrant",
        QDRANT_URL="http://127.0.0.1:1",
        OLLAMA_URL="http://127.0.0.1:1",
        PYTHONUNBUFFERED="1",
    )
    process = subprocess.Popen(
        [sys.executable, "-m", "osm_cli", "run"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
        cwd=Path(__file__).resolve().parent.parent,
    )
    try:
        assert process.stdin is not None and process.stdout is not None
        process.stdin.write(
            (
                json.dumps(
                    {
                        "jsonrpc": "2.0",
                        "id": 1,
                        "method": "initialize",
                        "params": {
                            "protocolVersion": "2025-06-18",
                            "capabilities": {},
                            "clientInfo": {"name": "stop-test", "version": "1"},
                        },
                    }
                )
                + "\n"
            ).encode()
        )
        process.stdin.flush()
        ready, _, _ = select.select([process.stdout], [], [], 10)
        assert ready, "MCP initialize did not respond with stdin open"
        assert json.loads(process.stdout.readline())["id"] == 1

        deadline = time.monotonic() + 5
        while not list(launcher._run_path().glob("osm-run-*.sock")):
            assert time.monotonic() < deadline, "OSM run socket did not appear"
            time.sleep(0.05)
        assert launcher.stop_runs()[0] == 1
        assert process.wait(timeout=5) == 0
        assert process.stderr is not None
        assert b"Event loop stopped before Future completed" not in process.stderr.read()
    finally:
        if process.poll() is None:
            process.kill()
            process.communicate()
