"""Native launcher configuration and startup tests."""

import json
import os
import threading
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
    settings = launcher.OSM_CONFIG_DIR / "native_runtime.json"
    assert settings.stat().st_mode & 0o777 == 0o600
    assert "OSM_DOCKER" not in json.loads(settings.read_text())["env"]


def test_obsolete_runtime_key_is_rejected(tmp_path):
    launcher.write_native_runtime([str(tmp_path / "vault")], "postgresql://localhost/osm")
    path = launcher.OSM_CONFIG_DIR / "native_runtime.json"
    saved = json.loads(path.read_text())
    saved["env"]["OSM_DOCKER"] = "0"
    path.write_text(json.dumps(saved))
    with pytest.raises(ValueError, match="invalid native runtime variables"):
        launcher._read_native_runtime(path)


def test_archive_choice_survives_runtime_reload(tmp_path):
    vault = tmp_path / "vault"
    vault.mkdir()
    launcher.write_native_runtime([str(vault)], "postgresql://localhost/osm", index_archive=True)
    saved = launcher._read_native_runtime(launcher.OSM_CONFIG_DIR / "native_runtime.json")
    assert saved["OBSIDIAN_INDEX_ARCHIVE"] == "yes"


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
    assert not (launcher.OSM_CONFIG_DIR / "native_runtime.json").exists()


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
