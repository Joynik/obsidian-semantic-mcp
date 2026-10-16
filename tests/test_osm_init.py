"""Native setup flag and dispatch tests."""

import pytest

import osm_cli


@pytest.fixture(autouse=True)
def reset_flags(monkeypatch):
    osm_cli._PARAMS.clear()
    monkeypatch.setattr(osm_cli, "DRY_RUN", False)


def test_native_setup_flags():
    rest, params = osm_cli._parse_flags(
        [
            "init",
            "--vault",
            "/tmp/notes",
            "--database-url",
            "postgresql://localhost/osm",
            "--ollama-url",
            "http://localhost:11434",
            "--embedding-model",
            "nomic-embed-text-v2-moe",
        ]
    )
    assert rest == ["init"]
    assert params["vault"] == "/tmp/notes"
    assert params["embedding_model"] == "nomic-embed-text-v2-moe"


def test_setup_rejects_unsupported_platform(monkeypatch):
    monkeypatch.setattr(osm_cli.platform, "system", lambda: "Linux")
    with pytest.raises(SystemExit):
        osm_cli.cmd_init()


def test_setup_dispatches_native_on_macos(monkeypatch):
    monkeypatch.setattr(osm_cli.platform, "system", lambda: "Darwin")
    called = []
    monkeypatch.setattr(osm_cli, "mode_native_macos", lambda: called.append("native"))
    osm_cli.cmd_init()
    assert called == ["native"]
