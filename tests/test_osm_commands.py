"""Native OSM command behavior."""

import json
import socket
from unittest.mock import MagicMock

import pytest

import osm_cli


@pytest.fixture(autouse=True)
def reset_cli(monkeypatch, tmp_path):
    monkeypatch.setattr(osm_cli, "OSM_CONFIG_DIR", tmp_path / "config")
    monkeypatch.setattr(osm_cli, "DRY_RUN", False)
    osm_cli._PARAMS.clear()


def test_only_native_commands_are_exposed():
    assert "init" in osm_cli.COMMANDS
    assert "status" in osm_cli.COMMANDS
    assert "version" in osm_cli.COMMANDS
    assert "rebuild" not in osm_cli.COMMANDS
    assert "tunnel" not in osm_cli.COMMANDS


@pytest.mark.parametrize("archive_name", ["Archive", "Old Notes"])
def test_native_setup_checks_services_before_writing(monkeypatch, tmp_path, archive_name):
    monkeypatch.setattr(osm_cli, "prompt_vault", lambda: [str(tmp_path)])
    monkeypatch.setattr(
        osm_cli,
        "prompt",
        lambda question, default=None, **kw: (
            archive_name if question == "Archive folder name to exclude" else default
        ),
    )
    monkeypatch.setattr(
        osm_cli,
        "_read_env",
        lambda: {
            "OSM_DATABASE_BACKEND": "postgresql",
            "DATABASE_URL": "postgresql://127.0.0.1:1/osm_pytest_test",
            "OLLAMA_URL": "http://127.0.0.1:11434",
            "DASHBOARD_PORT": "8484",
        },
    )
    monkeypatch.setattr(osm_cli, "_verify_ollama_model", lambda *_: True)
    monkeypatch.setattr(osm_cli.requests, "get", lambda *a, **k: MagicMock(status_code=200))
    monkeypatch.setattr("src.postgres.inspect_setup", lambda *_: ("0.8.0", False, None))
    monkeypatch.setattr(osm_cli, "register_with_clients", lambda *_: None)
    writes = []
    monkeypatch.setattr(osm_cli, "_write_native_runtime", lambda *a, **k: writes.append((a, k)))
    osm_cli.mode_native_macos()
    assert len(writes) == 1
    assert writes[0][1]["embedding_model"] == "qwen3-embedding:0.6b"
    assert writes[0][1]["archive_folder"] == archive_name


def test_dashboard_port_check_detects_listener():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        port = listener.getsockname()[1]
        assert not osm_cli._dashboard_port_available(str(port))
    assert osm_cli._dashboard_port_available(str(port))


def test_native_setup_reprompts_for_occupied_dashboard_port(monkeypatch, tmp_path):
    monkeypatch.setattr(osm_cli, "prompt_vault", lambda: [str(tmp_path)])
    monkeypatch.setattr(osm_cli, "confirm", lambda *a, **k: False)
    monkeypatch.setattr(osm_cli, "_read_env", lambda: {"DASHBOARD_PORT": "8484"})
    monkeypatch.setattr(osm_cli, "_dashboard_port_available", lambda port: port != "8484")
    monkeypatch.setattr(osm_cli, "_dashboard_control", lambda *_: None)
    ports = iter(["8484", "8485"])

    def fake_prompt(question, default=None, **kwargs):
        if question == "Dashboard port":
            return next(ports)
        if question == "Database (postgresql/qdrant)":
            return "postgresql"
        return default

    monkeypatch.setattr(osm_cli, "prompt", fake_prompt)
    monkeypatch.setattr("src.postgres.inspect_setup", lambda *_: ("0.8.0", False, None))
    monkeypatch.setattr(osm_cli, "_verify_ollama_model", lambda *_: True)
    monkeypatch.setattr(osm_cli.requests, "get", lambda *a, **k: MagicMock(status_code=200))
    monkeypatch.setattr(osm_cli, "register_with_clients", lambda *_: None)
    writes = []
    monkeypatch.setattr(osm_cli, "_write_native_runtime", lambda *a, **k: writes.append(k))
    osm_cli.mode_native_macos()
    assert writes[0]["dashboard_port"] == "8485"


def test_native_setup_does_not_write_on_database_failure(monkeypatch, tmp_path):
    monkeypatch.setattr(osm_cli, "prompt_vault", lambda: [str(tmp_path)])
    monkeypatch.setattr(osm_cli, "prompt", lambda question, default=None, **kw: default)
    monkeypatch.setattr(
        osm_cli,
        "_read_env",
        lambda: {
            "OSM_DATABASE_BACKEND": "postgresql",
            "DATABASE_URL": "postgresql://127.0.0.1:1/osm_pytest_test",
            "OLLAMA_URL": "http://127.0.0.1:11434",
            "DASHBOARD_PORT": "8484",
        },
    )
    monkeypatch.setattr(
        "src.postgres.inspect_setup", lambda *_: (_ for _ in ()).throw(OSError("unavailable"))
    )
    writes = []
    monkeypatch.setattr(osm_cli, "_write_native_runtime", lambda *a, **k: writes.append(1))
    with pytest.raises(SystemExit):
        osm_cli.mode_native_macos()
    assert writes == []


def test_native_setup_rejects_unidentified_existing_vectors(monkeypatch, tmp_path):
    monkeypatch.setattr(osm_cli, "prompt_vault", lambda: [str(tmp_path)])
    monkeypatch.setattr(osm_cli, "prompt", lambda question, default=None, **kw: default)
    monkeypatch.setattr(
        osm_cli,
        "_read_env",
        lambda: {
            "OSM_DATABASE_BACKEND": "postgresql",
            "DATABASE_URL": "postgresql://127.0.0.1:1/osm_pytest_test",
            "OLLAMA_URL": "http://127.0.0.1:11434",
            "DASHBOARD_PORT": "8484",
        },
    )
    monkeypatch.setattr(osm_cli, "_verify_ollama_model", lambda *_: True)
    monkeypatch.setattr(osm_cli.requests, "get", lambda *a, **k: MagicMock(status_code=200))
    monkeypatch.setattr("src.postgres.inspect_setup", lambda *_: ("0.8.0", True, None))
    writes = []
    monkeypatch.setattr(osm_cli, "_write_native_runtime", lambda *a, **k: writes.append(1))
    with pytest.raises(SystemExit):
        osm_cli.mode_native_macos()
    assert writes == []


def test_status_without_configuration_reports_missing_services(monkeypatch, capsys):
    monkeypatch.setattr(osm_cli, "_read_env", lambda: {})
    monkeypatch.setattr(osm_cli, "check_ollama_at", lambda *_: False)
    monkeypatch.setattr(osm_cli, "_dashboard_control", lambda *_: None)
    osm_cli.cmd_status()
    result = capsys.readouterr().out
    assert "Vault is not configured" in result
    assert "PostgreSQL" in result
    assert "Dashboard not running" in result


def test_status_reports_running_dashboard(monkeypatch, capsys):
    monkeypatch.setattr(osm_cli, "_read_env", lambda: {})
    monkeypatch.setattr(osm_cli, "_dashboard_control", lambda *_: "PONG 6543")
    monkeypatch.setattr(osm_cli, "_dashboard_port_open", lambda port: port == "6543")
    osm_cli.cmd_status()
    assert "Dashboard running: http://127.0.0.1:6543" in capsys.readouterr().out


def test_update_does_not_modify_services(monkeypatch, capsys):
    monkeypatch.setattr(osm_cli, "cmd_version", lambda: None)
    osm_cli.cmd_update()
    assert "brew upgrade joynik/tap/osm" in " ".join(capsys.readouterr().out.split())


def test_version_checks_fork_releases(monkeypatch):
    requested = []

    def fake_get(url, **kwargs):
        requested.append(url)
        return MagicMock(status_code=200, json=lambda: {"tag_name": "v0.41.7"})

    monkeypatch.setattr(osm_cli.requests, "get", fake_get)
    assert osm_cli._fetch_latest_release_tag() == "0.41.7"
    assert requested == [
        "https://api.github.com/repos/Joynik/obsidian-semantic-mcp/releases/latest"
    ]


@pytest.mark.parametrize("command", ["version", "-v", "-V"])
def test_version_commands_use_same_handler(monkeypatch, command):
    called = []
    monkeypatch.setattr(osm_cli.sys, "argv", ["osm", command])
    monkeypatch.setitem(osm_cli.COMMANDS, "version", (lambda: called.append(True), "Version"))
    osm_cli.main()
    assert called == [True]


def test_client_entries_use_osm_run():
    assert osm_cli._native_entry([], "") == {"command": "osm", "args": ["run"], "env": {}}


def test_osm_run_dispatches_without_cli_output(monkeypatch, capsys):
    called = []
    monkeypatch.setattr(osm_cli.sys, "argv", ["osm", "run"])
    monkeypatch.setattr(osm_cli, "cmd_run", lambda: called.append(True))
    osm_cli.main()
    assert called == [True]
    assert capsys.readouterr().out == ""


@pytest.mark.parametrize(
    "command", [*(name for name in osm_cli.COMMANDS if name not in {"run", "open"}), "-v", "-V"]
)
def test_commands_show_one_version_header(monkeypatch, capsys, command):
    monkeypatch.setattr(osm_cli.sys, "argv", ["osm", command])
    for name in osm_cli.COMMANDS:
        monkeypatch.setitem(osm_cli.COMMANDS, name, (lambda: None, "Test command"))

    osm_cli.main()

    output = capsys.readouterr().out
    assert output.count(f"OSM v{osm_cli.APP_VERSION}") == 1


def test_open_shows_version_header(monkeypatch, capsys):
    monkeypatch.setattr(osm_cli.sys, "argv", ["osm", "open", "--help"])
    monkeypatch.setattr(osm_cli, "cmd_open", lambda argv: None)

    osm_cli.main()

    assert capsys.readouterr().out.count(f"OSM v{osm_cli.APP_VERSION}") == 1


def test_codex_legacy_command_is_updated_without_touching_other_servers(monkeypatch, tmp_path):
    path = tmp_path / "config.toml"
    path.write_text(
        '[mcp_servers.obsidian-semantic]\ncommand = "obsidian-semantic-mcp"\nargs = []\n\n[mcp_servers.other]\ncommand = "other-mcp"\n'
    )
    monkeypatch.setattr(osm_cli, "_codex_cfg_path", lambda: path)
    osm_cli.update_codex_config(osm_cli._native_entry([], ""))
    content = path.read_text()
    assert 'command = "osm"' in content
    assert 'args = ["run"]' in content
    assert 'command = "other-mcp"' in content


def test_claude_desktop_legacy_command_is_updated(monkeypatch, tmp_path):
    path = tmp_path / "claude_desktop_config.json"
    path.write_text(
        json.dumps(
            {
                "mcpServers": {
                    "obsidian-semantic": {
                        "command": "obsidian-semantic-mcp",
                        "args": [],
                        "env": {},
                    },
                    "other": {"command": "other-mcp"},
                }
            }
        )
    )
    monkeypatch.setattr(osm_cli, "_claude_cfg_path", lambda: path)
    osm_cli.update_claude_config(osm_cli._native_entry([], ""))
    servers = json.loads(path.read_text())["mcpServers"]
    assert servers["obsidian-semantic"]["command"] == "osm"
    assert servers["obsidian-semantic"]["args"] == ["run"]
    assert servers["other"] == {"command": "other-mcp"}


def test_claude_code_preserves_existing_different_entry(monkeypatch):
    monkeypatch.setattr(osm_cli, "cmd_exists", lambda name: name == "claude")
    monkeypatch.setattr(osm_cli, "_claude_cli_already_registered", lambda: True)
    calls = []

    def fake_run(args, **kwargs):
        calls.append(args)
        if args[2] == "get":
            return MagicMock(
                returncode=0,
                stdout=(
                    "obsidian-semantic:\n  Scope: User config\n  Command: obsidian-semantic-mcp\n"
                    "  Args:\n  Environment:\n\n"
                ),
            )
        return MagicMock(returncode=0)

    monkeypatch.setattr(osm_cli, "run", fake_run)
    osm_cli.register_claude_cli(osm_cli._native_entry([], ""))
    assert calls == []
