#!/usr/bin/env python3
"""
osm_cli.py — Obsidian Semantic MCP command-line interface.

Usage from a checkout (after uv sync):
  uv run osm init      Interactive setup wizard
  uv run osm run       Start the MCP server
  uv run osm status    Check native services
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import platform
import secrets
import shutil
import socket
import stat
import subprocess
import sys
import tempfile
import time
import tomllib
import urllib.parse
import urllib.request
import webbrowser
from collections import deque
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import questionary
import requests
import tomlkit
from dotenv import dotenv_values
from rich.console import Console
from rich.rule import Rule
from rich.table import Table
from rich.text import Text
from src.config import DEFAULT_EMBED_MODEL
from src.embedding_profiles import embedding_input
from src.launcher import OSM_CONFIG_DIR, OSM_LOG_PATH
from tomlkit.exceptions import TOMLKitError

# Ensure stdout/stderr can handle Unicode on Windows (cp1252 etc.)
if sys.stdout.encoding and sys.stdout.encoding.lower().replace("-", "") != "utf8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
if sys.stderr.encoding and sys.stderr.encoding.lower().replace("-", "") != "utf8":
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")


def _version_from_pyproject() -> str | None:
    pyproject = Path(__file__).resolve().parent / "pyproject.toml"
    try:
        data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
        project = data.get("project", {})
        v = project.get("version")
        return str(v) if v else None
    except Exception:
        return None


if source_version := _version_from_pyproject():
    APP_VERSION = source_version
else:
    try:
        APP_VERSION = version("obsidian-semantic-mcp")
    except PackageNotFoundError:
        APP_VERSION = "0.0.0"


# ── Terminal output ───────────────────────────────────────────────────────────

_console = Console(no_color=not sys.stdout.isatty(), highlight=False)
_prompt_style = questionary.Style(
    [
        ("qmark", "fg:#66B2B2 bold"),
        ("question", "bold"),
        ("answer", "fg:#66B2B2 bold"),
        ("pointer", "fg:#66B2B2 bold"),
        ("highlighted", "fg:#66B2B2 bold"),
        ("instruction", "fg:#808080"),
    ]
)


def _message(symbol: str, message: str, style: str) -> None:
    line = Text("  ")
    line.append(symbol, style=style)
    line.append(f"  {message}")
    _console.print(line)


def ok(msg):
    # Under a dry run, run()/writes are faked (see run()'s DRY_RUN branch), so
    # an unprefixed "✓ Setup complete!" would assert an outcome that never
    # happened. Mark every success line as simulated instead.
    if DRY_RUN:
        _message("○", f"[dry-run] {msg}", "yellow")
        return
    _message("✓", str(msg), "green")


def warn(msg):
    _message("⚠", str(msg), "yellow")


def fail(msg):
    _message("✗", str(msg), "red")


def info(msg):
    _message("→", str(msg), "cyan")


def header(msg):
    _console.print()
    _console.print(Text(str(msg), style="bold"))


def hr():
    _console.print(Rule(style="dim", characters="─"))


# ── Dry-run state ─────────────────────────────────────────────────────────────

DRY_RUN = False
_INIT_VAULTS = None
_DRY_ACTIONS: list[str] = []  # collects every skipped action for the summary


def _dry(label, detail=""):
    """Print a dry-run notice and record it for the end-of-run summary."""
    line = f"{label}{('  # ' + detail) if detail else ''}"
    _message("○", f"[dry-run] {line}", "yellow")
    _DRY_ACTIONS.append(line)


# ── Subprocess helpers ────────────────────────────────────────────────────────


def run(cmd, check=True, capture=False, env=None):
    if DRY_RUN:
        cmd_str = cmd if isinstance(cmd, str) else " ".join(str(a) for a in cmd)
        _dry(cmd_str)
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
    kwargs: dict = {"shell": isinstance(cmd, str), "check": check}
    if env:
        kwargs["env"] = env
    if capture:
        kwargs.update(stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return subprocess.run(cmd, **kwargs)


def cmd_exists(name):
    return shutil.which(name) is not None


def _stdin_is_tty():
    return bool(getattr(sys.stdin, "isatty", lambda: False)())


def _dashboard_port_available(port: str) -> bool:
    """Check whether the dashboard can bind its local TCP address."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
            listener.bind(("127.0.0.1", int(port)))
    except OSError:
        return False
    return True


# ── Non-interactive params (set by CLI flags, consumed by prompt_*) ───────────

# Keys: vault, pg_password, mode, persistent, data_dir,
#       ssh_host, ssh_user, ssh_port, ssh_key, vault_remote
_PARAMS: dict = {}


# ── Prompts ───────────────────────────────────────────────────────────────────


def prompt(question, default=None, choices=None, param_key=None):
    # Non-interactive: use pre-supplied value from CLI flags
    if param_key and param_key in _PARAMS:
        val = str(_PARAMS[param_key])
        if choices and val not in choices:
            fail(f"--{param_key.replace('_', '-')} {val!r} is not one of: {', '.join(choices)}")
            sys.exit(1)
        info(f"{question}: {val}  (from --{param_key.replace('_', '-')})")
        return val

    _EXIT_WORDS = {"q", "quit", "exit", "skip"}
    while True:
        try:
            if _stdin_is_tty():
                _console.print()
                answer = questionary.text(
                    question,
                    default=str(default or ""),
                    style=_prompt_style,
                    instruction="(q to quit)",
                ).ask()
                if answer is None:
                    raise KeyboardInterrupt
                answer = answer.strip()
            else:
                hint = f" [{default}]" if default else ""
                answer = input(f"  {question}{hint}: ").strip()
        except EOFError, KeyboardInterrupt:
            print()
            sys.exit(0)
        if answer.lower() in _EXIT_WORDS:
            print()
            sys.exit(0)
        if not answer and default is not None:
            return default
        if choices and answer not in choices:
            warn(f"Please enter one of: {', '.join(choices)}")
            continue
        if answer:
            return answer
        warn("Please enter a value.")


def confirm(question, default="y", param_key=None):
    if param_key and param_key in _PARAMS:
        return prompt(question, default=default, choices=["y", "n"], param_key=param_key) == "y"
    if not _stdin_is_tty():
        return prompt(question, default=default, choices=["y", "n"]) == "y"
    choices = [
        questionary.Choice("Yes", value="y"),
        questionary.Choice("No", value="n"),
        questionary.Choice("Quit setup", value="q"),
    ]
    try:
        _console.print()
        answer = questionary.select(
            question,
            choices=choices,
            default=default,
            style=_prompt_style,
            instruction="(use ↑/↓ and Enter)",
        ).ask()
    except KeyboardInterrupt:
        raise SystemExit(0) from None
    if answer in (None, "q"):
        raise SystemExit(0)
    return answer == "y"


def _prompt_single_vault():
    """Prompt for a single vault path interactively. Returns resolved path string."""
    while True:
        raw = prompt("Absolute path to your Obsidian vault")
        p = Path(raw).expanduser().resolve()
        if p.is_dir():
            return str(p)
        fail(f"Directory not found: {p}")


def prompt_vault():
    """Prompt for one or more vault paths. Returns a list of path strings."""

    def finish(paths: list[str]) -> list[str]:
        header("Vault check")
        for path in paths:
            info(path)
            valid, message = _validate_vault(path)
            (ok if valid else warn)(message)
        return paths

    if _INIT_VAULTS is not None:
        return finish(list(_INIT_VAULTS))
    # --vault flag: single vault, no interactive multi-vault prompt
    if "vault" in _PARAMS:
        p = Path(_PARAMS["vault"]).expanduser().resolve()
        if not p.is_dir():
            fail(f"Vault not found: {p}")
            sys.exit(1)
        info(f"Vault: {p}  (from --vault)")
        return finish([str(p)])

    # Prefer the installed configuration when the wizard is run again.
    from src.launcher import load_native_runtime

    saved = load_native_runtime(OSM_CONFIG_DIR) or {}
    existing_multi = os.environ.get("OBSIDIAN_VAULTS", "")
    existing = os.environ.get("OBSIDIAN_VAULT", "")
    if not existing_multi and not existing:
        existing_multi = saved.get("OBSIDIAN_VAULTS", "")
        existing = saved.get("OBSIDIAN_VAULT", "")
    print()

    if existing_multi:
        paths = [v.strip() for v in existing_multi.split(",") if v.strip()]
        info(f"Configured vaults: {', '.join(paths)}")
        if confirm("Use these vaults?"):
            if confirm("Add another Obsidian vault?", default="n"):
                while True:
                    vault = _prompt_single_vault()
                    if vault not in paths:
                        paths.append(vault)
                    else:
                        warn("Vault is already configured")
                    if not confirm("Add another vault?", default="n"):
                        break
            return finish(paths)
    elif existing:
        info(f"Configured vault: {existing}")
        if confirm("Use this vault?"):
            if confirm("Add another Obsidian vault?", default="n"):
                vaults = [existing]
                while True:
                    v = _prompt_single_vault()
                    if v not in vaults:
                        vaults.append(v)
                    else:
                        warn("Vault is already configured")
                    if not confirm("Add another vault?", default="n"):
                        break
                return finish(vaults)
            return finish([existing])

    # Fresh prompt — collect first vault, then offer to add more
    v = _prompt_single_vault()
    vaults = [v]
    if confirm("Add another Obsidian vault?", default="n"):
        while True:
            v = _prompt_single_vault()
            if v not in vaults:
                vaults.append(v)
            else:
                warn("Vault is already configured")
            if not confirm("Add another vault?", default="n"):
                break
    return finish(vaults)


# ── Prerequisite checks ───────────────────────────────────────────────────────

# Directory holding this module. A source checkout can use its own .env.
_CODE_DIR = Path(__file__).parent.resolve()


EMBED_MODEL = DEFAULT_EMBED_MODEL


def check_ollama_at(base_url):
    url = f"{base_url.rstrip('/')}/api/tags"
    try:
        urllib.request.urlopen(url, timeout=4)
        ok(f"Ollama reachable at {base_url}")
        return True
    except Exception:
        fail(f"Ollama not reachable at {base_url}")
        return False


def _normalize_ollama_url(url: str) -> str:
    url = (url or "").strip()
    if "://" not in url:
        url = f"http://{url}"
    return url.rstrip("/")


def _ollama_error_detail(resp) -> str:
    try:
        data = resp.json()
        if isinstance(data, dict) and data.get("error"):
            return str(data["error"])
    except Exception:
        pass
    return (getattr(resp, "text", "") or f"HTTP {resp.status_code}").strip()


def _print_ollama_restart_hint():
    if platform.system() == "Darwin":
        info("Try: brew services restart ollama")
        info("Logs: /opt/homebrew/var/log/ollama.log")
    else:
        info("Restart the Ollama daemon and inspect its logs")


def check_ollama_inference_at(ollama_url: str, model: str = EMBED_MODEL) -> bool:
    """Probe the embeddings endpoint so status catches broken model execution."""
    base = _normalize_ollama_url(ollama_url)
    try:
        resp = requests.post(
            f"{base}/api/embed",
            json={"model": model, "input": "healthcheck"},
            timeout=30,
        )
    except requests.RequestException as exc:
        fail(f"Ollama embeddings not reachable at {base}")
        info(f"Reason: {exc}")
        return False

    if resp.status_code >= 400:
        detail = _ollama_error_detail(resp)
        fail(f"Ollama embeddings failing at {base}")
        info(f"Reason: {detail}")
        lower = detail.lower()
        if "model failed to load" in lower or "runner process has terminated" in lower:
            _print_ollama_restart_hint()
        return False

    try:
        data = resp.json()
    except ValueError as exc:
        fail(f"Ollama embeddings returned invalid JSON at {base}")
        info(f"Reason: {exc}")
        return False

    if not data.get("embeddings") or not data["embeddings"][0]:
        fail(f"Ollama embeddings returned no vector at {base}")
        info(f"Model: {model}")
        return False

    ok(f"Ollama embeddings responding at {base} ({model})")
    return True


# ── SSH tunnel helpers ────────────────────────────────────────────────────────

_SSH_KEY_CANDIDATES = [
    ".ssh/id_ed25519",
    ".ssh/id_rsa",
    ".ssh/id_ecdsa",
    ".ssh/id_ecdsa_sk",
]


# ── Vault and model helpers ───────────────────────────────────────────────────


def _validate_vault(vault_path: str) -> tuple[bool, str]:
    """Check that vault path is a valid Obsidian vault."""
    p = Path(vault_path)
    if not p.exists():
        return False, f"Path does not exist: {vault_path}"
    if not p.is_dir():
        return False, f"Not a directory: {vault_path}"
    md_files = list(p.rglob("*.md"))
    if len(md_files) == 0:
        return False, f"No .md files found in {vault_path}"
    return True, f"✓ Found {len(md_files)} markdown files"


def _verify_ollama_model(ollama_url: str, model: str) -> bool:
    """Verify the model is available by running a minimal embedding request."""
    try:
        resp = requests.post(
            f"{ollama_url}/api/embed",
            json={"model": model, "input": "test"},
            timeout=30,
        )
        data = resp.json()
        return bool(data.get("embeddings") and data["embeddings"][0])
    except Exception:
        return False


# ── .env writer (runtime only — gitignored) ───────────────────────────────────


# Use the same config directory as the installed launcher.
_OSM_CONFIG_DIR = OSM_CONFIG_DIR
_DASHBOARD_TOKEN_FILE = _OSM_CONFIG_DIR / "dashboard_token"


def _generate_dashboard_token() -> str:
    """Generate or reuse the dashboard bearer token that guards the
    mutating endpoints (/api/reindex, /api/reindex/full, /api/prune,
    /api/ollama/start).

    Reuses ~/.config/osm/dashboard_token if present
    so the token survives `osm init` re-runs and `osm update` /
    `osm rebuild` (no auth flap). On a fresh install, generates a new
    token_urlsafe(32) and persists it mode 0600 — never written under
    the repo checkout.

    Mirrors src/config.py:resolve_dashboard_token() precedence.
    """
    if env_token := os.environ.get("DASHBOARD_TOKEN"):
        return env_token
    if _DASHBOARD_TOKEN_FILE.exists():
        try:
            existing = _DASHBOARD_TOKEN_FILE.read_text(encoding="utf-8").strip()
            if existing:
                return existing
        except OSError:
            pass
    token = secrets.token_urlsafe(32)
    try:
        _OSM_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        _DASHBOARD_TOKEN_FILE.write_text(token, encoding="utf-8")
        _DASHBOARD_TOKEN_FILE.chmod(0o600)
    except OSError:
        pass
    return token


# ── Claude Code CLI registration ──────────────────────────────────────────────


def _claude_cli_already_registered() -> bool:
    """Return True if obsidian-semantic is already in the MCP list."""
    r = run(["claude", "mcp", "list"], check=False, capture=True)
    return "obsidian-semantic" in (r.stdout or "")


def register_claude_cli(entry):
    """
    Register the MCP server with Claude Code CLI via `claude mcp add --scope user`.

    This writes to $HOME/.claude.json (separate from claude_desktop_config.json).
    Silently skips if the `claude` CLI is not installed.

    obsidian-semantic is a single global server — shared across all projects.
    If it is already registered, this is a no-op with an informational message.
    """
    if not cmd_exists("claude"):
        return  # Claude Code CLI not installed — Desktop config is enough

    if DRY_RUN:
        cmd_args = entry.get("args", [])
        env_pairs = [f"{k}={v}" for k, v in entry.get("env", {}).items()]
        cli_cmd = ["claude", "mcp", "add", "--scope", "user"]
        for pair in env_pairs:
            cli_cmd += ["-e", pair]
        cli_cmd += ["obsidian-semantic", "--", entry["command"]] + cmd_args
        _dry(" ".join(str(a) for a in cli_cmd))
        return

    if _claude_cli_already_registered():
        previous = _read_claude_user_entry()
        if previous == entry:
            ok("Claude Code CLI: obsidian-semantic already configured")
            return
        if previous is None or not _is_osm_entry(previous):
            warn("Claude Code CLI: existing obsidian-semantic entry differs; leaving it unchanged")
            return
        if set(previous) - {"command", "args", "env"} or set(previous.get("env", {})) - {
            "OSM_ALLOW_WRITE"
        }:
            warn("Claude Code CLI: existing entry has custom settings; leaving it unchanged")
            return
        removed = run(
            ["claude", "mcp", "remove", "--scope", "user", "obsidian-semantic"], check=False
        )
        if removed.returncode != 0:
            warn("Claude Code CLI: could not replace the old command")
            return
        updated = run(_claude_add_command(entry), check=False)
        if updated.returncode == 0:
            ok("Claude Code CLI: obsidian-semantic updated to osm run")
        else:
            run(_claude_add_command(previous), check=False)
            warn("Claude Code CLI: command update failed; previous entry restored")
        return

    cli_cmd = _claude_add_command(entry)

    r = run(cli_cmd, check=False)
    if r.returncode == 0:
        ok("Claude Code CLI: obsidian-semantic registered  (claude mcp list to verify)")
    else:
        warn("claude mcp add failed — add manually:")
        info(f"  {' '.join(str(a) for a in cli_cmd)}")


def _claude_add_command(entry):
    command = ["claude", "mcp", "add", "--scope", "user"]
    for key, value in entry.get("env", {}).items():
        command += ["-e", f"{key}={value}"]
    return command + ["obsidian-semantic", "--", entry["command"]] + entry.get("args", [])


def _read_claude_user_entry():
    try:
        cfg = json.loads((Path.home() / ".claude.json").read_text(encoding="utf-8"))
        return cfg.get("mcpServers", {}).get("obsidian-semantic")
    except OSError, ValueError, AttributeError:
        return None


# ── Claude Desktop config ─────────────────────────────────────────────────────


def _claude_cfg_path():
    system = platform.system()
    if system == "Darwin":
        return (
            Path.home()
            / "Library"
            / "Application Support"
            / "Claude"
            / "claude_desktop_config.json"
        )
    if system == "Linux":
        return Path.home() / ".config" / "Claude" / "claude_desktop_config.json"
    if system == "Windows":
        appdata = os.environ.get("APPDATA", "")
        if appdata:
            return Path(appdata) / "Claude" / "claude_desktop_config.json"
    return None


def update_claude_config(entry):
    path = _claude_cfg_path()
    if not path:
        warn("Unknown platform — update claude_desktop_config.json manually")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    cfg = {}
    if path.exists():
        try:
            cfg = json.loads(path.read_text())
        except json.JSONDecodeError:
            warn(f"Could not parse {path} — leaving it unchanged")
            return
    existing = cfg.get("mcpServers", {}).get("obsidian-semantic")
    if existing:
        if existing == entry:
            ok("Claude Desktop: obsidian-semantic already configured")
            return
        if not _is_osm_entry(existing):
            warn("Claude Desktop: existing obsidian-semantic entry differs; leaving it unchanged")
            return
        entry = {**existing, **entry, "env": {**existing.get("env", {}), **entry.get("env", {})}}
    cfg.setdefault("mcpServers", {})["obsidian-semantic"] = entry
    # Show the full merged config so the user can see what other entries are preserved
    pretty = json.dumps(cfg, indent=2)
    if DRY_RUN:
        _dry(f"write {path}", "merged config shown below")
        print()
        for line in pretty.splitlines():
            _console.print(f"    {line}", style="dim")
        print()
        return
    path.write_text(json.dumps(cfg, indent=2) + "\n")
    ok(f"Updated {path}")
    info("Restart Claude Desktop to pick up the new server")


# ── OpenCode config ───────────────────────────────────────────────────────────


def _opencode_cfg_path() -> Path:
    """OpenCode reads MCP servers from ~/.config/opencode/opencode.json on every platform."""
    return Path.home() / ".config" / "opencode" / "opencode.json"


def update_opencode_config(entry):
    """Merge the obsidian-semantic entry into ~/.config/opencode/opencode.json.

    Converts the standard MCP config entry {command, args, env} into OpenCode's
    native format: command as a flat array, ``type: "local"``, ``enabled: true``.

    Idempotent — preserves other entries, warns on parse errors, and returns
    silently if OpenCode has never been run on this machine (no config file).
    """
    path = _opencode_cfg_path()
    cfg = {}
    if path.exists():
        try:
            cfg = json.loads(path.read_text())
        except json.JSONDecodeError:
            warn(f"Could not parse {path} — leaving it unchanged")
            return
    # Convert standard {command, args, env} to OpenCode's native format
    opencode_entry = {
        "command": [entry["command"]] + entry.get("args", []),
        "enabled": True,
        "type": "local",
        "environment": entry.get("env", {}),
    }
    existing = cfg.setdefault("mcp", {}).get("obsidian-semantic")
    if existing:
        if existing == opencode_entry:
            ok("OpenCode: obsidian-semantic already configured")
            return
        if not _is_osm_entry(existing):
            warn("OpenCode: existing obsidian-semantic entry differs; leaving it unchanged")
            return
        opencode_entry = {
            **existing,
            **opencode_entry,
            "environment": {**existing.get("environment", {}), **opencode_entry["environment"]},
        }
    cfg.setdefault("mcp", {})["obsidian-semantic"] = opencode_entry
    pretty = json.dumps(cfg, indent=2)
    if DRY_RUN:
        _dry(f"write {path}", "merged config shown below")
        print()
        for line in pretty.splitlines():
            _console.print(f"    {line}", style="dim")
        print()
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(pretty + "\n")
    ok(f"Updated {path}")
    info("Restart OpenCode to pick up the new server")


def remove_opencode_config():
    """Delete the obsidian-semantic entry from ~/.config/opencode/opencode.json if present."""
    path = _opencode_cfg_path()
    if not path.exists():
        info("OpenCode config not found — skipping")
        return
    try:
        cfg = json.loads(path.read_text())
        servers = cfg.get("mcp", {})
        existing = servers.get("obsidian-semantic")
        if existing and _is_osm_entry(existing):
            if DRY_RUN:
                _dry(f"remove obsidian-semantic entry from {path}")
                return
            del servers["obsidian-semantic"]
            path.write_text(json.dumps(cfg, indent=2) + "\n")
            ok(f"Removed obsidian-semantic from {path}")
            info("Restart OpenCode to apply")
        elif existing:
            warn("OpenCode: obsidian-semantic uses a different command; leaving it unchanged")
    except (OSError, ValueError, AttributeError) as exc:
        warn(f"OpenCode configuration unchanged ({type(exc).__name__})")


def _is_osm_entry(entry: object) -> bool:
    """Recognize only the launcher commands OSM has registered."""
    if not isinstance(entry, dict):
        return False
    command = entry.get("command")
    args = entry.get("args", [])
    if isinstance(command, list):
        return command in (["osm", "run"], ["obsidian-semantic-mcp"], ["osm-mcp"])
    return (command == "osm" and args == ["run"]) or (
        command in ("obsidian-semantic-mcp", "osm-mcp") and args == []
    )


# ── pi agent (earendil-works/pi) config ──────────────────────────────────────


def _pi_agent_mcp_path() -> Path:
    """pi agent MCP server config: ~/.pi/agent/mcp.json"""
    return Path.home() / ".pi" / "agent" / "mcp.json"


def _pi_enabled() -> bool:
    """Enable pi registration only when its CLI is available and not opted out."""
    if os.environ.get("OSM_SKIP_PI", "").strip().lower() in ("1", "true", "yes"):
        return False
    return cmd_exists("pi")


def _mcp_clients_label() -> str:
    """List clients that setup can configure on this host."""
    clients = "Claude Desktop, Claude Code CLI, OpenCode"
    return f"{clients}, pi" if _pi_enabled() else clients


def register_pi_agent(entry):
    """Register the installed OSM command in pi agent MCP settings."""
    if not _pi_enabled():
        warn("Pi CLI is not installed; see README for manual setup")
        return

    mcp_path = _pi_agent_mcp_path()
    if mcp_path.exists():
        try:
            cfg = json.loads(mcp_path.read_text(encoding="utf-8"))
            existing = cfg.get("mcpServers", {}).get("obsidian-semantic")
        except OSError, ValueError, AttributeError:
            warn("Pi MCP configuration could not be read; leaving it unchanged")
            return
        if existing:
            if existing == entry:
                ok("Pi: obsidian-semantic already configured")
                return
            if not _is_osm_entry(existing):
                warn("Pi: existing obsidian-semantic entry differs; leaving it unchanged")
                return
            if set(existing) - {"command", "args", "env"} or set(existing.get("env", {})) - {
                "OSM_ALLOW_WRITE"
            }:
                warn("Pi: existing entry has custom settings; leaving it unchanged")
                return

    if DRY_RUN:
        _dry(" ".join(_pi_add_command(entry)))
        return
    result = run(_pi_add_command(entry), check=False)
    if result.returncode == 0:
        ok("Pi: obsidian-semantic configured")
        info("Use /reload in Pi to pick up the new server")
    else:
        warn("Pi registration failed; see README for manual setup")


def _pi_add_command(entry):
    command = ["pi", "mcp", "add", "obsidian-semantic"]
    for key, value in entry.get("env", {}).items():
        command += ["--env", f"{key}={value}"]
    return command + ["--", entry["command"]] + entry.get("args", [])


# ── Cross-client registration fan-out ─────────────────────────────────────────


def _codex_cfg_path() -> Path:
    directory = os.environ.get("CODEX_HOME")
    return (Path(directory).expanduser() if directory else Path.home() / ".codex") / "config.toml"


def _codex_config_bytes(path):
    if path.is_symlink():
        raise ValueError("unsafe Codex configuration file")
    if not path.exists():
        return None
    if not stat.S_ISREG(path.stat().st_mode):
        raise ValueError("unsafe Codex configuration file")
    return path.read_bytes()


def _read_codex_config(path):
    original = _codex_config_bytes(path)
    cfg = tomlkit.document() if original is None else tomlkit.parse(original.decode("utf-8"))
    servers = cfg.get("mcp_servers", {})
    if not isinstance(servers, dict):
        raise TypeError("invalid Codex server table")
    if "obsidian-semantic" in servers and not isinstance(servers["obsidian-semantic"], dict):
        raise TypeError("invalid Codex OSM table")
    return cfg, original


def _write_codex_config(path, cfg, original):
    content = tomlkit.dumps(cfg)
    if DRY_RUN:
        _dry(f"write {path}", "preserving unrelated Codex settings")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".osm-codex-", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as target:
            if os.name == "posix":
                os.fchmod(target.fileno(), 0o600)
            target.write(content)
            target.flush()
            os.fsync(target.fileno())
        if _codex_config_bytes(path) != original:
            raise FileExistsError("Codex configuration changed concurrently")
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def update_codex_config(entry):
    """Register the installed launcher for CLI and same-host desktop clients."""
    path = _codex_cfg_path()
    if not cmd_exists("codex") and not path.exists() and not path.is_symlink():
        return
    try:
        cfg, original = _read_codex_config(path)
        servers = cfg.get("mcp_servers", {})
        existing = servers.get("obsidian-semantic")
        if existing:
            if (
                existing.get("command") == "osm"
                and existing.get("args", []) == ["run"]
                and existing.get("env", {}).get("OSM_ALLOW_WRITE", "0")
                == entry.get("env", {}).get("OSM_ALLOW_WRITE", "0")
            ):
                ok("Codex: obsidian-semantic already configured")
                return
            if not _is_osm_entry(existing):
                warn("Codex: existing obsidian-semantic entry differs; leaving it unchanged")
                return
        if (
            entry.get("command") != "osm"
            or entry.get("args", []) != ["run"]
            or not isinstance(entry.get("env", {}), dict)
        ):
            raise ValueError("nonportable Codex server entry")
        env = entry.get("env", {})
        server = existing if existing else tomlkit.table()
        server["command"] = "osm"
        server["args"] = ["run"]
        server["env"] = {**existing.get("env", {}), **env} if existing else env
        if "mcp_servers" not in cfg:
            cfg["mcp_servers"] = tomlkit.table()
        cfg["mcp_servers"]["obsidian-semantic"] = server
        _write_codex_config(path, cfg, original)
        ok("Codex CLI / ChatGPT Desktop: obsidian-semantic configured")
        info("Restart the desktop client; use codex mcp list or /mcp to verify")
    except (OSError, ValueError, TypeError, TOMLKitError) as exc:
        warn(f"Codex configuration unchanged; registration failed ({type(exc).__name__})")


def remove_codex_config():
    """Remove only OSM's server table, retaining all unrelated TOML content."""
    path = _codex_cfg_path()
    if not path.exists() and not path.is_symlink():
        return
    try:
        cfg, original = _read_codex_config(path)
        servers = cfg.get("mcp_servers", {})
        existing = servers.get("obsidian-semantic")
        if not existing:
            return
        if not _is_osm_entry(existing):
            warn("Codex: obsidian-semantic uses a different command; leaving it unchanged")
            return
        del servers["obsidian-semantic"]
        _write_codex_config(path, cfg, original)
        if not DRY_RUN:
            ok("Removed obsidian-semantic from Codex configuration")
    except (OSError, ValueError, TypeError, TOMLKitError) as exc:
        warn(f"Codex configuration unchanged; removal failed ({type(exc).__name__})")


def remove_claude_desktop_config():
    """Remove only the OSM entry from Claude Desktop's user configuration."""
    path = _claude_cfg_path()
    if path is None or not path.exists():
        return
    try:
        cfg = json.loads(path.read_text(encoding="utf-8"))
        servers = cfg.get("mcpServers", {})
        existing = servers.get("obsidian-semantic")
        if not existing:
            return
        if not _is_osm_entry(existing):
            warn("Claude Desktop: obsidian-semantic uses a different command; leaving it unchanged")
            return
        if DRY_RUN:
            _dry(f"remove obsidian-semantic from {path}")
            return
        del servers["obsidian-semantic"]
        path.write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")
        ok("Claude Desktop: OSM disconnected")
    except (OSError, ValueError, AttributeError) as exc:
        warn(f"Claude Desktop configuration unchanged ({type(exc).__name__})")


def remove_claude_code_config():
    """Remove OSM only from Claude Code's user scope."""
    if not cmd_exists("claude"):
        return
    path = Path.home() / ".claude.json"
    if not path.exists():
        return
    try:
        cfg = json.loads(path.read_text(encoding="utf-8"))
        entry = cfg.get("mcpServers", {}).get("obsidian-semantic")
    except (OSError, ValueError, AttributeError) as exc:
        warn(f"Claude Code configuration unchanged ({type(exc).__name__})")
        return
    if not entry:
        return
    if not _is_osm_entry(entry):
        warn("Claude Code: existing user entry uses a different command; leaving it unchanged")
        return
    if DRY_RUN:
        _dry("claude mcp remove --scope user obsidian-semantic")
        return
    removed = run(["claude", "mcp", "remove", "--scope", "user", "obsidian-semantic"], check=False)
    (ok if removed.returncode == 0 else warn)(
        "Claude Code: OSM disconnected"
        if removed.returncode == 0
        else "Claude Code: could not remove OSM entry"
    )


def remove_pi_agent_config():
    """Remove OSM from Pi's user-level MCP configuration."""
    path = _pi_agent_mcp_path()
    if not path.exists():
        return
    try:
        cfg = json.loads(path.read_text(encoding="utf-8"))
        servers = cfg.get("mcpServers", {})
        existing = servers.get("obsidian-semantic")
        if existing:
            if not _is_osm_entry(existing):
                warn("Pi: obsidian-semantic uses a different command; leaving it unchanged")
                return
            if DRY_RUN:
                _dry(f"remove obsidian-semantic from {path}")
                return
            del servers["obsidian-semantic"]
            path.write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")
            ok("Pi: OSM disconnected")
            return
        legacy = cfg.get("servers")
        if isinstance(legacy, list):
            kept = [
                item
                for item in legacy
                if not (
                    isinstance(item, dict)
                    and item.get("name") == "obsidian-semantic"
                    and _is_osm_entry(item)
                )
            ]
            if len(kept) != len(legacy):
                if DRY_RUN:
                    _dry(f"remove legacy obsidian-semantic from {path}")
                    return
                cfg["servers"] = kept
                path.write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")
                ok("Pi: legacy OSM entry disconnected")
    except (OSError, ValueError, AttributeError) as exc:
        warn(f"Pi configuration unchanged ({type(exc).__name__})")


def register_with_clients(entry):
    """Offer each client connection separately; leave all unchecked by default."""
    header("MCP clients")
    info("Each client starts OSM with: osm run")
    if not _stdin_is_tty():
        warn("Non-interactive setup: client registration skipped; see README")
        return

    def client_entry(client):
        allow = confirm(f"Allow {client} to modify Obsidian notes?", default="n")
        return {**entry, "env": {"OSM_ALLOW_WRITE": "1" if allow else "0"}}

    if confirm("Connect Codex?", default="n"):
        update_codex_config(client_entry("Codex"))
    if confirm("Connect Claude Code?", default="n"):
        register_claude_cli(client_entry("Claude Code"))
    if confirm("Connect Pi?", default="n"):
        register_pi_agent(client_entry("Pi"))
    if confirm("Connect OpenCode?", default="n"):
        update_opencode_config(client_entry("OpenCode"))
    if confirm("Connect Claude Desktop?", default="n"):
        update_claude_config(client_entry("Claude Desktop"))
    info("Manual connection commands are in README.md")


# ── MCP client command ────────────────────────────────────────────────────────


def _native_entry(vault, db_url):
    """Runtime configuration belongs to OSM, not its MCP clients."""
    return {"command": "osm", "args": ["run"], "env": {}}


def _write_native_runtime(
    vault,
    db_url,
    ollama_url=None,
    embedding_model=None,
    database_backend="postgresql",
    qdrant_url=None,
    dashboard_port=None,
    search_language=None,
    index_archive=None,
    archive_folder=None,
):
    from src.launcher import write_native_runtime

    if DRY_RUN:
        _dry(
            f"write {OSM_CONFIG_DIR / 'runtime.json'}",
            "owner-only native runtime configuration",
        )
        return
    write_native_runtime(
        vault,
        db_url,
        config_dir=OSM_CONFIG_DIR,
        ollama_url=ollama_url,
        embedding_model=embedding_model,
        database_backend=database_backend,
        qdrant_url=qdrant_url,
        dashboard_port=dashboard_port,
        search_language=search_language,
        index_archive=index_archive,
        archive_folder=archive_folder,
    )


# ── Native setup ──────────────────────────────────────────────────────────────


def mode_native_macos():
    """Configure a native Qdrant or PostgreSQL index and local Ollama."""
    from src.launcher import load_native_runtime

    previous = load_native_runtime(OSM_CONFIG_DIR) or _read_env()
    vaults = prompt_vault()
    index_archive = confirm(
        "Include notes in archive folders in search?",
        default="y" if previous.get("OBSIDIAN_INDEX_ARCHIVE") == "yes" else "n",
    )
    archive_folder = previous.get("OBSIDIAN_ARCHIVE_FOLDER", "Archive")
    if not index_archive:
        while True:
            archive_folder = prompt("Archive folder name to exclude", default=archive_folder)
            if archive_folder not in {".", ".."} and not any(
                char in archive_folder for char in ("/", "\\", "\x00", "\r", "\n")
            ):
                break
            warn("Enter one folder name without path separators")
    if previous.get("OBSIDIAN_INDEX_ARCHIVE") == "yes" and not index_archive:
        warn("Archive notes already indexed remain searchable until Safe Rebuild completes")
    if (
        previous
        and not index_archive
        and previous.get("OBSIDIAN_ARCHIVE_FOLDER", "archive") != archive_folder
    ):
        warn(
            "Notes from the previous archive folder remain searchable until Safe Rebuild completes"
        )
    default_backend = previous.get("OSM_DATABASE_BACKEND", "qdrant")
    if _stdin_is_tty() and "database_backend" not in _PARAMS:
        backend = questionary.select(
            "Select the local database",
            choices=[
                questionary.Choice("Qdrant (no PostgreSQL)", value="qdrant"),
                questionary.Choice("PostgreSQL + pgvector", value="postgresql"),
                questionary.Choice("Quit setup", value="q"),
            ],
            default=default_backend,
            style=_prompt_style,
            instruction="(use ↑/↓ and Enter)",
        ).ask()
        if backend in (None, "q"):
            raise SystemExit(0)
    else:
        backend = prompt(
            "Database (postgresql/qdrant)",
            default=default_backend,
            choices=("postgresql", "qdrant"),
            param_key="database_backend",
        )
    db_url = None
    qdrant_url = None
    if backend == "postgresql":
        db_url = prompt(
            "PostgreSQL URL",
            default=previous.get("DATABASE_URL") or os.environ.get("DATABASE_URL"),
            param_key="database_url",
        )
    else:
        qdrant_url = prompt(
            "Local Qdrant URL",
            default=previous.get("QDRANT_URL") or os.environ.get("QDRANT_URL"),
            param_key="qdrant_url",
        ).rstrip("/")
        parts = urllib.parse.urlsplit(qdrant_url)
        if (
            parts.scheme != "http"
            or parts.hostname not in ("localhost", "127.0.0.1", "::1")
            or parts.username
            or parts.password
            or parts.path not in ("", "/")
            or parts.query
            or parts.fragment
        ):
            fail("Qdrant URL must point to the local native service")
            raise SystemExit(1)
    ollama_url = prompt(
        "Ollama URL",
        default=previous.get("OLLAMA_URL") or os.environ.get("OLLAMA_URL"),
        param_key="ollama_url",
    ).rstrip("/")
    model = prompt(
        "Embedding model",
        default=previous.get("EMBEDDING_MODEL", EMBED_MODEL),
        param_key="embedding_model",
    )
    search_language = previous.get("OSM_SEARCH_LANGUAGE", "russian")
    if backend == "postgresql":
        if _stdin_is_tty() and "search_language" not in _PARAMS:
            search_language = questionary.select(
                "PostgreSQL search language",
                choices=[
                    questionary.Choice("Russian (word forms + exact words)", value="russian"),
                    questionary.Choice("English (word forms + exact words)", value="english"),
                    questionary.Choice("Simple (exact words, any language)", value="simple"),
                ],
                default=search_language,
                style=_prompt_style,
                instruction="(use ↑/↓ and Enter)",
            ).ask()
            if search_language is None:
                raise SystemExit(0)
        else:
            search_language = prompt(
                "PostgreSQL search language (russian/english/simple)",
                default=search_language,
                choices=("russian", "english", "simple"),
                param_key="search_language",
            )
    from src.launcher import _valid_dashboard_port

    default_port = previous.get("DASHBOARD_PORT") or os.environ.get("DASHBOARD_PORT")
    while True:
        dashboard_port = prompt(
            "Dashboard port",
            default=default_port,
            param_key="dashboard_port",
        )
        if not _valid_dashboard_port(dashboard_port):
            fail("Dashboard port must be between 1 and 65535")
        elif _dashboard_port_available(dashboard_port):
            break
        elif _dashboard_control(b"OSM_DASHBOARD_PING\n") == f"PONG {dashboard_port}":
            info(f"Dashboard is already running on port {dashboard_port}")
            break
        else:
            fail(f"Dashboard port {dashboard_port} is unavailable; choose another port")
        if "dashboard_port" in _PARAMS:
            raise SystemExit(1)
        default_port = None

    # Finish all checks before changing user configuration or registering clients.
    existing_vectors = False
    active_model = None
    if backend == "postgresql":
        try:
            from src.postgres import inspect_setup

            vector_version, existing_vectors, active_model = inspect_setup(db_url or "")
            ok(f"PostgreSQL and pgvector available (vector {vector_version})")
        except Exception as exc:
            fail(f"PostgreSQL check failed: {exc}")
            sys.exit(1)
    else:
        try:
            from src.qdrant import QdrantStore

            store = QdrantStore(qdrant_url or "", model)
            store.collections()
            target = store.alias_target()
            if target:
                profile = store.profile()
                if not profile or not profile.get("model"):
                    raise RuntimeError("Existing OSM Qdrant index has no model identity")
                active_model = profile["model"]
                existing_vectors = bool(store.notes())
            ok(f"Local Qdrant available: {qdrant_url}")
        except Exception as exc:
            fail(f"Qdrant check failed: {exc}")
            sys.exit(1)
    try:
        response = requests.get(f"{ollama_url}/api/tags", timeout=5)
        response.raise_for_status()
        if not _verify_ollama_model(ollama_url, model):
            raise RuntimeError(f"model {model!r} is unavailable or cannot embed")
        ok(f"Ollama and {model} available")
    except Exception as exc:
        fail(f"Ollama check failed: {exc}")
        sys.exit(1)

    if active_model and active_model != model:
        fail("Embedding model changed; reindex migration is required before activation")
        sys.exit(1)
    if (
        previous.get("OSM_DATABASE_BACKEND", "postgresql") == backend
        and previous.get("EMBEDDING_MODEL")
        and previous["EMBEDDING_MODEL"] != model
        and not active_model
    ):
        fail("Existing model identity is unknown; run the current OSM server before migration")
        sys.exit(1)
    if (
        backend == "postgresql"
        and existing_vectors
        and not previous.get("EMBEDDING_MODEL")
        and not active_model
    ):
        fail(
            "Existing vectors have no recorded embedding model; choose a verified migration before activation"
        )
        sys.exit(1)
    if backend == "qdrant":
        if DRY_RUN:
            _dry("create or verify the OSM Qdrant collection", "model profile checked")
        else:
            try:
                response = requests.post(
                    f"{ollama_url}/api/embed",
                    json={
                        "model": model,
                        "input": embedding_input("dimension probe", "document", model),
                    },
                    timeout=30,
                )
                response.raise_for_status()
                dimension = len(response.json()["embeddings"][0])
                if not dimension:
                    raise ValueError("Ollama returned an empty embedding")
                store.ensure(dimension)
                ok("OSM Qdrant collection is ready")
            except Exception as exc:
                fail(f"Cannot initialize Qdrant collection: {exc}")
                raise SystemExit(1) from exc
    _write_native_runtime(
        vaults,
        db_url,
        ollama_url=ollama_url,
        embedding_model=model,
        database_backend=backend,
        qdrant_url=qdrant_url,
        dashboard_port=dashboard_port,
        search_language=search_language,
        index_archive=index_archive,
        archive_folder=archive_folder,
    )
    entry = _native_entry(vaults, db_url)
    installed_command = shutil.which("osm")
    if installed_command and not Path(installed_command).resolve().is_relative_to(_CODE_DIR):
        register_with_clients(entry)
    else:
        warn("MCP client registration skipped; install a path-independent launcher first")
    _done_native(vaults)


# ── Summary printers ──────────────────────────────────────────────────────────


def _done_dry_run():
    hr()
    warn("DRY RUN — no changes were made")
    if _DRY_ACTIONS:
        header("Actions that would have run")
        for i, action in enumerate(_DRY_ACTIONS, 1):
            _console.print(f"  {i}.  {action}", style="dim")
    else:
        info("No actions would have run")
    info("Re-run without --dry-run to apply")
    hr()


def _done_native(vault):
    hr()
    ok("Setup complete!")
    info(f"Vault: {vault}")
    info("Restart any MCP client you chose to connect")
    hr()


# ── Tunnel command ────────────────────────────────────────────────────────────


def _read_env():
    """Parse .env into a dict (simple KEY=VALUE, ignores comments)."""
    from src.launcher import load_native_runtime

    saved = load_native_runtime(OSM_CONFIG_DIR)
    if saved is not None:
        return dict(saved)
    if not (_CODE_DIR / "pyproject.toml").is_file():
        return {}
    env_path = _CODE_DIR / ".env"
    result = {}
    if not env_path.exists():
        return result
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, _, v = line.partition("=")
            key, value = k.strip(), v.strip()
            result[key] = value
    return result


# ── Vaults command ────────────────────────────────────────────────────────────


def cmd_vaults():
    """List the Obsidian vault(s) this install is indexing."""
    hr()

    env = _read_env()
    raw_multi = env.get("OBSIDIAN_VAULTS", "")
    vaults = [v.strip() for v in raw_multi.split(",") if v.strip()]
    if not vaults:
        single = env.get("OBSIDIAN_VAULT", "")
        if single:
            vaults = [single]

    if not vaults:
        warn("No vault configured — run osm init")
        return

    table = Table(show_header=True, box=None, padding=(0, 2))
    table.add_column("Vault", style="bold")
    table.add_column("Path", overflow="fold")
    table.add_column("State")
    for index, vault in enumerate(vaults, 1):
        exists = Path(vault).is_dir()
        table.add_row(
            str(index),
            vault,
            Text("Available", style="green") if exists else Text("Path not found", style="red"),
        )
    _console.print(table)


# ── Status command ────────────────────────────────────────────────────────────


def _report_dashboard_status() -> None:
    dashboard = _dashboard_control(b"OSM_DASHBOARD_PING\n")
    if dashboard and dashboard.startswith("PONG "):
        port = dashboard.removeprefix("PONG ")
        if port.isdecimal() and _dashboard_port_open(port):
            ok(f"Dashboard running: http://127.0.0.1:{port}")
        else:
            fail("Dashboard control socket is active, but its HTTP port is unavailable")
    else:
        warn("Dashboard not running; start with osm dashboard")


def cmd_status():
    """Report health of the configured native services."""
    hr()
    try:
        env = _read_env()
    except (OSError, ValueError, TypeError) as exc:
        fail(f"Native configuration is invalid ({type(exc).__name__})")
        _report_dashboard_status()
        return
    vaults = [v.strip() for v in env.get("OBSIDIAN_VAULTS", "").split(",") if v.strip()]
    if not vaults and env.get("OBSIDIAN_VAULT"):
        vaults = [env["OBSIDIAN_VAULT"]]
    if vaults:
        for vault in vaults:
            (ok if Path(vault).is_dir() else fail)(f"Vault: {vault}")
    else:
        warn("Vault is not configured; run osm init")

    backend = env.get("OSM_DATABASE_BACKEND", "postgresql")
    db_url = env.get("DATABASE_URL") or os.environ.get("DATABASE_URL")
    if backend == "qdrant":
        if not env.get("QDRANT_URL"):
            warn("Qdrant URL is not configured")
        else:
            try:
                from src.qdrant import QdrantStore

                store = QdrantStore(env["QDRANT_URL"], env.get("EMBEDDING_MODEL", EMBED_MODEL))
                store.collections()
                ok(f"Qdrant reachable: {store.url}")
                if store.alias_target():
                    profile = store.profile()
                    if not profile:
                        fail("Qdrant index has no embedding profile")
                    elif profile.get("model") != store.model:
                        fail(
                            f"Qdrant model differs: index={profile.get('model')}, configured={store.model}"
                        )
                    else:
                        ok(f"Qdrant index: {len(store.notes())} notes; model {store.model}")
                        for state in store.states():
                            status = state["status"]
                            report = (
                                fail
                                if status in ("failed", "dimension_mismatch")
                                else (warn if status in ("indexing", "pending") else ok)
                            )
                            detail = f": {state['error']}" if state.get("error") else ""
                            report(f"Index {status}: {state['vault_id']}{detail}")
                else:
                    warn("Qdrant index not initialized; osm run will build it")
            except Exception as exc:
                fail(f"Qdrant unavailable: {exc}")
    elif db_url:
        try:
            from src.postgres import status_snapshot

            snapshot = status_snapshot(db_url, vaults)
            if snapshot["vector_version"]:
                ok(f"PostgreSQL reachable; pgvector {snapshot['vector_version']}")
            else:
                fail("PostgreSQL reachable; pgvector extension missing")
            profile = snapshot["profile"]
            if profile:
                info(f"Active embedding model: {profile[0]}")
                if profile[1]:
                    warn(f"Unfinished model migration: {profile[1]}")
            if snapshot["vaults"] is not None:
                for vault, (indexed, status, error) in snapshot["vaults"].items():
                    suffix = " (existing index available)" if indexed else " (no indexed notes yet)"
                    if status == "indexing":
                        warn(f"Indexing: {vault}{suffix}")
                    elif status in ("failed", "dimension_mismatch"):
                        fail(f"Index error: {vault}: {error or status}{suffix}")
                    elif status == "idle":
                        ok(f"Index ready: {vault}{suffix}" if indexed else f"Index empty: {vault}")
                    else:
                        warn(f"Index pending: {vault}{suffix}")
            else:
                warn("Index not initialized; osm run will initialize it after Ollama is ready")
        except Exception as exc:
            fail(f"PostgreSQL unavailable: {exc}")
    else:
        warn("PostgreSQL URL is not configured")

    ollama_url = env.get("OLLAMA_URL", "").rstrip("/")
    if not ollama_url:
        warn("Ollama URL is not configured")
    elif check_ollama_at(ollama_url):
        if not check_ollama_inference_at(ollama_url, env.get("EMBEDDING_MODEL", EMBED_MODEL)):
            warn("Semantic search is waiting for Ollama; a running osm run retries automatically")
    else:
        fail(f"Ollama unavailable: {ollama_url}")
        warn("Semantic search is waiting for Ollama; a running osm run retries automatically")

    _report_dashboard_status()


# ── Rebuild command ───────────────────────────────────────────────────────────


# ── Model migration ──────────────────────────────────────────────────────────


def cmd_migrate():
    """Re-embed into a separate column, then activate the new model."""
    from src.launcher import load_native_runtime, write_native_runtime

    target = _PARAMS.get("embedding_model")
    if not target:
        fail("Usage: osm migrate --embedding-model MODEL [--dry-run]")
        raise SystemExit(2)
    settings = load_native_runtime(OSM_CONFIG_DIR)
    if not settings:
        fail("Run osm init first; migration needs installed native settings")
        raise SystemExit(1)
    current = settings.get("EMBEDDING_MODEL", DEFAULT_EMBED_MODEL)
    if current == target:
        fail(f"{target} is already the configured model")
        raise SystemExit(1)
    info(f"Current: {current}; target: {target}")
    if not DRY_RUN and not confirm(
        f"Re-embed all notes with {target} and activate it after completion?",
        default="n",
    ):
        info("Cancelled.")
        return
    backend = settings.get("OSM_DATABASE_BACKEND", "postgresql")
    if backend == "qdrant":
        env = {**os.environ, **settings, "EMBEDDING_MODEL": target}
        if DRY_RUN:
            try:
                from src.qdrant import QdrantStore

                store = QdrantStore(settings["QDRANT_URL"], current)
                store.assert_profile()
                response = requests.post(
                    f"{settings['OLLAMA_URL'].rstrip('/')}/api/embed",
                    json={
                        "model": target,
                        "input": embedding_input("dimension probe", "document", target),
                    },
                    timeout=30,
                )
                response.raise_for_status()
                dimension = len(response.json()["embeddings"][0])
                if not dimension:
                    raise ValueError("target model returned an empty embedding")
                info(f"Would rebuild {len(store.notes())} notes into a separate Qdrant collection")
            except Exception as exc:
                fail(f"Qdrant migration check failed: {exc}")
                raise SystemExit(1) from exc
            return
        code = (
            "from src import server\n"
            "with server.reindex_lock() as acquired:\n"
            "    if not acquired: raise RuntimeError('reindex already in progress')\n"
            "    print(server.rebuild_index_safely(server.VAULT_PATHS))\n"
        )
        result = run([sys.executable, "-c", code], env=env, check=False)
    else:
        result = _run_migration_snippet(target, dry_run=DRY_RUN, settings=settings)
    if result.returncode != 0:
        fail("Migration failed; the previous model and index remain active")
        raise SystemExit(1)
    if DRY_RUN:
        return
    vaults = settings.get("OBSIDIAN_VAULTS", settings.get("OBSIDIAN_VAULT", "")).split(",")
    write_native_runtime(
        vaults,
        settings.get("DATABASE_URL"),
        config_dir=OSM_CONFIG_DIR,
        ollama_url=settings.get("OLLAMA_URL"),
        embedding_model=target,
        database_backend=backend,
        qdrant_url=settings.get("QDRANT_URL"),
        dashboard_port=settings.get("DASHBOARD_PORT"),
        search_language=settings.get("OSM_SEARCH_LANGUAGE"),
    )
    ok(f"Activated {target}. Restart MCP clients to load the new setting.")


def _migration_snippet(target: str, *, dry_run: bool) -> str:
    setup = "from src import server, migrations\n"
    if not dry_run:
        setup += "server.init_db(server.get_embed_dim())\n"
    return setup + (
        "conn = server._get_pool().getconn()\n"
        "try:\n"
        "    with server.reindex_lock() as acquired:\n"
        "        if not acquired: raise RuntimeError('reindex already in progress')\n"
        f"        target = {target!r}\n"
        "        dimension = len(server.embed('dimension probe', model=target))\n"
        f"        result = migrations.migrate_embedding_model(conn, target, dimension, "
        f"lambda text: server.embed(text, model=target), dry_run={dry_run!r}, "
        "chunk_fn=server.split_note)\n"
        "        print(result)\n"
        "finally:\n"
        "    server._get_pool().putconn(conn)\n"
    )


def _run_migration_snippet(target: str, *, dry_run: bool, settings: dict):
    env = {**os.environ, **settings}
    return run(
        [sys.executable, "-c", _migration_snippet(target, dry_run=dry_run)], env=env, check=False
    )


# ── Version command ───────────────────────────────────────────────────────────

_GITHUB_REPO = "Joynik/obsidian-semantic-mcp"
_LATEST_RELEASE_URL = f"https://api.github.com/repos/{_GITHUB_REPO}/releases/latest"


def _fetch_latest_release_tag(timeout: float = 3.0):
    """Return the latest GitHub release tag (without leading 'v'), or None on failure."""
    try:
        resp = requests.get(_LATEST_RELEASE_URL, timeout=timeout)
        if resp.status_code != 200:
            return None
        tag = (resp.json().get("tag_name") or "").lstrip("v").strip()
        return tag or None
    except Exception:
        return None


def _version_tuple(v: str):
    parts = []
    for p in v.split("."):
        try:
            parts.append(int(p))
        except ValueError:
            # non-numeric segment (e.g., rc1) — treat as -1 so numeric wins
            parts.append(-1)
    return tuple(parts)


def cmd_version():
    hr()
    info(f"Installed:  {APP_VERSION}")
    latest = _fetch_latest_release_tag()
    if latest is None:
        warn("Could not reach GitHub to check for updates")
        return
    info(f"Latest:     {latest}")
    if _version_tuple(latest) > _version_tuple(APP_VERSION):
        warn("A newer release is available — run: osm update")
    else:
        ok("You are on the latest release")


# ── Update command ────────────────────────────────────────────────────────────


def cmd_update():
    """Show the package-manager command; updates belong to Homebrew."""
    cmd_version()
    info("When installed from the tap, run: brew update && brew upgrade joynik/tap/osm")


# ── Remove command ────────────────────────────────────────────────────────────


# ── Install mode tables ───────────────────────────────────────────────────────


def cmd_init():
    global _INIT_VAULTS
    if platform.system() != "Darwin":
        fail("This setup wizard currently supports native macOS only")
        sys.exit(1)
    _INIT_VAULTS = None
    mode_native_macos()


# ── Dashboard command ─────────────────────────────────────────────────────────


def _dashboard_control(command: bytes) -> str | None:
    from src.launcher import _run_path

    path = _run_path() / "osm-dashboard.sock"
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
            connection.settimeout(1)
            connection.connect(str(path))
            connection.sendall(command)
            return connection.recv(32).decode("ascii").strip()
    except OSError, UnicodeError:
        return None


def _dashboard_port_open(port: str) -> bool:
    try:
        with socket.create_connection(("127.0.0.1", int(port)), timeout=0.3):
            return True
    except OSError:
        return False


def _stop_running_dashboard(port: str) -> None:
    if _dashboard_control(b"OSM_DASHBOARD_STOP\n") != "OK":
        fail("Dashboard did not acknowledge the stop request")
        raise SystemExit(1)
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        if _dashboard_control(b"OSM_DASHBOARD_PING\n") is None and not _dashboard_port_open(port):
            return
        time.sleep(0.1)
    fail(f"Dashboard did not release port {port}; check its process")
    raise SystemExit(1)


def cmd_dashboard():
    """Start or stop the local dashboard independently of MCP clients."""
    if _PARAMS.get("dashboard_stop"):
        running = _dashboard_control(b"OSM_DASHBOARD_PING\n")
        if not running or not running.startswith("PONG "):
            warn("Dashboard is not running; an open browser tab may still show the page")
            return
        port = running.split(" ", 1)[1]
        _stop_running_dashboard(port)
        ok(f"Dashboard stopped; port {port} is closed")
        return

    settings = _read_env()
    settings.update(
        {
            key: value
            for key, value in dotenv_values(OSM_CONFIG_DIR / ".env").items()
            if value is not None
        }
    )
    settings.update(os.environ)
    from src.launcher import _valid_dashboard_port

    port = settings.get("DASHBOARD_PORT", "")
    if not _valid_dashboard_port(port):
        fail("Set DASHBOARD_PORT in .env, run osm init, or export it in the shell")
        raise SystemExit(1)
    running = _dashboard_control(b"OSM_DASHBOARD_PING\n")
    restarting = bool(running and running.startswith("PONG "))
    if restarting:
        _stop_running_dashboard(running.split(" ", 1)[1])
    log_path = OSM_LOG_PATH.with_name("dashboard.log")
    log_path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    fd = os.open(log_path, os.O_WRONLY | os.O_APPEND | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    try:
        details = os.fstat(fd)
        if (
            not stat.S_ISREG(details.st_mode)
            or details.st_uid != os.geteuid()
            or stat.S_IMODE(details.st_mode) != 0o600
            or details.st_nlink != 1
        ):
            raise ValueError("unsafe dashboard log file")
        process = subprocess.Popen(
            [sys.executable, "-m", "src.dashboard"],
            env=settings,
            cwd="/",
            stdin=subprocess.DEVNULL,
            stdout=fd,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    finally:
        os.close(fd)
    for _ in range(50):
        if _dashboard_control(b"OSM_DASHBOARD_PING\n") == f"PONG {port}":
            break
        if process.poll() is not None:
            fail(f"Dashboard did not become ready; see {log_path}")
            raise SystemExit(1)
        time.sleep(0.1)
    else:
        process.terminate()
        fail(f"Dashboard did not become ready; see {log_path}")
        raise SystemExit(1)
    ok("Dashboard restarted" if restarting else "Dashboard started")
    url = f"http://127.0.0.1:{port}"
    info(f"Opening {url}")
    webbrowser.open(url)


# ── Help command ──────────────────────────────────────────────────────────────


def _desktop_uri_command(uri):
    """Return an argument list for the native URI handler."""
    system = platform.system()
    if system == "Darwin":
        handler = shutil.which("open")
        app = Path("/Applications/Obsidian.app")
        if not handler or not app.is_dir():
            raise ValueError("Obsidian.app and the macOS open command are required")
        return [handler, "-a", str(app), uri]
    if system == "Linux":
        handler = shutil.which("xdg-open")
        if not handler:
            raise ValueError("xdg-open is required; register Obsidian as the obsidian URI handler")
        return [handler, uri]
    if system == "Windows" and hasattr(os, "startfile"):
        return None
    raise ValueError(f"Desktop note opening is unsupported on {system}")


def cmd_open(argv=None):
    """Dispatch existing Markdown notes to Obsidian without creating content."""
    parser = argparse.ArgumentParser(
        prog="osm open", description="Open existing Markdown notes as Obsidian desktop tabs"
    )
    parser.add_argument("--vault", required=True, help="Absolute path to the local Obsidian vault")
    parser.add_argument(
        "--base",
        default=".",
        help="Explicit note base directory within the vault (default: vault root)",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Validate and print URIs without launching"
    )
    parser.add_argument(
        "paths", nargs="+", help="Exact existing note paths; use -- before names starting with -"
    )
    args = parser.parse_args(sys.argv[2:] if argv is None else argv)
    try:
        vault_arg = Path(args.vault).expanduser()
        if not vault_arg.is_absolute():
            raise ValueError("--vault must be an absolute path")
        vault = vault_arg.resolve(strict=True)
        if not vault.is_dir() or not (vault / ".obsidian").is_dir():
            raise ValueError(f"Not an Obsidian vault: {vault}")
        base = (vault / Path(args.base).expanduser()).resolve(strict=True)
        if not base.is_dir() or not base.is_relative_to(vault):
            raise ValueError("--base must be a directory inside the vault")
        targets = []
        for supplied in args.paths:
            path = (base / Path(supplied).expanduser()).resolve(strict=True)
            if not path.is_relative_to(vault):
                raise ValueError(f"Note is outside the vault: {supplied}")
            if not path.is_file() or path.suffix.lower() != ".md":
                raise ValueError(f"Not a regular Markdown note: {supplied}")
            if "#" in str(path):
                raise ValueError(
                    f"Cannot safely open a path containing a heading delimiter (#): {supplied}"
                )
            uri = "obsidian://open?" + urllib.parse.urlencode(
                {"path": str(path), "paneType": "tab"}, quote_via=urllib.parse.quote, safe=""
            )
            targets.append((path, hashlib.sha256(path.read_bytes()).digest(), uri))
        commands = [_desktop_uri_command(uri) for _, _, uri in targets]
        if args.dry_run:
            for _, _, uri in targets:
                info(f"[dry-run] {uri}")
            return
        dispatched = 0
        try:
            for (path, digest, uri), command in zip(targets, commands, strict=True):
                if hashlib.sha256(path.read_bytes()).digest() != digest:
                    raise ValueError(f"Note changed after validation: {path}")
                if command is None:
                    # startfile exists only on Windows; this branch is selected there.
                    startfile = getattr(os, "startfile", None)
                    if startfile is None:
                        raise OSError("Windows desktop launcher is unavailable")
                    startfile(uri)
                else:
                    subprocess.run(command, check=True, timeout=10)
                dispatched += 1
            for path, digest, _ in targets:
                if hashlib.sha256(path.read_bytes()).digest() != digest:
                    raise ValueError(f"Note changed during opening: {path}")
        except (OSError, ValueError, subprocess.SubprocessError) as exc:
            fail(f"Desktop dispatch failed after {dispatched} of {len(targets)} requests: {exc}")
            raise SystemExit(1) from exc
        ok(f"Dispatched {dispatched} desktop opening requests; note bytes unchanged")
        info("Verify tab selection in Obsidian; URI dispatch alone does not prove the tabs opened")
    except (OSError, ValueError, RuntimeError) as exc:
        fail(f"Cannot open notes: {exc}")
        raise SystemExit(1) from exc


def cmd_help():
    info("Usage: osm <command> [flags]")
    table = Table(show_header=True, box=None, padding=(0, 2))
    table.add_column("Command", style="cyan bold")
    table.add_column("Purpose", overflow="fold")
    for name, (_, description) in COMMANDS.items():
        table.add_row(f"osm {name}", description)
    _console.print(table)
    header("Native setup")
    _console.print("  osm init")
    _console.print("  osm init --vault PATH --database-backend qdrant --qdrant-url URL")
    _console.print("  osm init --vault PATH --database-backend postgresql --database-url URL")
    _console.print("  osm init --ollama-url URL --embedding-model MODEL")
    _console.print("  osm init --search-language russian|english|simple")
    _console.print("  osm init --dry-run")
    _console.print("  osm migrate --embedding-model MODEL [--dry-run]")


def cmd_run():
    """Start the MCP server without writing to standard output."""
    from src.launcher import main as run_server

    run_server()


def cmd_stop():
    """Request shutdown of locally tracked osm run processes."""
    from src.launcher import stop_runs

    try:
        stopped, stale = stop_runs()
    except (OSError, ValueError) as exc:
        fail(f"Could not stop OSM server: {exc}")
        raise SystemExit(1) from exc
    info(f"Stop signal sent to {stopped} OSM server process(es)")
    if stale:
        info(f"Removed {stale} stale process record(s)")
    if stopped:
        info("An MCP client may start a new server process when it reconnects")


def cmd_disconnect():
    """Remove OSM's user-level MCP registrations from supported clients."""
    remove_codex_config()
    remove_claude_code_config()
    remove_pi_agent_config()
    remove_opencode_config()
    remove_claude_desktop_config()
    info("Restart connected clients to apply; OSM settings and indexed data remain available")


def _open_log():
    """Open the owner-only log without following a symlink."""
    fd = os.open(OSM_LOG_PATH, os.O_RDONLY | os.O_NOFOLLOW)
    details = os.fstat(fd)
    if (
        not stat.S_ISREG(details.st_mode)
        or details.st_uid != os.geteuid()
        or stat.S_IMODE(details.st_mode) != 0o600
        or details.st_nlink != 1
    ):
        os.close(fd)
        raise ValueError("unsafe OSM log file")
    return os.fdopen(fd, "r", encoding="utf-8", errors="replace")


def cmd_logs():
    """Show recent server messages, optionally following new lines."""
    try:
        lines = int(_PARAMS.get("lines", 100))
        if not 1 <= lines <= 1000:
            raise ValueError("--lines must be between 1 and 1000")
    except ValueError as exc:
        fail(str(exc))
        raise SystemExit(2) from exc
    follow = bool(_PARAMS.get("follow"))
    try:
        stream = _open_log()
    except FileNotFoundError:
        warn("No OSM log yet; start osm run to create it")
        return
    except (OSError, ValueError) as exc:
        fail(f"Cannot read OSM log ({type(exc).__name__})")
        raise SystemExit(1) from exc
    info(f"Log: {OSM_LOG_PATH}")
    try:
        for line in deque(stream, maxlen=lines):
            _console.print(Text(line.rstrip("\n")))
        if not follow:
            return
        info("Following log; press Ctrl+C to stop")
        try:
            while True:
                line = stream.readline()
                if line:
                    _console.print(Text(line.rstrip("\n")))
                    continue
                time.sleep(0.5)
                if OSM_LOG_PATH.stat().st_ino != os.fstat(stream.fileno()).st_ino:
                    stream.close()
                    stream = _open_log()
        except KeyboardInterrupt:
            _console.print()
        except (OSError, ValueError) as exc:
            fail(f"Log follow stopped ({type(exc).__name__})")
    finally:
        stream.close()


# ── Entry point ───────────────────────────────────────────────────────────────

COMMANDS = {
    "init": (cmd_init, "Interactive setup wizard"),
    "run": (cmd_run, "Start the MCP server"),
    "stop": (cmd_stop, "Stop running local MCP server processes"),
    "disconnect": (cmd_disconnect, "Remove OSM from user-level MCP client settings"),
    "logs": (cmd_logs, "Show recent server logs; use --follow for live output"),
    "status": (cmd_status, "Check service health"),
    "vaults": (cmd_vaults, "List configured Obsidian vault(s)"),
    "dashboard": (cmd_dashboard, "Start the dashboard and open it in a browser"),
    "open": (cmd_open, "Open existing Markdown notes as Obsidian desktop tabs"),
    "update": (cmd_update, "Show the Homebrew update command"),
    "version": (cmd_version, "Show installed version and check for updates (-v, -V)"),
    "migrate": (cmd_migrate, "Safely re-embed notes with a new model"),
    "help": (cmd_help, "Show this help message"),
}


def _menu_command() -> str | None:
    """Choose a CLI command when osm is started in a terminal without arguments."""
    choices = [
        questionary.Choice(f"osm {name:<18} {description}", value=name)
        for name, (_, description) in COMMANDS.items()
    ]
    choices.insert(
        choices.index(next(choice for choice in choices if choice.value == "open")),
        questionary.Choice(
            f"{'osm dashboard stop':<22} Stop the dashboard", value="dashboard_stop"
        ),
    )
    choices.append(questionary.Choice("Quit", value="quit"))
    try:
        selected = questionary.select(
            "Choose a command",
            choices=choices,
            style=_prompt_style,
            instruction="(↑/↓ and Enter)",
        ).ask()
    except KeyboardInterrupt:
        return None
    if selected in (None, "quit"):
        return None
    if selected == "dashboard_stop":
        _PARAMS["dashboard_stop"] = True
        return "dashboard"
    if selected == "migrate":
        _PARAMS["embedding_model"] = prompt("New embedding model")
    if selected == "open":
        settings = _read_env()
        vaults = [v.strip() for v in settings.get("OBSIDIAN_VAULTS", "").split(",") if v.strip()]
        if not vaults and settings.get("OBSIDIAN_VAULT"):
            vaults = [settings["OBSIDIAN_VAULT"]]
        if len(vaults) > 1:
            vault = questionary.select("Choose a vault", choices=vaults, style=_prompt_style).ask()
            if vault is None:
                return None
        else:
            vault = vaults[0] if vaults else prompt("Absolute path to Obsidian vault")
        note = prompt("Markdown note path relative to vault")
        cmd_open(["--vault", vault, note])
        return None
    return selected


_FLAG_MAP = {
    # flag name (without --) → _PARAMS key
    "vault": "vault",
    "database-url": "database_url",
    "database-backend": "database_backend",
    "qdrant-url": "qdrant_url",
    "ollama-url": "ollama_url",
    "embedding-model": "embedding_model",
    "search-language": "search_language",
    "dashboard-port": "dashboard_port",
    "lines": "lines",
}


def _parse_flags(args):
    """
    Pull --key=value / --key value / boolean --flag pairs out of args.
    Returns (remaining_args, params_dict).
    """
    global DRY_RUN
    params = {}
    remaining = []
    i = 0
    while i < len(args):
        a = args[i]
        if a == "--dry-run":
            DRY_RUN = True
            i += 1
            continue
        if a == "--follow":
            params["follow"] = True
            i += 1
            continue
        if a.startswith("--"):
            key = a[2:]
            if "=" in key:
                k, v = key.split("=", 1)
                if k in _FLAG_MAP:
                    params[_FLAG_MAP[k]] = v
                    i += 1
                    continue
            elif key in _FLAG_MAP and i + 1 < len(args) and not args[i + 1].startswith("--"):
                params[_FLAG_MAP[key]] = args[i + 1]
                i += 2
                continue
        remaining.append(a)
        i += 1
    return remaining, params


def main():
    global DRY_RUN, _PARAMS

    if len(sys.argv) > 1 and sys.argv[1] == "run":
        if len(sys.argv) != 2:
            print("osm run: unexpected arguments", file=sys.stderr)
            raise SystemExit(2)
        cmd_run()
        return

    if len(sys.argv) > 1 and sys.argv[1] == "open":
        header(f"OSM v{APP_VERSION}")
        cmd_open(sys.argv[2:])
        return

    args, _PARAMS = _parse_flags(sys.argv[1:])

    header(f"OSM v{APP_VERSION}")

    if args in (["-v"], ["-V"]):
        args = ["version"]

    if DRY_RUN:
        info("Dry-run mode — no changes will be made")
        print()

    if not args and _stdin_is_tty():
        selected = _menu_command()
        if selected is None:
            return
        args = [selected]

    # Non-interactive invocation or explicit help request
    if not args or args[0] in ("--help", "-h"):
        cmd_help()
        sys.exit(0)

    cmd = args[0]

    if cmd == "dashboard":
        if len(args) > 2 or (len(args) == 2 and args[1] != "stop"):
            fail("Usage: osm dashboard [stop]")
            raise SystemExit(2)
        _PARAMS["dashboard_stop"] = len(args) == 2

    if cmd not in COMMANDS:
        fail(f"Unknown command: {cmd!r}")
        print()
        cmd_help()
        sys.exit(1)

    COMMANDS[cmd][0]()

    if DRY_RUN:
        _done_dry_run()


if __name__ == "__main__":
    main()
