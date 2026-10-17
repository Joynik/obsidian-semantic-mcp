"""
launcher.py — path-agnostic runtime entry point for osm run.

The MCP server runs in-process. Installed settings live in the user config
directory; a source checkout may also use its own gitignored .env file.

Environment variables:
  OBSIDIAN_VAULT      vault path (required unless OBSIDIAN_VAULTS is set)
  OBSIDIAN_VAULTS     comma-separated vault paths
  DATABASE_URL        postgres connection string (or set POSTGRES_PASSWORD)
  POSTGRES_PASSWORD   postgres password
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import secrets
import signal
import socket
import stat
import sys
import tempfile
import threading
from logging.handlers import RotatingFileHandler
from pathlib import Path
from urllib.parse import urlsplit

from dotenv import load_dotenv

OSM_CONFIG_DIR = Path.home() / ".config" / "osm"
RUNTIME_FILE_NAME = "runtime.json"
LEGACY_RUNTIME_FILE_NAME = "native_runtime.json"
OSM_LOG_PATH = Path.home() / ".local" / "state" / "obsidian-semantic-mcp" / "osm.log"


class _PrivateRotatingHandler(RotatingFileHandler):
    """Create each log file with owner-only permissions and no symlink traversal."""

    def _open(self):
        flags = os.O_WRONLY | os.O_APPEND | os.O_CREAT | os.O_NOFOLLOW
        fd = os.open(self.baseFilename, flags, 0o600)
        details = os.fstat(fd)
        if (
            not stat.S_ISREG(details.st_mode)
            or details.st_uid != os.geteuid()
            or stat.S_IMODE(details.st_mode) != 0o600
            or details.st_nlink != 1
        ):
            os.close(fd)
            raise ValueError("unsafe OSM log file")
        return os.fdopen(fd, "a", encoding="utf-8")


class _SecretFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        for secret in (os.environ.get("DATABASE_URL"), os.environ.get("POSTGRES_PASSWORD")):
            if secret:
                message = message.replace(secret, "[redacted]")
        record.msg = message
        record.args = ()
        return True


def _configure_logging() -> None:
    """Keep server logs on stderr and in a bounded private file."""
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] [pid=%(process)d] %(message)s")
    secrets_filter = _SecretFilter()
    if not any(
        isinstance(handler, logging.StreamHandler) and not isinstance(handler, RotatingFileHandler)
        for handler in root.handlers
    ):
        stream = logging.StreamHandler(sys.stderr)
        stream.setFormatter(formatter)
        stream.addFilter(secrets_filter)
        root.addHandler(stream)
    try:
        directory = OSM_LOG_PATH.parent
        directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        details = directory.lstat()
        if (
            not stat.S_ISDIR(details.st_mode)
            or details.st_uid != os.geteuid()
            or stat.S_IMODE(details.st_mode) & 0o077
        ):
            raise ValueError("unsafe OSM log directory")
        handler = _PrivateRotatingHandler(OSM_LOG_PATH, maxBytes=2_000_000, backupCount=3)
        handler.setFormatter(formatter)
        handler.addFilter(secrets_filter)
        root.addHandler(handler)
    except (OSError, ValueError) as exc:
        logging.getLogger(__name__).warning("File logging unavailable (%s)", type(exc).__name__)


def _validate_native_runtime(data):
    if not isinstance(data, dict) or set(data) != {"version", "env"}:
        raise ValueError("invalid native runtime schema")
    if (
        type(data["version"]) is not int
        or data["version"] != 1
        or not isinstance(data["env"], dict)
    ):
        raise ValueError("invalid native runtime schema")
    env = data["env"]
    vault_keys = set(env) & {"OBSIDIAN_VAULT", "OBSIDIAN_VAULTS"}
    allowed_env = {
        "DATABASE_URL",
        "OLLAMA_URL",
        "EMBEDDING_MODEL",
        "OSM_DATABASE_BACKEND",
        "QDRANT_URL",
        "DASHBOARD_PORT",
        "OSM_SEARCH_LANGUAGE",
        "OBSIDIAN_INDEX_ARCHIVE",
        "OBSIDIAN_ARCHIVE_FOLDER",
    } | vault_keys
    backend = env.get("OSM_DATABASE_BACKEND", "postgresql")
    required_env = ({"DATABASE_URL"} if backend == "postgresql" else {"QDRANT_URL"}) | vault_keys
    if len(vault_keys) != 1 or not required_env <= set(env) or set(env) - allowed_env:
        raise ValueError("invalid native runtime variables")
    if any(not isinstance(value, str) or not value or "\x00" in value for value in env.values()):
        raise ValueError("invalid native runtime values")
    if backend not in ("postgresql", "qdrant"):
        raise ValueError("invalid native runtime values")
    if backend == "postgresql" and not env["DATABASE_URL"].startswith(
        ("postgresql://", "postgres://")
    ):
        raise ValueError("invalid PostgreSQL URL")
    if backend == "qdrant":
        if "DATABASE_URL" in env:
            raise ValueError("Qdrant settings must not include PostgreSQL")
        parts = urlsplit(env["QDRANT_URL"])
        if (
            parts.scheme != "http"
            or parts.hostname not in ("localhost", "127.0.0.1", "::1")
            or parts.username
            or parts.password
            or parts.path not in ("", "/")
            or parts.query
            or parts.fragment
        ):
            raise ValueError("invalid local Qdrant URL")
    if "OLLAMA_URL" in env:
        parts = urlsplit(env["OLLAMA_URL"])
        if (
            parts.scheme not in ("http", "https")
            or not parts.hostname
            or parts.username
            or parts.password
            or parts.path not in ("", "/")
            or parts.query
            or parts.fragment
        ):
            raise ValueError("invalid Ollama URL")
    if any(
        "\r" in env[name] or "\n" in env[name]
        for name in ("OLLAMA_URL", "EMBEDDING_MODEL")
        if name in env
    ):
        raise ValueError("invalid embedding settings")
    if "DASHBOARD_PORT" in env and not _valid_dashboard_port(env["DASHBOARD_PORT"]):
        raise ValueError("invalid dashboard port")
    if env.get("OSM_SEARCH_LANGUAGE", "russian") not in ("russian", "english", "simple"):
        raise ValueError("invalid text search language")
    if env.get("OBSIDIAN_INDEX_ARCHIVE", "no") not in ("yes", "no"):
        raise ValueError("invalid archive indexing setting")
    archive_folder = env.get("OBSIDIAN_ARCHIVE_FOLDER")
    if archive_folder is not None and (
        archive_folder != archive_folder.strip()
        or archive_folder in {".", ".."}
        or any(char in archive_folder for char in ("/", "\\", "\r", "\n"))
    ):
        raise ValueError("invalid archive folder name")
    vaults = (
        env[next(iter(vault_keys))].split(",")
        if "OBSIDIAN_VAULTS" in vault_keys
        else [env["OBSIDIAN_VAULT"]]
    )
    if any(not Path(vault).is_absolute() for vault in vaults):
        raise ValueError("invalid native vault paths")
    return env


def _read_native_runtime(path):
    """Read an owner-only regular file without following links."""
    if os.name != "posix":
        if path.exists() or path.is_symlink():
            raise ValueError("native runtime permission validation requires POSIX")
        return None
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise ValueError("unsafe native runtime file") from exc
    with os.fdopen(fd, "r", encoding="utf-8") as source:
        info = os.fstat(source.fileno())
        if (
            not stat.S_ISREG(info.st_mode)
            or stat.S_IMODE(info.st_mode) != 0o600
            or info.st_uid != os.geteuid()
            or info.st_nlink != 1
            or info.st_size > 65536
        ):
            raise ValueError("unsafe native runtime file")
        return _validate_native_runtime(json.loads(source.read(65537)))


def load_native_runtime(config_dir=None):
    """Load current settings, copying a validated legacy file on first use."""
    directory = OSM_CONFIG_DIR if config_dir is None else Path(config_dir)
    path = directory / RUNTIME_FILE_NAME
    env = _read_native_runtime(path)
    if env is not None:
        return env

    legacy = directory.parent / "obsidian-semantic-mcp" / LEGACY_RUNTIME_FILE_NAME
    env = _read_native_runtime(legacy)
    if env is None:
        return None

    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    info = directory.lstat()
    if (
        not stat.S_ISDIR(info.st_mode)
        or info.st_uid != os.geteuid()
        or stat.S_IMODE(info.st_mode) & 0o022
    ):
        raise ValueError("unsafe native runtime directory")
    encoded = json.dumps({"version": 1, "env": env}, ensure_ascii=False, indent=2) + "\n"
    try:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    except FileExistsError:
        return _read_native_runtime(path)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as target:
            target.write(encoded)
            target.flush()
            os.fsync(target.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise
    return _read_native_runtime(path)


def _valid_dashboard_port(value: str) -> bool:
    return value.isdecimal() and 1 <= int(value) <= 65535


def write_native_runtime(
    vaults,
    db_url=None,
    config_dir=None,
    ollama_url=None,
    embedding_model=None,
    database_backend="postgresql",
    qdrant_url=None,
    dashboard_port=None,
    search_language=None,
    index_archive=None,
    archive_folder=None,
):
    """Atomically persist native settings outside every MCP client's config."""
    vaults = [vaults] if isinstance(vaults, str) else list(vaults)
    env = {"OSM_DATABASE_BACKEND": database_backend}
    if database_backend == "qdrant":
        env["QDRANT_URL"] = qdrant_url
    else:
        env["DATABASE_URL"] = db_url
    if ollama_url is not None:
        env["OLLAMA_URL"] = ollama_url
    if embedding_model is not None:
        env["EMBEDDING_MODEL"] = embedding_model
    if dashboard_port is not None:
        env["DASHBOARD_PORT"] = dashboard_port
    if search_language is not None:
        env["OSM_SEARCH_LANGUAGE"] = search_language
    if index_archive is not None:
        env["OBSIDIAN_INDEX_ARCHIVE"] = "yes" if index_archive else "no"
    if archive_folder is not None:
        env["OBSIDIAN_ARCHIVE_FOLDER"] = archive_folder
    if not vaults:
        raise ValueError("native runtime requires a vault")
    env["OBSIDIAN_VAULTS" if len(vaults) > 1 else "OBSIDIAN_VAULT"] = ",".join(vaults)
    data = {"version": 1, "env": env}
    _validate_native_runtime(data)
    encoded = json.dumps(data, ensure_ascii=False, indent=2) + "\n"
    if len(encoded.encode("utf-8")) > 65536 or os.name != "posix":
        raise ValueError("unsupported native runtime configuration")
    directory = OSM_CONFIG_DIR if config_dir is None else Path(config_dir)
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    info = directory.lstat()
    if (
        not stat.S_ISDIR(info.st_mode)
        or info.st_uid != os.geteuid()
        or stat.S_IMODE(info.st_mode) & 0o022
    ):
        raise ValueError("unsafe native runtime directory")
    path = directory / RUNTIME_FILE_NAME
    _read_native_runtime(path)
    fd, temporary = tempfile.mkstemp(prefix=".native-runtime-", dir=directory)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as target:
            os.fchmod(target.fileno(), 0o600)
            target.write(encoded)
            target.flush()
            os.fsync(target.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def _load_native_runtime() -> bool:
    env = load_native_runtime()
    if env is None:
        return False
    for name, value in env.items():
        if name in ("OBSIDIAN_VAULT", "OBSIDIAN_VAULTS") and (
            "OBSIDIAN_VAULT" in os.environ or "OBSIDIAN_VAULTS" in os.environ
        ):
            continue
        if name == "DATABASE_URL" and (
            "DATABASE_URL" in os.environ or "POSTGRES_PASSWORD" in os.environ
        ):
            continue
        os.environ.setdefault(name, value)
    return True


def remove_native_runtime(config_dir=None):
    """Remove only the native runtime file, leaving other OSM configuration intact."""
    directory = OSM_CONFIG_DIR if config_dir is None else Path(config_dir)
    try:
        info = directory.lstat()
    except FileNotFoundError:
        return
    if not stat.S_ISDIR(info.st_mode) or (os.name == "posix" and info.st_uid != os.geteuid()):
        raise ValueError("unsafe native runtime directory")
    path = directory / RUNTIME_FILE_NAME
    try:
        info = path.lstat()
    except FileNotFoundError:
        return
    if os.name == "posix" and info.st_uid != os.geteuid():
        raise ValueError("unsafe native runtime file")
    path.unlink()


def _load_checkout_env() -> None:
    """Use .env only when running directly from a source checkout."""
    root = Path(__file__).resolve().parent.parent
    if (root / "pyproject.toml").is_file():
        load_dotenv(root / ".env", override=False)


def _validate_env() -> None:
    if not os.environ.get("OBSIDIAN_VAULTS") and not os.environ.get("OBSIDIAN_VAULT"):
        logging.getLogger(__name__).error("Missing Obsidian vault configuration")
        print("osm run: missing OBSIDIAN_VAULTS or OBSIDIAN_VAULT", file=sys.stderr)
        sys.exit(1)
    if not os.environ.get("OLLAMA_URL"):
        print("osm run: missing OLLAMA_URL", file=sys.stderr)
        sys.exit(1)
    if os.environ.get("OSM_DATABASE_BACKEND", "postgresql") == "qdrant":
        if not os.environ.get("QDRANT_URL"):
            print("osm run: missing QDRANT_URL", file=sys.stderr)
            sys.exit(1)
        return
    if not os.environ.get("DATABASE_URL") and not os.environ.get("POSTGRES_PASSWORD"):
        logging.getLogger(__name__).error("Missing PostgreSQL configuration")
        print("osm run: missing DATABASE_URL or POSTGRES_PASSWORD", file=sys.stderr)
        sys.exit(1)


def _run_server() -> None:
    from src.server import run_server

    run_server()


def _run_path() -> Path:
    namespace = hashlib.sha256(str(OSM_CONFIG_DIR).encode("utf-8")).hexdigest()[:8]
    return Path("/private/tmp") / f"osm-{os.geteuid()}-{namespace}"


def _run_directory() -> Path:
    directory = _run_path()
    directory.mkdir(mode=0o700, exist_ok=True)
    details = directory.lstat()
    if (
        not stat.S_ISDIR(details.st_mode)
        or details.st_uid != os.geteuid()
        or stat.S_IMODE(details.st_mode) & 0o077
    ):
        raise ValueError("unsafe OSM run directory")
    return directory


def _register_run() -> tuple[socket.socket, Path]:
    path = _run_directory() / f"osm-run-{os.getpid()}-{secrets.token_hex(4)}.sock"
    listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        listener.bind(str(path))
        path.chmod(0o600)
        listener.listen(1)
        listener.settimeout(0.5)

        def serve_stop() -> None:
            while listener.fileno() >= 0:
                try:
                    connection, _ = listener.accept()
                except TimeoutError:
                    continue
                except OSError:
                    return
                with connection:
                    connection.settimeout(1)
                    try:
                        if connection.recv(16) == b"OSM_STOP\n":
                            connection.sendall(b"OK\n")
                            os.kill(os.getpid(), signal.SIGTERM)
                            return
                    except OSError:
                        continue

        threading.Thread(target=serve_stop, daemon=True, name="osm-stop").start()
        return listener, path
    except BaseException:
        listener.close()
        path.unlink(missing_ok=True)
        raise


def stop_runs() -> tuple[int, int]:
    """Signal only tracked processes whose identity still matches."""
    if not _run_path().exists():
        return 0, 0
    directory = _run_directory()
    stopped = stale = 0
    for path in directory.glob("osm-run-*.sock"):
        try:
            details = path.lstat()
        except FileNotFoundError:
            continue
        if not stat.S_ISSOCK(details.st_mode) or details.st_uid != os.geteuid():
            continue
        try:
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
                connection.settimeout(1)
                connection.connect(str(path))
                connection.sendall(b"OSM_STOP\n")
                if connection.recv(16) == b"OK\n":
                    stopped += 1
        except ConnectionRefusedError, FileNotFoundError:
            path.unlink(missing_ok=True)
            stale += 1
    return stopped, stale


def main() -> None:
    _configure_logging()
    logging.getLogger(__name__).info("OSM server starting")
    try:
        loaded = _load_native_runtime()
    except (OSError, ValueError, TypeError) as exc:
        logging.getLogger(__name__).error(
            "Invalid native runtime configuration (%s)", type(exc).__name__
        )
        print(
            f"osm run: invalid native runtime configuration ({type(exc).__name__})", file=sys.stderr
        )
        sys.exit(1)
    if not loaded:
        _load_checkout_env()
    _validate_env()
    try:
        listener, marker = _register_run()
    except (OSError, ValueError) as exc:
        logging.getLogger(__name__).error("Cannot track server process (%s)", type(exc).__name__)
        print(f"osm run: cannot track server process ({type(exc).__name__})", file=sys.stderr)
        raise SystemExit(1) from exc
    try:
        _run_server()
    finally:
        listener.close()
        marker.unlink(missing_ok=True)
        try:
            marker.parent.rmdir()
        except OSError:
            pass


if __name__ == "__main__":
    main()
