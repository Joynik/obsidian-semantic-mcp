"""Child-process jobs shared by the Python entry point and the frozen binary."""

from __future__ import annotations

import shutil
import sys
from pathlib import Path


def child_command(*args: str, detached: bool = False) -> list[str]:
    compiled = globals().get("__uncompiled__", globals().get("__compiled__"))
    if compiled is not None:
        executable = compiled.process_exe
        if detached:
            invoked = sys.argv[0]
            found = shutil.which(invoked) if Path(invoked).parent == Path(".") else invoked
            if not found:
                raise FileNotFoundError(f"Cannot locate OSM executable: {invoked}")
            executable = str(Path(found).resolve())
        return [executable, "--osm-worker", *args]
    return [sys.executable, "-m", "src.workers", *args]


def run_worker(args: list[str]) -> None:
    if args == ["dashboard"]:
        from .dashboard import serve_dashboard

        serve_dashboard()
        return
    if args == ["qdrant-rebuild"]:
        from . import server

        with server.reindex_lock() as acquired:
            if not acquired:
                raise RuntimeError("reindex already in progress")
            print(server.rebuild_index_safely(server.VAULT_PATHS))
        return
    if len(args) == 3 and args[0] == "postgres-migrate" and args[2] in {"apply", "dry-run"}:
        from . import migrations, server

        target = args[1]
        dry_run = args[2] == "dry-run"
        if not dry_run:
            server.init_db(server.get_embed_dim())
        conn = server._get_pool().getconn()
        try:
            with server.reindex_lock() as acquired:
                if not acquired:
                    raise RuntimeError("reindex already in progress")
                dimension = len(server.embed("dimension probe", model=target))
                result = migrations.migrate_embedding_model(
                    conn,
                    target,
                    dimension,
                    lambda text: server.embed(text, model=target),
                    dry_run=dry_run,
                    chunk_fn=server.split_note,
                )
                print(result)
        finally:
            server._get_pool().putconn(conn)
        return
    raise SystemExit("invalid OSM worker command")


if __name__ == "__main__":
    run_worker(sys.argv[1:])
