"""Versioned schema migrations for OSM.

The notes table stays in server.py because its vector dimension comes from the
configured Ollama model. Each migration and its schema_version record commit
in one transaction, preserving earlier migrations if a later one fails.
"""

from __future__ import annotations

import logging
import re
import uuid
from collections.abc import Callable
from dataclasses import dataclass

import psycopg2
from psycopg2 import sql

try:
    from .config import search_language
except ImportError:
    from config import search_language

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Migration:
    version: int
    name: str
    apply: Callable  # Callable[[psycopg2 connection], None] — runs its own cursor(s)


def _migration_1_baseline(conn) -> None:
    """Create note_links and the pre-existing indexes, IF NOT EXISTS
    throughout. Safe to run against both a fresh database and an
    established one that already has these objects."""
    with conn.cursor() as cur:
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS note_links (
                source_path TEXT NOT NULL,
                target_name TEXT NOT NULL,
                target_path TEXT,
                PRIMARY KEY (source_path, target_name)
            );
            """
        )
        cur.execute("CREATE INDEX IF NOT EXISTS note_links_target_idx ON note_links (target_path);")
        cur.execute("CREATE INDEX IF NOT EXISTS notes_vault_idx ON notes (vault_id);")
        cur.execute("CREATE INDEX IF NOT EXISTS notes_tsv_idx ON notes USING GIN (content_tsv);")


def _migration_2_index_state(conn) -> None:
    """Create index_state — one row per vault, tracking indexing status and
    the paths that failed to embed on the most recent pass (see
    server.set_index_state / get_index_state)."""
    with conn.cursor() as cur:
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS index_state (
                vault_id     TEXT PRIMARY KEY,
                status       TEXT NOT NULL DEFAULT 'idle',
                started_at   TIMESTAMP,
                finished_at  TIMESTAMP,
                failed_paths TEXT[] NOT NULL DEFAULT '{}',
                error        TEXT
            );
            """
        )


def _migration_3_embedding_profile(conn) -> None:
    """Track the active model and an interrupted replacement separately."""
    with conn.cursor() as cur:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS embedding_profile (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                active_model TEXT NOT NULL,
                staging_model TEXT,
                staging_dim INTEGER
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS embedding_archives (
                column_name TEXT PRIMARY KEY,
                model TEXT NOT NULL,
                archived_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )
        """)


def _migration_4_note_chunks(conn) -> None:
    """Store ordered passages without discarding complete note content."""
    with conn.cursor() as cur:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS note_chunks (
                path TEXT NOT NULL REFERENCES notes(path) ON DELETE CASCADE,
                model TEXT NOT NULL,
                note_hash TEXT NOT NULL,
                chunk_index INTEGER NOT NULL,
                start_offset INTEGER NOT NULL,
                end_offset INTEGER NOT NULL,
                content TEXT NOT NULL,
                embedding vector NOT NULL,
                content_tsv tsvector,
                PRIMARY KEY (path, model, chunk_index)
            )
        """)
        cur.execute("CREATE INDEX IF NOT EXISTS note_chunks_model_idx ON note_chunks (model, path)")


def _migration_5_russian_text_search(conn) -> None:
    """Rebuild only lexical vectors; existing note text and embeddings remain intact."""
    with conn.cursor() as cur:
        cur.execute("""
            UPDATE notes SET content_tsv =
                setweight(to_tsvector('russian', content), 'A') ||
                setweight(to_tsvector('simple', content), 'B')
        """)
        cur.execute("""
            UPDATE note_chunks SET content_tsv =
                setweight(to_tsvector('russian', content), 'A') ||
                setweight(to_tsvector('simple', content), 'B')
        """)


def _migration_6_text_search_profile(conn) -> None:
    """Record the language used by the lexical vectors for future changes."""
    with conn.cursor() as cur:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS text_search_profile (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                language TEXT NOT NULL
            )
        """)
        cur.execute("""
            INSERT INTO text_search_profile (id, language) VALUES (1, 'russian')
            ON CONFLICT (id) DO NOTHING
        """)


MIGRATIONS: list[Migration] = [
    Migration(1, "baseline", _migration_1_baseline),
    Migration(2, "index_state", _migration_2_index_state),
    Migration(3, "embedding_profile", _migration_3_embedding_profile),
    Migration(4, "note_chunks", _migration_4_note_chunks),
    Migration(5, "russian_text_search", _migration_5_russian_text_search),
    Migration(6, "text_search_profile", _migration_6_text_search_profile),
]


def read_embedding_profile(conn) -> tuple[str, str | None, int | None] | None:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT active_model, staging_model, staging_dim FROM embedding_profile WHERE id = 1"
        )
        return cur.fetchone()


def establish_embedding_profile(conn, model: str) -> str:
    """Record a model once; never silently relabel an existing index."""
    with conn:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO embedding_profile (id, active_model) VALUES (1, %s) "
                "ON CONFLICT (id) DO NOTHING",
                (model,),
            )
    with conn:
        profile = read_embedding_profile(conn)
    assert profile is not None
    return profile[0]


def migrate_embedding_model(
    conn,
    target_model: str,
    target_dim: int,
    embed_fn,
    *,
    dry_run: bool = False,
    progress_callback=None,
    chunk_fn=None,
) -> dict:
    """Stage vectors for a new model and atomically activate them when complete.

    A changed note invalidates its staged vector through the stored content hash.
    The final table lock makes the completeness check and switch one operation.
    """
    if not target_model or target_dim <= 0:
        raise ValueError("a model name and positive dimension are required")
    with conn:
        profile = read_embedding_profile(conn)
        if profile is None:
            raise RuntimeError("active embedding model is unknown; run osm init first")
        active_model, staging_model, staging_dim = profile
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM notes")
            total = cur.fetchone()[0]
    plan = {
        "active_model": active_model,
        "target_model": target_model,
        "target_dim": target_dim,
        "notes_to_reembed": total,
        "dry_run": dry_run,
    }
    if active_model == target_model:
        raise ValueError("target model is already active")
    if staging_model and (staging_model != target_model or staging_dim != target_dim):
        raise RuntimeError(
            "another model migration is staged; finish it before starting a different one"
        )
    if dry_run:
        return plan

    with conn:
        with conn.cursor() as cur:
            cur.execute("SET LOCAL lock_timeout = '5s'")
            cur.execute("ALTER TABLE notes ADD COLUMN IF NOT EXISTS embedding_next vector")
            cur.execute("ALTER TABLE notes ADD COLUMN IF NOT EXISTS embedding_next_hash TEXT")
            cur.execute(
                "UPDATE embedding_profile SET staging_model = %s, staging_dim = %s WHERE id = 1",
                (target_model, target_dim),
            )

    done = 0
    last_id = 0
    while True:
        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT id, content, hash FROM notes
                    WHERE id > %s AND (
                        embedding_next IS NULL OR embedding_next_hash IS DISTINCT FROM hash
                        OR NOT EXISTS (
                            SELECT 1 FROM note_chunks c
                            WHERE c.path = notes.path AND c.model = %s AND c.note_hash = notes.hash
                        )
                    )
                    ORDER BY id LIMIT 50
                """,
                    (last_id, target_model),
                )
                rows = cur.fetchall()
        if not rows:
            break
        for note_id, content, content_hash in rows:
            parts = (
                chunk_fn(content) if chunk_fn else [(0, min(len(content), 2000), content[:2000])]
            )
            vectors = [embed_fn(passage or " ") for _, _, passage in parts]
            if not vectors or any(len(vector) != target_dim for vector in vectors):
                raise ValueError("embedding model returned a different dimension")
            with conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        UPDATE notes SET embedding_next = %s::vector, embedding_next_hash = %s
                        WHERE id = %s AND hash = %s
                        RETURNING path
                    """,
                        (
                            "[" + ",".join(map(str, vectors[0])) + "]",
                            content_hash,
                            note_id,
                            content_hash,
                        ),
                    )
                    updated = cur.fetchone()
                    if updated:
                        path = updated[0]
                        cur.execute(
                            "DELETE FROM note_chunks WHERE path = %s AND model = %s",
                            (path, target_model),
                        )
                        for index, ((start, end, passage), vector) in enumerate(
                            zip(parts, vectors, strict=True)
                        ):
                            cur.execute(
                                """
                                INSERT INTO note_chunks
                                    (path, model, note_hash, chunk_index, start_offset,
                                     end_offset, content, embedding, content_tsv)
                                VALUES (%s, %s, %s, %s, %s, %s, %s, %s::vector,
                                        setweight(to_tsvector(%s::regconfig, %s), 'A') ||
                                        setweight(to_tsvector('simple', %s), 'B'))
                            """,
                                (
                                    path,
                                    target_model,
                                    content_hash,
                                    index,
                                    start,
                                    end,
                                    passage,
                                    "[" + ",".join(map(str, vector)) + "]",
                                    search_language(),
                                    passage,
                                    passage,
                                ),
                            )
            done += 1
            last_id = note_id
        if progress_callback:
            progress_callback(done, total)

    with conn:
        with conn.cursor() as cur:
            cur.execute("SET LOCAL lock_timeout = '5s'")
            cur.execute(
                "SELECT active_model, staging_model FROM embedding_profile WHERE id = 1 FOR UPDATE"
            )
            locked_profile = cur.fetchone()
            if locked_profile != (active_model, target_model):
                raise RuntimeError("embedding model changed during migration")
            cur.execute("LOCK TABLE notes IN ACCESS EXCLUSIVE MODE")
            cur.execute(
                """
                SELECT COUNT(*) FROM notes
                WHERE embedding_next IS NULL OR embedding_next_hash IS DISTINCT FROM hash
                   OR NOT EXISTS (
                       SELECT 1 FROM note_chunks c
                       WHERE c.path = notes.path AND c.model = %s AND c.note_hash = notes.hash
                   )
            """,
                (target_model,),
            )
            remaining = cur.fetchone()[0]
            if remaining:
                raise RuntimeError(
                    f"{remaining} note(s) changed during migration; rerun to finish staging"
                )
            archive_col = "embedding_archive_" + uuid.uuid4().hex[:16]
            cur.execute("DROP INDEX IF EXISTS notes_embedding_idx")
            cur.execute(
                sql.SQL("ALTER TABLE notes RENAME COLUMN embedding TO {}").format(
                    sql.Identifier(archive_col)
                )
            )
            cur.execute("ALTER TABLE notes RENAME COLUMN embedding_next TO embedding")
            cur.execute("ALTER TABLE notes DROP COLUMN embedding_next_hash")
            cur.execute(
                "INSERT INTO embedding_archives (column_name, model) VALUES (%s, %s)",
                (archive_col, active_model),
            )
            cur.execute(
                "UPDATE embedding_profile SET active_model = %s, staging_model = NULL, staging_dim = NULL WHERE id = 1",
                (target_model,),
            )
            cur.execute(
                "CREATE INDEX notes_embedding_idx ON notes USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100)"
            )
    plan.update(status="complete", backfilled=done)
    return plan


def _assert_versions_contiguous() -> None:
    versions = sorted(m.version for m in MIGRATIONS)
    expected = list(range(1, len(versions) + 1))
    if versions != expected:
        raise AssertionError(
            f"MIGRATIONS versions must be contiguous starting at 1, unique, "
            f"and sorted: got {versions}, expected {expected}"
        )


def _ensure_schema_version_table(conn) -> None:
    with conn.cursor() as cur:
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS schema_version (
                version    INTEGER PRIMARY KEY,
                name       TEXT NOT NULL,
                applied_at TIMESTAMP DEFAULT NOW()
            );
            """
        )


def current_version(conn) -> int:
    """Return the highest applied migration version, or 0 if none have been
    applied yet (including when schema_version itself does not exist)."""
    with conn.cursor() as cur:
        cur.execute("SELECT to_regclass('public.schema_version')")
        if cur.fetchone()[0] is None:
            return 0
        cur.execute("SELECT COALESCE(MAX(version), 0) FROM schema_version")
        return cur.fetchone()[0]


def apply_pending(conn) -> list[int]:
    """Apply every migration with version > current_version(), in ascending
    order. Returns the list of versions actually applied this call (empty
    if the database was already fully up to date).

    Each migration's DDL and its schema_version stamp commit together, in
    one transaction, immediately after that migration succeeds — not
    batched with any other migration. A migration that raises rolls back
    only its own transaction; every earlier migration in this call (and
    any from a previous call) stays committed and recorded.
    """
    _assert_versions_contiguous()

    with conn:
        _ensure_schema_version_table(conn)

    applied: list[int] = []
    already_at = current_version(conn)
    for migration in sorted(MIGRATIONS, key=lambda m: m.version):
        if migration.version <= already_at:
            continue
        with conn:
            migration.apply(conn)
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO schema_version (version, name) VALUES (%s, %s) "
                    "ON CONFLICT (version) DO NOTHING",
                    (migration.version, migration.name),
                )
        log.info("Applied migration %d (%s)", migration.version, migration.name)
        applied.append(migration.version)
    return applied


# ─────────────────── Embedding-dimension migration ───────────────────────────
#
# Legacy dimension-only helpers retained for callers that imported them.
# The CLI uses migrate_embedding_model(), which also records model identity.
#
# Add a column, never mutate the old one. The old column and old model
# stay authoritative until cutover explicitly (and irreversibly) succeeds,
# so a failed or interrupted migration leaves the server fully functional
# on the original dimension.


def _new_column_name(new_dim: int) -> str:
    return f"embedding_{int(new_dim)}"


# ALTER TABLE takes ACCESS EXCLUSIVE on `notes`. Without a bound, it waits
# indefinitely behind any open reader transaction — and while it waits, every
# subsequent reader queues behind IT, so a migration that merely blocks takes
# the whole table down with it. That is the opposite of this migration's
# stated guarantee that search stays available throughout.
#
# Observed 2026-07-20 running the pg suite against a real Postgres: a test
# holding an idle-in-transaction SELECT on `notes` stalled ADD COLUMN until
# the run was killed 10 minutes later. Mocks cannot reproduce this; only a
# real lock manager can.
#
# 5s is chosen to be longer than any healthy search transaction and far
# shorter than a human's patience. On timeout the DDL raises LockNotAvailable
# and the migration aborts cleanly, leaving the old column authoritative.
DDL_LOCK_TIMEOUT = "5s"


def _execute_ddl_with_lock_timeout(cur, statements: list[str | sql.Composable]) -> None:
    """Run DDL under a bounded lock wait, converting a timeout into a clear,
    actionable error rather than an unbounded stall."""
    # DDL_LOCK_TIMEOUT is a module constant ('5s'), never request input.
    # lock_timeout cannot be %s-parameterized. Not a SQLi vector.
    cur.execute(sql.SQL("SET LOCAL lock_timeout = {}").format(sql.Literal(DDL_LOCK_TIMEOUT)))
    try:
        for stmt in statements:
            cur.execute(stmt)
    except psycopg2.OperationalError as e:
        if e.pgcode != "55P03":  # lock_not_available
            raise
        raise RuntimeError(
            f"Could not acquire the table lock within {DDL_LOCK_TIMEOUT} — another "
            "connection is holding an open transaction on `notes` (a long-running "
            "search, or a session left idle in transaction). No changes were made "
            "and the existing embedding column is untouched. Stop the dashboard "
            "and MCP server, or wait for the reader to finish, then re-run."
        ) from e


def add_embedding_column(conn, new_dim: int) -> None:
    """ADD COLUMN embedding_<new_dim> vector(new_dim), IF NOT EXISTS.

    Purely additive — never touches the existing `embedding` column, so a
    failure here (or anywhere before cutover_embedding_column) leaves
    search fully functional on the original column.
    """
    col = _new_column_name(new_dim)
    # new_dim is an int validated by the caller (osm_cli.py's cmd_migrate
    # rejects non-integer/non-positive input before this is ever called),
    # never request input, and pgvector's vector(N) type doesn't accept a
    # %s parameter for its dimension — DDL, not a SQLi vector.
    with conn:
        with conn.cursor() as cur:
            _execute_ddl_with_lock_timeout(
                cur,
                [
                    # Identifiers are composed with psycopg2.sql.Identifier, which
                    # quotes and escapes them at the driver level -- injection-safe by
                    # construction rather than by argument. `new_dim` is an int the
                    # caller validated; vector(N) is a type, not a value, so it cannot
                    # be %s-parameterized.
                    sql.SQL("ALTER TABLE notes ADD COLUMN IF NOT EXISTS {} vector({})").format(
                        sql.Identifier(col), sql.SQL(str(int(new_dim)))
                    ),
                ],
            )


def backfill_embedding_column(
    conn,
    new_dim: int,
    embed_fn,
    *,
    batch_size: int = 50,
    progress_callback=None,
) -> int:
    """Re-embed every row whose new column is still NULL, in batches of
    `batch_size` (default 50, comfortably under the plan's "at least every
    100 notes" progress-reporting requirement).

    `embed_fn(text: str) -> list[float]` is injected rather than imported
    directly, so callers (and tests) can stub it without touching Ollama.

    `progress_callback(done: int, total: int)`, if given, is called after
    every batch — the caller (typically osm_cli.py's cmd_migrate, wiring
    server.set_index_state) uses this to make an hours-long run observable
    rather than opaque.

    Returns the number of rows backfilled. Raises (without ever reaching
    cutover_embedding_column) if embed_fn raises for any row — e.g. Ollama
    becomes unreachable mid-run. The old `embedding` column and all its
    data are untouched by this function regardless of how it exits; the
    old model stays authoritative because cutover never runs.
    """
    col = _new_column_name(new_dim)
    with conn.cursor() as cur:
        # `col` is _new_column_name(new_dim), built from an int the caller
        # validated (osm_cli rejects non-integer/non-positive input) — never
        # request input. pgvector's vector(N) and SQL identifiers cannot be
        # %s-parameterized; every real value is. Not a SQLi vector.
        cur.execute(
            sql.SQL("SELECT id, content FROM notes WHERE {} IS NULL ORDER BY id").format(
                sql.Identifier(col)
            )
        )
        rows = cur.fetchall()

    total = len(rows)
    done = 0
    for i in range(0, total, batch_size):
        chunk = rows[i : i + batch_size]
        with conn:
            with conn.cursor() as cur:
                for note_id, content in chunk:
                    vec = embed_fn(content)
                    vec_literal = "[" + ",".join(str(v) for v in vec) + "]"
                    # `col` is _new_column_name(new_dim), built from an int the caller
                    # validated (osm_cli rejects non-integer/non-positive input) — never
                    # request input. pgvector's vector(N) and SQL identifiers cannot be
                    # %s-parameterized; every real value is. Not a SQLi vector.
                    cur.execute(
                        sql.SQL("UPDATE notes SET {} = %s::vector WHERE id = %s").format(
                            sql.Identifier(col)
                        ),
                        (vec_literal, note_id),
                    )
        done += len(chunk)
        if progress_callback is not None:
            progress_callback(done, total)
    return done


def cutover_embedding_column(conn, new_dim: int) -> None:
    """Drop the old `embedding` column and rename embedding_<new_dim> to
    `embedding`, in one transaction, then rebuild the IVFFlat index.

    Refuses to run while any row's new column is still NULL — a
    partially-backfilled migration must never cut over. This is the ONLY
    irreversible step in the whole migration; everything before it is
    additive and safe to abandon.
    """
    col = _new_column_name(new_dim)
    with conn.cursor() as cur:
        # `col` is _new_column_name(new_dim), built from an int the caller
        # validated (osm_cli rejects non-integer/non-positive input) — never
        # request input. pgvector's vector(N) and SQL identifiers cannot be
        # %s-parameterized; every real value is. Not a SQLi vector.
        cur.execute(
            sql.SQL("SELECT COUNT(*) FROM notes WHERE {} IS NULL").format(sql.Identifier(col))
        )
        remaining = cur.fetchone()[0]
    if remaining > 0:
        raise RuntimeError(
            f"cutover refused: {remaining} row(s) still have a NULL {col} — "
            f"backfill is incomplete. Re-run the migration to finish "
            f"backfilling before cutover is attempted again."
        )

    with conn:
        with conn.cursor() as cur:
            # Same ACCESS EXCLUSIVE hazard as add_embedding_column, and worse
            # here: this is the irreversible step, so stalling mid-cutover is
            # the least acceptable place to block every reader.
            _execute_ddl_with_lock_timeout(
                cur,
                [
                    sql.SQL("ALTER TABLE notes DROP COLUMN embedding"),
                    # `col` is _new_column_name(new_dim), built from an int the caller
                    # validated (osm_cli rejects non-integer/non-positive input) — never
                    # request input. pgvector's vector(N) and SQL identifiers cannot be
                    # %s-parameterized; every real value is. Not a SQLi vector.
                    sql.SQL("ALTER TABLE notes RENAME COLUMN {} TO embedding").format(
                        sql.Identifier(col)
                    ),
                    sql.SQL("DROP INDEX IF EXISTS notes_embedding_idx"),
                    sql.SQL(
                        "CREATE INDEX notes_embedding_idx ON notes "
                        "USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100)"
                    ),
                ],
            )


def migrate_embedding_dimension(
    conn,
    old_dim: int | None,
    new_dim: int,
    embed_fn,
    *,
    dry_run: bool = False,
    progress_callback=None,
) -> dict:
    """Orchestrate a non-destructive embedding-dimension change.

    `old_dim`, if given, is validated against the CURRENT `notes.embedding`
    column's actual dimension before anything else runs — a guard against
    a caller invoking this legacy helper with a stale
    assumption about the current dimension. Pass `None` to skip this check
    (the add/backfill/cutover steps below don't otherwise need it — the
    new dimension alone determines what they do).

    `dry_run=True` reports the plan (row count to re-embed, whether a
    dimension mismatch actually exists) and writes nothing.

    Real run: add_embedding_column -> backfill_embedding_column ->
    cutover_embedding_column, in that order. If backfill raises (e.g.
    Ollama becomes unreachable), this function lets the exception
    propagate WITHOUT calling cutover — the old column and old model stay
    authoritative. Search remains available throughout: the old
    `embedding` column is never touched until cutover's single, fast
    (metadata-only DROP + RENAME) transaction.
    """
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT format_type(a.atttypid, a.atttypmod)
            FROM pg_attribute a
            JOIN pg_class c ON c.oid = a.attrelid
            WHERE c.relname = 'notes' AND a.attname = 'embedding'
              AND a.attnum > 0 AND NOT a.attisdropped
            """
        )
        row = cur.fetchone()
    current_dim = None
    if row:
        m = re.search(r"vector\((\d+)\)", row[0])
        if m:
            current_dim = int(m.group(1))

    if old_dim is not None and current_dim is not None and old_dim != current_dim:
        raise RuntimeError(
            f"migrate_embedding_dimension: expected current dimension "
            f"vector({old_dim}) but notes.embedding is actually "
            f"vector({current_dim}) — refusing to proceed with a stale "
            f"assumption. Re-check the database's current vector dimension."
        )

    plan = {
        "current_dim": current_dim,
        "new_dim": new_dim,
        "dry_run": dry_run,
    }

    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM notes")
        plan["notes_to_reembed"] = cur.fetchone()[0]

    if dry_run:
        plan["status"] = "dry_run"
        return plan

    add_embedding_column(conn, new_dim)
    backfilled = backfill_embedding_column(
        conn, new_dim, embed_fn, progress_callback=progress_callback
    )
    cutover_embedding_column(conn, new_dim)

    plan["backfilled"] = backfilled
    plan["status"] = "complete"
    return plan
