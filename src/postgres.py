"""PostgreSQL/pgvector storage operations for the Obsidian index."""

from __future__ import annotations

import re
import time
from collections.abc import Callable
from contextlib import AbstractContextManager
from typing import Any

import psycopg2

try:
    from .config import search_language
except ImportError:
    from config import search_language


def inspect_setup(url: str) -> tuple[str, bool, str | None]:
    """Check pgvector and the model identity without changing the database."""
    with psycopg2.connect(url, connect_timeout=5) as conn, conn.cursor() as cur:
        cur.execute("SELECT extversion FROM pg_extension WHERE extname = 'vector'")
        row = cur.fetchone()
        if not row:
            raise RuntimeError("pgvector extension is missing in the selected database")
        version = str(row[0])
        cur.execute("SELECT to_regclass('public.notes')")
        notes_table = cur.fetchone()
        existing_vectors = False
        if notes_table and notes_table[0]:
            cur.execute("SELECT EXISTS (SELECT 1 FROM notes WHERE embedding IS NOT NULL)")
            existing_vectors = bool(cur.fetchone()[0])
        cur.execute("SELECT to_regclass('public.embedding_profile')")
        active_model = None
        if cur.fetchone()[0]:
            cur.execute("SELECT active_model FROM embedding_profile WHERE id = 1")
            active_row = cur.fetchone()
            active_model = active_row[0] if active_row else None
        return version, existing_vectors, active_model


def status_snapshot(url: str, vaults: list[str]) -> dict[str, Any]:
    """Read PostgreSQL health and index state for osm status."""
    with psycopg2.connect(url, connect_timeout=5) as conn, conn.cursor() as cur:
        cur.execute("SELECT extversion FROM pg_extension WHERE extname = 'vector'")
        row = cur.fetchone()
        result: dict[str, Any] = {
            "vector_version": row[0] if row else None,
            "profile": None,
            "vaults": None,
        }
        cur.execute("SELECT to_regclass('public.embedding_profile')")
        if cur.fetchone()[0]:
            cur.execute("SELECT active_model, staging_model FROM embedding_profile WHERE id = 1")
            result["profile"] = cur.fetchone()
        cur.execute("SELECT to_regclass('public.index_state'), to_regclass('public.notes')")
        state_table, notes_table = cur.fetchone()
        if state_table and notes_table:
            cur.execute("SELECT vault_id, status, error FROM index_state")
            states = {vault_id: (status, error) for vault_id, status, error in cur.fetchall()}
            vault_state = {}
            for vault in vaults:
                cur.execute("SELECT EXISTS (SELECT 1 FROM notes WHERE vault_id = %s)", (vault,))
                vault_state[vault] = (
                    bool(cur.fetchone()[0]),
                    *states.get(vault, ("pending", None)),
                )
            result["vaults"] = vault_state
        return result


class PostgresStore:
    def __init__(self, connection: Callable[[], AbstractContextManager[Any]], model: str):
        self.connection = connection
        self.model = model

    @staticmethod
    def _vector(value: list[float]) -> str:
        if not value:
            raise ValueError("Cannot convert empty list to vector literal")
        return "[" + ",".join(str(x) for x in value) + "]"

    def require_model(self, cursor: Any) -> None:
        cursor.execute("SELECT active_model FROM embedding_profile WHERE id = 1 FOR SHARE")
        row = cursor.fetchone()
        if not row or row[0] != self.model:
            raise RuntimeError("Embedding model differs from the active database index")

    @staticmethod
    def acquire_reindex_lock(conn: Any, key: int) -> bool:
        with conn.cursor() as cur:
            cur.execute("SELECT pg_try_advisory_lock(%s)", (key,))
            return bool(cur.fetchone()[0])

    @staticmethod
    def release_reindex_lock(conn: Any, key: int) -> None:
        with conn.cursor() as cur:
            cur.execute("SELECT pg_advisory_unlock(%s)", (key,))

    def rebuild_vector_index(self) -> None:
        with self.connection() as conn:
            with conn, conn.cursor() as cur:
                cur.execute("REINDEX INDEX notes_embedding_idx;")

    def initialize(self, embed_dim: int, primary_vault: str) -> tuple[int, int | None]:
        """Create/upgrade schema and report an existing vector dimension mismatch."""
        try:
            from . import migrations
        except ImportError:
            import migrations

        with self.connection() as conn:
            with conn, conn.cursor() as cur:
                cur.execute("SET LOCAL lock_timeout = '5s';")
                cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
                cur.execute(f"""
                    CREATE TABLE IF NOT EXISTS notes (
                        id          SERIAL PRIMARY KEY,
                        path        TEXT UNIQUE NOT NULL,
                        content     TEXT NOT NULL,
                        hash        TEXT NOT NULL,
                        embedding   vector({embed_dim}),
                        content_tsv tsvector,
                        vault_id    TEXT,
                        indexed_at  TIMESTAMP DEFAULT NOW()
                    );
                """)
                cur.execute("ALTER TABLE notes ADD COLUMN IF NOT EXISTS content_tsv tsvector;")
                cur.execute("ALTER TABLE notes ADD COLUMN IF NOT EXISTS vault_id TEXT;")
                if primary_vault:
                    cur.execute(
                        "UPDATE notes SET vault_id = %s WHERE vault_id IS NULL", (primary_vault,)
                    )

        with self.connection() as conn:
            migrations.apply_pending(conn)
        self.sync_search_language()
        with self.connection() as conn:
            active_model = migrations.establish_embedding_profile(conn, self.model)
        if active_model != self.model:
            raise RuntimeError(
                f"Database index uses {active_model}; configured model is {self.model}. "
                "Run osm migrate --embedding-model MODEL before changing osm init settings."
            )

        with self.connection() as conn:
            with conn, conn.cursor() as cur:
                cur.execute("""
                    SELECT format_type(a.atttypid, a.atttypmod)
                    FROM pg_attribute a
                    JOIN pg_class c ON c.oid = a.attrelid
                    WHERE c.relname = 'notes' AND a.attname = 'embedding'
                      AND a.attnum > 0 AND NOT a.attisdropped
                """)
                row = cur.fetchone()
                existing_dim = None
                if row:
                    match = re.search(r"vector\((\d+)\)", row[0])
                    if match and int(match.group(1)) != embed_dim:
                        existing_dim = int(match.group(1))
                cur.execute("SELECT COUNT(*) FROM notes")
                count = cur.fetchone()[0]
                lists = max(10, min(count // 50, 500)) if count > 0 else 100
                cur.execute(
                    "CREATE INDEX IF NOT EXISTS notes_embedding_idx "
                    "ON notes USING ivfflat (embedding vector_cosine_ops) "
                    f"WITH (lists = {int(lists)});"
                )
        return int(lists), existing_dim

    def sync_search_language(self) -> None:
        """Atomically refresh lexical vectors when the selected language changes."""
        language = search_language()
        with self.connection() as conn:
            with conn, conn.cursor() as cur:
                cur.execute("SELECT language FROM text_search_profile WHERE id = 1 FOR UPDATE")
                row = cur.fetchone()
                if row and row[0] == language:
                    return
                for table in ("notes", "note_chunks"):
                    cur.execute(
                        f"""
                        UPDATE {table} SET content_tsv =
                            setweight(to_tsvector(%s::regconfig, content), 'A') ||
                            setweight(to_tsvector('simple', content), 'B')
                    """,
                        (language,),
                    )
                cur.execute(
                    """
                    INSERT INTO text_search_profile (id, language) VALUES (1, %s)
                    ON CONFLICT (id) DO UPDATE SET language = EXCLUDED.language
                """,
                    (language,),
                )

    def hashes(self, paths: list[str]) -> dict[str, str]:
        if not paths:
            return {}
        with self.connection() as conn, conn.cursor() as cur:
            cur.execute(
                """
                SELECT n.path, n.hash FROM notes n
                WHERE n.path = ANY(%s) AND EXISTS (
                    SELECT 1 FROM note_chunks c
                    WHERE c.path = n.path AND c.model = %s AND c.note_hash = n.hash
                )
            """,
                (paths, self.model),
            )
            return dict(cur.fetchall())

    def note_is_current(self, path: str, content_hash: str) -> bool:
        with self.connection() as conn, conn.cursor() as cur:
            cur.execute(
                """
                SELECT n.hash, EXISTS (
                    SELECT 1 FROM note_chunks c
                    WHERE c.path = n.path AND c.model = %s AND c.note_hash = n.hash
                ) FROM notes n WHERE n.path = %s
            """,
                (self.model, path),
            )
            row = cur.fetchone()
            return bool(row and row[0] == content_hash and row[1])

    def upsert_note(
        self,
        path: str,
        content: str,
        content_hash: str,
        vault_id: str,
        chunks: list[tuple[int, int, str]],
        vectors: list[list[float]],
        links: dict[str, str | None],
    ) -> None:
        if len(chunks) != len(vectors) or not vectors:
            raise ValueError("missing note chunk embeddings")
        for attempt in range(3):
            try:
                with self.connection() as conn:
                    with conn, conn.cursor() as cur:
                        self.require_model(cur)
                        cur.execute(
                            """
                            INSERT INTO notes (path, content, hash, embedding, content_tsv, vault_id, indexed_at)
                            VALUES (%s, %s, %s, %s::vector,
                                    setweight(to_tsvector(%s::regconfig, %s), 'A') ||
                                    setweight(to_tsvector('simple', %s), 'B'), %s, NOW())
                            ON CONFLICT (path) DO UPDATE
                                SET content     = EXCLUDED.content,
                                    hash        = EXCLUDED.hash,
                                    embedding   = EXCLUDED.embedding,
                                    content_tsv = EXCLUDED.content_tsv,
                                    vault_id    = EXCLUDED.vault_id,
                                    indexed_at  = NOW()
                        """,
                            (
                                path,
                                content,
                                content_hash,
                                self._vector(vectors[0]),
                                search_language(),
                                content,
                                content,
                                vault_id or None,
                            ),
                        )
                        cur.execute(
                            "DELETE FROM note_chunks WHERE path = %s AND model = %s",
                            (path, self.model),
                        )
                        for index, ((start, end, passage), vector) in enumerate(
                            zip(chunks, vectors, strict=True)
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
                                    self.model,
                                    content_hash,
                                    index,
                                    start,
                                    end,
                                    passage,
                                    self._vector(vector),
                                    search_language(),
                                    passage,
                                    passage,
                                ),
                            )
                        cur.execute("DELETE FROM note_links WHERE source_path = %s", (path,))
                        if links:
                            cur.executemany(
                                "INSERT INTO note_links (source_path, target_name, target_path) "
                                "VALUES (%s, %s, %s)",
                                [(path, name, target) for name, target in links.items()],
                            )
                return
            except psycopg2.Error as exc:
                if exc.pgcode == "40P01" and attempt < 2:
                    time.sleep(0.1 * (attempt + 1))
                    continue
                raise

    def delete_note(self, path: str) -> None:
        with self.connection() as conn:
            with conn, conn.cursor() as cur:
                cur.execute("DELETE FROM notes WHERE path = %s", (path,))
                cur.execute("DELETE FROM note_links WHERE source_path = %s", (path,))

    def delete_notes(self, paths: list[str]) -> None:
        if not paths:
            return
        with self.connection() as conn:
            with conn, conn.cursor() as cur:
                cur.execute("DELETE FROM notes WHERE path = ANY(%s)", (paths,))

    def note_paths(self) -> list[str]:
        with self.connection() as conn, conn.cursor() as cur:
            cur.execute("SELECT path FROM notes")
            return [row[0] for row in cur.fetchall()]

    def indexed_notes(self) -> list[tuple[str, Any]]:
        with self.connection() as conn:
            with conn, conn.cursor() as cur:
                cur.execute("SELECT path, indexed_at FROM notes ORDER BY indexed_at DESC")
                return cur.fetchall()

    def dashboard_snapshot(self) -> dict[str, Any]:
        with self.connection() as conn, conn.cursor() as cur:
            cur.execute("SELECT version();")
            pg_version = cur.fetchone()[0].split(",")[0]
            cur.execute("SELECT extversion FROM pg_extension WHERE extname = 'vector';")
            vector_row = cur.fetchone()
            cur.execute("SELECT COUNT(*), MAX(indexed_at), MIN(indexed_at) FROM notes;")
            count, last, oldest = cur.fetchone()
            cur.execute("SELECT pg_total_relation_size('notes');")
            size = cur.fetchone()[0]
            cur.execute("SELECT path, indexed_at FROM notes ORDER BY indexed_at DESC LIMIT 10;")
            recent = cur.fetchall()
            cur.execute("SELECT path FROM notes")
            paths = [row[0] for row in cur.fetchall()]
        return {
            "pg_version": pg_version,
            "pgvector_version": vector_row[0] if vector_row else "not installed",
            "indexed_count": count or 0,
            "last_indexed": last,
            "oldest_indexed": oldest,
            "db_size_bytes": size,
            "recent_notes": recent,
            "paths": paths,
        }

    def init_search_metrics(self) -> None:
        with self.connection() as conn, conn, conn.cursor() as cur:
            cur.execute("SET LOCAL lock_timeout = '100ms'")
            cur.execute("SET LOCAL statement_timeout = '1s'")
            cur.execute("""
                CREATE TABLE IF NOT EXISTS search_metrics_hourly (
                    hour TIMESTAMPTZ NOT NULL,
                    backend TEXT NOT NULL CHECK (backend = 'local'),
                    mode TEXT NOT NULL CHECK (mode IN ('hybrid', 'semantic', 'keyword')),
                    outcome TEXT NOT NULL CHECK (outcome IN ('success', 'error')),
                    requests BIGINT NOT NULL DEFAULT 0 CHECK (requests >= 0),
                    results BIGINT NOT NULL DEFAULT 0 CHECK (results >= 0),
                    duration_ms DOUBLE PRECISION NOT NULL DEFAULT 0 CHECK (duration_ms >= 0),
                    PRIMARY KEY (hour, backend, mode, outcome)
                )
            """)

    def record_search_metric(
        self, hour: Any, cutoff: Any, state: dict[str, Any], duration_ms: float
    ) -> None:
        with self.connection() as conn, conn, conn.cursor() as cur:
            cur.execute("SET LOCAL lock_timeout = '100ms'")
            cur.execute("SET LOCAL statement_timeout = '1s'")
            cur.execute("DELETE FROM search_metrics_hourly WHERE hour < %s", (cutoff,))
            cur.execute(
                """
                INSERT INTO search_metrics_hourly
                    (hour, backend, mode, outcome, requests, results, duration_ms)
                VALUES (%s, %s, %s, %s, 1, %s, %s)
                ON CONFLICT (hour, backend, mode, outcome) DO UPDATE SET
                    requests = search_metrics_hourly.requests + 1,
                    results = search_metrics_hourly.results + EXCLUDED.results,
                    duration_ms = search_metrics_hourly.duration_ms + EXCLUDED.duration_ms
            """,
                (
                    hour,
                    state["backend"],
                    state["mode"],
                    state["outcome"],
                    state["results"],
                    duration_ms,
                ),
            )

    def search_metric_rows(self, start: Any, end: Any) -> list[tuple[Any, ...]]:
        with self.connection() as conn, conn, conn.cursor() as cur:
            cur.execute("SET LOCAL statement_timeout = '1s'")
            cur.execute(
                """
                SELECT backend, mode, outcome, SUM(requests), SUM(results),
                       SUM(duration_ms)
                FROM search_metrics_hourly
                WHERE hour >= %s AND hour < %s AND backend = 'local'
                GROUP BY backend, mode, outcome
                ORDER BY backend, mode, outcome
            """,
                (start, end),
            )
            return cur.fetchall()

    def keyword_search(
        self, query: str, limit: int, vault_ids: list[str] | None = None
    ) -> list[tuple[str, str, float]]:
        vault_clause = "AND vault_id = ANY(%s)" if vault_ids else ""
        vault_param = (vault_ids,) if vault_ids else ()
        with self.connection() as conn:
            with conn, conn.cursor() as cur:
                cur.execute(
                    f"""
                    SELECT path, content,
                           ts_rank(content_tsv,
                                   plainto_tsquery(%s::regconfig, %s) ||
                                   plainto_tsquery('simple', %s)) AS similarity
                    FROM notes
                    WHERE content_tsv @@ (plainto_tsquery(%s::regconfig, %s) ||
                                          plainto_tsquery('simple', %s))
                    {vault_clause}
                    ORDER BY similarity DESC LIMIT %s
                """,
                    (search_language(), query, query, search_language(), query, query)
                    + vault_param
                    + (limit,),
                )
                return cur.fetchall()

    def search_chunks(
        self,
        query: str,
        vector: list[float],
        mode: str,
        limit: int,
        min_similarity: float = 0.0,
        vault_ids: list[str] | None = None,
    ) -> list[tuple[str, str, float]]:
        vec_str = self._vector(vector)
        vault_clause = "AND n.vault_id = ANY(%s)" if vault_ids else ""
        vault_param = (vault_ids,) if vault_ids else ()
        with self.connection() as conn:
            with conn, conn.cursor() as cur:
                self.require_model(cur)
                if mode == "semantic":
                    cur.execute(
                        f"""
                        WITH scored AS (
                            SELECT c.path, c.chunk_index, c.content,
                                   1 - (c.embedding <=> %s::vector) AS similarity
                            FROM note_chunks c JOIN notes n ON n.path = c.path
                            WHERE c.model = %s AND c.note_hash = n.hash {vault_clause}
                        ), ranked AS (
                            SELECT *, ROW_NUMBER() OVER (
                                PARTITION BY path ORDER BY similarity DESC
                            ) AS rank_in_note FROM scored
                        )
                        SELECT path, chunk_index, content, similarity FROM ranked
                        WHERE rank_in_note = 1 ORDER BY similarity DESC LIMIT %s
                    """,
                        (vec_str, self.model) + vault_param + (limit,),
                    )
                else:
                    cur.execute(
                        f"""
                        WITH scored AS (
                            SELECT c.path, c.chunk_index, c.content,
                                   (1 - (c.embedding <=> %s::vector)) * 0.7 +
                                   COALESCE(ts_rank(c.content_tsv,
                                       plainto_tsquery(%s::regconfig, %s) ||
                                       plainto_tsquery('simple', %s)), 0) * 0.3 AS similarity
                            FROM note_chunks c JOIN notes n ON n.path = c.path
                            WHERE c.model = %s AND c.note_hash = n.hash {vault_clause}
                        ), ranked AS (
                            SELECT *, ROW_NUMBER() OVER (
                                PARTITION BY path ORDER BY similarity DESC
                            ) AS rank_in_note FROM scored
                        )
                        SELECT path, chunk_index, content, similarity FROM ranked
                        WHERE rank_in_note = 1 ORDER BY similarity DESC LIMIT %s
                    """,
                        (vec_str, search_language(), query, query, self.model)
                        + vault_param
                        + (limit,),
                    )
                results: list[tuple[str, str, float]] = []
                seen: set[str] = set()
                for path, index, passage, similarity in cur.fetchall():
                    if path in seen or similarity is None or similarity < min_similarity:
                        continue
                    seen.add(path)
                    cur.execute(
                        """
                        WITH span AS (
                            SELECT MIN(start_offset) AS start_pos, MAX(end_offset) AS end_pos
                            FROM note_chunks
                            WHERE path = %s AND model = %s
                              AND chunk_index BETWEEN %s AND %s
                        )
                        SELECT substring(n.content FROM span.start_pos + 1
                                                 FOR span.end_pos - span.start_pos)
                        FROM notes n CROSS JOIN span WHERE n.path = %s
                    """,
                        (path, self.model, max(0, index - 1), index + 1, path),
                    )
                    context = cur.fetchone()
                    results.append((path, context[0] if context else passage, float(similarity)))
                    if len(results) >= limit:
                        break
                return results

    def linked_notes(self, paths: list[str], hops: int = 1) -> list[tuple[str, str, str]]:
        seen: set[str] = set(paths)
        frontier: set[str] = set(paths)
        expansions: list[tuple[str, str, str]] = []
        for _ in range(hops):
            if not frontier:
                break
            with self.connection() as conn, conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT nl.target_path, n.content, nl.source_path
                    FROM note_links nl JOIN notes n ON n.path = nl.target_path
                    WHERE nl.source_path = ANY(%s)
                      AND nl.target_path IS NOT NULL
                      AND nl.target_path != ALL(%s)
                """,
                    (list(frontier), list(seen)),
                )
                outgoing = cur.fetchall()
                cur.execute(
                    """
                    SELECT nl.source_path, n.content, nl.target_path
                    FROM note_links nl JOIN notes n ON n.path = nl.source_path
                    WHERE nl.target_path = ANY(%s)
                      AND nl.source_path != ALL(%s)
                """,
                    (list(frontier), list(seen)),
                )
                incoming = cur.fetchall()
            next_frontier: set[str] = set()
            for path, content, via in outgoing + incoming:
                if path not in seen:
                    expansions.append((path, content, via))
                    seen.add(path)
                    next_frontier.add(path)
            frontier = next_frontier
        return expansions

    def set_index_state(
        self, vault_id: str, status: str, failed_paths: list[str], error: str | None
    ) -> None:
        with self.connection() as conn:
            with conn, conn.cursor() as cur:
                if status == "indexing":
                    cur.execute(
                        """
                        INSERT INTO index_state
                            (vault_id, status, started_at, finished_at, failed_paths, error)
                        VALUES (%s, %s, NOW(), NULL, %s, NULL)
                        ON CONFLICT (vault_id) DO UPDATE
                            SET status       = EXCLUDED.status,
                                started_at   = NOW(),
                                finished_at  = NULL,
                                failed_paths = EXCLUDED.failed_paths,
                                error        = NULL
                    """,
                        (vault_id, status, failed_paths),
                    )
                else:
                    cur.execute(
                        """
                        INSERT INTO index_state
                            (vault_id, status, started_at, finished_at, failed_paths, error)
                        VALUES (%s, %s, NULL, NOW(), %s, %s)
                        ON CONFLICT (vault_id) DO UPDATE
                            SET status       = EXCLUDED.status,
                                finished_at  = NOW(),
                                failed_paths = EXCLUDED.failed_paths,
                                error        = EXCLUDED.error
                    """,
                        (vault_id, status, failed_paths, error),
                    )

    def index_state(self, vault_id: str | None = None) -> list[tuple[Any, ...]]:
        with self.connection() as conn:
            with conn, conn.cursor() as cur:
                if vault_id is not None:
                    cur.execute(
                        "SELECT vault_id, status, started_at, finished_at, failed_paths, error "
                        "FROM index_state WHERE vault_id = %s",
                        (vault_id,),
                    )
                else:
                    cur.execute(
                        "SELECT vault_id, status, started_at, finished_at, failed_paths, error "
                        "FROM index_state ORDER BY vault_id"
                    )
                return cur.fetchall()
