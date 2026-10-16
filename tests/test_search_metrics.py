"""Synthetic search outcomes and aggregate persistence never retain content."""

import asyncio
import contextlib
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

import pytest


@pytest.fixture
def metrics(monkeypatch):
    from src import server

    recorded = []
    monkeypatch.setattr(
        server, "_record_search_metric", lambda *args: recorded.append(args), raising=False
    )
    monkeypatch.setattr(server, "_readiness_state", "ready")
    monkeypatch.setattr(server, "get_index_state", lambda *_: [])
    server._search_cache.invalidate()
    monkeypatch.setattr(server, "VAULT_PATHS", ["/v/main"])
    monkeypatch.setattr(server, "VAULT_PATH", "/v/main")
    yield recorded
    server._search_cache.invalidate()


def local_db(monkeypatch, rows):
    from src import server

    store = MagicMock()
    store.keyword_search.return_value = rows
    monkeypatch.setattr(server, "_pgstore", lambda: store)
    monkeypatch.setattr(server, "search_chunk_rows", lambda *_: rows)
    monkeypatch.setattr(server, "embed", lambda query, **kw: [0.1])
    monkeypatch.setattr(server, "_rerank", lambda query, rows, limit: rows[:limit])
    return store


def search(**args):
    from src import server

    return asyncio.run(server.call_tool("search_vault", {"query": "private query", **args}))


def test_direct_local_and_cache_hit_each_record_once(metrics, monkeypatch):
    local_db(
        monkeypatch,
        [("/v/main/a.md", "private result", 0.8), ("/v/main/b.md", "private result", 0.7)],
    )
    first = search(mode="keyword")
    second = search(mode="keyword")
    assert first == second
    assert len(metrics) == 2
    assert all(
        state["backend"] == "local" and state["results"] == 2 and state["outcome"] == "success"
        for state, _ in metrics
    )


def test_zero_results_is_success(metrics, monkeypatch):
    local_db(monkeypatch, [])
    search(mode="untrusted label")
    assert metrics[0][0]["mode"] == "hybrid"
    assert metrics[0][0]["outcome"] == "success"
    assert metrics[0][0]["results"] == 0


def test_metrics_exception_does_not_change_search_result(metrics, monkeypatch, caplog):
    from src import server

    local_db(monkeypatch, [("/v/main/a.md", "passage text", 0.8)])

    def fail(*args):
        raise RuntimeError("private content must not appear in warning")

    monkeypatch.setattr(server, "_record_search_metric", fail)
    assert "passage text" in search()[0].text
    assert "private content" not in caplog.text
    assert "search metrics" in caplog.text


def state():
    return {"backend": "local", "mode": "keyword", "outcome": "success", "results": 2}


def persistence_db(monkeypatch):
    from src import server

    cur = MagicMock()
    conn = MagicMock()
    conn.cursor.return_value.__enter__.return_value = cur

    @contextlib.contextmanager
    def connection():
        yield conn

    monkeypatch.setattr(server, "db_conn", connection)
    return cur


def test_hourly_upsert_utc_bucket_retention_and_parameters(monkeypatch):
    from src import server

    cur = persistence_db(monkeypatch)
    now = datetime(2026, 10, 7, 13, 59, tzinfo=timezone(timedelta(hours=2)))
    server._record_search_metric(state(), 12.5, now=now)
    calls = cur.execute.call_args_list
    upsert = next(
        call.args for call in calls if "INSERT INTO search_metrics_hourly" in call.args[0]
    )
    hour = datetime(2026, 10, 7, 11, tzinfo=timezone.utc)
    assert upsert[1] == (hour, "local", "keyword", "success", 2, 12.5)
    assert "ON CONFLICT" in upsert[0] and "requests + 1" in upsert[0]
    deletion = next(call.args for call in calls if "DELETE FROM" in call.args[0])
    assert deletion[1] == (hour - timedelta(hours=719),)
    server._record_search_metric(state(), 5, now=now + timedelta(minutes=1))
    assert cur.execute.call_args_list[-1].args[1][0] == hour + timedelta(hours=1)


@pytest.mark.parametrize(
    "field,value",
    [
        ("backend", "private query"),
        ("mode", "private query"),
        ("outcome", "private query"),
        ("results", -1),
    ],
)
def test_invalid_metric_dimensions_fail_closed(monkeypatch, caplog, field, value):
    from src import server

    cur = persistence_db(monkeypatch)
    record = state()
    record[field] = value
    server._record_search_metric(record, 1)
    assert not cur.execute.called
    assert "private" not in caplog.text


def test_schema_has_only_bounded_aggregate_fields(monkeypatch):
    from src import server

    cur = persistence_db(monkeypatch)
    server._init_search_metrics()
    ddl = next(
        call.args[0] for call in cur.execute.call_args_list if "CREATE TABLE" in call.args[0]
    )
    assert "PRIMARY KEY (hour, backend, mode, outcome)" in ddl
    assert "CHECK" in ddl
    assert all(word not in ddl.lower() for word in ("query", "path", "content", "trace"))


def test_metrics_schema_waits_for_embedding_service(monkeypatch):
    from src import server
    import requests
    import threading

    initialized = []
    stopping = threading.Event()
    monkeypatch.setattr(server, "_STOPPING", stopping)
    monkeypatch.setattr(stopping, "wait", lambda *_: False)
    monkeypatch.setattr(server, "_init_search_metrics", lambda: initialized.append(True))

    def unavailable():
        stopping.set()
        raise requests.ConnectionError("synthetic embedding outage")

    monkeypatch.setattr(server, "get_embed_dim", unavailable)
    server.background_init([])
    assert initialized == []
