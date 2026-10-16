"""Local Qdrant storage for notes, passages, links, and index metadata."""

from __future__ import annotations

import hashlib
import os
import secrets
import uuid
from datetime import UTC, datetime
from typing import Any
from urllib.parse import quote

import requests

ALIAS = "osm_notes"
_DENSE = "dense"
_BM25 = "bm25"
_SEARCH_PROFILE = "bm25-russian-v1"
_CONTROL_ID = str(uuid.uuid5(uuid.NAMESPACE_URL, "obsidian-semantic-mcp:profile"))


def point_id(kind: str, path: str, suffix: str = "") -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"osm:{kind}:{path}:{suffix}"))


class QdrantStore:
    def __init__(self, url: str, model: str):
        self.url = url.rstrip("/")
        self.model = model

    def request(self, method: str, path: str, body: dict[str, Any] | None = None) -> Any:
        response = requests.request(method, f"{self.url}{path}", json=body, timeout=30)
        response.raise_for_status()
        data = response.json()
        if data.get("status") != "ok":
            raise RuntimeError(f"Qdrant request failed: {data.get('status')}")
        return data.get("result")

    @staticmethod
    def _name(name: str) -> str:
        return quote(name, safe="")

    def collections(self) -> set[str]:
        result = self.request("GET", "/collections")
        return {item["name"] for item in result["collections"]}

    def alias_target(self) -> str | None:
        result = self.request("GET", "/aliases")
        return next(
            (item["collection_name"] for item in result["aliases"] if item["alias_name"] == ALIAS),
            None,
        )

    def create_collection(self, name: str, dimension: int) -> None:
        self.request(
            "PUT",
            f"/collections/{self._name(name)}",
            {
                "vectors": {_DENSE: {"size": dimension, "distance": "Cosine"}},
                "sparse_vectors": {_BM25: {"modifier": "idf"}},
            },
        )
        self.upsert(
            [
                {
                    "id": _CONTROL_ID,
                    "vector": {_DENSE: [0.0] * dimension},
                    "payload": {
                        "kind": "profile",
                        "model": self.model,
                        "dimension": dimension,
                        "search_profile": _SEARCH_PROFILE,
                    },
                }
            ],
            name,
        )

    def profile(self, collection: str = ALIAS) -> dict[str, Any] | None:
        points = self.request(
            "POST",
            f"/collections/{self._name(collection)}/points",
            {"ids": [_CONTROL_ID], "with_payload": True, "with_vector": False},
        )
        return points[0]["payload"] if points else None

    def ensure(self, dimension: int) -> None:
        target = self.alias_target()
        if target is None:
            name = f"osm_notes_{hashlib.sha256(self.model.encode()).hexdigest()[:10]}_{secrets.token_hex(4)}"
            self.create_collection(name, dimension)
            try:
                self.swap(name)
            except requests.HTTPError:
                # Another process may have initialized the alias meanwhile.
                if self.alias_target() != name:
                    self.assert_profile(dimension)
            return
        self.assert_profile(dimension)
        profile = self.profile()
        if profile and profile.get("search_profile") != _SEARCH_PROFILE:
            self.upgrade_search_index(target, dimension)

    def upgrade_search_index(self, source: str, dimension: int) -> None:
        """Copy legacy points into a BM25 collection before changing the alias."""
        stage = f"osm_notes_{hashlib.sha256(self.model.encode()).hexdigest()[:10]}_{secrets.token_hex(4)}"
        self.create_collection(stage, dimension)
        offset = None
        copied = 0
        while True:
            body: dict[str, Any] = {"limit": 128, "with_payload": True, "with_vector": True}
            if offset is not None:
                body["offset"] = offset
            page = self.request("POST", f"/collections/{self._name(source)}/points/scroll", body)
            points = []
            for point in page["points"]:
                vector = point["vector"]
                if isinstance(vector, dict):
                    vector = vector.get(_DENSE)
                if not isinstance(vector, list) or len(vector) != dimension:
                    raise RuntimeError("Cannot copy a Qdrant point with an unexpected vector")
                payload = point["payload"]
                if payload.get("kind") == "profile":
                    payload = {**payload, "search_profile": _SEARCH_PROFILE}
                vectors: dict[str, Any] = {_DENSE: vector}
                if payload.get("kind") == "chunk":
                    vectors[_BM25] = self.bm25_document(payload["content"])
                points.append({"id": point["id"], "vector": vectors, "payload": payload})
            if points:
                self.upsert(points, stage)
                copied += len(points)
            offset = page.get("next_page_offset")
            if offset is None:
                break
        source_notes = {row["path"]: row["hash"] for row in self.notes(source)}
        staged_notes = {row["path"]: row["hash"] for row in self.notes(stage)}
        if source_notes != staged_notes or copied != len(self.scroll(stage)):
            raise RuntimeError("Qdrant BM25 copy incomplete; active collection preserved")
        if self.alias_target() != source:
            raise RuntimeError("Qdrant alias changed during BM25 copy; active collection preserved")
        self.swap(stage)

    @staticmethod
    def bm25_document(content: str) -> dict[str, Any]:
        return {"text": content or " ", "model": "qdrant/bm25", "options": {"language": "russian"}}

    def assert_profile(self, dimension: int | None = None) -> None:
        profile = self.profile()
        if not profile or profile.get("model") != self.model:
            raise RuntimeError("Qdrant contains vectors from another model; run osm migrate")
        if dimension is not None and profile.get("dimension") != dimension:
            raise RuntimeError("Qdrant embedding dimension changed; run osm migrate")

    def swap(self, collection: str) -> None:
        previous = self.alias_target()
        actions = []
        if previous:
            actions.append({"delete_alias": {"alias_name": ALIAS}})
        actions.append({"create_alias": {"alias_name": ALIAS, "collection_name": collection}})
        self.request("POST", "/collections/aliases", {"actions": actions})

    def upsert(self, points: list[dict[str, Any]], collection: str = ALIAS) -> None:
        self.request(
            "PUT", f"/collections/{self._name(collection)}/points?wait=true", {"points": points}
        )

    def scroll(
        self,
        collection: str = ALIAS,
        filt: dict[str, Any] | None = None,
        limit: int = 256,
        fields: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        offset = None
        while True:
            body: dict[str, Any] = {
                "limit": limit,
                "with_payload": fields if fields is not None else True,
                "with_vector": False,
            }
            if filt:
                body["filter"] = filt
            if offset is not None:
                body["offset"] = offset
            page = self.request(
                "POST", f"/collections/{self._name(collection)}/points/scroll", body
            )
            result.extend(page["points"])
            offset = page.get("next_page_offset")
            if offset is None:
                return result

    @staticmethod
    def kind_filter(kind: str) -> dict[str, Any]:
        return {"must": [{"key": "kind", "match": {"value": kind}}]}

    def notes(
        self, collection: str = ALIAS, fields: list[str] | None = None
    ) -> list[dict[str, Any]]:
        return [
            point["payload"]
            for point in self.scroll(collection, self.kind_filter("note"), fields=fields)
        ]

    def states(self, collection: str = ALIAS) -> list[dict[str, Any]]:
        return [point["payload"] for point in self.scroll(collection, self.kind_filter("state"))]

    def set_state(
        self, vault_id: str, status: str, failed_paths: list[str], error: str | None, dimension: int
    ) -> None:
        self.upsert(
            [
                {
                    "id": point_id("state", vault_id),
                    "vector": {_DENSE: [0.0] * dimension},
                    "payload": {
                        "kind": "state",
                        "vault_id": vault_id,
                        "status": status,
                        "failed_paths": failed_paths,
                        "error": error,
                        "updated_at": datetime.now(UTC).isoformat(),
                    },
                }
            ]
        )

    def note(self, path: str, collection: str = ALIAS) -> dict[str, Any] | None:
        points = self.request(
            "POST",
            f"/collections/{self._name(collection)}/points",
            {"ids": [point_id("note", path)], "with_payload": True, "with_vector": False},
        )
        return points[0]["payload"] if points else None

    def delete_path(self, path: str, collection: str = ALIAS) -> None:
        self.request(
            "POST",
            f"/collections/{self._name(collection)}/points/delete?wait=true",
            {"filter": {"must": [{"key": "path", "match": {"value": path}}]}},
        )

    def write_note(
        self,
        path: str,
        content: str,
        content_hash: str,
        vault_id: str,
        chunks: list[tuple[int, int, str]],
        vectors: list[list[float]],
        links: dict[str, str | None],
        collection: str = ALIAS,
    ) -> None:
        if len(chunks) != len(vectors) or not vectors:
            raise ValueError("missing passage embeddings")
        self.assert_profile(len(vectors[0])) if collection == ALIAS else None
        old = self.note(path, collection)
        points = []
        for index, ((start, end, passage), vector) in enumerate(zip(chunks, vectors, strict=True)):
            points.append(
                {
                    "id": point_id("chunk", path, f"{content_hash}:{index}"),
                    "vector": {_DENSE: vector, _BM25: self.bm25_document(passage)},
                    "payload": {
                        "kind": "chunk",
                        "path": path,
                        "vault_id": vault_id,
                        "hash": content_hash,
                        "index": index,
                        "start": start,
                        "end": end,
                        "content": passage,
                        "model": self.model,
                    },
                }
            )
        self.upsert(points, collection)
        self.upsert(
            [
                {
                    "id": point_id("note", path),
                    "vector": {_DENSE: vectors[0]},
                    "payload": {
                        "kind": "note",
                        "path": path,
                        "vault_id": vault_id,
                        "content": content,
                        "hash": content_hash,
                        "links": links,
                        "model": self.model,
                        "indexed_at": datetime.now(UTC).isoformat(),
                    },
                }
            ],
            collection,
        )
        if old and old.get("hash") != content_hash:
            self.request(
                "POST",
                f"/collections/{self._name(collection)}/points/delete?wait=true",
                {
                    "filter": {
                        "must": [
                            {"key": "kind", "match": {"value": "chunk"}},
                            {"key": "path", "match": {"value": path}},
                        ],
                        "must_not": [{"key": "hash", "match": {"value": content_hash}}],
                    }
                },
            )

    def search(
        self,
        vector: list[float] | None,
        limit: int,
        vault_ids: list[str] | None = None,
        *,
        query: str | None = None,
        mode: str = "semantic",
    ) -> list[dict[str, Any]]:
        if vector is not None:
            self.assert_profile(len(vector))
        elif mode != "keyword":
            raise ValueError("Dense or hybrid search requires an embedding")
        conditions: list[dict[str, Any]] = [{"key": "kind", "match": {"value": "chunk"}}]
        if vault_ids:
            conditions.append({"key": "vault_id", "match": {"any": vault_ids}})
        filt = {"must": conditions}
        body: dict[str, Any] = {"limit": limit, "with_payload": True}
        if mode == "keyword":
            body.update({"query": self.bm25_document(query or ""), "using": _BM25, "filter": filt})
        elif mode == "hybrid":
            depth = max(limit * 2, 40)
            body.update(
                {
                    "prefetch": [
                        {"query": vector, "using": _DENSE, "filter": filt, "limit": depth},
                        {
                            "query": self.bm25_document(query or ""),
                            "using": _BM25,
                            "filter": filt,
                            "limit": depth,
                        },
                    ],
                    "query": {"fusion": "rrf"},
                }
            )
        else:
            body.update({"query": vector, "using": _DENSE, "filter": filt})
        result = self.request("POST", f"/collections/{ALIAS}/points/query", body)
        return result["points"]


def local_url() -> str:
    url = os.environ.get("QDRANT_URL", "").strip()
    if not url:
        raise RuntimeError("QDRANT_URL is required for the Qdrant backend")
    return url
