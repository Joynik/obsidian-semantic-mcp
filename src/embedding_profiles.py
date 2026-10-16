"""Load the input templates shipped for each embedding model family."""

from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=32)
def _templates(family: str) -> dict[str, str] | None:
    if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_.-]*", family):
        raise ValueError("invalid embedding model family")
    packaged = Path(__file__).resolve().parent / "embedding_profiles" / f"{family}.json"
    source = Path(__file__).resolve().parent.parent / "embedding-profiles" / f"{family}.json"
    path = packaged if packaged.is_file() else source
    if not path.is_file():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    if (
        not isinstance(data, dict)
        or set(data) != {"document", "query"}
        or any(not isinstance(value, str) or value.count("{text}") != 1 for value in data.values())
    ):
        raise ValueError(f"invalid embedding profile for {family}")
    return data


def embedding_input(text: str, task: str, model: str) -> str:
    if task not in ("document", "query"):
        raise ValueError("invalid embedding task")
    family = model.split(":", 1)[0]
    profile = _templates(family)
    if profile is None:
        return text
    return profile[task].replace("{text}", text)
