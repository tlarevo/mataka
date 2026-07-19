"""Load and manage the differential-test corpus."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

CORPUS_PATH = Path(__file__).parent / "fixtures" / "corpus.jsonl"


@dataclass
class Fixture:
    id: str
    name: str
    items: list[dict[str, Any]]
    queries: list[dict[str, Any]]
    isolation_test: bool = False
    retain_user: str | None = None
    query_users: list[str] = field(default_factory=list)


def load_corpus(path: Path | None = None) -> list[Fixture]:
    """Load fixtures from corpus.jsonl."""
    path = path or CORPUS_PATH
    fixtures = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            fixtures.append(Fixture(
                id=data["id"],
                name=data["name"],
                items=data["items"],
                queries=data["queries"],
                isolation_test=data.get("isolation_test", False),
                retain_user=data.get("retain_user"),
                query_users=data.get("query_users", []),
            ))
    return fixtures
