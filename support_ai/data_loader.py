"""Load the starter repo data and knowledge-base files."""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"


def _knowledge_base_dir() -> Path:
    candidates = [PROJECT_ROOT / "Knowledge-base", PROJECT_ROOT / "knowledge-base"]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


KB_DIR = _knowledge_base_dir()


@dataclass(frozen=True)
class KnowledgeDocument:
    path: str
    title: str
    text: str


def _read_json(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


@lru_cache(maxsize=1)
def load_tickets() -> list[dict[str, Any]]:
    return _read_json(DATA_DIR / "tickets.json")


@lru_cache(maxsize=1)
def load_accounts() -> list[dict[str, Any]]:
    return _read_json(DATA_DIR / "accounts.json")


@lru_cache(maxsize=1)
def load_knowledge_documents() -> list[KnowledgeDocument]:
    docs: list[KnowledgeDocument] = []
    for path in sorted(KB_DIR.rglob("*.md"), key=lambda item: str(item).lower()):
        text = path.read_text(encoding="utf-8")
        title = path.stem.replace("-", " ").title()
        for line in text.splitlines():
            if line.startswith("# "):
                title = line[2:].strip()
                break
        docs.append(
            KnowledgeDocument(
                path=path.relative_to(PROJECT_ROOT).as_posix(),
                title=title,
                text=text,
            )
        )
    return docs


def account_lookup() -> dict[str, dict[str, Any]]:
    return {account["account_id"]: account for account in load_accounts()}
