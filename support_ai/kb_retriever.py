"""Deterministic markdown knowledge-base retrieval."""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache

from .data_loader import KnowledgeDocument, load_knowledge_documents
from .text_utils import best_sentence, extract_error_codes, normalize_whitespace, tokenize


@dataclass(frozen=True)
class KnowledgeChunk:
    chunk_id: str
    path: str
    title: str
    heading: str
    text: str
    token_counts: Counter[str]


@dataclass(frozen=True)
class SearchResult:
    path: str
    title: str
    heading: str
    score: float
    snippet: str

    def to_dict(self) -> dict[str, object]:
        return {
            "path": self.path,
            "title": self.title,
            "heading": self.heading,
            "score": round(self.score, 3),
            "snippet": self.snippet,
        }


HEADING_RE = re.compile(r"^(#{1,4})\s+(.+?)\s*$")
RULE_RE = re.compile(r"^\s*-{3,}\s*$")


def _split_document(doc: KnowledgeDocument) -> list[KnowledgeChunk]:
    chunks: list[KnowledgeChunk] = []
    heading = doc.title
    buffer: list[str] = []

    def flush() -> None:
        nonlocal buffer
        text = "\n".join(buffer).strip()
        if not text:
            buffer = []
            return
        chunk_id = f"{doc.path}#{len(chunks) + 1}"
        chunks.append(
            KnowledgeChunk(
                chunk_id=chunk_id,
                path=doc.path,
                title=doc.title,
                heading=heading,
                text=text,
                token_counts=Counter(tokenize(f"{doc.title} {heading} {text}")),
            )
        )
        buffer = []

    for line in doc.text.splitlines():
        heading_match = HEADING_RE.match(line)
        if RULE_RE.match(line):
            flush()
            continue
        if heading_match and buffer:
            flush()
        if heading_match:
            heading = heading_match.group(2).strip()
        buffer.append(line)

    flush()
    return chunks


class KnowledgeRetriever:
    def __init__(self, docs: list[KnowledgeDocument] | None = None) -> None:
        self.docs = docs or load_knowledge_documents()
        self.chunks = [chunk for doc in self.docs for chunk in _split_document(doc)]
        self._idf = self._build_idf(self.chunks)

    @staticmethod
    def _build_idf(chunks: list[KnowledgeChunk]) -> dict[str, float]:
        doc_frequency: Counter[str] = Counter()
        for chunk in chunks:
            doc_frequency.update(chunk.token_counts.keys())
        total = max(len(chunks), 1)
        return {
            token: math.log((total + 1) / (frequency + 1)) + 1.0
            for token, frequency in doc_frequency.items()
        }

    def _score_chunk(self, query: str, chunk: KnowledgeChunk) -> float:
        query_tokens = tokenize(query)
        if not query_tokens:
            return 0.0

        query_counts = Counter(query_tokens)
        token_score = 0.0
        for token, count in query_counts.items():
            if token in chunk.token_counts:
                token_score += self._idf.get(token, 1.0) * min(count, 3) * min(chunk.token_counts[token], 3)

        chunk_text_lower = chunk.text.lower()
        query_lower = query.lower()
        error_score = 0.0
        for code in extract_error_codes(query):
            if code.lower() in chunk_text_lower:
                error_score += 8.0

        phrase_score = 0.0
        for phrase in [chunk.title, chunk.heading]:
            normalized = normalize_whitespace(phrase).lower()
            if normalized and normalized in query_lower:
                phrase_score += 4.0

        length_penalty = math.sqrt(sum(chunk.token_counts.values()) + 35)
        return (token_score / length_penalty) + error_score + phrase_score

    def search(self, query: str, top_k: int = 3, min_score: float = 0.15) -> list[SearchResult]:
        scored = [
            (self._score_chunk(query, chunk), chunk)
            for chunk in self.chunks
        ]
        results: list[SearchResult] = []
        for score, chunk in sorted(scored, key=lambda item: item[0], reverse=True):
            if score < min_score:
                continue
            results.append(
                SearchResult(
                    path=chunk.path,
                    title=chunk.title,
                    heading=chunk.heading,
                    score=score,
                    snippet=best_sentence(chunk.text, query),
                )
            )
            if len(results) >= top_k:
                break
        return results


@lru_cache(maxsize=1)
def get_retriever() -> KnowledgeRetriever:
    return KnowledgeRetriever()
