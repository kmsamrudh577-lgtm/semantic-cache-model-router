"""
cache.py — prompt normalization, embeddings and a metadata-aware semantic cache.

Lookup pipeline for a query:
  1. normalize + validate the prompt and its metadata tags
  2. embed it (sentence-transformers all-MiniLM-L6-v2, or an offline hashing
     embedder if the model can't be loaded)
  3. keep only cache entries whose context key (canonical tag set) matches
     -> a "python" question never collides with a "java" question
  4. cosine similarity against the remaining entries (vectors are L2-normalized,
     so cosine = dot product)
  5. best candidate >= threshold  AND  passes the optional qualifier guard
     (numbers / units / negations / named entities must agree)  ->  CACHE_HIT

Entries + embeddings persist in SQLite, so the cache survives restarts.
"""
from __future__ import annotations

import json
import os
import re
import sqlite3
import threading
import time
from dataclasses import dataclass, field

import numpy as np

MAX_PROMPT_CHARS = 4000
ALLOWED_TAG_KEYS = ("topic", "language", "context_tag")


# --------------------------------------------------------------------------- #
# Normalization & validation
# --------------------------------------------------------------------------- #
def normalize_prompt(text: str) -> str:
    """Canonical form used for embedding: trimmed, single-spaced, lower-case,
    trailing punctuation removed. The original text is kept for display."""
    if text is None:
        return ""
    text = re.sub(r"\s+", " ", str(text)).strip()
    text = text.lower().rstrip(" ?.!")
    return text


def validate_prompt(text: str) -> str:
    """Raise ValueError for unusable prompts; return the cleaned original."""
    if text is None or not str(text).strip():
        raise ValueError("Prompt is empty.")
    cleaned = re.sub(r"\s+", " ", str(text)).strip()
    if len(cleaned) > MAX_PROMPT_CHARS:
        raise ValueError(f"Prompt too long ({len(cleaned)} chars > {MAX_PROMPT_CHARS}).")
    if not re.search(r"[A-Za-z0-9]", cleaned):
        raise ValueError("Prompt has no alphanumeric content.")
    return cleaned


def normalize_tags(tags: dict | None) -> dict[str, str]:
    """Lower-case, slugify and drop empty / unknown tag keys."""
    out: dict[str, str] = {}
    for key, value in (tags or {}).items():
        key = str(key).strip().lower()
        if key not in ALLOWED_TAG_KEYS or value is None:
            continue
        value = re.sub(r"[^a-z0-9+#._-]+", "-", str(value).strip().lower()).strip("-")
        if value and value not in {"none", "any", "general"}:
            out[key] = value
    return out


def context_key(tags: dict[str, str]) -> str:
    """Canonical string for a tag set; entries only match on identical keys."""
    return "|".join(f"{k}={tags[k]}" for k in sorted(tags)) or "global"


# --------------------------------------------------------------------------- #
# Qualifier guard — blocks "near-miss" false hits
# --------------------------------------------------------------------------- #
_ALIASES = {
    "km": "kilometers", "kms": "kilometers", "kilometres": "kilometers", "kilometre": "kilometers",
    "kilometer": "kilometers", "mi": "miles", "mile": "miles", "kg": "kilograms",
    "kilogram": "kilograms", "lb": "pounds", "lbs": "pounds", "pound": "pounds",
    "°c": "celsius", "°f": "fahrenheit", "c++": "cpp", "js": "javascript",
    "isn't": "not", "don't": "not", "doesn't": "not", "can't": "not", "won't": "not",
    "never": "not", "without": "not", "cons": "disadvantages", "pros": "advantages",
    "risks": "side-effects", "risk": "side-effects", "downsides": "disadvantages",
}
# Words that flip the meaning of an otherwise-similar prompt.
_CRITICAL_WORDS = {
    # units
    "kilometers", "miles", "meters", "feet", "inches", "centimeters", "kilograms", "pounds",
    "grams", "ounces", "celsius", "fahrenheit", "kelvin", "liters", "gallons", "usd", "eur", "inr",
    # negation / polarity
    "not", "advantages", "disadvantages", "benefits", "side-effects",
    # comparatives / ordering
    "minimum", "maximum", "min", "max", "ascending", "descending", "top", "bottom",
    "first", "last", "largest", "smallest", "highest", "lowest", "before", "after",
    # programming languages & data structures
    "python", "java", "javascript", "typescript", "cpp", "rust", "go", "sql", "c#",
    "list", "string", "dict", "dictionary", "tuple", "array", "set", "map",
}


def _tokens(text: str) -> list[str]:
    text = re.sub(r"side[\s-]+effects?", "side-effects", text, flags=re.I)
    raw = re.findall(r"[A-Za-z][\w+#'-]*|\d+(?:\.\d+)?%?|°[cf]", text)
    out = []
    for tok in raw:
        low = _ALIASES.get(tok.lower(), tok.lower())
        out.append(low)
    return out


def _qualifiers(text: str) -> tuple[list[str], set[str]]:
    """Return (ordered critical tokens incl. numbers, named-entity set)."""
    toks = _tokens(text)
    critical = [t for t in toks if t in _CRITICAL_WORDS or re.fullmatch(r"\d+(?:\.\d+)?%?", t)]
    # Named entities: capitalized words that are not the first word of the text
    words = re.findall(r"[A-Za-z][\w'-]*", text)
    ents = set()
    for i, w in enumerate(words):
        if i > 0 and (w[0].isupper() or w.isupper()) and len(w) > 1:
            w = w.lower()
            ents.add(w[:-1] if w.endswith("s") and len(w) > 3 else w)
    return critical, ents


def qualifier_conflict(query: str, candidate: str) -> str | None:
    """Return a human-readable reason if the two prompts disagree on a
    qualifier that would make a cached answer wrong, else None."""
    q_crit, q_ent = _qualifiers(query)
    c_crit, c_ent = _qualifiers(candidate)
    q_set, c_set = set(q_crit), set(c_crit)
    if q_set != c_set:
        diff = sorted(q_set ^ c_set)
        return f"qualifier mismatch: {', '.join(diff[:6])}"
    if len(q_crit) >= 2 and q_crit != c_crit:
        return f"qualifier order differs: {q_crit} vs {c_crit}"
    # Entities are compared case-insensitively against the other prompt's words.
    q_words = {w[:-1] if w.endswith("s") and len(w) > 3 else w for w in _tokens(query)}
    c_words = {w[:-1] if w.endswith("s") and len(w) > 3 else w for w in _tokens(candidate)}
    missing = sorted((q_ent - c_words) | (c_ent - q_words))
    if missing:
        return f"entity mismatch: {', '.join(missing[:6])}"
    return None


# --------------------------------------------------------------------------- #
# Embedders
# --------------------------------------------------------------------------- #
class HashingEmbedder:
    """Offline fallback: hashed word uni/bi-grams + char 3-grams -> 1024-d,
    L2-normalized. No downloads, deterministic, weaker on paraphrases."""

    name = "hashing-1024"
    dim = 1024

    def encode(self, texts: list[str]) -> np.ndarray:
        import hashlib

        mat = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            words = re.findall(r"\w+", text.lower())
            feats = words + [f"{a}_{b}" for a, b in zip(words, words[1:])]
            padded = f" {' '.join(words)} "
            feats += [padded[i:i + 3] for i in range(len(padded) - 2)]
            for f in feats:
                h = int(hashlib.md5(f.encode()).hexdigest()[:8], 16)
                mat[row, h % self.dim] += 1.0 if h & 1 else -1.0
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        return mat / np.where(norms == 0, 1, norms)


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name)
        self.name = f"st:{model_name}"
        dim_fn = getattr(self.model, "get_embedding_dimension", None) or \
            self.model.get_sentence_embedding_dimension
        self.dim = dim_fn()

    def encode(self, texts: list[str]) -> np.ndarray:
        vecs = self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return np.asarray(vecs, dtype=np.float32)


def load_embedder(prefer: str | None = None):
    """Load sentence-transformers if possible, else the hashing fallback.

    EMBEDDER=hashing forces the offline embedder; EMBEDDING_MODEL picks the ST model.
    """
    prefer = (prefer or os.getenv("EMBEDDER", "auto")).lower()
    if prefer != "hashing":
        try:
            return SentenceTransformerEmbedder(os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2"))
        except Exception as exc:  # not installed / no internet for first download
            print(f"[cache] sentence-transformers unavailable ({exc!s:.120}); using hashing embedder")
    return HashingEmbedder()


# --------------------------------------------------------------------------- #
# Semantic cache
# --------------------------------------------------------------------------- #
@dataclass
class CacheEntry:
    id: int
    query: str
    normalized: str
    response: str
    model: str
    tags: dict
    context_key: str
    input_tokens: int
    output_tokens: int
    created_at: float
    hits: int = 0


@dataclass
class CacheResult:
    hit: bool
    similarity: float = 0.0               # best similarity among compatible entries
    entry: CacheEntry | None = None       # the matched entry (hit) or best candidate (miss)
    blocked_reason: str | None = None     # set when the qualifier guard vetoed a hit
    candidates_checked: int = 0
    embedding: np.ndarray | None = field(default=None, repr=False)
    lookup_ms: float = 0.0


class SemanticCache:
    """Metadata-aware semantic cache backed by SQLite + an in-memory numpy matrix."""

    def __init__(self, db_path: str = "data/semantic_cache.db", embedder=None,
                 ttl_seconds: float | None = None):
        self.db_path = db_path
        if db_path != ":memory:":
            os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
        self.embedder = embedder or load_embedder()
        self.ttl_seconds = ttl_seconds
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.execute(
            """CREATE TABLE IF NOT EXISTS cache_entries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                query TEXT NOT NULL, normalized TEXT NOT NULL, response TEXT NOT NULL,
                model TEXT, tags TEXT, context_key TEXT, embedder TEXT,
                input_tokens INTEGER, output_tokens INTEGER,
                created_at REAL, hits INTEGER DEFAULT 0, embedding BLOB)"""
        )
        self._conn.commit()
        self._load()

    # ------------------------------------------------------------------ #
    def _load(self) -> None:
        """Load entries created with the *current* embedder into memory."""
        rows = self._conn.execute(
            "SELECT id, query, normalized, response, model, tags, context_key, input_tokens, "
            "output_tokens, created_at, hits, embedding FROM cache_entries WHERE embedder = ? "
            "ORDER BY id", (self.embedder.name,)
        ).fetchall()
        self.entries: list[CacheEntry] = []
        vecs = []
        for r in rows:
            self.entries.append(CacheEntry(r[0], r[1], r[2], r[3], r[4], json.loads(r[5] or "{}"),
                                           r[6], r[7] or 0, r[8] or 0, r[9], r[10]))
            vecs.append(np.frombuffer(r[11], dtype=np.float32))
        self.matrix = np.vstack(vecs) if vecs else np.zeros((0, self.embedder.dim), np.float32)

    def __len__(self) -> int:
        return len(self.entries)

    def embed(self, text: str) -> np.ndarray:
        return self.embedder.encode([normalize_prompt(text)])[0]

    # ------------------------------------------------------------------ #
    def lookup(self, query: str, tags: dict | None = None, threshold: float = 0.88,
               qualifier_guard: bool = True) -> CacheResult:
        t0 = time.perf_counter()
        tags = normalize_tags(tags)
        ckey = context_key(tags)
        vec = self.embed(query)
        with self._lock:
            now = time.time()
            idx = [i for i, e in enumerate(self.entries)
                   if e.context_key == ckey
                   and (self.ttl_seconds is None or now - e.created_at <= self.ttl_seconds)]
            result = CacheResult(hit=False, candidates_checked=len(idx), embedding=vec)
            if idx:
                sims = self.matrix[idx] @ vec
                order = np.argsort(-sims)
                best = idx[int(order[0])]
                result.similarity = float(sims[order[0]])
                result.entry = self.entries[best]
                if result.similarity >= threshold:
                    reason = qualifier_conflict(query, result.entry.query) if qualifier_guard else None
                    if reason:
                        result.blocked_reason = reason
                    else:
                        result.hit = True
                        self.entries[best].hits += 1
                        self._conn.execute("UPDATE cache_entries SET hits = hits + 1 WHERE id = ?",
                                           (result.entry.id,))
                        self._conn.commit()
        result.lookup_ms = (time.perf_counter() - t0) * 1000
        return result

    def add(self, query: str, response: str, model: str, tags: dict | None = None,
            input_tokens: int = 0, output_tokens: int = 0,
            embedding: np.ndarray | None = None) -> CacheEntry:
        tags = normalize_tags(tags)
        vec = (embedding if embedding is not None else self.embed(query)).astype(np.float32)
        now = time.time()
        with self._lock:
            cur = self._conn.execute(
                "INSERT INTO cache_entries (query, normalized, response, model, tags, context_key, "
                "embedder, input_tokens, output_tokens, created_at, hits, embedding) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,0,?)",
                (query, normalize_prompt(query), response, model, json.dumps(tags),
                 context_key(tags), self.embedder.name, input_tokens, output_tokens, now,
                 vec.tobytes()),
            )
            self._conn.commit()
            entry = CacheEntry(cur.lastrowid, query, normalize_prompt(query), response, model,
                               tags, context_key(tags), input_tokens, output_tokens, now)
            self.entries.append(entry)
            self.matrix = np.vstack([self.matrix, vec[None, :]])
        return entry

    def clear(self) -> None:
        with self._lock:
            self._conn.execute("DELETE FROM cache_entries")
            self._conn.commit()
            self.entries = []
            self.matrix = np.zeros((0, self.embedder.dim), np.float32)

    def to_records(self) -> list[dict]:
        return [{"id": e.id, "query": e.query, "context": e.context_key, "model": e.model,
                 "hits": e.hits, "output_tokens": e.output_tokens,
                 "created": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(e.created_at))}
                for e in self.entries]


if __name__ == "__main__":
    print(qualifier_conflict("Convert 5 km to miles", "Convert 5 kilometers to miles."))      # None
    print(qualifier_conflict("Convert 5 miles to km", "Convert 5 kilometers to miles."))      # order
    print(qualifier_conflict("What is the capital of Germany?", "What is the capital of France?"))
