"""Embedding backends behind one interface.

- SentenceTransformerEmbedder: local semantic model (default for real use; needs a one-time model download).
- HashingEmbedder: dependency-free lexical fallback (signed hashed bag of stems + bigrams). It works fully
  offline and is deterministic, but it matches *words*, not *meaning*; it is a fallback, not a substitute.

Pick with EMBEDDER=auto|local|hashing (default auto: try local, fall back to hashing with a warning).
"""
import hashlib
import logging
import os
import re
from abc import ABC, abstractmethod

import numpy as np

log = logging.getLogger(__name__)

_STOP = set("""a an and are as at be but by can do find for from give has have how i in is it me my of on or please
show some suggest that the their there these this to want what which with would you your about any need looking
resources resource material materials books book""".split())
# Note: 'book(s)' are dropped because the question's *type* ("books on X") is a filter, not topical content.


# Level/style words describe *how* a user wants a resource, not *what* it is about.
MODIFIERS = {"beginner", "beginners", "beginning", "introductory", "introduction", "intro", "basic", "basics",
             "advanced", "intermediate", "friendly", "easy", "simple", "guide", "recent", "latest", "best", "good", "new",
             "related", "approach", "approaches", "compare", "paper", "papers", "using", "used", "learning path",
             "semester", "course", "courses", "class", "study", "reading", "list", "path", "currently"}


def stem(tok: str) -> str:
    for suf in ("ing", "es", "s", "ed"):
        if len(tok) > len(suf) + 3 and tok.endswith(suf):
            tok = tok[: -len(suf)]
            break
    if len(tok) > 3 and tok[-1] == tok[-2]:  # programm -> program
        tok = tok[:-1]
    return tok


def tokens(text: str, keep_stop: bool = False) -> list[str]:
    return [stem(t) for t in re.findall(r"[a-z0-9]+", text.lower()) if keep_stop or t not in _STOP]


MODIFIER_STEMS = {stem(m) for m in MODIFIERS}


class Embedder(ABC):
    name: str
    lexical_only: bool = False  # True => no semantic generalisation, so require a topical word match to avoid false hits
    min_score: float  # abstention threshold on the fused score; calibrate per model in the Stage 7 evaluation

    @abstractmethod
    def encode(self, texts: list[str]) -> np.ndarray:
        """Return L2-normalised float32 matrix (len(texts), dim)."""


class HashingEmbedder(Embedder):
    name = "hashing-v1"
    lexical_only = True
    min_score = 0.18

    def __init__(self, dim: int = 2048):
        self.dim = dim

    def encode(self, texts):
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            toks = tokens(text)
            feats = [(t, 1.0) for t in toks] + [(f"{a}_{b}", 0.5) for a, b in zip(toks, toks[1:])]
            for f, w in feats:
                h = int.from_bytes(hashlib.blake2b(f.encode(), digest_size=8).digest(), "little")
                out[row, h % self.dim] += w if (h >> 63) & 1 else -w
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.where(norms == 0, 1, norms)


class SentenceTransformerEmbedder(Embedder):
    min_score = 0.30  # initial guess for MiniLM; NOT yet calibrated on this catalog

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer  # lazy: heavy import
        self.name = f"st:{model_name}"
        self._model = SentenceTransformer(model_name)

    def encode(self, texts):
        return np.asarray(self._model.encode(texts, normalize_embeddings=True, show_progress_bar=False), dtype=np.float32)


def get_embedder(kind: str | None = None) -> Embedder:
    emb = _make_embedder(kind)
    if os.environ.get("SEARCH_MIN_SCORE"):   # set from `python -m eval.calibrate` output
        emb.min_score = float(os.environ["SEARCH_MIN_SCORE"])
    return emb


def _make_embedder(kind: str | None = None) -> Embedder:
    kind = (kind or os.environ.get("EMBEDDER", "auto")).lower()
    if kind == "hashing":
        return HashingEmbedder()
    try:
        return SentenceTransformerEmbedder(os.environ.get("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"))
    except Exception as exc:  # noqa: BLE001 - offline / model not downloadable
        if kind == "local":
            raise
        log.warning("Local semantic model unavailable (%s: %s); falling back to lexical HashingEmbedder.",
                    type(exc).__name__, str(exc)[:120])
        return HashingEmbedder()
