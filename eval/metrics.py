"""Pure metric helpers (unit-tested)."""
import math


def hit_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    return 1.0 if any(t in relevant for t in ranked[:k]) else 0.0


def recall_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    return len(set(ranked[:k]) & relevant) / len(relevant) if relevant else 0.0


def precision_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    top = ranked[:k]
    return sum(t in relevant for t in top) / len(top) if top else 0.0


def reciprocal_rank(ranked: list[str], relevant: set[str]) -> float:
    for i, t in enumerate(ranked, 1):
        if t in relevant:
            return 1.0 / i
    return 0.0


def percentile(values: list[float], p: float) -> float:
    """Nearest-rank percentile (p in 0..100)."""
    if not values:
        return float("nan")
    xs = sorted(values)
    return xs[max(0, math.ceil(p / 100 * len(xs)) - 1)]


def mean(xs) -> float:
    xs = list(xs)
    return sum(xs) / len(xs) if xs else float("nan")
