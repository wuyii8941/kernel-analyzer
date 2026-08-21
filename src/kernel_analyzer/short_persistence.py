"""Low-cost short-trajectory persistence screening.

The screen is deliberately generic: it consumes one effective-update residual
vector per ordered reference state and case, then keeps only a small CountSketch
per state.  It is not a replacement for an exact matched-repair trajectory.  A
positive screen is a risk candidate; the exact F+B/repair protocol remains the
confirmation oracle.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


def _splitmix64(values: np.ndarray) -> np.ndarray:
    """Vectorized SplitMix64 for deterministic coordinate hashing."""

    x = values.astype(np.uint64, copy=False)
    x = (x + np.uint64(0x9E3779B97F4A7C15)).astype(np.uint64)
    x = ((x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)).astype(np.uint64)
    x = ((x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)).astype(np.uint64)
    return (x ^ (x >> np.uint64(31))).astype(np.uint64)


def count_sketch_chunks(
    chunks: Iterable[Sequence[float] | np.ndarray],
    *,
    projection_dim: int,
    seed: int,
    chunk_size: int = 1 << 20,
) -> tuple[np.ndarray, int]:
    """CountSketch ordered chunks without concatenating parameter tensors."""

    if projection_dim < 4 or chunk_size < 1:
        raise ValueError("projection_dim must be >=4 and chunk_size must be positive")
    output = np.zeros(projection_dim, dtype=np.float64)
    coordinate_count = 0
    saw_chunk = False
    for chunk in chunks:
        values = np.asarray(chunk, dtype=np.float64).reshape(-1)
        if not np.isfinite(values).all():
            raise ValueError("screen vector is nonfinite")
        saw_chunk = True
        for local_start in range(0, values.size, chunk_size):
            local_stop = min(local_start + chunk_size, values.size)
            start = coordinate_count + local_start
            stop = coordinate_count + local_stop
            indices = np.arange(start, stop, dtype=np.uint64)
            hashed = _splitmix64(indices + np.uint64(seed))
            buckets = (hashed % np.uint64(projection_dim)).astype(np.int64)
            signs = np.where((hashed & np.uint64(1)) == 0, 1.0, -1.0)
            np.add.at(output, buckets, signs * values[local_start:local_stop])
        coordinate_count += int(values.size)
    if not saw_chunk or coordinate_count == 0:
        raise ValueError("screen vector has zero coordinates")
    return output / math.sqrt(float(projection_dim)), coordinate_count


def count_sketch(
    vector: Sequence[float] | np.ndarray,
    *,
    projection_dim: int,
    seed: int,
    chunk_size: int = 1 << 20,
) -> np.ndarray:
    """Return a deterministic signed CountSketch of a dense vector."""

    result, _ = count_sketch_chunks(
        [vector], projection_dim=projection_dim, seed=seed, chunk_size=chunk_size
    )
    return result


def _prefixes(steps: int) -> list[int]:
    return sorted({stop for stop in (2, 4, 8, 16, 32, steps) if stop <= steps})


def _amplification(vectors: np.ndarray) -> float:
    energy = float(np.square(vectors).sum())
    if energy <= 0.0:
        return 0.0
    return float(np.linalg.norm(vectors.sum(axis=0)) / math.sqrt(energy))


def _lag_curve(vectors: np.ndarray, max_lag: int) -> list[dict[str, float | int]]:
    diagonal = np.linalg.norm(vectors, axis=1)
    result: list[dict[str, float | int]] = []
    for lag in range(1, min(max_lag, len(vectors) - 1) + 1):
        values = np.sum(vectors[:-lag] * vectors[lag:], axis=1)
        denominator = diagonal[:-lag] * diagonal[lag:]
        result.append({
            "lag": lag,
            "pairs": int(len(values)),
            "normalized_correlation": float(values.sum() / max(float(denominator.sum()), 1e-30)),
            "mean_inner_product": float(values.mean()),
        })
    return result


def _sign_flip_null(
    vectors: np.ndarray,
    *,
    draws: int,
    seed: int,
) -> np.ndarray:
    generator = np.random.default_rng(seed)
    result = np.empty(draws, dtype=np.float64)
    energy = float(np.square(vectors).sum())
    for index in range(draws):
        signs = generator.choice(np.array([-1.0, 1.0]), size=len(vectors))
        result[index] = (
            float(np.linalg.norm((vectors * signs[:, None]).sum(axis=0)) / math.sqrt(energy))
            if energy > 0.0 else 0.0
        )
    return result


@dataclass
class ShortPersistencePath:
    """One case's projected ordered effective-update residual path."""

    case_id: str
    projection_dim: int
    projection_seed: int
    expected_steps: int
    null_draws: int = 2000
    null_seed: int = 20260822
    max_lag: int = 4
    _vectors: list[np.ndarray] = field(default_factory=list)
    _coordinate_count: int | None = None

    def add_vector(self, vector: Sequence[float] | np.ndarray) -> None:
        self.add_chunks([vector])

    def add_chunks(self, chunks: Iterable[Sequence[float] | np.ndarray]) -> None:
        sketch, coordinate_count = count_sketch_chunks(
            chunks, projection_dim=self.projection_dim, seed=self.projection_seed
        )
        if self._coordinate_count is None:
            self._coordinate_count = coordinate_count
        elif coordinate_count != self._coordinate_count:
            raise ValueError(f"{self.case_id}: coordinate count changed")
        self._vectors.append(sketch)
        if len(self._vectors) > self.expected_steps:
            raise ValueError(f"{self.case_id}: too many states")

    def finalize(self) -> dict[str, Any]:
        if len(self._vectors) != self.expected_steps:
            raise ValueError(f"{self.case_id}: incomplete short path")
        vectors = np.stack(self._vectors, axis=0)
        if not np.isfinite(vectors).all():
            raise ValueError(f"{self.case_id}: projected path is nonfinite")
        energy = float(np.square(vectors).sum())
        if energy <= 0.0:
            return {
                "schema": "kernel-analyzer-short-persistence-screen-v1",
                "case_id": self.case_id,
                "status": "UNRESOLVED_ZERO_ENERGY",
                "steps": self.expected_steps,
                "coordinate_count": self._coordinate_count,
            }
        prefixes: dict[str, dict[str, float]] = {}
        for stop in _prefixes(self.expected_steps):
            block = vectors[:stop]
            prefixes[str(stop)] = {
                "amplification": _amplification(block),
                "diffusive_scale": math.sqrt(float(np.square(block).sum())),
                "resultant_l2": float(np.linalg.norm(block.sum(axis=0))),
            }
        null = _sign_flip_null(vectors, draws=self.null_draws, seed=self.null_seed)
        observed = _amplification(vectors)
        lag = _lag_curve(vectors, self.max_lag)
        positive_lags = [row for row in lag if row["normalized_correlation"] > 0.0]
        # Requiring the first lag to be positive prevents an alternating
        # sequence from passing merely because its even lags are positive.
        lag1_positive = bool(lag and lag[0]["normalized_correlation"] > 0.0)
        null_upper = float(np.quantile(null, 0.95))
        later_prefixes = [
            values["amplification"] for stop, values in prefixes.items()
            if int(stop) >= min(8, self.expected_steps)
        ]
        prefix_growth = (
            later_prefixes[-1] > later_prefixes[0] if len(later_prefixes) >= 2 else False
        )
        risk_candidate = bool(
            observed > null_upper
            and lag1_positive
            and len(positive_lags) >= 2
            and prefix_growth
        )
        return {
            "schema": "kernel-analyzer-short-persistence-screen-v1",
            "case_id": self.case_id,
            "status": "RISK_CANDIDATE" if risk_candidate else "NULL_LIKE_OR_UNRESOLVED",
            "steps": self.expected_steps,
            "coordinate_count": self._coordinate_count,
            "projection": {
                "kind": "SIGNED_COUNT_SKETCH",
                "dimension": self.projection_dim,
                "seed": self.projection_seed,
                "coordinate_hashing": "SPLITMIX64",
            },
            "observed_amplification": observed,
            "sign_flip_null": {
                "draws": self.null_draws,
                "seed": self.null_seed,
                "median": float(np.quantile(null, 0.5)),
                "upper_95": null_upper,
                "one_sided_p": float((1 + np.count_nonzero(null >= observed)) / (self.null_draws + 1)),
            },
            "prefixes": prefixes,
            "lag_correlation": lag,
            "positive_lag_count": len(positive_lags),
            "lag1_positive": lag1_positive,
            "prefix_growth_after_short_warmup": prefix_growth,
            "screen_rule": {
                "requires_observed_above_sign_flip_95": True,
                "requires_lag1_and_at_least_two_positive_lags": True,
                "requires_late_prefix_growth": True,
                "is_confirmation": False,
            },
            "raw_vectors_retained": False,
        }


class SharedShortPersistenceScreen:
    """Run the same sketch protocol for many endpoints on shared states."""

    def __init__(self, *, projection_dim: int = 64, projection_seed: int = 20260822,
                 expected_steps: int = 8, null_draws: int = 2000) -> None:
        if expected_steps < 4:
            raise ValueError("short screen needs at least four ordered states")
        self.projection_dim = int(projection_dim)
        self.projection_seed = int(projection_seed)
        self.expected_steps = int(expected_steps)
        self.null_draws = int(null_draws)
        self._paths: dict[str, ShortPersistencePath] = {}

    def add(self, case_id: str, vector: Sequence[float] | np.ndarray) -> None:
        self.add_chunks(case_id, [vector])

    def add_chunks(
        self,
        case_id: str,
        chunks: Iterable[Sequence[float] | np.ndarray],
    ) -> None:
        path = self._paths.setdefault(case_id, ShortPersistencePath(
            case_id=case_id,
            projection_dim=self.projection_dim,
            projection_seed=self.projection_seed,
            expected_steps=self.expected_steps,
            null_draws=self.null_draws,
        ))
        path.add_chunks(chunks)

    def finalize(self) -> dict[str, Any]:
        return {
            "schema": "kernel-analyzer-shared-short-persistence-screen-v1",
            "status": "COMPLETE",
            "protocol": {
                "projection_dim": self.projection_dim,
                "projection_seed": self.projection_seed,
                "expected_steps": self.expected_steps,
                "null_draws": self.null_draws,
                "shared_reference_states": True,
                "raw_vectors_retained": False,
                "positive_screen_requires_exact_confirmation": True,
            },
            "cases": [self._paths[key].finalize() for key in sorted(self._paths)],
        }
