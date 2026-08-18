"""Complete-coordinate directional statistics without retaining JSON tensors."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


def direction_certificate_from_gram(
    gram: np.ndarray, *, coordinates: int, state_ids: Sequence[str],
    state_vector_sha256: Mapping[str, str], bootstrap_draws: int = 4000,
    seed: int = 0,
) -> Mapping[str, Any]:
    """Compute the de-biased U statistic and distinct-cluster bootstrap.

    When a state is sampled more than once, pairs of two copies of that same
    original state are excluded. Including those diagonal pairs adds squared
    norms that are absent from the U statistic and can manufacture a positive
    confidence interval for a non-coherent carrier.
    """
    matrix = np.asarray(gram, dtype=np.float64)
    states = len(state_ids)
    if matrix.shape != (states, states) or states < 2 or coordinates < 1:
        raise ValueError("invalid complete Gram certificate inputs")
    off_diagonal = matrix.sum() - np.trace(matrix)
    statistic = float(off_diagonal / (states * (states - 1)))
    rng = np.random.default_rng(seed)
    samples = []
    diagonal = np.diag(matrix)
    while len(samples) < bootstrap_draws:
        indices = rng.integers(0, states, size=states)
        counts = np.bincount(indices, minlength=states).astype(np.float64)
        denominator = float(states * states - counts @ counts)
        if denominator <= 0.0:
            continue
        numerator = float(counts @ matrix @ counts - (counts * counts) @ diagonal)
        samples.append(numerator / denominator)
    confidence = {
        "lower_95": float(np.quantile(samples, 0.025)),
        "median": float(np.quantile(samples, 0.5)),
        "upper_95": float(np.quantile(samples, 0.975)),
    }
    return {
        "status": "PASS" if confidence["lower_95"] > 0.0 else "FAIL_CAUSAL_NONCOHERENT",
        "streamed_complete_gram": True,
        "complete_coordinates": True,
        "independent_states": True,
        "bootstrap_excludes_same_original_cluster_pairs": True,
        "state_ids": list(state_ids),
        "state_vector_sha256": dict(state_vector_sha256),
        "states": states,
        "coordinates": coordinates,
        "cross_state_inner_product_u": statistic,
        "cluster_bootstrap_95": confidence,
        "gram": matrix.tolist(),
    }


def direction_certificate_from_vector_files(
    rows: Sequence[Mapping[str, Any]], *, chunk_elements: int = 1_048_576,
    bootstrap_draws: int = 4000, seed: int = 0,
) -> Mapping[str, Any]:
    """Build one complete Gram directly from retained float32/float64 spools."""
    if len(rows) < 2:
        raise ValueError("at least two state vectors are required")
    coordinates = int(rows[0]["coordinates"])
    if coordinates < 1 or any(int(row["coordinates"]) != coordinates for row in rows):
        raise ValueError("state vectors have different coordinate counts")
    arrays = []
    for row in rows:
        if row.get("constant_zero"):
            arrays.append(None)
            continue
        dtype = np.float32 if row.get("storage_dtype") == "float32" else np.float64
        arrays.append(np.memmap(row["path"], dtype=dtype, mode="r", shape=(coordinates,)))
    states = len(rows)
    gram = np.zeros((states, states), dtype=np.float64)
    for start in range(0, coordinates, chunk_elements):
        stop = min(coordinates, start + chunk_elements)
        block = np.stack([
            np.zeros(stop - start, dtype=np.float64) if array is None
            else np.asarray(array[start:stop], dtype=np.float64)
            for array in arrays
        ])
        gram += block @ block.T
    del arrays
    return direction_certificate_from_gram(
        gram, coordinates=coordinates,
        state_ids=[str(row["state_id"]) for row in rows],
        state_vector_sha256={str(row["state_id"]): row["sha256"] for row in rows},
        bootstrap_draws=bootstrap_draws, seed=seed,
    )


class StreamingGramAccumulator:
    """Store temporary vectors and reduce them to a complete Gram certificate.

    Temporary vectors are raw float64 files under an explicitly supplied
    directory.  They are removed after a certificate is finalized unless the
    caller opts into retaining them for debugging.
    """

    def __init__(self, root: Path, certificate_id: str, chunk_elements: int = 1_048_576) -> None:
        if chunk_elements < 1:
            raise ValueError("chunk_elements must be positive")
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self.certificate_id = certificate_id
        self.chunk_elements = chunk_elements
        self._rows: list[tuple[str, Path, int, str]] = []
        self._coordinates: int | None = None

    def add_chunks(self, state_id: str, chunks: Iterable[np.ndarray]) -> Mapping[str, Any]:
        if any(row[0] == state_id for row in self._rows):
            raise ValueError("duplicate state ID: %s" % state_id)
        path = self.root / (hashlib.sha256(
            (self.certificate_id + "\0" + state_id).encode()
        ).hexdigest() + ".f64")
        count = 0
        digest = hashlib.sha256()
        with path.open("wb") as handle:
            for chunk in chunks:
                values = np.asarray(chunk, dtype=np.float64).reshape(-1)
                encoded = values.tobytes(order="C")
                handle.write(encoded)
                digest.update(encoded)
                count += values.size
        if self._coordinates is None:
            self._coordinates = count
        elif count != self._coordinates:
            path.unlink(missing_ok=True)
            raise ValueError("state vectors have different coordinate counts")
        self._rows.append((state_id, path, count, digest.hexdigest()))
        return {"state_id": state_id, "coordinates": count, "sha256": digest.hexdigest()}

    def add_array(self, state_id: str, values: Sequence[float] | np.ndarray) -> Mapping[str, Any]:
        array = np.asarray(values, dtype=np.float64).reshape(-1)
        return self.add_chunks(
            state_id,
            (array[start:start + self.chunk_elements]
             for start in range(0, array.size, self.chunk_elements)),
        )

    def finalize(
        self, *, bootstrap_draws: int = 4000, seed: int = 0,
        cleanup: bool = True,
    ) -> Mapping[str, Any]:
        if len(self._rows) < 2 or not self._coordinates:
            raise ValueError("at least two nonempty state vectors are required")
        vectors = [np.memmap(path, dtype=np.float64, mode="r", shape=(count,))
                   for _, path, count, _ in self._rows]
        states = len(vectors)
        gram = np.zeros((states, states), dtype=np.float64)
        for start in range(0, self._coordinates, self.chunk_elements):
            stop = min(self._coordinates, start + self.chunk_elements)
            block = np.stack([np.asarray(row[start:stop]) for row in vectors])
            gram += block @ block.T
        certificate = direction_certificate_from_gram(
            gram, coordinates=self._coordinates,
            state_ids=[row[0] for row in self._rows],
            state_vector_sha256={row[0]: row[3] for row in self._rows},
            bootstrap_draws=bootstrap_draws, seed=seed,
        )
        del vectors
        if cleanup:
            for _, path, _, _ in self._rows:
                path.unlink(missing_ok=True)
        return certificate
