"""Cross-fit geometry diagnostics for effective-update persistence.

The routines in this module are intentionally exploratory: they never alter a
mainline verdict.  They operate on compact CPU spools containing one update
vector per state and fit every reported subspace from calibration rows only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import torch


TensorTree = Mapping[str, torch.Tensor]


def _flatten(tree: TensorTree, order: Sequence[str] | None = None) -> torch.Tensor:
    keys = list(order) if order is not None else sorted(tree)
    return torch.cat([tree[key].detach().float().reshape(-1).cpu() for key in keys])


def _row_matrix(rows: Sequence[Mapping[str, Any]], field: str,
               order: Sequence[str] | None = None) -> torch.Tensor:
    values = [_flatten(row[field], order) for row in rows]
    if not values:
        raise ValueError(f"no rows for field {field}")
    return torch.stack(values)


def normalized_gram(matrix: torch.Tensor) -> torch.Tensor:
    matrix = matrix.float()
    norms = torch.linalg.vector_norm(matrix, dim=1, keepdim=True)
    # Zero rows are valid for a decomposition component that is undefined in
    # calibration (for example feedback before the two arms have diverged).
    # They contribute a zero row/column instead of turning an exploratory
    # report into a hard failure.
    normalized = matrix / norms.clamp_min(1e-30)
    return normalized @ normalized.T


def _basis_from_rows(matrix: torch.Tensor, rank: int) -> tuple[torch.Tensor, list[float]]:
    """Return an orthonormal basis represented as ``rank x dimension`` rows.

    The eigendecomposition is performed on the small row Gram matrix, so the
    implementation never materialises a d-by-d covariance matrix.
    """
    if rank < 1:
        raise ValueError("rank must be positive")
    matrix = matrix.float()
    norms = torch.linalg.vector_norm(matrix, dim=1, keepdim=True)
    keep = norms[:, 0] > 0
    if not bool(keep.any()):
        return matrix.new_zeros((0, matrix.shape[1])), []
    normalized = matrix[keep] / norms[keep]
    gram = normalized @ normalized.T
    eigenvalues, eigenvectors = torch.linalg.eigh(gram)
    order = torch.argsort(eigenvalues, descending=True)
    vectors: list[torch.Tensor] = []
    values: list[float] = []
    for index in order.tolist():
        value = float(eigenvalues[index].item())
        if value <= 1e-10 or len(vectors) >= rank:
            continue
        # v^T X / sqrt(lambda) is a unit principal direction.  ``normalized``
        # has shape (number_of_rows, dimension), so the multiplication is
        # X^T v rather than the other way round.
        basis = torch.mv(normalized.T, eigenvectors[:, index]) / (value ** 0.5)
        vectors.append(basis / torch.linalg.vector_norm(basis))
        values.append(value)
    return (torch.stack(vectors) if vectors else matrix.new_zeros((0, matrix.shape[1]))), values


def subspace_overlap(left: torch.Tensor, right: torch.Tensor) -> float:
    if left.ndim != 2 or right.ndim != 2 or left.shape[1] != right.shape[1]:
        raise ValueError("subspaces must be rank-by-dimension matrices")
    rank = min(left.shape[0], right.shape[0])
    if rank == 0:
        return 0.0
    return float(torch.sum((left[:rank] @ right[:rank].T) ** 2).item() / rank)


def projected_persistence(matrix: torch.Tensor, basis: torch.Tensor) -> dict[str, float]:
    if basis.shape[0] == 0:
        return {"persistence": 0.0, "energy_capture": 0.0, "projected_sum_l2": 0.0}
    projected = (matrix.float() @ basis.T) @ basis
    norms = torch.linalg.vector_norm(projected, dim=1)
    total = float(norms.sum().item())
    summed = float(torch.linalg.vector_norm(projected.sum(dim=0)).item())
    energy = float((torch.sum(projected * projected) /
                    torch.clamp(torch.sum(matrix.float() * matrix.float()), min=1e-30)).item())
    return {
        "persistence": summed / max(total, 1e-30),
        "energy_capture": energy,
        "projected_sum_l2": summed,
    }


def _permutation_null(values: torch.Tensor, seed: int = 3407) -> float:
    generator = torch.Generator().manual_seed(seed)
    permutation = torch.randperm(values.shape[0], generator=generator)
    return float(values[permutation].mean().item())


@dataclass(frozen=True)
class GeometryAnalyzer:
    ranks: tuple[int, ...] = (1, 2, 4, 8)
    windows: tuple[int, ...] = (2, 4, 8)

    def rank_report(self, calibration: Sequence[Mapping[str, Any]],
                    evaluation: Sequence[Mapping[str, Any]],
                    field: str = "local", order: Sequence[str] | None = None,
                    calibration_field: str | None = None) -> dict[str, Any]:
        calibration_field = calibration_field or field
        cal = _row_matrix(calibration, calibration_field, order)
        ev = _row_matrix(evaluation, field, order)
        odd, even = cal[::2], cal[1::2]
        report: dict[str, Any] = {
            "field": field,
            "calibration_field": calibration_field,
            "calibration_rows": int(cal.shape[0]),
            "evaluation_rows": int(ev.shape[0]),
            "normalized_gram_calibration": normalized_gram(cal).tolist(),
            "singular_values_normalized_calibration": torch.linalg.svdvals(
                cal / torch.linalg.vector_norm(cal, dim=1, keepdim=True).clamp_min(1e-30)
            ).tolist(),
            "ranks": {},
        }
        for rank in self.ranks:
            q_odd, odd_values = _basis_from_rows(odd, rank)
            q_even, even_values = _basis_from_rows(even, rank)
            q_all, all_values = _basis_from_rows(cal, rank)
            persistence = projected_persistence(ev, q_all)
            report["ranks"][str(rank)] = {
                "effective_rank": int(q_all.shape[0]),
                "odd_even_subspace_overlap": subspace_overlap(q_odd, q_even),
                "calibration_odd_eigenvalues": odd_values,
                "calibration_even_eigenvalues": even_values,
                "calibration_all_eigenvalues": all_values,
                "evaluation": persistence,
            }
        return report

    def window_report(self, rows: Sequence[Mapping[str, Any]], field: str = "local",
                      order: Sequence[str] | None = None, seed: int = 3407) -> dict[str, Any]:
        matrix = _row_matrix(rows, field, order)
        entries: list[dict[str, Any]] = []
        for window in self.windows:
            if len(rows) <= window:
                continue
            predictions: list[float] = []
            overlaps: list[float] = []
            for index in range(window, len(rows)):
                basis, _ = _basis_from_rows(matrix[index - window:index], 1)
                if basis.shape[0] == 0:
                    continue
                current = matrix[index:index + 1]
                predictions.append(projected_persistence(current, basis)["energy_capture"])
                previous, _ = _basis_from_rows(matrix[max(0, index - 2 * window):index - window], 1)
                if previous.shape[0]:
                    overlaps.append(subspace_overlap(previous, basis))
            entries.append({
                "window": window,
                "next_step_projection_mean": sum(predictions) / max(len(predictions), 1),
                "next_step_projection_values": predictions,
                "adjacent_window_overlap_mean": sum(overlaps) / max(len(overlaps), 1),
                "steps": len(predictions),
            })
        # This is a fixed shuffle diagnostic, not a post-hoc best permutation.
        generator = torch.Generator().manual_seed(seed)
        permutation = torch.randperm(matrix.shape[0], generator=generator).tolist()
        shuffled = [rows[index] for index in permutation]
        shuffled_report = self.window_report_no_shuffle(shuffled, field, order)
        return {"field": field, "original": entries, "shuffle_seed": seed,
                "shuffled": shuffled_report}

    def window_report_no_shuffle(self, rows: Sequence[Mapping[str, Any]], field: str,
                                 order: Sequence[str] | None = None) -> list[dict[str, Any]]:
        matrix = _row_matrix(rows, field, order)
        entries: list[dict[str, Any]] = []
        for window in self.windows:
            if len(rows) <= window:
                continue
            values: list[float] = []
            for index in range(window, len(rows)):
                basis, _ = _basis_from_rows(matrix[index - window:index], 1)
                if basis.shape[0]:
                    values.append(projected_persistence(matrix[index:index + 1], basis)["energy_capture"])
            entries.append({"window": window, "next_step_projection_mean": sum(values) / max(len(values), 1),
                            "next_step_projection_values": values, "steps": len(values)})
        return entries


def load_spool(path: str) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or payload.get("schema") != "kernel-analyzer-seup-geometry-spool-v1":
        raise ValueError("invalid SEUP geometry spool")
    return payload
