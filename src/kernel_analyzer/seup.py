"""Signed Effective-Update Persistence (SEUP).

This module contains the measurement protocol used by the mainline property
experiment.  It deliberately separates three objects which were previously
mixed together in the trajectory scripts:

* a carrier fitted on an independent calibration lane;
* the symmetric endpoint/local versus state-feedback decomposition; and
* the norm-preserving sign intervention used only for the Liger anchor.

The tensors are accepted either as one tensor or as a mapping of parameter
names to tensors.  No candidate verdict is consumed by this module.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
import random
from typing import Any, Dict, Mapping, Sequence, Tuple, Union


TensorTree = Union[Any, Mapping[str, Any]]


def _torch():
    import torch

    return torch


def _as_tree(value: TensorTree) -> Dict[str, Any]:
    torch = _torch()
    if isinstance(value, torch.Tensor):
        return {"value": value}
    if not isinstance(value, Mapping) or not value:
        raise ValueError("an effective update must be a tensor or nonempty tensor mapping")
    result = {str(key): tensor for key, tensor in value.items()}
    if any(not isinstance(tensor, torch.Tensor) for tensor in result.values()):
        raise TypeError("every effective-update leaf must be a torch.Tensor")
    return result


def _validate_same(left: Mapping[str, Any], right: Mapping[str, Any]) -> None:
    torch = _torch()
    _validate_structure(left, right)
    for key in left:
        if not bool(torch.isfinite(left[key]).all()) or not bool(torch.isfinite(right[key]).all()):
            raise ValueError("effective updates must be finite")


def _validate_structure(left: Mapping[str, Any], right: Mapping[str, Any]) -> None:
    if set(left) != set(right):
        raise ValueError("effective-update tensor populations differ")
    for key in left:
        if left[key].shape != right[key].shape:
            raise ValueError("effective-update tensor shapes differ at %s" % key)


def _clone_float(value: Mapping[str, Any]) -> Dict[str, Any]:
    return {key: tensor.detach().float().clone() for key, tensor in value.items()}


def _tree_sub(left: Mapping[str, Any], right: Mapping[str, Any]) -> Dict[str, Any]:
    _validate_structure(left, right)
    return {key: left[key].detach().float() - right[key].detach().float() for key in left}


def _tree_avg(left: Mapping[str, Any], right: Mapping[str, Any]) -> Dict[str, Any]:
    _validate_structure(left, right)
    return {key: (left[key].detach().float() + right[key].detach().float()).mul_(0.5)
            for key in left}


def _tree_dot(left: Mapping[str, Any], right: Mapping[str, Any], chunk: int = 1_048_576) -> float:
    """Accurate tree dot product without materialising a flattened vector."""
    torch = _torch()
    _validate_structure(left, right)
    total = 0.0
    for key in sorted(left):
        a = left[key].detach().reshape(-1)
        b = right[key].detach().reshape(-1)
        for start in range(0, a.numel(), chunk):
            stop = min(a.numel(), start + chunk)
            total += float(torch.sum(a[start:stop].double() * b[start:stop].double()).item())
    return total


def _tree_norm(value: Mapping[str, Any]) -> float:
    return math.sqrt(max(0.0, _tree_dot(value, value)))


def _tree_add_scaled_(target: Mapping[str, Any], source: Mapping[str, Any], scale: float) -> None:
    _validate_structure(target, source)
    for key in target:
        target[key].add_(source[key], alpha=float(scale))


def _tree_scaled(value: Mapping[str, Any], scale: float) -> Dict[str, Any]:
    return {key: tensor.detach().float().mul(float(scale)) for key, tensor in value.items()}


def _basis_digest(value: Mapping[str, Any], chunk: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    for key in sorted(value):
        tensor = value[key].detach().float().cpu().contiguous().reshape(-1)
        digest.update(key.encode())
        digest.update(str(tuple(value[key].shape)).encode())
        block = max(1, chunk // tensor.element_size())
        for start in range(0, tensor.numel(), block):
            digest.update(tensor[start:start + block].numpy().tobytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class FrozenCarrier:
    """Frozen calibration result; ``basis`` is intentionally not serialised."""

    basis: Mapping[str, Any] | None
    certificate: Dict[str, Any]

    @property
    def stable(self) -> bool:
        return bool(self.certificate["stable_carrier"])


class SEUPCalibrator:
    """Fit a carrier from an independent calibration lane.

    Odd/even cross-fitting is a gate, not a second fit: the final direction is
    the normalised sum over all calibration updates only after the gate passes.
    """

    def __init__(self, calibration_steps: int = 16, gate_cosine: float = 0.5):
        if calibration_steps < 1:
            raise ValueError("calibration_steps must be positive")
        if not 0.0 < gate_cosine <= 1.0:
            raise ValueError("gate_cosine must be in (0, 1]")
        self.calibration_steps = int(calibration_steps)
        self.gate_cosine = float(gate_cosine)
        self._steps: list[str] = []
        self._odd: Dict[str, Any] | None = None
        self._even: Dict[str, Any] | None = None
        self._total: Dict[str, Any] | None = None

    def add(self, step_id: str, delta_update: TensorTree) -> None:
        if len(self._steps) >= self.calibration_steps:
            raise RuntimeError("calibration lane is already complete")
        if not step_id:
            raise ValueError("step_id must be nonempty")
        tree = _clone_float(_as_tree(delta_update))
        if any(not bool(_torch().isfinite(tensor).all()) for tensor in tree.values()):
            raise ValueError("effective updates must be finite")
        if self._total is None:
            self._total = _clone_float(tree)
        else:
            _tree_add_scaled_(self._total, tree, 1.0)
        target = self._odd if len(self._steps) % 2 == 0 else self._even
        if target is None:
            target = _clone_float(tree)
            if len(self._steps) % 2 == 0:
                self._odd = target
            else:
                self._even = target
        else:
            _tree_add_scaled_(target, tree, 1.0)
        self._steps.append(str(step_id))

    def freeze(self) -> FrozenCarrier:
        if len(self._steps) != self.calibration_steps:
            raise RuntimeError("calibration lane is incomplete")
        assert self._odd is not None and self._total is not None
        epsilon = 1e-30
        odd_norm = _tree_norm(self._odd)
        even_norm = _tree_norm(self._even) if self._even is not None else 0.0
        total_norm = _tree_norm(self._total)
        stable = odd_norm > 0.0 and even_norm > 0.0 and total_norm > 0.0
        odd_dir = _tree_scaled(self._odd, 1.0 / max(odd_norm, epsilon))
        even_dir = _tree_scaled(self._even, 1.0 / max(even_norm, epsilon)) if self._even is not None else None
        cosine = _tree_dot(odd_dir, even_dir) if stable and even_dir is not None else 0.0
        odd_on_even = (_tree_dot(odd_dir, self._even) / (self.calibration_steps / 2)
                       if self._even is not None else 0.0)
        even_on_odd = (_tree_dot(even_dir, self._odd) / (self.calibration_steps / 2)
                       if even_dir is not None else 0.0)
        stable = bool(stable and cosine >= self.gate_cosine and odd_on_even > 0.0 and even_on_odd > 0.0)
        basis = _tree_scaled(self._total, 1.0 / max(total_norm, epsilon)) if total_norm else None
        certificate = {
            "schema": "kernel-analyzer-seup-carrier-v2",
            "calibration_steps": self.calibration_steps,
            "calibration_state_ids": list(self._steps),
            "gate_cosine_threshold": self.gate_cosine,
            "odd_even_cosine": cosine,
            "odd_direction_on_even_sum": odd_on_even,
            "even_direction_on_odd_sum": even_on_odd,
            "odd_sum_l2": odd_norm,
            "even_sum_l2": even_norm,
            "all_sum_l2": total_norm,
            "stable_carrier": stable,
            "status": "STABLE_CARRIER" if stable else "NO_STABLE_CARRIER",
            "basis_rank": 1,
            "basis_sha256": None if basis is None else _basis_digest(basis),
            "fit_uses_evaluation_values": False,
        }
        return FrozenCarrier(basis if stable else None, certificate)


@dataclass(frozen=True)
class SEUPRecord:
    step_id: str
    update_l2: float
    signed_projection: float
    residual_l2: float
    carrier_fraction: float

    def as_dict(self) -> Dict[str, Any]:
        return {
            "step_id": self.step_id,
            "update_l2": self.update_l2,
            "signed_projection": self.signed_projection,
            "residual_l2": self.residual_l2,
            "carrier_fraction": self.carrier_fraction,
        }


class SEUPAccumulator:
    """Backward-compatible one-sided scorer using the new calibration gate."""

    def __init__(self, calibration_steps: int = 16, evaluation_steps: int = 16,
                 gate_cosine: float = 0.5):
        self.calibration_steps = int(calibration_steps)
        self.evaluation_steps = int(evaluation_steps)
        self._calibrator = SEUPCalibrator(calibration_steps, gate_cosine)
        self._carrier: FrozenCarrier | None = None
        self._records: list[SEUPRecord] = []
        self._sum_update_energy = 0.0
        self._sum_carrier_energy = 0.0
        self._sum_projection = 0.0
        self._sum_abs_projection = 0.0
        self._residual_sum: Dict[str, Any] | None = None
        self._sum_update: Dict[str, Any] | None = None

    @property
    def basis(self) -> Mapping[str, Any]:
        if self._carrier is None or self._carrier.basis is None:
            raise RuntimeError("carrier basis is not stable or not frozen")
        return self._carrier.basis

    def add(self, step_id: str, delta_update: TensorTree) -> str:
        count = len(self._calibrator._steps)
        if count >= self.calibration_steps and len(self._records) >= self.evaluation_steps:
            raise RuntimeError("SEUP window is already complete")
        if count < self.calibration_steps:
            self._calibrator.add(step_id, delta_update)
            if count + 1 == self.calibration_steps:
                self._carrier = self._calibrator.freeze()
                if self._carrier.basis is None:
                    raise RuntimeError("calibration effective updates have no stable signed carrier")
                if self._carrier.basis is not None:
                    self._residual_sum = {key: value.new_zeros(value.shape) for key, value in self.basis.items()}
                    self._sum_update = {key: value.new_zeros(value.shape) for key, value in self.basis.items()}
            return "CALIBRATION"
        if self._carrier is None:
            raise RuntimeError("carrier was not frozen")
        tree = _as_tree(delta_update)
        self._records.append(self._score(step_id, tree))
        return "EVALUATION"

    def _score(self, step_id: str, tree: Mapping[str, Any]) -> SEUPRecord:
        if self._carrier is None or self._carrier.basis is None:
            raise RuntimeError("cannot score one-sided update without stable carrier")
        basis = self._carrier.basis
        _validate_same(tree, basis)
        update_l2 = _tree_norm(tree)
        projection = _tree_dot(tree, basis)
        residual = _clone_float(tree)
        _tree_add_scaled_(residual, basis, -projection)
        residual_l2 = _tree_norm(residual)
        assert self._residual_sum is not None and self._sum_update is not None
        _tree_add_scaled_(self._residual_sum, residual, 1.0)
        _tree_add_scaled_(self._sum_update, tree, 1.0)
        self._sum_update_energy += update_l2 * update_l2
        self._sum_carrier_energy += projection * projection
        self._sum_projection += projection
        self._sum_abs_projection += abs(projection)
        return SEUPRecord(str(step_id), update_l2, projection, residual_l2,
                          projection * projection / max(update_l2 * update_l2, 1e-30))

    def finalize(self) -> Dict[str, Any]:
        if len(self._records) != self.evaluation_steps:
            raise RuntimeError("SEUP evaluation window is incomplete")
        if self._carrier is None:
            raise RuntimeError("SEUP carrier was not frozen")
        if self._carrier.basis is None:
            return {
                "schema": "kernel-analyzer-seup-certificate-v2",
                "status": "NO_STABLE_CARRIER",
                "carrier": self._carrier.certificate,
                "evaluation_steps": self.evaluation_steps,
                "evaluation_records": [row.as_dict() for row in self._records],
            }
        epsilon = 1e-30
        residual_accumulation = _tree_norm(self._residual_sum or {})
        carrier_accumulation = abs(self._sum_projection)
        return {
            "schema": "kernel-analyzer-seup-certificate-v2",
            "status": "MEASURED",
            "carrier": self._carrier.certificate,
            "evaluation_steps": self.evaluation_steps,
            "carrier_energy_capture": self._sum_carrier_energy / max(self._sum_update_energy, epsilon),
            "signed_persistence": abs(self._sum_projection) / max(self._sum_abs_projection, epsilon),
            "carrier_accumulation_abs": carrier_accumulation,
            "residual_accumulation_l2": residual_accumulation,
            "residual_to_carrier_accumulation": residual_accumulation / max(carrier_accumulation, epsilon),
            "total_accumulated_update_l2": _tree_norm(self._sum_update or {}),
            "evaluation_records": [row.as_dict() for row in self._records],
            "t4_verdict_used_to_fit_carrier": False,
        }


class SymmetricSEUPEvaluator:
    """Evaluate the exact four-counterfactual symmetric decomposition."""

    def __init__(self, carrier: FrozenCarrier, evaluation_steps: int = 16):
        self.carrier = carrier
        self.evaluation_steps = int(evaluation_steps)
        self._records: list[Dict[str, Any]] = []
        self._local_sum: Dict[str, Any] | None = None
        self._feedback_sum: Dict[str, Any] | None = None
        self._last_drift: Dict[str, Any] | None = None
        self._sum_abs_a = 0.0
        self._sum_a = 0.0
        self._sum_abs_b = 0.0
        self._sum_b = 0.0
        self._sum_a2 = 0.0
        self._sum_local2 = 0.0
        self._sum_feedback2 = 0.0
        self._max_recurrence_relative = 0.0

    def add(self, step_id: str, uc_sc: TensorTree, ur_sc: TensorTree,
            uc_sr: TensorTree, ur_sr: TensorTree,
            drift_before: TensorTree, drift_after: TensorTree,
            *, endpoint_repair_nonzero: bool = True) -> None:
        if len(self._records) >= self.evaluation_steps:
            raise RuntimeError("evaluation lane is already complete")
        ucc, urc, ucr, urr = (_as_tree(uc_sc), _as_tree(ur_sc), _as_tree(uc_sr), _as_tree(ur_sr))
        before, after = _as_tree(drift_before), _as_tree(drift_after)
        local = _tree_avg(_tree_sub(ucc, urc), _tree_sub(ucr, urr))
        feedback = _tree_avg(_tree_sub(ucc, ucr), _tree_sub(urc, urr))
        expected = _tree_add(_tree_add(before, local), feedback)
        recurrence = _tree_sub(after, expected)
        local_l2 = _tree_norm(local)
        feedback_l2 = _tree_norm(feedback)
        recurrence_l2 = _tree_norm(recurrence)
        denom = max(_tree_norm(after), _tree_norm(before), local_l2, 1e-30)
        relative = recurrence_l2 / denom
        self._max_recurrence_relative = max(self._max_recurrence_relative, relative)
        row: Dict[str, Any] = {
            "step_id": str(step_id),
            "local_l2": local_l2,
            "feedback_l2": feedback_l2,
            "recurrence_residual_l2": recurrence_l2,
            "recurrence_relative": relative,
            "endpoint_repair_nonzero": bool(endpoint_repair_nonzero),
        }
        if self.carrier.basis is not None:
            basis = self.carrier.basis
            a = _tree_dot(local, basis)
            b = _tree_dot(feedback, basis)
            row.update({"local_projection": a, "feedback_projection": b})
            self._sum_a += a; self._sum_abs_a += abs(a); self._sum_a2 += a * a
            self._sum_b += b; self._sum_abs_b += abs(b)
        self._sum_local2 += local_l2 * local_l2
        self._sum_feedback2 += feedback_l2 * feedback_l2
        self._local_sum = local if self._local_sum is None else _tree_add(self._local_sum, local)
        self._feedback_sum = feedback if self._feedback_sum is None else _tree_add(self._feedback_sum, feedback)
        self._last_drift = _clone_float(after)
        self._records.append(row)

    def finalize(self) -> Dict[str, Any]:
        if len(self._records) != self.evaluation_steps:
            raise RuntimeError("evaluation lane is incomplete")
        payload: Dict[str, Any] = {
            "schema": "kernel-analyzer-seup-symmetric-v2",
            "status": "MEASURED" if self.carrier.stable else "NO_STABLE_CARRIER",
            "carrier": self.carrier.certificate,
            "evaluation_steps": self.evaluation_steps,
            "max_recurrence_relative_residual": self._max_recurrence_relative,
            "local_accumulation_l2": _tree_norm(self._local_sum or {}),
            "feedback_accumulation_l2": _tree_norm(self._feedback_sum or {}),
            "local_energy": self._sum_local2,
            "feedback_energy": self._sum_feedback2,
            "evaluation_records": self._records,
        }
        if self.carrier.basis is not None:
            epsilon = 1e-30
            final_projection = _tree_dot(self._last_drift or {}, self.carrier.basis)
            final_drift_l2 = _tree_norm(self._last_drift or {})
            payload.update({
                "signed_persistence": abs(self._sum_a) / max(self._sum_abs_a, epsilon),
                "local_projection_accumulation_abs": abs(self._sum_a),
                "feedback_projection_accumulation_abs": abs(self._sum_b),
                "local_fraction_of_projected_accumulation": abs(self._sum_a) / max(abs(self._sum_a) + abs(self._sum_b), epsilon),
                "carrier_energy_capture": self._sum_a2 / max(self._sum_local2, epsilon),
                "final_carrier_projection": final_projection,
                "final_drift_l2": final_drift_l2,
                "final_carrier_fraction": final_projection * final_projection / max(final_drift_l2 * final_drift_l2, epsilon),
                "local_and_final_carrier_same_sign": bool(self._sum_a == 0.0 or final_projection == 0.0 or self._sum_a * final_projection > 0.0),
            })
        return payload


def _tree_add(left: Mapping[str, Any], right: Mapping[str, Any]) -> Dict[str, Any]:
    _validate_structure(left, right)
    return {key: left[key].detach().float() + right[key].detach().float() for key in left}


def sgd_effective_update_delta(candidate_gradient: TensorTree, repair_gradient: TensorTree,
                               *, learning_rate: float, weight_decay: float = 0.0) -> Dict[str, Any]:
    """Candidate-minus-repair SGD update at one identical pre-step state."""
    if not math.isfinite(learning_rate) or learning_rate <= 0.0:
        raise ValueError("learning_rate must be finite and positive")
    if not math.isfinite(weight_decay):
        raise ValueError("weight_decay must be finite")
    candidate, repair = _as_tree(candidate_gradient), _as_tree(repair_gradient)
    _validate_same(candidate, repair)
    return {key: (candidate[key].detach().float() - repair[key].detach().float()).mul(-learning_rate)
            for key in candidate}


def _adam_update(gradient: Mapping[str, Any], first: Mapping[str, Any], second: Mapping[str, Any],
                 parameter: Mapping[str, Any], *, step: int, learning_rate: float,
                 betas: Tuple[float, float], epsilon: float, weight_decay: float) -> Dict[str, Any]:
    _validate_same(gradient, first); _validate_same(gradient, second); _validate_same(gradient, parameter)
    if step < 1:
        raise ValueError("AdamW step must be positive")
    b1, b2 = betas
    if not (0.0 <= b1 < 1.0 and 0.0 <= b2 < 1.0):
        raise ValueError("AdamW betas must lie in [0, 1)")
    result = {}
    for key in gradient:
        grad = gradient[key].detach().float()
        m = first[key].detach().float().mul(b1).add(grad, alpha=1.0 - b1)
        v = second[key].detach().float().mul(b2).addcmul(grad, grad, value=1.0 - b2)
        update = (m / (1.0 - b1 ** step)) / ((v / (1.0 - b2 ** step)).sqrt().add(epsilon))
        if weight_decay:
            update = update.add(parameter[key].detach().float(), alpha=weight_decay)
        result[key] = update.mul(-learning_rate)
    return result


def adamw_update(gradient: TensorTree, first_moment: TensorTree, second_moment: TensorTree,
                 parameter: TensorTree, *, step: int, learning_rate: float,
                 betas: Tuple[float, float] = (0.9, 0.95), epsilon: float = 1e-8,
                 weight_decay: float = 0.0) -> Dict[str, Any]:
    return _adam_update(_as_tree(gradient), _as_tree(first_moment), _as_tree(second_moment),
                        _as_tree(parameter), step=step, learning_rate=learning_rate,
                        betas=betas, epsilon=epsilon, weight_decay=weight_decay)


def adamw_effective_update_delta(candidate_gradient: TensorTree, repair_gradient: TensorTree,
                                 first_moment: TensorTree, second_moment: TensorTree,
                                 parameter: TensorTree, *, step: int, learning_rate: float,
                                 betas: Tuple[float, float] = (0.9, 0.95), epsilon: float = 1e-8,
                                 weight_decay: float = 0.0) -> Dict[str, Any]:
    """Candidate-minus-repair AdamW update from the same pre-step moments."""
    candidate, repair = _as_tree(candidate_gradient), _as_tree(repair_gradient)
    first, second, parameters = _as_tree(first_moment), _as_tree(second_moment), _as_tree(parameter)
    left = _adam_update(candidate, first, second, parameters, step=step, learning_rate=learning_rate,
                        betas=betas, epsilon=epsilon, weight_decay=weight_decay)
    right = _adam_update(repair, first, second, parameters, step=step, learning_rate=learning_rate,
                         betas=betas, epsilon=epsilon, weight_decay=weight_decay)
    return {key: left[key] - right[key] for key in left}


def force_carrier_sign(delta_update: TensorTree, basis: TensorTree, sign: int) -> Dict[str, Any]:
    """Set, rather than multiply, the carrier coefficient to ``sign*abs(a)``."""
    if sign not in (-1, 1):
        raise ValueError("carrier sign must be -1 or +1")
    delta, direction = _as_tree(delta_update), _as_tree(basis)
    _validate_structure(delta, direction)
    norm = _tree_norm(direction)
    if abs(norm - 1.0) > 1e-5:
        raise ValueError("carrier basis must have unit norm")
    projection = _tree_dot(delta, direction)
    result = _clone_float(delta)
    _tree_add_scaled_(result, direction, float(sign) * abs(projection) - projection)
    if not math.isclose(_tree_norm(delta), _tree_norm(result), rel_tol=2e-6, abs_tol=1e-8):
        raise RuntimeError("carrier sign operation failed to preserve update norm")
    return result


def flip_carrier_component(delta_update: TensorTree, basis: TensorTree, sign: int) -> Dict[str, Any]:
    """Compatibility alias for :func:`force_carrier_sign`."""
    return force_carrier_sign(delta_update, basis, sign)


def alternating_sign_schedule(length: int, first: int = 1) -> Tuple[int, ...]:
    if length < 1 or first not in (-1, 1):
        raise ValueError("invalid alternating sign schedule")
    return tuple(first if index % 2 == 0 else -first for index in range(length))


def balanced_sign_schedule(length: int, seed: int = 3407) -> Tuple[int, ...]:
    if length < 2 or length % 2:
        raise ValueError("balanced sign schedule requires a positive even length")
    values = [1] * (length // 2) + [-1] * (length // 2)
    random.Random(seed).shuffle(values)
    return tuple(values)
