"""Deterministic semantic-rule registry; no name or LLM fallback is allowed."""

from __future__ import annotations

from dataclasses import dataclass, field
import gzip
import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional, Sequence


Witness = Callable[[], Mapping[str, Any]]


def _digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


@dataclass(frozen=True)
class SemanticRule:
    overload: str
    rule_id: str
    forward_map: str
    vjp_map: str
    finite_arithmetic: str
    error_relation: str
    assumptions: Sequence[str]
    required_non_tensor_arguments: Sequence[str] = ()
    applicability: Mapping[str, Any] = field(default_factory=dict)
    witness: Optional[Witness] = None

    def __post_init__(self) -> None:
        for name in ("overload", "rule_id", "forward_map", "vjp_map",
                     "finite_arithmetic", "error_relation"):
            if not str(getattr(self, name)).strip():
                raise ValueError("%s must be non-empty" % name)
        if not self.assumptions:
            raise ValueError("semantic rule assumptions must be non-empty")

    def as_dict(self, run_witness: bool = False) -> Dict[str, Any]:
        payload = {
            "overload": self.overload,
            "rule_id": self.rule_id,
            "forward_map": self.forward_map,
            "vjp_map": self.vjp_map,
            "finite_arithmetic": self.finite_arithmetic,
            "error_relation": self.error_relation,
            "assumptions": list(self.assumptions),
            "required_non_tensor_arguments": list(self.required_non_tensor_arguments),
            "applicability": dict(self.applicability),
        }
        payload["witness"] = (
            dict(self.witness()) if run_witness and self.witness is not None
            else {"status": "NOT_RUN" if self.witness is not None else "UNAVAILABLE"}
        )
        payload["rule_sha256"] = _digest(payload)
        return payload


class SemanticRegistry:
    def __init__(self) -> None:
        self._rules: Dict[str, SemanticRule] = {}

    def register(self, rule: SemanticRule) -> None:
        if rule.overload in self._rules:
            raise ValueError("duplicate exact overload rule: %s" % rule.overload)
        self._rules[rule.overload] = rule

    def resolve(self, overload: str) -> Optional[SemanticRule]:
        """Exact dispatcher-overload lookup only."""
        return self._rules.get(overload)

    def instantiate(
        self, overload: str, argument_bindings: Sequence[Mapping[str, Any]],
        concrete_program_proof: Optional[Mapping[str, Any]] = None,
    ) -> Mapping[str, Any]:
        rule = self.resolve(overload)
        if rule is None:
            return {
                "status": "UNRESOLVED_MISSING_EXACT_SEMANTIC_RULE",
                "overload": overload,
                "argument_bindings": list(argument_bindings),
                "name_or_family_fallback_used": False,
            }
        witness = rule.witness() if rule.witness is not None else None
        witness_pass = (
            witness is not None and str(witness.get("status", "")).startswith("PASS_")
        )
        by_name = {str(row.get("name")): row for row in argument_bindings}
        required_present = all(
            name in by_name
            and by_name[name].get("source") != "MISSING_REQUIRED_ARGUMENT"
            and by_name[name].get("value_type") != "UNSUPPORTED"
            for name in rule.required_non_tensor_arguments
        )
        try:
            json.dumps(list(argument_bindings), sort_keys=True)
            bindings_serializable = True
        except (TypeError, ValueError):
            bindings_serializable = False
        applicable = required_present and bindings_serializable
        if concrete_program_proof is not None and hasattr(concrete_program_proof, "as_dict"):
            proof = dict(concrete_program_proof.as_dict())
        else:
            proof = dict(concrete_program_proof or {})
        required_proof_checks = (
            "saved_tensor_origins_exact",
            "cotangent_edge_exact",
            "backward_program_matches_analytic_vjp",
            "non_tensor_arguments_exact",
            "output_edges_exact",
        )
        concrete_proof_complete = bool(proof) and all(
            proof.get(name) is True for name in required_proof_checks
        ) and all(bool(proof.get(name)) for name in (
            "forward_program_sha256", "backward_program_sha256",
            "analytic_derivation_sha256",
        ))
        formula_bound = witness_pass and applicable
        return {
            "status": (
                "INSTANTIATED_CERTIFIED_EXACT_SEMANTIC_RULE"
                if formula_bound and concrete_proof_complete
                else "UNRESOLVED_RULE_APPLICABILITY"
                if not applicable else
                "FORMULA_REGISTERED_EXECUTABLE_WITNESS_ONLY"
                if formula_bound else
                "BOUND_UNCERTIFIED_EXACT_SEMANTIC_RULE"
            ),
            "overload": overload,
            "rule_id": rule.rule_id,
            "rule": rule.as_dict(),
            "executable_witness": dict(witness) if witness is not None else None,
            "argument_bindings": list(argument_bindings),
            "applicability_checks": {
                "required_non_tensor_arguments_present": required_present,
                "argument_bindings_serializable": bindings_serializable,
                "checked_on_current_invocation": True,
            },
            "analytic_proof_status": (
                "ANALYTICALLY_PROVED"
                if concrete_proof_complete and formula_bound else
                "UNRESOLVED_NO_CONCRETE_BACKWARD_PROGRAM_PROOF"
            ),
            "concrete_program_proof": proof or None,
            "required_concrete_proof_checks": list(required_proof_checks),
            "name_or_family_fallback_used": False,
        }

    def load_catalog(self, path: Path, witness_path: Path = None) -> None:
        """Load the existing deterministic template schema as exact rules."""
        if path.suffix == ".gz":
            with gzip.open(path, "rt", encoding="utf-8") as handle:
                data = json.load(handle)
        else:
            data = json.loads(path.read_text())
        witnesses = {}
        if witness_path is not None:
            witness_data = json.loads(witness_path.read_text())
            witnesses = {str(row["overload"]): row for row in witness_data["witness_rows"]}
        templates = data.get("templates", data.get("mathematical_templates", []))
        for row in templates:
            derivation = row.get("derivation", row)
            witness_row = witnesses.get(str(row["overload"]))
            self.register(SemanticRule(
                overload=str(row["overload"]),
                rule_id=str(row.get("template_id", row.get("rule_id"))),
                forward_map=str(derivation.get("exact_forward_map", "")),
                vjp_map=str(derivation.get("exact_vjp_map", "")),
                finite_arithmetic=str(derivation.get("finite_arithmetic_realization", "")),
                error_relation=(
                    str(derivation.get("forward_error_relation", "")) + "\n" +
                    str(derivation.get("vjp_error_relation", ""))
                ).strip(),
                assumptions=tuple(derivation.get("assumptions", ("catalog-stated assumptions",))),
                required_non_tensor_arguments=tuple(
                    derivation.get("required_non_tensor_arguments", ())
                ),
                applicability={"catalog_source": str(path)},
                witness=(lambda value=dict(witness_row): value) if witness_row else None,
            ))

    def as_dict(self) -> Mapping[str, Any]:
        rules = [self._rules[key].as_dict() for key in sorted(self._rules)]
        return {
            "schema": "kernel-analyzer-semantic-registry-v1",
            "lookup": "EXACT_DISPATCHER_OVERLOAD_ONLY",
            "rule_count": len(rules),
            "rules": rules,
            "registry_sha256": _digest(rules),
        }
