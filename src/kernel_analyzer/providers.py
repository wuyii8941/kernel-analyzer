"""Reference providers and deterministic adapters for retained evidence."""

from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

from .api import AnalysisSpec, ReferenceAnalysis, ReferenceProvider
from .semantics import SemanticRegistry


class LedgerReferenceProvider(ReferenceProvider):
    """Read a retained, execution-derived F+B ledger without changing verdicts."""

    def __init__(
        self, ledger: Path, model_key: str, template_catalog: Path = None,
        case_audit: Path = None, additional_case_targets: Sequence[Mapping[str, Any]] = (),
    ) -> None:
        self.ledger = ledger
        self.model_key = model_key
        self.template_catalog = template_catalog
        self.case_audit = case_audit
        self.additional_case_targets = list(additional_case_targets)

    @staticmethod
    def _read(path: Path) -> Mapping[str, Any]:
        if path.suffix == ".gz":
            with gzip.open(path, "rt", encoding="utf-8") as handle:
                return json.load(handle)
        return json.loads(path.read_text())

    def analyze(self, spec: AnalysisSpec, run_dir: Path) -> ReferenceAnalysis:
        data = self._read(self.ledger)
        units = [
            row for row in data["units"]
            if row["model"] == self.model_key
            and row["denominator_role"] == "PRIMARY_FB_PROOF"
        ]
        templates = []
        if self.template_catalog is not None:
            catalog = self._read(self.template_catalog)
            templates = catalog.get("templates", catalog.get("mathematical_templates", []))
        targets = []
        if self.case_audit is not None:
            audit = self._read(self.case_audit)
            targets = [{
                "unit_id": "retained-case::" + row["case"],
                "case_name": row["case"],
                "closed_semantic_target": True,
                "source_classification": row.get(
                    "classification", row["flash_style"]["verdict"]
                ),
            } for row in audit["rows"]]
        targets.extend(dict(row) for row in self.additional_case_targets)
        unresolved = [
            {"unit_id": row["unit_id"], "reason": "MATH_UNRESOLVED"}
            for row in units
            if row["mathematics"]["status"] not in {
                "ANALYTICALLY_PROVED",
                # Read compatibility for retained artifacts generated before
                # the strict proof-level split. New artifacts never emit it.
                "MATH_CLOSED",
            }
        ]
        return ReferenceAnalysis(
            subject=spec.subject,
            proof_units=units,
            census={
                "source": str(self.ledger),
                "model": self.model_key,
                "primary_fb_proof_units": len(units),
                "execution_derived": True,
            },
            templates=templates,
            unresolved=unresolved,
            case_targets=targets,
        )


class EagerReferenceProvider(ReferenceProvider):
    """Capture one real step and bind exact overload rules automatically.

    This provider intentionally fails closed when exact forward/backward origin
    metadata or an exact semantic rule is absent.
    """

    def __init__(self, registry: SemanticRegistry) -> None:
        self.registry = registry

    def analyze(self, spec: AnalysisSpec, run_dir: Path) -> ReferenceAnalysis:
        if spec.model_factory is None or spec.step_builder is None:
            raise ValueError("eager capture requires model_factory and step_builder")
        if not spec.states:
            raise ValueError("eager capture requires at least one state")
        from scripts.op_inventory import observe_full_forward_backward_step

        model = spec.model_factory()
        step = spec.step_builder(model, spec.states[0])
        trace = observe_full_forward_backward_step(
            loss_closure=step.loss_closure,
            endpoint_closure=step.endpoint_closure,
            model=model,
            capture_autograd_sequence_numbers=True,
            retain_forward_outputs_for_origin_binding=True,
        )
        events = [event.as_dict() for event in trace.events]
        parent = list(range(len(events)))

        def find(index: int) -> int:
            while parent[index] != index:
                parent[index] = parent[parent[index]]
                index = parent[index]
            return index

        def union(left: int, right: int) -> None:
            a, b = find(left), find(right)
            if a != b:
                parent[max(a, b)] = min(a, b)

        forward_by_seq: Dict[int, Sequence[int]] = {}
        mutable_forward: Dict[int, list] = {}
        for index, event in enumerate(events):
            if str(event["phase"]).endswith("FORWARD"):
                for seq in event.get("forward_output_autograd_sequence_nrs", ()):
                    if seq is not None:
                        mutable_forward.setdefault(int(seq), []).append(index)
        forward_by_seq = mutable_forward
        dangling = []
        for index, event in enumerate(events):
            if not str(event["phase"]).endswith("BACKWARD"):
                continue
            seq = event.get("backward_autograd_sequence_nr")
            if seq is None:
                continue
            origins = forward_by_seq.get(int(seq), ())
            if not origins:
                dangling.append(index)
            for origin in origins:
                union(index, origin)
            for origin in origins[1:]:
                union(origins[0], origin)

        groups: Dict[int, list] = {}
        for index in range(len(events)):
            groups.setdefault(find(index), []).append(index)
        proof_units = []
        unresolved = []
        auxiliary_components = []
        for members in groups.values():
            forward = [i for i in members if str(events[i]["phase"]).endswith("FORWARD")]
            if not forward:
                auxiliary_components.append(members)
                continue
            bindings = []
            for index in members:
                event = events[index]
                arguments = event.get("argument_bindings", ())
                bindings.append(
                    self.registry.instantiate(
                        str(event["overload"]), arguments,
                        concrete_program_proof=event.get("concrete_program_proof"),
                    )
                )
            unit_id = "fb::" + __import__("hashlib").sha256(
                json.dumps(members).encode()).hexdigest()[:20]
            origin_statuses = [str(events[index].get("sequence_binding_status", ""))
                               for index in members]
            origin_exact = all(value.startswith("EXACT_") for value in origin_statuses)
            row = {
                "unit_id": unit_id,
                "member_ordinals": members,
                "forward_ordinals": forward,
                "backward_ordinals": [i for i in members if i not in forward],
                "semantic_bindings": bindings,
                "actual_backward_origin_binding": {
                    "status": "EXACT" if origin_exact else "UNRESOLVED",
                    "member_statuses": origin_statuses,
                    "name_shape_or_family_pairing_used": False,
                },
                "status": (
                    "ANALYTICALLY_PROVED" if origin_exact and all(
                        item.get("analytic_proof_status") == "ANALYTICALLY_PROVED"
                        for item in bindings
                    ) else
                    "ORIGIN_BOUND_FORMULA_REGISTERED_ONLY" if origin_exact and all(
                        item["status"] in {
                            "FORMULA_REGISTERED_EXECUTABLE_WITNESS_ONLY",
                            "INSTANTIATED_CERTIFIED_EXACT_SEMANTIC_RULE",
                        }
                        for item in bindings
                    ) else "UNRESOLVED"
                ),
            }
            proof_units.append(row)
            if row["status"] != "ANALYTICALLY_PROVED":
                unresolved.append({
                    "unit_id": unit_id,
                    "reason": (
                        "CONCRETE_SAVED_TENSOR_COTANGENT_BACKWARD_PROOF_MISSING"
                        if row["status"] == "ORIGIN_BOUND_FORMULA_REGISTERED_ONLY"
                        else "MISSING_RULE_OR_ORIGIN"
                    ),
                })
        unresolved.extend({"ordinal": index, "reason": "DANGLING_BACKWARD_ORIGIN"}
                          for index in dangling)
        return ReferenceAnalysis(
            subject=spec.subject,
            proof_units=proof_units,
            census={
                "execution_invocations": len(events),
                "primary_fb_proof_units": len(proof_units),
                "auxiliary_backward_components": len(auxiliary_components),
                "auxiliary_backward_invocations": sum(map(len, auxiliary_components)),
                "execution_derived": True,
            },
            templates=self.registry.as_dict()["rules"],
            unresolved=unresolved,
        )
