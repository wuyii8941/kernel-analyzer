"""Structure of a contract (schema 1.1 as reviewed on 2026-10-15; 1.2 adds author, covers, readings and changes).

Structure only: a contract that passes here is well formed, not correct. Review status, implementation status and
runtime activation are separate (CONTRACT_PROTOCOL section 2) and never follow from this check.
"""
from __future__ import annotations

from . import expr
from .num import GUARANTEES, OUTCOMES

REVIEW_STATUSES = {"reviewed_for_implementation", "drafted"}
REQUIRED = ("id", "role", "title", "review", "activation", "profiles", "outputs", "signature", "domain", "clauses",
            "examples", "mutation_requirements")
EXPECTED_OUTCOMES = set(OUTCOMES) - {"implementation_error"}


def profile_outputs(contract: dict, profile: str) -> dict:
    """The complete output interface fixed by the contract, never by a query.

    Without an explicit list a profile promises every contract output. A list
    selects a nonempty subset with the original guarantees unchanged.
    """
    if not isinstance(profile, str) or profile not in contract['profiles']:
        raise ValueError('profile outputs: unknown profile')
    declared = contract['profiles'][profile]
    outputs = contract['outputs']
    if 'outputs' not in declared:
        return outputs
    names = declared['outputs']
    if (not isinstance(names, list) or not names
            or any(not isinstance(n, str) or n not in outputs for n in names)
            or len(set(names)) != len(names)):
        raise ValueError(f'profile outputs {profile}: expected nonempty distinct declared output names')
    return {name: outputs[name] for name in names}


def known_bases(contract: dict, sources: dict) -> set:
    return ({f"{sid}.{item}" for sid, s in sources.items() for item in s.get("items", {})}
            | {d.get("id") for d in contract.get("decisions", [])})


def decision_review_errors(contract: dict) -> list[str]:
    """A contract-level review must not silently review its draft decisions."""
    errors = []
    reviewed = contract.get("review", {}).get("status") == "reviewed_for_implementation"
    for decision in contract.get("decisions", []):
        status = decision.get("status")
        if status not in ("reviewed", "needs_binding", "drafted"):
            errors.append(f"{decision.get('id')}: invalid decision status {status!r}")
        elif reviewed and status == "drafted":
            errors.append(f"{decision.get('id')}: drafted decision in a reviewed contract")
    return errors


def profile_decision_errors(contract: dict) -> list[str]:
    """Connect task-selected decisions to the profiles that use them."""
    decisions = {d.get("id"): d for d in contract.get("decisions", [])
                 if isinstance(d.get("id"), str)}
    required = {name for name, d in decisions.items() if d.get("status") == "needs_binding"}
    covered, errors = set(), []
    for name, profile in contract.get("profiles", {}).items():
        refs = profile.get("basis", [])
        pending = list(refs) if isinstance(refs, list) else []
        seen = set()
        while pending:
            ref = pending.pop()
            if not isinstance(ref, str) or ref not in decisions or ref in seen:
                continue
            seen.add(ref)
            bases = decisions[ref].get("basis", [])
            if isinstance(bases, list):
                pending.extend(bases)
        choices = seen & required
        covered.update(choices)
        if choices and profile.get("task_choice_required") is not True:
            errors.append(f"profile {name}: decisions {', '.join(sorted(choices))} require task_choice_required=true")
    for name in sorted(required - covered):
        errors.append(f"{name}: needs_binding decision has no explicit profile basis")
    return errors


def decision_basis_errors(contract: dict) -> list[str]:
    """Decision references must terminate in source clauses, not a circular justification.

    Source IDs are resolved separately against SOURCES; this checks the part of
    the chain formed by locally declared decisions.
    """
    decisions, errors, done, active = {}, [], set(), []
    for decision in contract.get("decisions", []):
        name = decision.get("id")
        if not isinstance(name, str) or not name:
            errors.append("decision without a valid ID")
            continue
        if name in decisions:
            errors.append(f"duplicate decision ID: {name}")
        decisions[name] = decision.get("basis", [])

    def visit(name):
        if name in active:
            errors.append("circular basis: " + " -> ".join(active[active.index(name):] + [name]))
            return
        if name in done:
            return
        active.append(name)
        refs = decisions[name]
        if not isinstance(refs, list) or not refs:
            errors.append(f"{name}: no source basis (empty or malformed decision basis)")
        else:
            for ref in refs:
                if isinstance(ref, str) and ref in decisions:
                    visit(ref)
        active.pop()
        done.add(name)

    for name in decisions:
        visit(name)
    return errors


def _check_expected(value, prefix, errors, boolean=False):
    """boolean: True (only booleans), False (no booleans) or None (kind 'discrete': booleans and integers both allowed)."""
    if value is None:
        # Explicitly absent positions are allowed inside structured outputs.
        # The reference and contract-specific complete-output checker must agree.
        return
    if isinstance(value, dict) and "set" in value:
        for v in value["set"]:
            _check_expected(v, prefix, errors, boolean)
        return
    if isinstance(value, dict):
        for v in value.values():
            _check_expected(v, prefix, errors, boolean)
        return
    if isinstance(value, list):
        for v in value:
            _check_expected(v, prefix, errors, boolean)
        return
    if boolean is not None and isinstance(value, bool) != boolean:
        errors.append(f"{prefix}: {'non-boolean' if boolean else 'boolean'} value {value!r} for a "
                      f"{'boolean' if boolean else 'numeric'} output")
        return
    try:
        expr.parse(value)
    except expr.ExpressionError as err:
        errors.append(f"{prefix}: {err}")


def _check_block(block, outputs, prefix, errors):
    if not isinstance(block, dict) or block.get("outcome") not in EXPECTED_OUTCOMES:
        errors.append(f"{prefix}: invalid or missing outcome")
        return
    if block["outcome"] == "value":
        fields = block.get("fields", {})
        if set(fields) != set(outputs):
            errors.append(f"{prefix}: output keys {sorted(fields)} do not match the declared outputs {sorted(outputs)}")
        for k, v in fields.items():
            kind = (outputs.get(k) or {}).get("kind")
            _check_expected(v, f"{prefix}.{k}", errors, None if kind == "discrete" else kind == "boolean")
    elif "fields" in block:
        errors.append(f"{prefix}: a non-value outcome must not carry value fields")


def validate(c: dict, sources: dict) -> list[str]:
    errors: list[str] = []

    def need(ok, msg):
        if not ok:
            errors.append(msg)

    need(c.get("schema_version") in ("1.1", "1.2"), "schema_version must be 1.1 or 1.2")
    for f in REQUIRED:
        need(bool(c.get(f)), f"missing or empty {f}")
    if errors:
        return errors
    status = c["review"].get("status")
    need(status in REVIEW_STATUSES, f"review status {status!r} not in {sorted(REVIEW_STATUSES)}")
    if status == "drafted":
        need(bool(c["review"].get("author")), "a drafted contract names its author")
    need(c["activation"].get("implementation_status") in {"pending", "validated"}, "invalid implementation_status")
    need(c["activation"].get("runtime_binding_status") in {"pending", "validated"}, "invalid runtime_binding_status")
    for name, info in c["outputs"].items():
        need(isinstance(info, dict) and info.get("guarantee") in GUARANTEES, f"output {name}: invalid guarantee")
        need(isinstance(info, dict) and info.get("kind") in (None, "boolean", "discrete"), f"output {name}: invalid kind")
    ids = [cl.get("id") for cl in c["clauses"]]
    need(len(ids) == len(set(ids)), "duplicate clause IDs")
    dids = [d.get("id") for d in c.get("decisions", [])]
    need(len(dids) == len(set(dids)), "duplicate decision IDs")
    errors.extend(decision_basis_errors(c))
    known = known_bases(c, sources)
    for node in [*c["clauses"], *c.get("decisions", [])]:
        refs = node.get("basis", [])
        need(isinstance(refs, list) and refs, f"{node.get('id')}: empty or non-list basis")
        for ref in refs if isinstance(refs, list) else []:
            need(ref in known, f"{node.get('id')}: unresolved basis {ref}")
    errors.extend(decision_review_errors(c))
    errors.extend(profile_decision_errors(c))
    for cl in c["clauses"]:
        need(bool(cl.get("observes")), f"{cl['id']}: no observed field")
        need(set(cl.get("observes", [])) <= set(c["outputs"]), f"{cl['id']}: unknown observed field")
    readings = c.get("readings", {}) or {}
    interfaces = {}
    for pid, profile in c["profiles"].items():
        try:
            interfaces[pid] = profile_outputs(c, pid)
        except ValueError as error:
            errors.append(str(error))
        need(type(profile.get("task_choice_required")) is bool,
             f"profile {pid}: task_choice_required must be an explicit boolean")
        refs = profile.get("basis", [])
        need(isinstance(refs, list), f"profile {pid}: basis must be a list")
        for ref in refs if isinstance(refs, list) else []:
            need(isinstance(ref, str) and ref in known, f"profile {pid}: unresolved basis {ref}")
    for rid, r in readings.items():
        need(isinstance(r, dict) and r.get("meaning"), f"reading {rid}: no meaning")
        for ref in (r.get("basis") or []):
            need(ref in known, f"reading {rid}: unresolved basis {ref}")
    eids = [e.get("id") for e in c["examples"]]
    need(len(eids) == len(set(eids)), "duplicate example IDs")
    covered = set()
    for e in c["examples"]:
        p = e.get("id", "example")
        need(e.get("profile") in c["profiles"], f"{p}: unknown profile")
        need(bool(e.get("clauses")) and set(e.get("clauses", [])) <= set(ids), f"{p}: missing or unknown clause")
        covered.update(e.get("clauses", []))
        outputs = interfaces.get(e.get('profile'), c['outputs'])
        _check_block(e.get("expected"), outputs, p, errors)
        for rid, block in (e.get("readings") or {}).items():
            need(rid in readings, f"{p}: unknown reading {rid}")
            _check_block(block, outputs, f"{p}[{rid}]", errors)
    need(set(ids) <= covered, f"clauses without any example: {sorted(set(ids) - covered)}")
    mids = [m.get("id") for m in c["mutation_requirements"]]
    need(len(mids) == len(set(mids)), "duplicate mutation IDs")
    for m in c["mutation_requirements"]:
        ws = m.get("witnesses", [])
        need(bool(ws) and set(ws) <= set(eids), f"{m.get('id')}: missing or invalid witness")
    return errors
