"""The single runtime entry to the contracts.

Only `contracts/*.yaml` (top level) are loaded; `contracts/draft/` and `drafts/` are history and are never imported.
Each contract binds to `adapters/<ID>.py`. A query is answered only when the contract may be used for it:

- review: `reviewed_for_implementation`, or `drafted` when the caller passes allow_draft=True (the result is then
  labelled as coming from a draft contract);
- implementation: the contract explicitly records implementation_status=validated, and the exact contract and adapter
  files retained by the independent implementation review equal the current files. Development checks never create
  these reviewed copies and never activate a reference;
- sources: every source a clause, decision or diagnostic reading rests on resolves to a retained source;
- task choice: a profile with task_choice_required needs an explicit nonblank textual statement from the caller
  (this choice alone is not evidence of a kernel's input/output/state binding);
- profile: must be declared in the contract.

None of these is upgraded automatically from another (CONTRACT_PROTOCOL sections 2 and 7).
"""
from __future__ import annotations

import importlib.util
import json
from dataclasses import dataclass
from pathlib import Path

from .num import Result, NotEstablished, evaluate
from .contracts_io import load_contract
from .schema import known_bases, decision_basis_errors, decision_review_errors, profile_decision_errors, profile_outputs
from .compare import validate_reference_values

ROOT = Path(__file__).resolve().parents[1]
CONTRACTS = ROOT / "contracts"
ADAPTERS = ROOT / "adapters"
VALIDATED = ROOT / "validated"
SOURCES = ROOT / "SOURCES.json"


def load_sources() -> dict:
    return json.loads(SOURCES.read_text())


def contract_paths() -> list[Path]:
    return sorted(CONTRACTS.glob("*.yaml"))


def frozen_source(entry: dict) -> bool:
    """The cited source is available locally, or is a standard cited by edition.

    A metadata entry alone does not establish that a source was retained. This
    availability check is not a review of the source's meaning.
    """
    snapshot = entry.get("snapshot")
    if snapshot:
        path = (ROOT / snapshot).resolve()
        return ROOT.resolve() in path.parents and path.is_file()
    return entry.get("kind") == "standard" and bool(entry.get("edition"))


@dataclass
class Contract:
    id: str
    data: dict
    path: Path
    adapter_path: Path
    adapter: object

def import_adapter(cid: str, adapters: Path = ADAPTERS):
    path = adapters / f"{cid}.py"
    if not path.is_file():
        raise FileNotFoundError(f"no adapter {path}")
    spec = importlib.util.spec_from_file_location(f"kernel_spec_adapter_{cid}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return path, mod


class ContractNotFound(FileNotFoundError):
    """No current contract was selected; distinct from a missing adapter/dependency."""


def load(cid: str, contracts: Path = CONTRACTS, adapters: Path = ADAPTERS) -> Contract:
    path = contracts / f"{cid}.yaml"
    if path.parent != contracts or not path.is_file():
        raise ContractNotFound(f"no contract {cid} in {contracts}")
    data = load_contract(path)
    if not isinstance(data, dict):
        raise ValueError(f"{path}: contract must be a mapping")
    if data.get("id") != cid:
        raise ValueError(f"{path}: id {data.get('id')!r} does not match the file name")
    apath, mod = import_adapter(cid, adapters)
    return Contract(cid, data, path, apath, mod)


def _validated_file_reasons(c: Contract, root: Path = None) -> list[str]:
    """Bind implementation validation to retained, human-reviewable files.

    The reviewed pair is deliberately stored as the original files rather than
    as digests. Nothing in the development checker writes this directory.
    """
    root = VALIDATED if root is None else Path(root)
    contract = root / "contracts" / f"{c.id}.yaml"
    adapter = root / "adapters" / f"{c.id}.py"
    reasons = []
    for kind, reviewed, current in (("contract", contract, c.path), ("adapter", adapter, c.adapter_path)):
        if not reviewed.is_file():
            reasons.append(f"{kind} implementation review file is not retained")
            continue
        try:
            same = reviewed.read_bytes() == current.read_bytes()
        except OSError as error:
            reasons.append(f"cannot compare reviewed {kind} file: {error}")
            continue
        if not same:
            reasons.append(f"current {kind} differs from the independently reviewed file")
    return reasons


def activation(c: Contract, profile: str, task_bindings=None, allow_draft=False):
    """(usable, reasons, labels) for one query."""
    reasons, labels = [], {"contract": c.id, "profile": profile,
                           "review_status": c.data["review"]["status"]}
    implementation_status = c.data.get("activation", {}).get("implementation_status")
    labels["implementation_status"] = implementation_status
    labels["runtime_binding_status"] = c.data.get("activation", {}).get("runtime_binding_status")
    # A reference answer and the caller's task declaration are not a kernel verdict.
    labels["semantic_verdict"] = "not_established"
    labels["task_choice"] = None
    if type(allow_draft) is not bool:
        reasons.append("allow_draft must be an explicit boolean")
    if task_bindings is None:
        task_bindings = {}
    elif not isinstance(task_bindings, dict):
        reasons.append("task bindings must map profile names to nonblank task statements")
        task_bindings = {}
    for name, choice in task_bindings.items():
        if not isinstance(name, str) or name not in c.data["profiles"]:
            reasons.append(f"task binding names an unknown profile: {name!r}")
        if not isinstance(choice, str) or not choice.strip():
            reasons.append(f"task binding for {name!r} must be a nonblank textual statement")
    valid_profile_name = isinstance(profile, str) and bool(profile.strip())
    if not valid_profile_name:
        reasons.append("profile must be a nonblank name")
    choice = task_bindings.get(profile) if valid_profile_name else None
    if isinstance(choice, str) and choice.strip():
        labels["task_choice"] = choice
    if implementation_status != "validated":
        reasons.append(f"implementation status {implementation_status!r}: independent implementation validation is not recorded")
    else:
        reasons.extend(_validated_file_reasons(c))
    status = c.data["review"]["status"]
    if status == "drafted":
        labels["draft_contract"] = True
        if allow_draft is not True:
            reasons.append("contract is drafted, not reviewed (pass allow_draft to use it with a draft label)")
    elif status != "reviewed_for_implementation":
        reasons.append(f"review status {status!r}")
    sources = load_sources()
    reasons.extend(decision_basis_errors(c.data))
    reasons.extend(decision_review_errors(c.data))
    reasons.extend(profile_decision_errors(c.data))
    try:
        profile_outputs(c.data, profile)
    except ValueError as error:
        reasons.append(str(error))
    bases = []
    for node in [*c.data.get("clauses", []), *c.data.get("decisions", []),
                 *c.data.get("profiles", {}).values(), *(c.data.get("readings") or {}).values()]:
        refs = node.get("basis", [])
        if not isinstance(refs, list) or any(not isinstance(b, str) for b in refs):
            reasons.append("invalid basis declaration: expected a list of source or decision IDs")
            continue
        bases.extend(refs)
    unresolved = sorted(set(bases) - known_bases(c.data, sources))
    if unresolved:
        reasons.append(f"unresolved basis: {', '.join(unresolved)}")
    used = {b.split(".")[0] for b in bases}
    unfrozen = sorted(s for s in used if s in sources and not frozen_source(sources[s]))
    if unfrozen:
        reasons.append(f"sources without an archived snapshot: {', '.join(unfrozen)}")
    prof = c.data["profiles"].get(profile) if valid_profile_name else None
    if prof is None:
        reasons.append(f"unknown profile {profile!r}")
    elif type(prof.get("task_choice_required")) is not bool:
        reasons.append(f"profile {profile!r}: task_choice_required must be an explicit boolean")
    elif prof.get("task_choice_required") and labels["task_choice"] is None:
        reasons.append(f"profile {profile!r} needs an explicit task binding")
    return not reasons, reasons, labels


def query(cid: str, profile: str, inputs: dict, task_bindings=None, allow_draft=False) -> Result:
    """Answer one query from the contract's reference, or refuse with the reasons."""
    labels = {"contract": cid, "profile": profile, "semantic_verdict": "not_established"}
    stage = "contract loading"
    try:
        c = load(cid)
        stage = "contract admission"
        ok, reasons, labels = activation(c, profile, task_bindings, allow_draft)
    except (ContractNotFound, NotEstablished) as error:
        return Result("not_established", note=f"reference unavailable: {error}", labels=labels)
    except Exception as error:
        # Installation/metadata/programming errors are not mathematical outcomes.
        return Result("implementation_error", note=f"{stage}: {type(error).__name__}: {error}", labels=labels)
    if not ok:
        return Result("not_established", note="refused: " + "; ".join(reasons), labels=labels)
    r = evaluate(c.adapter.reference, profile, inputs)
    if r.outcome == "value":
        outputs = profile_outputs(c.data, profile)
        required = set(outputs)
        if not isinstance(r.fields, dict) or set(r.fields) != required:
            got = sorted(r.fields, key=repr) if isinstance(r.fields, dict) else type(r.fields).__name__
            r = Result("implementation_error", note=f"output fields {got}; contract requires {sorted(required)}")
        else:
            try:
                validate_reference_values(r.fields, outputs)
            except (ValueError, TypeError) as error:
                r = Result("implementation_error", note=f"output guarantee: {error}")
    r.labels.update(labels)
    return r


def check_output(cid: str, profile: str, inputs: dict, observed: dict, task_bindings=None, allow_draft=False):
    """(verdict, note, labels) for a complete output of the supplied task.

    Contract activation and reference evaluation are mandatory. No generic
    elementwise fallback is used: it would lose dependencies in allowed sets.
    This compares against a supplied task; kernel binding is still required
    before using a contradiction as a kernel's semantic-error conclusion.
    """
    ref = query(cid, profile, inputs, task_bindings, allow_draft)
    if ref.outcome != "value":
        return "not_established", f"reference {ref.outcome}: {ref.note}", ref.labels
    # query has already checked the reference's complete profile-specific fields.
    # A comparison adapter cannot weaken that interface by ignoring an output.
    if not isinstance(observed, dict) or set(observed) != set(ref.fields):
        return "contradiction", "observed output fields differ from the supplied task", ref.labels
    def compare():
        fn = getattr(load(cid).adapter, "check_output", None)
        if fn is None:
            raise NotEstablished("contract has no complete-output comparison")
        return Result("value", {"matches": fn(profile, inputs, observed)})

    result = evaluate(compare)
    if result.outcome != "value":
        return "not_established", f"output comparison {result.outcome}: {result.note}", ref.labels
    matches = result.fields["matches"]
    if type(matches) is not bool:
        return "not_established", "output comparison did not return a boolean", ref.labels
    return ("consistent" if matches else "contradiction"), "complete output against the supplied task", ref.labels
