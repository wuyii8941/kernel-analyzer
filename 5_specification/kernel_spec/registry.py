"""The single runtime entry to the contracts.

Only `contracts/*.yaml` (top level) are loaded; `contracts/draft/` and `drafts/` are history and are never imported.
Each contract binds to `adapters/<ID>.py`. A query is answered only when the contract may be used for it:

- review: `reviewed_for_implementation`, or `drafted` when the caller passes allow_draft=True (the result is then
  labelled as coming from a draft contract);
- implementation: the contract and adapter hashes equal the ones in `contracts/LOCK.json`, which `check_contracts.py
  --lock` writes only after every example, reading and mutation requirement of the contract passed;
- sources: every source a clause or decision rests on has an archived snapshot (SOURCES.json snapshot_sha256);
- task choice: a profile with task_choice_required needs an explicit binding from the caller;
- profile: must be declared in the contract.

None of these is upgraded automatically from another (CONTRACT_PROTOCOL sections 2 and 7).
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
from dataclasses import dataclass
from pathlib import Path

import yaml

from .num import Result, evaluate

ROOT = Path(__file__).resolve().parents[1]
CONTRACTS = ROOT / "contracts"
ADAPTERS = ROOT / "adapters"
LOCK = CONTRACTS / "LOCK.json"
SOURCES = ROOT / "SOURCES.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_sources() -> dict:
    return json.loads(SOURCES.read_text())


def contract_paths() -> list[Path]:
    return sorted(CONTRACTS.glob("*.yaml"))


def frozen_source(entry: dict) -> bool:
    """An archived snapshot with its hash, or a published standard cited by edition and section (not archivable)."""
    return bool(entry.get("snapshot_sha256")) or (entry.get("kind") == "standard" and bool(entry.get("edition")))


def load_lock() -> dict:
    return json.loads(LOCK.read_text()) if LOCK.is_file() else {}


@dataclass
class Contract:
    id: str
    data: dict
    path: Path
    adapter_path: Path
    adapter: object

    @property
    def contract_sha256(self):
        return sha256(self.path)

    @property
    def adapter_sha256(self):
        return sha256(self.adapter_path)


def import_adapter(cid: str, adapters: Path = ADAPTERS):
    path = adapters / f"{cid}.py"
    if not path.is_file():
        raise FileNotFoundError(f"no adapter {path}")
    spec = importlib.util.spec_from_file_location(f"kernel_spec_adapter_{cid}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return path, mod


def load(cid: str, contracts: Path = CONTRACTS, adapters: Path = ADAPTERS) -> Contract:
    path = contracts / f"{cid}.yaml"
    if path.parent != contracts or not path.is_file():
        raise FileNotFoundError(f"no contract {cid} in {contracts}")
    data = yaml.safe_load(path.read_text())
    if data.get("id") != cid:
        raise ValueError(f"{path}: id {data.get('id')!r} does not match the file name")
    apath, mod = import_adapter(cid, adapters)
    return Contract(cid, data, path, apath, mod)


def activation(c: Contract, profile: str, task_bindings=None, allow_draft=False, lock=None):
    """(usable, reasons, labels) for one query."""
    lock = load_lock() if lock is None else lock
    reasons, labels = [], {"contract": c.id, "profile": profile, "contract_sha256": c.contract_sha256,
                           "adapter_sha256": c.adapter_sha256, "review_status": c.data["review"]["status"]}
    status = c.data["review"]["status"]
    if status == "drafted":
        labels["draft_contract"] = True
        if not allow_draft:
            reasons.append("contract is drafted, not reviewed (pass allow_draft to use it with a draft label)")
    elif status != "reviewed_for_implementation":
        reasons.append(f"review status {status!r}")
    sources = load_sources()
    used = {b.split(".")[0] for node in [*c.data.get("clauses", []), *c.data.get("decisions", [])]
            for b in node.get("basis", [])}
    unfrozen = sorted(s for s in used if s in sources and not frozen_source(sources[s]))
    if unfrozen:
        reasons.append(f"sources without an archived snapshot: {', '.join(unfrozen)}")
    entry = lock.get(c.id)
    if not entry:
        reasons.append("implementation not validated (no LOCK entry)")
    else:
        if entry.get("contract_sha256") != c.contract_sha256:
            reasons.append("contract changed since the implementation was validated (stale LOCK hash)")
        if entry.get("adapter_sha256") != c.adapter_sha256:
            reasons.append("adapter changed since the implementation was validated (stale LOCK hash)")
    prof = c.data["profiles"].get(profile)
    if prof is None:
        reasons.append(f"unknown profile {profile!r}")
    elif prof.get("task_choice_required") and profile not in (task_bindings or {}):
        reasons.append(f"profile {profile!r} needs an explicit task binding")
    return not reasons, reasons, labels


def query(cid: str, profile: str, inputs: dict, task_bindings=None, allow_draft=False) -> Result:
    """Answer one query from the contract's reference, or refuse with the reasons."""
    c = load(cid)
    ok, reasons, labels = activation(c, profile, task_bindings, allow_draft)
    if not ok:
        return Result("not_established", note="refused: " + "; ".join(reasons), labels=labels)
    r = evaluate(c.adapter.reference, profile, inputs)
    r.labels.update(labels)
    return r
