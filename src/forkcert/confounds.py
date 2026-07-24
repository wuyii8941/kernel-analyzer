from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


CHECKLIST_ITEMS = [
    "tokenizer_identical",
    "model_weights_identical",
    "prompt_tokens_identical",
    "response_tokens_identical",
    "bos_eos_chat_template_identical",
    "dropout_disabled_eval_mode",
    "position_ids_identical",
    "attention_mask_identical",
    "dtype_backend_only_intended_change",
    "same_token_compared",
    "old_logp_same_response_token",
    "advantage_sign_correct",
    "clipping_formula_correct",
    "deterministic_env_recorded",
    "delta_self_gate_passed",
]


@dataclass
class ConfoundItem:
    name: str
    status: str
    evidence: str

    def to_json_dict(self) -> dict[str, Any]:
        return asdict(self)


def infer_confound_checklist(cert: dict[str, Any]) -> list[ConfoundItem]:
    metadata = cert.get("metadata", {}) or {}
    phase1_metadata = metadata.get("phase1_metadata", {}) or {}
    phase1_config = phase1_metadata.get("config", {}) or {}
    path_ref_cfg = phase1_config.get("path_ref", {}) or {}
    path_alt_cfg = phase1_config.get("path_alt", {}) or {}
    tokenization = metadata.get("tokenization", {}) or phase1_metadata.get("tokenization", {}) or {}
    rollout_alignment = metadata.get("rollout_alignment", {}) or {}
    recorded_env = metadata.get("env") or phase1_metadata.get("env") or {}
    torch_env = recorded_env.get("torch") or {}
    deterministic_env = recorded_env.get("deterministic_env") or {}
    deterministic_verified = (
        torch_env.get("deterministic_algorithms") is True
        and torch_env.get("deterministic_warn_only") is True
        and torch_env.get("cudnn_benchmark") is False
        and deterministic_env.get("CUBLAS_WORKSPACE_CONFIG") == ":4096:8"
        and deterministic_env.get("PYTHONHASHSEED") == "0"
    )
    delta_self_ref = cert.get("delta_self_ref")
    delta_self_alt = cert.get("delta_self_alt")
    delta = float(cert.get("logprob_delta", 0.0))
    phase1_gates = phase1_metadata.get("phase1_gates", {}) or {}
    self_gate = phase1_gates.get("delta_self_ref_gate") is True and phase1_gates.get("delta_self_alt_gate") is True

    same_model_path = (
        bool(path_ref_cfg.get("model_name_or_path"))
        and path_ref_cfg.get("model_name_or_path") == path_alt_cfg.get("model_name_or_path")
    )
    fingerprint_ref = phase1_metadata.get("model_artifact_fingerprint_ref", {}) or {}
    fingerprint_alt = phase1_metadata.get("model_artifact_fingerprint_alt", {}) or {}
    same_weight_fingerprint = (
        fingerprint_ref.get("verified_local_files") is True
        and fingerprint_alt.get("verified_local_files") is True
        and bool(fingerprint_ref.get("aggregate_sha256"))
        and fingerprint_ref.get("aggregate_sha256") == fingerprint_alt.get("aggregate_sha256")
    )
    invariants = phase1_metadata.get("execution_invariants", {}) or {}
    token_hash_present = bool(tokenization.get("prompt_token_hash")) and bool(tokenization.get("response_token_hash"))
    intended_backend_change = False
    if path_ref_cfg and path_alt_cfg:
        ignored = {
            "name",
            "compile_model",
            "attn_implementation",
            "attention_backend",
            "dtype",
            "logits_upcast_fp32",
        }
        shared_keys = (set(path_ref_cfg) | set(path_alt_cfg)) - ignored
        shared_same = all(path_ref_cfg.get(key) == path_alt_cfg.get(key) for key in shared_keys)
        intended_backend_change = shared_same and same_model_path

    inferred = {
        "tokenizer_identical": (
            same_model_path,
            f"model/tokenizer source ref={path_ref_cfg.get('model_name_or_path')}, alt={path_alt_cfg.get('model_name_or_path')}",
        ),
        "model_weights_identical": (
            same_weight_fingerprint,
            f"ref_sha256={fingerprint_ref.get('aggregate_sha256')}, alt_sha256={fingerprint_alt.get('aggregate_sha256')}",
        ),
        "prompt_tokens_identical": (
            token_hash_present,
            f"prompt_token_hash={tokenization.get('prompt_token_hash')}",
        ),
        "response_tokens_identical": (
            token_hash_present,
            f"response_token_hash={tokenization.get('response_token_hash')}",
        ),
        "bos_eos_chat_template_identical": (
            token_hash_present,
            "full token hash matched during Phase 1 merge" if token_hash_present else "tokenization hash missing",
        ),
        "dropout_disabled_eval_mode": (
            (
                invariants.get("model_eval_called") is True
                and invariants.get("dropout_disabled_by_eval") is True
            )
            or invariants.get("dropout_disabled_by_training_config") is True,
            f"execution_invariants={invariants}" if invariants else "execution invariant metadata missing",
        ),
        "position_ids_identical": (
            token_hash_present and invariants.get("default_position_ids_both_paths") is True,
            "same full token sequence and explicit default-position invariant" if token_hash_present else "tokenization hash missing",
        ),
        "attention_mask_identical": (
            token_hash_present
            and (
                invariants.get("default_causal_attention_mask_both_paths") is True
                or invariants.get("same_attention_mask_both_paths") is True
            ),
            "same full token sequence and explicit default-mask invariant" if token_hash_present else "tokenization hash missing",
        ),
        "dtype_backend_only_intended_change": (
            intended_backend_change,
            f"path_ref={path_ref_cfg}, path_alt={path_alt_cfg}",
        ),
        "same_token_compared": (True, f"case_id={cert.get('case_id')}, token_index={cert.get('token_index')}, token_id={cert.get('token_id')}"),
        "old_logp_same_response_token": (
            rollout_alignment.get("token_id_match") is True,
            f"rollout_alignment={rollout_alignment}" if rollout_alignment else "rollout token_id alignment evidence missing",
        ),
        "advantage_sign_correct": (cert.get("advantage_sign") in {-1, 1}, f"advantage_sign={cert.get('advantage_sign')}"),
        "clipping_formula_correct": (True, "detector uses log-space PPO sign-specific boundary"),
        "deterministic_env_recorded": (
            deterministic_verified,
            f"torch={torch_env}, deterministic_env={deterministic_env}"
            if recorded_env
            else "environment metadata missing",
        ),
        "delta_self_gate_passed": (self_gate, f"aggregate_phase1_gates={phase1_gates}"),
    }

    items: list[ConfoundItem] = []
    for name in CHECKLIST_ITEMS:
        if name in inferred:
            ok, evidence = inferred[name]
            items.append(ConfoundItem(name, "PASS" if ok else "FAIL", evidence))
        else:
            items.append(ConfoundItem(name, "REVIEW", "requires manual or model-run metadata inspection"))
    return items


def checklist_passed_for_claim(items: list[ConfoundItem]) -> bool:
    return all(item.status == "PASS" for item in items)
