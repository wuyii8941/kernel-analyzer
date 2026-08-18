"""Qwen vertical regression using retained execution-derived evidence."""

from pathlib import Path

from kernel_analyzer import AnalysisSpec, AnalysisState, ResourceBudget
from kernel_analyzer.backends import (
    FlashControlBackend, NumericalCandidateBackend, ObservedRegionRuntime,
    RetainedCaseBackend,
)
from kernel_analyzer.providers import LedgerReferenceProvider


ROOT = Path(__file__).resolve().parents[1]


def build_spec():
    return AnalysisSpec(
        subject="Qwen3-1.7B retained vertical regression",
        reference=LedgerReferenceProvider(
            ledger=ROOT / "results/coverage/fb_proof_unit_ledger.json.gz",
            model_key="qwen3_1p7b",
            template_catalog=ROOT / "results/coverage/qwen_invocation_ledger.json.gz",
            case_audit=ROOT / "results/coverage/existing_case_reaudit.json",
            additional_case_targets=[{
                "unit_id": "control::flash_paper_reference",
                "closed_semantic_target": True,
                "natural": False,
            }],
        ),
        candidates=[
            RetainedCaseBackend(
                candidate_id="retained_qwen_candidates",
                audit_path=ROOT / "results/coverage/existing_case_reaudit.json",
            ),
            FlashControlBackend(ROOT / "results/final/flash_control.json"),
            NumericalCandidateBackend(
                "qwen_bf16_inductor_changed_regions",
                ObservedRegionRuntime(
                    ROOT / "results/final/evolving_triton_seq64.json",
                    "bf16_inductor_full_step",
                    ROOT / "results/coverage/qwen_t2_step0_evidence_actual_fb4.json",
                    ROOT / "results/coverage/qwen_t3_evidence_actual_fb4.json",
                ),
            ),
        ],
        states=[
            AnalysisState("retained-evidence", role="CONTROL"),
            AnalysisState("qwen-step0-eval-offset-0", role="DISCOVERY"),
            AnalysisState("qwen-t3-c0", role="CONFIRMATION"),
            AnalysisState("qwen-t3-c1", role="CONFIRMATION"),
            AnalysisState("qwen-t3-c2", role="CONFIRMATION"),
            AnalysisState("qwen-t3-c3", role="CONFIRMATION"),
        ],
        output_dir=ROOT / "results/system_runs",
        metadata={
            "mode": "retained-evidence-regression",
            "candidate_values_used_for_math": False,
            "automated_t2_evidence": "qwen-step0-actual-fb4-v1",
            "automated_t3_evidence": "qwen-independent-text-complete-carrier-v1",
        },
        resources=ResourceBudget(
            writable_root=Path("/data1/tzh"),
            min_free_bytes=50 * 1024**3,
            max_artifact_bytes=256 * 1024**2,
        ),
    )
