"""Tests for the Bias-Variance Oracle.

These tests verify that the oracle correctly:
1. Detects systematic bias (and rejects it)
2. Separates mean effect, state heterogeneity, runtime variability and uncertainty
3. Measures input heterogeneity
4. Handles edge cases (INVALID, UNINSTANTIATED)
5. Step-level propagation works correctly
"""

import math
import numpy as np
import pytest

from forkcert.oracle import (
    AcceptanceCriteria,
    TrainingOracle,
    Oracle,
    OperatorMeasurement,
    Verdict,
    VerdictResult,
    collect_operator_measurements,
    compute_operator_profile,
    compute_step_profile,
    judge_operator,
    judge_step,
    format_operator_report,
    format_step_report,
)


# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------

def make_measurements(n_inputs, n_repeats, ref_fn, cand_fn):
    ms = []
    for i in range(n_inputs):
        for r in range(n_repeats):
            ref = ref_fn(i, r)
            cand = cand_fn(i, r)
            ms.append(OperatorMeasurement(
                input_id=i, repeat_id=r,
                ref_output=np.array(ref, dtype=np.float64),
                cand_output=np.array(cand, dtype=np.float64),
            ))
    return ms


# -----------------------------------------------------------------------
# Layer 2: Statistical profile
# -----------------------------------------------------------------------

class TestBiasDetection:
    def test_constant_bias_detected(self):
        """Candidate has a fixed +0.01 shift on every output."""
        bias = 0.01
        ms = make_measurements(
            n_inputs=50, n_repeats=1,
            ref_fn=lambda i, r: [1.0 + 0.1 * i],
            cand_fn=lambda i, r: [1.0 + 0.1 * i + bias],
        )
        p = compute_operator_profile("const_bias", ms)
        assert abs(p.bias - bias) < 1e-10
        assert p.relative_bias > 0
        assert p.heterogeneity < 1e-20  # bias is constant across inputs
        assert p.runtime_var == 0.0  # deterministic

    def test_zero_bias_for_identical(self):
        """Identical implementations should have zero bias."""
        ms = make_measurements(
            n_inputs=30, n_repeats=1,
            ref_fn=lambda i, r: [float(i) * 0.5],
            cand_fn=lambda i, r: [float(i) * 0.5],
        )
        p = compute_operator_profile("identical", ms)
        assert p.bias == 0.0
        assert p.relative_bias == 0.0
        assert p.heterogeneity == 0.0

    def test_symmetric_noise_low_bias(self):
        """Symmetric random noise should produce near-zero bias."""
        rng = np.random.RandomState(42)
        noise = rng.randn(100) * 0.01
        ms = make_measurements(
            n_inputs=100, n_repeats=1,
            ref_fn=lambda i, r: [1.0],
            cand_fn=lambda i, r: [1.0 + noise[i]],
        )
        p = compute_operator_profile("symmetric_noise", ms)
        assert abs(p.bias) < 0.005  # should be close to zero
        assert p.heterogeneity > 1e-6  # but heterogeneity is nonzero


class TestHeterogeneityDetection:
    def test_input_dependent_bias(self):
        """Bias depends on input: positive for large inputs, negative for small."""
        ms = make_measurements(
            n_inputs=100, n_repeats=1,
            ref_fn=lambda i, r: [float(i)],
            cand_fn=lambda i, r: [float(i) + (i - 50) * 0.001],  # bias proportional to input
        )
        p = compute_operator_profile("heterogeneous", ms)
        assert p.heterogeneity > 0  # bias varies with input
        assert abs(p.bias) < 0.001  # average bias is small (cancels out)

    def test_constant_bias_no_heterogeneity(self):
        """Constant bias has zero heterogeneity."""
        ms = make_measurements(
            n_inputs=50, n_repeats=1,
            ref_fn=lambda i, r: [1.0 + 0.01 * i],
            cand_fn=lambda i, r: [1.0 + 0.01 * i + 0.05],
        )
        p = compute_operator_profile("no_het", ms)
        assert p.heterogeneity < 1e-20


class TestRuntimeVariance:
    def test_deterministic_zero_variance(self):
        ms = make_measurements(
            n_inputs=20, n_repeats=5,
            ref_fn=lambda i, r: [float(i)],
            cand_fn=lambda i, r: [float(i) + 0.01],
        )
        p = compute_operator_profile("det", ms)
        assert p.runtime_var == 0.0

    def test_stochastic_nonzero_variance(self):
        rng = np.random.RandomState(123)
        noise = rng.randn(20, 5) * 0.01
        ms = make_measurements(
            n_inputs=20, n_repeats=5,
            ref_fn=lambda i, r: [1.0],
            cand_fn=lambda i, r: [1.0 + noise[i, r]],
        )
        p = compute_operator_profile("stoch", ms)
        assert p.runtime_var > 0


class TestMultiElement:
    def test_vector_output(self):
        """Profile works with multi-element outputs."""
        ms = make_measurements(
            n_inputs=30, n_repeats=1,
            ref_fn=lambda i, r: [1.0, 2.0, 3.0],
            cand_fn=lambda i, r: [1.01, 2.0, 3.0],  # bias only on first element
        )
        p = compute_operator_profile("vec", ms)
        assert p.element_bias is not None
        assert abs(p.element_bias[0] - 0.01) < 1e-10
        assert abs(p.element_bias[1]) < 1e-10
        assert abs(p.element_bias[2]) < 1e-10


# -----------------------------------------------------------------------
# Layer 4: Verdict
# -----------------------------------------------------------------------

class TestVerdictLogic:
    def test_accept_within_bounds(self):
        ms = make_measurements(
            n_inputs=50, n_repeats=1,
            ref_fn=lambda i, r: [10.0],
            cand_fn=lambda i, r: [10.0 + 1e-7],
        )
        p = compute_operator_profile("ok", ms)
        criteria = AcceptanceCriteria(max_relative_bias=1e-5)
        v = judge_operator(p, criteria)
        assert v.verdict == Verdict.ACCEPT

    def test_reject_exceeds_bias(self):
        ms = make_measurements(
            n_inputs=50, n_repeats=1,
            ref_fn=lambda i, r: [1.0],
            cand_fn=lambda i, r: [1.0 + 0.1],  # 10% bias
        )
        p = compute_operator_profile("bad", ms)
        criteria = AcceptanceCriteria(max_relative_bias=0.01)
        v = judge_operator(p, criteria)
        assert v.verdict == Verdict.REJECT

    def test_uninstantiated_no_criteria(self):
        ms = make_measurements(
            n_inputs=50, n_repeats=1,
            ref_fn=lambda i, r: [1.0],
            cand_fn=lambda i, r: [1.1],
        )
        p = compute_operator_profile("no_crit", ms)
        criteria = AcceptanceCriteria()
        v = judge_operator(p, criteria)
        assert v.verdict == Verdict.UNINSTANTIATED

    def test_invalid_single_input(self):
        ms = make_measurements(
            n_inputs=1, n_repeats=1,
            ref_fn=lambda i, r: [1.0],
            cand_fn=lambda i, r: [1.1],
        )
        p = compute_operator_profile("one", ms)
        criteria = AcceptanceCriteria(max_relative_bias=0.01)
        v = judge_operator(p, criteria)
        assert v.verdict == Verdict.INVALID

    def test_indeterminate_noisy_bias(self):
        """Large noise makes bias estimate unreliable -> INDETERMINATE."""
        rng = np.random.RandomState(99)
        ms = make_measurements(
            n_inputs=5, n_repeats=1,
            ref_fn=lambda i, r: [1.0],
            cand_fn=lambda i, r: [1.0 + rng.randn() * 0.5 + 0.02],
        )
        p = compute_operator_profile("noisy", ms)
        criteria = AcceptanceCriteria(max_relative_bias=0.01)
        v = judge_operator(p, criteria)
        assert v.verdict in (Verdict.INDETERMINATE, Verdict.REJECT)


class TestHighLevelOracle:
    def test_measure_and_judge(self):
        oracle = Oracle(AcceptanceCriteria(max_relative_bias=1e-4))
        inputs = list(range(30))
        ref_fn = lambda x: np.array([float(x) + 1.0])
        cand_fn = lambda x: np.array([float(x) + 1.0 + 1e-6])
        profile, verdict = oracle.measure_and_judge("test_op", ref_fn, cand_fn, inputs)
        assert verdict.verdict == Verdict.ACCEPT
        assert profile.n_inputs == 30


# -----------------------------------------------------------------------
# Layer 3: Step profile
# -----------------------------------------------------------------------

class TestStepProfile:
    def test_zero_diff_step(self):
        n = 10
        d = 5
        step_ms = []
        for i in range(n):
            g = np.random.randn(d)
            step_ms.append({
                "input_id": i, "repeat_id": 0,
                "ref_loss": 1.0, "cand_loss": 1.0,
                "ref_grad": g, "cand_grad": g.copy(),
                "ref_param_update": g * -0.01, "cand_param_update": g * -0.01,
            })
        sp = compute_step_profile(step_ms)
        assert abs(sp.loss_bias) < 1e-15
        assert sp.grad_bias_norm < 1e-15
        assert sp.param_update_bias_norm < 1e-15

    def test_biased_step(self):
        n = 20
        d = 10
        step_ms = []
        for i in range(n):
            g = np.random.randn(d)
            bias_g = np.ones(d) * 0.01
            step_ms.append({
                "input_id": i, "repeat_id": 0,
                "ref_loss": 2.0, "cand_loss": 2.05,
                "ref_grad": g, "cand_grad": g + bias_g,
                "ref_param_update": g * -0.01, "cand_param_update": (g + bias_g) * -0.01,
            })
        sp = compute_step_profile(step_ms)
        assert sp.loss_bias > 0
        assert sp.grad_bias_norm > 0
        assert sp.param_update_bias_norm > 0

    def test_step_verdict(self):
        step_ms = []
        for i in range(10):
            step_ms.append({
                "input_id": i, "repeat_id": 0,
                "ref_loss": 2.0, "cand_loss": 2.0 + 0.5,
                "ref_grad": np.ones(5), "cand_grad": np.ones(5) * 1.1,
                "ref_param_update": np.ones(5) * -0.01,
                "cand_param_update": np.ones(5) * -0.011,
            })
        sp = compute_step_profile(step_ms)
        criteria = AcceptanceCriteria(max_step_loss_bias=0.01)
        v = judge_step(sp, criteria)
        assert v.verdict == Verdict.REJECT


# -----------------------------------------------------------------------
# Key property: bias vs noise discrimination
# -----------------------------------------------------------------------

class TestBiasVsNoiseSeparation:
    """Finite-grid profiles separate a mean from H; they do not prove safety."""

    def test_large_state_heterogeneity_passes_only_declared_global_b_bound(self):
        """Mean cancellation can pass a B-only bound while H stays unchecked."""
        rng = np.random.RandomState(42)
        ms = make_measurements(
            n_inputs=200, n_repeats=1,
            ref_fn=lambda i, r: [1.0],
            cand_fn=lambda i, r: [1.0 + rng.randn() * 0.1],  # large noise, zero mean
        )
        p = compute_operator_profile("noisy_ok", ms)
        criteria = AcceptanceCriteria(max_relative_bias=0.05)
        v = judge_operator(p, criteria)
        assert v.verdict == Verdict.ACCEPT
        assert "H" in v.details["unchecked_components"]
        assert v.details["conditional_effects_tested"] is False
        assert v.details["semantic_effects_tested"] is False

    def test_small_bias_detected_and_rejected(self):
        """Small but consistent bias -> should reject."""
        ms = make_measurements(
            n_inputs=200, n_repeats=1,
            ref_fn=lambda i, r: [1.0],
            cand_fn=lambda i, r: [1.0 + 0.001],  # tiny but perfectly consistent
        )
        p = compute_operator_profile("small_bias", ms)
        criteria = AcceptanceCriteria(max_relative_bias=1e-4)
        v = judge_operator(p, criteria)
        assert v.verdict == Verdict.REJECT

    def test_raw_diff_would_mislead(self):
        """Case where raw diff magnitude is misleading.

        Operator A: large diff, near-zero global mean (H unresolved)
        Operator B: small diff, consistent finite-grid mean shift

        A B-only verdict can pass while B fails the same bound, even though A
        has larger raw diffs. This does not establish that A is benign.
        """
        rng = np.random.RandomState(7)
        noise_vals = rng.randn(500) * 0.5

        # A: large noise, zero-mean — generate all noise upfront for reproducibility
        ms_a = make_measurements(
            n_inputs=500, n_repeats=1,
            ref_fn=lambda i, r: [1.0],
            cand_fn=lambda i, r: [1.0 + noise_vals[i]],
        )
        p_a = compute_operator_profile("large_diff_ok", ms_a)

        # B: consistent 0.02 bias — small diff but systematic
        ms_b = make_measurements(
            n_inputs=500, n_repeats=1,
            ref_fn=lambda i, r: [1.0],
            cand_fn=lambda i, r: [1.02],
        )
        p_b = compute_operator_profile("small_diff_bad", ms_b)

        # Raw max diff: A >> B
        max_diff_a = max(abs(m.diff[0]) for m in ms_a)
        max_diff_b = max(abs(m.diff[0]) for m in ms_b)
        assert max_diff_a > max_diff_b

        # Bias significance: B is certain, A has high standard error
        assert p_b.bias_std_err == 0.0  # perfectly consistent
        assert p_a.bias_std_err > 0.01  # noisy estimate

        criteria = AcceptanceCriteria(max_relative_bias=0.01)
        v_a = judge_operator(p_a, criteria)
        v_b = judge_operator(p_b, criteria)
        # These verdicts only compare finite-grid summaries with the supplied
        # descriptive bound.  They do not establish danger, benignity, or
        # long-run training safety.
        assert v_b.verdict == Verdict.REJECT
        assert v_a.verdict in (Verdict.ACCEPT, Verdict.INDETERMINATE)


class TestCriticalCounterexamples:
    def test_vector_sign_cancellation_is_not_zero_bias(self):
        """A [+d,-d] deterministic effect must not disappear by element averaging."""
        ms = make_measurements(
            n_inputs=40, n_repeats=2,
            ref_fn=lambda i, r: [1.0, 1.0],
            cand_fn=lambda i, r: [2.0, 0.0],
        )
        p = compute_operator_profile("vector_cancel", ms)
        assert abs(p.bias) < 1e-15  # legacy signed scalar summary cancels
        assert p.bias_norm > 0.9
        v = judge_operator(p, AcceptanceCriteria(max_relative_bias=0.1))
        assert v.verdict == Verdict.REJECT

    def test_zero_sample_mean_does_not_prove_safety(self):
        """A wide state distribution centered at zero needs precision, not ACCEPT."""
        effects = [-1.0, 1.0, -1.0, 1.0]
        ms = make_measurements(
            n_inputs=4, n_repeats=1,
            ref_fn=lambda i, r: [1.0],
            cand_fn=lambda i, r: [1.0 + effects[i]],
        )
        p = compute_operator_profile("wide_zero_mean", ms)
        v = judge_operator(p, AcceptanceCriteria(max_relative_bias=0.1))
        assert v.verdict == Verdict.INDETERMINATE

    def test_runtime_bound_without_repeats_is_invalid(self):
        ms = make_measurements(
            n_inputs=20, n_repeats=1,
            ref_fn=lambda i, r: [1.0],
            cand_fn=lambda i, r: [1.0],
        )
        p = compute_operator_profile("no_repeats", ms)
        v = judge_operator(p, AcceptanceCriteria(max_runtime_cv=0.01))
        assert v.verdict == Verdict.INVALID

    def test_repeat_corrected_h_without_repeats_is_invalid(self):
        ms = make_measurements(
            n_inputs=20, n_repeats=1,
            ref_fn=lambda i, r: [1.0],
            cand_fn=lambda i, r: [1.0 + (-1) ** i * 0.1],
        )
        p = compute_operator_profile("h_without_repeats", ms)
        v = judge_operator(p, AcceptanceCriteria(max_heterogeneity_cv=0.2))
        assert v.verdict == Verdict.INVALID

    def test_operator_only_criteria_cannot_accept_a_step(self):
        step_ms = [{
            "input_id": i, "repeat_id": 0,
            "ref_loss": 1.0, "cand_loss": 100.0,
            "ref_grad": np.ones(2), "cand_grad": np.ones(2) * 100,
            "ref_param_update": np.ones(2), "cand_param_update": np.ones(2) * 100,
        } for i in range(3)]
        sp = compute_step_profile(step_ms)
        v = judge_step(sp, AcceptanceCriteria(max_relative_bias=1e-3))
        assert v.verdict == Verdict.UNINSTANTIATED


class TestMatchedTrainingOracle:
    def make_oracle(self):
        return TrainingOracle(
            AcceptanceCriteria(max_relative_bias=0.2, max_runtime_cv=0.2),
            query_id="unit-query",
            state_distribution="four fixed matched states",
            randomness_protocol="two deterministic repeats",
            coupling_protocol="same repeat id",
        )

    def test_multi_module_states_are_aggregated_without_free_running(self):
        oracle = self.make_oracle()
        for state in range(4):
            oracle.record_operator_state(
                "layer_a", f"s{state}",
                [np.array([1.0]), np.array([1.0])],
                [np.array([1.01]), np.array([1.01])],
            )
            oracle.record_operator_state(
                "layer_b", f"s{state}",
                [np.array([2.0]), np.array([2.0])],
                [np.array([2.0 + (-1) ** state * 0.1])] * 2,
            )
        profiles = oracle.operator_profiles()
        assert set(profiles) == {"layer_a", "layer_b"}
        assert profiles["layer_a"].heterogeneity < 1e-20
        assert profiles["layer_b"].heterogeneity > 0
        assert profiles["layer_a"].runtime_var == 0
        assert oracle.n_states == 4

    def test_duplicate_state_is_invalid(self):
        oracle = self.make_oracle()
        oracle.record_operator_state("layer", "s0", [np.array([1.0])], [np.array([1.0])])
        try:
            oracle.record_operator_state("layer", "s0", [np.array([1.0])], [np.array([1.0])])
        except ValueError:
            pass
        else:
            raise AssertionError("duplicate matched state was accepted")

    def test_query_contract_is_required(self):
        try:
            TrainingOracle(
                AcceptanceCriteria(max_relative_bias=0.1),
                query_id="", state_distribution="states",
                randomness_protocol="fixed", coupling_protocol="paired",
            )
        except ValueError:
            pass
        else:
            raise AssertionError("unnamed query was accepted")

    def test_multi_module_coverage_must_match(self):
        oracle = self.make_oracle()
        oracle.record_operator_state("a", "s0", [np.array([1.0])], [np.array([1.0])])
        oracle.record_operator_state("b", "s1", [np.array([1.0])], [np.array([1.0])])
        try:
            oracle.operator_profiles()
        except ValueError as error:
            assert "same matched state bank" in str(error)
        else:
            raise AssertionError("mismatched module state coverage was accepted")

    def test_step_profile_uses_same_matched_state_bank(self):
        oracle = TrainingOracle(
            AcceptanceCriteria(max_relative_bias=0.2, max_step_param_bias=0.2),
            query_id="step-query", state_distribution="four matched pre-step states",
            randomness_protocol="two repeats", coupling_protocol="paired",
        )
        for state in range(4):
            state_id = f"s{state}"
            oracle.record_operator_state(
                "scorer", state_id,
                [np.array([1.0]), np.array([1.0])],
                [np.array([1.01]), np.array([1.01])],
            )
            repeats = []
            for _ in range(2):
                repeats.append({
                    "ref_loss": 1.0,
                    "cand_loss": 1.001,
                    "ref_grad": np.array([1.0, 2.0]),
                    "cand_grad": np.array([1.001, 2.001]),
                    "ref_param_update": np.array([-0.1, -0.2]),
                    "cand_param_update": np.array([-0.1001, -0.2001]),
                })
            oracle.record_step_state(state_id, repeats)
        profile = oracle.step_profile()
        assert profile.n_inputs == 4
        assert profile.n_repeats == 2
        assert set(profile.operator_profiles) == {"scorer"}
        assert oracle.step_verdict().verdict == Verdict.ACCEPT


# -----------------------------------------------------------------------
# Reporting
# -----------------------------------------------------------------------

class TestReporting:
    def test_operator_report(self):
        ms = make_measurements(
            n_inputs=10, n_repeats=1,
            ref_fn=lambda i, r: [1.0],
            cand_fn=lambda i, r: [1.01],
        )
        p = compute_operator_profile("test", ms)
        criteria = AcceptanceCriteria(max_relative_bias=0.1)
        v = judge_operator(p, criteria)
        report = format_operator_report(p, v)
        assert "test" in report
        assert "ACCEPT" in report

    def test_step_report(self):
        step_ms = [{
            "input_id": i, "repeat_id": 0,
            "ref_loss": 1.0, "cand_loss": 1.0,
            "ref_grad": np.ones(3), "cand_grad": np.ones(3),
            "ref_param_update": np.ones(3) * -0.01,
            "cand_param_update": np.ones(3) * -0.01,
        } for i in range(5)]
        sp = compute_step_profile(step_ms)
        report = format_step_report(sp)
        assert "Training Step" in report


# -----------------------------------------------------------------------
# collect_operator_measurements
# -----------------------------------------------------------------------

class TestCollect:
    def test_basic_collect(self):
        ref_fn = lambda x: np.array([float(x)])
        cand_fn = lambda x: np.array([float(x) + 0.1])
        ms = collect_operator_measurements(ref_fn, cand_fn, [1, 2, 3], n_repeats=2)
        assert len(ms) == 6
        assert ms[0].input_id == 0
        assert ms[1].input_id == 0
        assert ms[1].repeat_id == 1
