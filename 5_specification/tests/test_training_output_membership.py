"""Complete outputs against reviewed training contracts, independently of interval midpoints."""
import copy
import itertools
import sys
import unittest
from fractions import Fraction as Q
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kernel_spec import registry
from kernel_spec.num import NotEstablished, hull


class TrainingOutputs(unittest.TestCase):
    def setUp(self):
        self.aw = registry.load("ADAMW").adapter
        self.rm = registry.load("RANKMEAN").adapter
        self.inp = dict(state=dict(theta=[1], m=[0], v=[0], t=0), gradient=[2],
                        lr=Q(1, 10), beta1=Q(9, 10), beta2=Q(99, 100), eps=0,
                        weight_decay=Q(1, 10), maximize=False, amsgrad=False)
        self.out = dict(theta=[Q(89, 100)], m=[Q(1, 5)], v=[Q(1, 25)], t=1)

    def test_adamw_all_state_fields_are_required(self):
        self.assertTrue(self.aw.check_output("base", self.inp, self.out))
        for field in self.out:
            missing = {k: v for k, v in self.out.items() if k != field}
            self.assertFalse(self.aw.check_output("base", self.inp, missing))
            wrong = copy.deepcopy(self.out)
            if field == "t":
                wrong[field] = 0
            else:
                wrong[field][0] += 1
            self.assertFalse(self.aw.check_output("base", self.inp, wrong))

    def test_adamw_unknown_coordinate_does_not_hide_state_contradiction(self):
        for field in ('m', 'v', 't'):
            candidate=copy.deepcopy(self.out)
            candidate['theta']=[hull(0,1)]
            if field=='t':candidate[field]=9
            else:candidate[field][0]+=1
            self.assertFalse(self.aw.check_output('base',self.inp,candidate))
            with patch.object(registry,'activation',return_value=(True,[],{})):
                self.assertEqual(registry.check_output('ADAMW','base',self.inp,candidate)[0],'contradiction')

    def test_adamw_unknown_state_does_not_hide_parameter_contradiction(self):
        for field,value in (('m',[hull(0,1)]),('t',hull(0,2))):
            candidate=dict(self.out,theta=[0]);candidate[field]=value
            self.assertFalse(self.aw.check_output('base',self.inp,candidate))
            candidate=dict(self.out);candidate[field]=value
            with self.assertRaises(NotEstablished):self.aw.check_output('base',self.inp,candidate)

    def test_adamw_missing_gradient_still_compares_all_state(self):
        inp=dict(self.inp,gradient=None)
        candidate=copy.deepcopy(inp['state']);candidate['theta']=[hull(0,2)];candidate['v']=[1]
        self.assertFalse(self.aw.check_output('base',inp,candidate))
        candidate['v']=[0]
        with self.assertRaises(NotEstablished):self.aw.check_output('base',inp,candidate)

    def test_adamw_rejects_wrong_square_root_sign(self):
        wrong = dict(self.out, theta=[Q(109, 100)])  # would satisfy a squared equation with sqrt(v_hat)=-2
        self.assertFalse(self.aw.check_output("base", self.inp, wrong))

    def test_adamw_exact_square_root_cases_and_perturbations(self):
        for g, eps, lr, b1, t, m in itertools.product((-2, 0, 2), (0, 1), (0, Q(1, 10)),
                                                     (0, Q(1, 2)), (0, 2), (-1, 0, 3)):
            if g == 0 and eps == 0:
                continue
            inp = dict(self.inp, gradient=[g], eps=eps, lr=lr, beta1=b1, beta2=0,
                       state=dict(theta=[2], m=[m], v=[7], t=t))
            moment = b1 * m + (1 - b1) * g
            corrected = Q(moment) / (1 - b1 ** (t + 1))
            # beta2=0 makes the independently known positive root |g|.
            value = 2 * (1 - lr * Q(1, 10)) - lr * corrected / (abs(g) + eps)
            out = dict(theta=[value], m=[moment], v=[g * g], t=t + 1)
            self.assertTrue(self.aw.check_output("base", inp, out))
            for error in (Q(1, 10**60), -Q(1, 10**60)):
                self.assertFalse(self.aw.check_output("base", inp, dict(out, theta=[value + error])))

    def test_adamw_irrational_target_is_not_certified_by_nearby_float(self):
        inp = dict(self.inp, state=dict(theta=[1], m=[1], v=[1], t=1), gradient=[0],
                   beta1=Q(1, 2), beta2=Q(1, 2), eps=0, weight_decay=0)
        near = 1 - 0.1 * (2 / 3) / (2 / 3) ** 0.5
        out = dict(theta=[near], m=[Q(1, 2)], v=[Q(1, 2)], t=2)
        self.assertFalse(self.aw.check_output("base", inp, out))

    def test_none_and_zero_gradient_have_different_post_state(self):
        inp = dict(self.inp, gradient=None)
        self.assertTrue(self.aw.check_output("base", inp, inp["state"]))
        self.assertFalse(self.aw.check_output("base", inp, self.out))
        zero = dict(self.inp, gradient=[0], eps=1)
        self.assertFalse(self.aw.check_output("base", zero, zero["state"]))
        self.assertTrue(self.aw.check_output("base", zero, dict(theta=[Q(99, 100)], m=[0], v=[0], t=1)))

    def test_rankmean_checks_exact_global_mean(self):
        inp = dict(S=[2, 9, 0], n=[1, 3, 0], participant_ids=["a", "b", "empty"])
        self.assertTrue(self.rm.check_output("base", inp, {"loss": Q(11, 4)}))
        for observed in ({"loss": Q(5, 2)}, {}, {"loss": Q(11, 4), "extra": 1}):
            self.assertFalse(self.rm.check_output("base", inp, observed))
        inp = dict(S=[1], n=[3], participant_ids=[0])
        self.assertTrue(self.rm.check_output("base", inp, {"loss": Q(1, 3)}))
        self.assertFalse(self.rm.check_output("base", inp, {"loss": 1 / 3}))

    def test_registry_keeps_comparison_relative_to_selected_task(self):
        # Isolate comparison from existing implementation-approval records.
        with patch.object(registry, "activation", return_value=(True, [], {"scope": "test task"})):
            self.assertEqual(registry.check_output("ADAMW", "base", self.inp, self.out)[0], "consistent")
            self.assertEqual(registry.check_output("ADAMW", "base", self.inp, {"theta": self.out["theta"]})[0],
                             "contradiction")


if __name__ == "__main__":
    unittest.main()
