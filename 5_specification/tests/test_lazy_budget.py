import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kernel_spec import lazy
from kernel_spec.num import Invalid, NotEstablished


class LazyBudget(unittest.TestCase):
    def test_dependencies_consume_budget_and_exhaustion_does_not_cache_an_answer(self):
        source = lazy.Lazy((3,), "i64", lambda i: i[0])
        total = lazy.Lazy((), "i64", lambda _: sum(source.get((i,)) for i in range(3)))
        with lazy.evaluation_budget(3) as budget:
            with self.assertRaises(NotEstablished):
                total.get(())
        self.assertEqual(budget.used, 3)
        self.assertNotIn((), total._memo)
        with lazy.evaluation_budget(2) as resumed:
            self.assertEqual(total.get(()), 3)
            self.assertEqual(total.get(()), 3)
        self.assertEqual(resumed.used, 2)

    def test_nested_budgets_cannot_reset_outer_limit(self):
        source = lazy.Lazy((3,), "i64", lambda i: i[0])
        with lazy.evaluation_budget(1):
            with lazy.evaluation_budget(10):
                source.get((0,))
                with self.assertRaises(NotEstablished):
                    source.get((1,))
        self.assertEqual(source.get((1,)), 1)

    def test_invalid_limits_are_not_coerced(self):
        for limit in (True, 1.5, 0, -1):
            with self.assertRaises(Invalid):
                with lazy.evaluation_budget(limit):
                    pass


if __name__ == "__main__":
    unittest.main()
