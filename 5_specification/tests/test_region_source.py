import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "corpus"))
from region_source import check_launch_region, map_return_values


class RegionSource(unittest.TestCase):
    def test_explicit_absent_return_is_preserved_and_must_agree(self):
        source = 'def call():\n    a.run(x, out, 4, stream=s)\n    return (None, out)\n'
        graph = {"nodes": [{"name": "gradient", "op": "call_function"},
                            {"name": "output", "op": "output", "expression": "return (None, gradient)"}]}
        region = check_launch_region(source, ["a"])
        mapping = map_return_values(source, graph, region)
        self.assertEqual(mapping[0], {"position": 0, "constant": None})
        self.assertEqual(mapping[1]["position"], 1)
        self.assertEqual(mapping[1]["node"], "gradient")
        for bad in (source.replace('(None, out)', '(out, out)'),
                    source.replace('(None, out)', '(None, None)'),
                    source.replace('(None, out)', '(0, out)')):
            with self.assertRaises(ValueError):
                map_return_values(bad, graph, region)

    def setUp(self):
        self.source = '''def call():
    a.run(x, tmp, 4, stream=s)
    tmp2 = empty_strided_cuda((4,), (1,), torch.float32)
    old = tmp
    del tmp
    s = get_raw_stream(0)
    b.run(old, tmp2, 4, stream=s)
'''

    def test_explicit_region_preserves_order_arguments_and_all_intervening_statements(self):
        result = check_launch_region(self.source, ["a", "b"])
        self.assertEqual([x["kernel"] for x in result["calls"]], ["a", "b"])
        self.assertEqual(result["calls"][1]["args"], ["old", "tmp2", "4"])
        self.assertEqual(len(result["housekeeping"]), 4)
        self.assertEqual(result["semantic_verdict"], "not_established")

    def test_hidden_calls_mutations_and_control_flow_are_refused(self):
        for statement in ("extern.mm(x, old)", "c.run(x, old)", "old.copy_(x)",
                          "if flag:\n        c.run(x, old)", "x = helper()",
                          "x = reinterpret_tensor(helper(), (4,), (1,), 0)"):
            source = self.source.replace("    old = tmp", "    " + statement + "\n    old = tmp")
            self.assertEqual(check_launch_region(source, ["a", "b"])["status"], "not_established")

    def test_missing_duplicate_reordered_or_evaluated_launches_are_refused(self):
        for source, names in ((self.source, ["b", "a"]), (self.source, ["a", "missing"]),
                              (self.source + "    a.run(x)\n", ["a", "b"]),
                              (self.source.replace("b.run(old", "b.run(helper()"), ["a", "b"]),
                              (self.source, ["a", "a"])):
            self.assertEqual(check_launch_region(source, names)["status"], "not_established")

    def test_return_positions_preserve_tuple_identity_views_and_duplicates(self):
        source = self.source + "    return (tmp2, reinterpret_tensor(old, (4,), (1,), 0), tmp2)\n"
        graph = {"nodes": [{"name": "mean", "op": "call_function"}, {"name": "var", "op": "call_function"},
                            {"name": "output", "op": "output", "expression": "return (mean, var, mean)"}]}
        region = check_launch_region(source, ["a", "b"])
        mapping = map_return_values(source, graph, region)
        self.assertEqual([r["node"] for r in mapping], ["mean", "var", "mean"])
        self.assertEqual(mapping[1]["view"], {"shape": [4], "stride": [1], "offset": 0})
        for bad in (source.replace(", tmp2)", ")"), source.replace("(1,), 0)", "(1,), 1)"),
                    source.replace("return (tmp2,", "return (unknown(),"),
                    source.replace("    return", "    tmp2 = other\n    return"),
                    source.replace("    return", "    c.run(tmp2)\n    return"),
                    source + "    return (tmp2,)\n"):
            with self.assertRaises((ValueError, SyntaxError)):
                map_return_values(bad, graph, region)


if __name__ == "__main__":
    unittest.main()
