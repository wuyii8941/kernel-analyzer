import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "corpus"))
from region_source import check_launch_region, check_local_temporary


class RegionTemporary(unittest.TestCase):
    def setUp(self):
        self.source = '''def call():
    temp = empty_strided_cuda((4,), (1,), torch.float32)
    a.run(x, temp, stream=s)
    alias = temp
    del temp
    b.run(alias, out, stream=s)
    del alias
    return (out,)
'''

    def check(self, source):
        region = check_launch_region(source, ["a", "b"])
        return check_local_temporary(source, region, ["temp", "alias"])

    def test_local_allocation_aliases_and_deletions_are_checked(self):
        result = self.check(self.source)
        self.assertEqual(result["status"], "wrapper local temporary checked")
        self.assertEqual(result["aliases"], ["alias", "temp"])
        self.assertEqual(len(result["events"]), 6)
        self.assertEqual(result["binding_status"], "not_established")

    def test_visible_or_unclosed_lifetimes_are_refused(self):
        sources = [self.source.replace("    del alias\n", ""),
                   self.source.replace("return (out,)", "return (out, alias)"),
                   self.source.replace("    del alias", "    observer(alias)\n    del alias"),
                   self.source.replace("    del alias", "    escaped = alias\n    del alias"),
                   self.source.replace("temp = empty_strided_cuda((4,), (1,), torch.float32)", "temp = x"),
                   self.source.replace("    del alias", "    alias = x\n    del alias")]
        for source in sources:
            with self.subTest(source=source):
                self.assertEqual(self.check(source)["status"], "not_established")


if __name__ == "__main__":
    unittest.main()
