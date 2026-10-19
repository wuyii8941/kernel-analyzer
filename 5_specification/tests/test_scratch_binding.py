import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "corpus"))
import dynamic_three_way as D


class ScratchBinding(unittest.TestCase):
    def launch(self, body):
        return SimpleNamespace(asm={"ttir": '''module {
  tt.func public @kernel(%p: !tt.ptr<f32>, %v: f32) {
''' + body + '''
    tt.return
  }
}'''}, ttir_params=lambda: [SimpleNamespace(name="out"), SimpleNamespace(name="value")])

    def test_store_only_pointer_is_proven(self):
        self.assertEqual(D.write_only_parameters(self.launch("    tt.store %p, %v : !tt.ptr<f32>")), {"out"})
        prefix = "    %i = arith.constant 1 : i32\n    %q = tt.addptr %p, %i : !tt.ptr<f32>, i32\n"
        self.assertEqual(D.write_only_parameters(self.launch(prefix + "    tt.store %q, %v : !tt.ptr<f32>")), {"out"})
        self.assertEqual(D.write_only_parameters(self.launch(prefix + "    %r = tt.load %q : !tt.ptr<f32>\n"
                                                             "    tt.store %p, %v : !tt.ptr<f32>")), set())

    def test_load_atomic_escape_and_unused_pointer_are_not_proven(self):
        for body in ("    %x = tt.load %p : !tt.ptr<f32>\n    tt.store %p, %v : !tt.ptr<f32>",
                     "    %x = tt.atomic_rmw fadd, relaxed, gpu, %p, %v : (!tt.ptr<f32>, f32) -> f32",
                     "    tt.call @other(%p) : (!tt.ptr<f32>) -> ()",
                     ""):
            with self.subTest(body=body):
                self.assertEqual(D.write_only_parameters(self.launch(body)), set())

    def test_only_unused_synthetic_alias_can_be_removed(self):
        prov = {"params": {"out": {"role": "inplace", "other_names": ["scratch", "state", "result"],
                                    "other_origin_nodes": {"scratch": None, "state": "pre", "result": "post"}}}}
        frag = [{"name": "scratch", "kind": "placeholder"}, {"name": "state", "kind": "placeholder"}]
        new, metadata, removed = D.remove_unused_scratch(frag, prov, {"out"})
        self.assertEqual(removed, ["scratch"])
        self.assertEqual([n["name"] for n in new], ["state"])
        self.assertEqual(metadata["params"]["out"]["role"], "output")
        self.assertEqual(prov["params"]["out"]["role"], "inplace")
        self.assertEqual(D.remove_unused_scratch(frag, prov, set())[0], frag)
        used = frag + [{"name": "post", "kind": "call", "args": [{"ref": "scratch"}], "kwargs": {}}]
        self.assertEqual(D.remove_unused_scratch(used, prov, {"out"})[0], used)


if __name__ == "__main__":
    unittest.main()
