"""Binding must use recorded origins and account for writes, not shape/name guesses."""
import copy
import importlib.util
import itertools
import sys
import unittest
import numpy as np
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "corpus"))
import dynamic_three_way as D


class CompleteBinding(unittest.TestCase):
    def test_write_scope_rejects_missing_storage_and_wrong_program_identity(self):
        arg = SimpleNamespace(kind="tensor", name="out", storage_ptr=100)
        launch = SimpleNamespace(args=[arg], implicit=[], grid=(1, 1, 1))
        ref = SimpleNamespace(aborted={}, reasons=set(), notes=[], programs=[(0, 0, 0)], buffers={})
        self.assertIn("every captured storage", D.inspect_reference_writes(launch, {"out"}, ref)["note"])
        for programs in ([(1, 0, 0)], [(0, 0, 0), (0, 0, 0)], [(False, 0, 0)]):
            ref.programs = programs
            self.assertIn("complete grid", D.inspect_reference_writes(launch, {"out"}, ref)["note"])

    def test_ir_detects_unchanged_value_write_outside_output(self):
        import torch
        from kernel_analyzer.reference_eval.capture import CapturedArg, CapturedLaunch
        def launch(offset):
            arg = CapturedArg(0, "out", "tensor", False, "*i32", dtype="int32", shape=(1,), stride=(1,),
                              data_ptr=100, storage_ptr=100, element_size=4, storage_nbytes=12,
                              before=torch.zeros(12, dtype=torch.uint8), after=torch.zeros(12, dtype=torch.uint8))
            ir = '''module {
  tt.func public @write(%arg0: !tt.ptr<i32>) {
    %zero = arith.constant 0 : i32
    %offset = arith.constant OFFSET : i32
    %ptr = tt.addptr %arg0, %offset : !tt.ptr<i32>, i32
    tt.store %ptr, %zero : !tt.ptr<i32>
    tt.return
  }
}'''.replace("OFFSET", str(offset))
            return CapturedLaunch(0, "write", "", (1, 1, 1), [arg], {"ttir": ir}, None, {}, None)
        inside, outside = launch(0), launch(1)
        self.assertIsNone(D.check_storage_changes(outside.args[0], True))  # all bytes still zero
        self.assertEqual(D.reference_write_scope(inside, {"out"})["outcome"], "checked")
        bad = D.reference_write_scope(outside, {"out"})
        self.assertEqual(bad["outcome"], "not_established")
        self.assertIn("outside", bad["note"])
        self.assertEqual(D.reference_write_scope(inside, set())["outcome"], "not_established")
        self.assertEqual(D.reference_write_scope(inside, {"out"}, max_programs=0)["outcome"], "not_established")
        self.assertEqual(D.reference_write_scope(launch(4), {"out"})["outcome"], "not_established")

    def test_changes_outside_declared_outputs_are_rejected(self):
        arg = SimpleNamespace(shape=(2,), stride=(2,), element_size=2, data_ptr=102, storage_ptr=100,
                              storage_nbytes=10, before=np.zeros(10, dtype=np.uint8),
                              after=np.zeros(10, dtype=np.uint8), window=None)
        arg.after[[2, 3, 6, 7]] = 1
        self.assertIsNone(D.check_storage_changes(arg, writable=True))
        self.assertIsNotNone(D.check_storage_changes(arg, writable=False))
        for i in (0, 1, 4, 5, 8, 9):
            with self.subTest(byte=i):
                arg.after[i] = 1
                self.assertIsNotNone(D.check_storage_changes(arg, writable=True))
                arg.after[i] = 0

    def test_storage_check_transpose_empty_and_incomplete(self):
        arg = SimpleNamespace(shape=(2, 3), stride=(1, 2), element_size=1, data_ptr=100, storage_ptr=100,
                              storage_nbytes=6, before=np.zeros(6, dtype=np.uint8),
                              after=np.ones(6, dtype=np.uint8), window=None)
        self.assertIsNone(D.check_storage_changes(arg, writable=True))
        for updates in ({"shape": (0,), "stride": (1,)}, {"after": None}, {"window": [0]},
                        {"stride": (1, 1)}, {"storage_nbytes": 7}):
            changed = SimpleNamespace(**(vars(arg) | updates))
            self.assertIsNotNone(D.check_storage_changes(changed, writable=True))

    def test_storage_membership_matches_explicit_coordinate_enumeration(self):
        # Independent enumeration includes gap bytes, both bytes of each element,
        # transposition, singleton axes and storage before/after the view.
        for shape, stride in (((2, 3), (7, 2)), ((3, 2), (2, 7)), ((1, 3), (0, 2)), ((), ())):
            addresses = {2 + 2 * sum(i * s for i, s in zip(index, stride)) + byte
                         for index in itertools.product(*(range(n) for n in shape)) for byte in range(2)}
            length = max(addresses) + 3
            arg = SimpleNamespace(shape=shape, stride=stride, element_size=2, data_ptr=102, storage_ptr=100,
                                  storage_nbytes=length, before=np.zeros(length, dtype=np.uint8),
                                  after=np.zeros(length, dtype=np.uint8), window=None)
            for byte in range(length):
                arg.after[byte] = 1
                with self.subTest(shape=shape, byte=byte):
                    self.assertEqual(D.check_storage_changes(arg, True) is None, byte in addresses)
                arg.after[byte] = 0

    def test_capture_requires_execution_and_all_memory(self):
        launch = SimpleNamespace(executed=True, implicit_recorded=True, implicit=[])
        self.assertIsNone(D.capture_task_state(launch))
        for changes in ({"executed": False}, {"executed": None}, {"implicit_recorded": False},
                        {"implicit": [object()]}):
            bad = SimpleNamespace(**(vars(launch) | changes))
            self.assertIsNotNone(D.capture_task_state(bad))

    def test_extra_grid_needs_source_metadata_and_captured_grid(self):
        launch = SimpleNamespace(args=[SimpleNamespace(name="x", kind="tensor", constexpr=False)], grid=(48, 1, 1))
        meta = D.launcher_metadata("@template(inductor_meta={'grid_type': 'FixedGrid', "
                                  "'fixed_grid': ['g0','g1','g2'], 'extra_launcher_args': ['g0','g1','g2']})\n"
                                  "def kernel(x): pass")
        args = ["buf", "48", "1", "1"]
        self.assertEqual(D.bind_call_scalars(["x"], args, launch, meta), ({"g0": 48, "g1": 1, "g2": 1}, None))
        for changed in ({}, dict(meta, fixed_grid=["g1", "g0", "g2"]), dict(meta, grid_type="Other")):
            self.assertIsNotNone(D.bind_call_scalars(["x"], args, launch, changed)[1])
        self.assertIsNotNone(D.bind_call_scalars(["x"], ["buf", "49", "1", "1"], launch, meta)[1])
        self.assertEqual(D.launcher_metadata("@template(inductor_meta=make_metadata())\ndef k(): pass"), {})

    def test_call_scalars_require_exact_captured_values(self):
        launch = SimpleNamespace(args=[SimpleNamespace(name="x", kind="tensor", constexpr=False),
                                      SimpleNamespace(name="n", kind="int", constexpr=False, value=128),
                                      SimpleNamespace(name="BLOCK", kind="int", constexpr=True, value=32)])
        params = ["x", "n", "BLOCK"]
        values, why = D.bind_call_scalars(params, ["buf0", "128"], launch)
        self.assertIsNone(why)
        self.assertEqual(values, {"n": 128})
        for args in (["buf0", "129"], ["buf0", "True"], ["buf0", "size"], ["buf0"],
                     ["buf0", "128", "32", "1"]):
            with self.subTest(args=args):
                self.assertIsNotNone(D.bind_call_scalars(params, args, launch)[1])

    def test_call_signature_cannot_omit_or_duplicate_captured_parameters(self):
        launch = SimpleNamespace(args=[SimpleNamespace(name="n", kind="int", constexpr=False, value=2)])
        for params, args in ((["m"], ["2"]), (["n", "n"], ["2", "2"]), ([], [])):
            self.assertIsNotNone(D.bind_call_scalars(params, args, launch)[1])

    def test_scalar_sign_and_integer_precision_are_preserved(self):
        for kind, value, good, bad in (("float", -0.0, "-0.0", "0.0"),
                                       ("int", 2**53 + 1, str(2**53 + 1), str(2**53))):
            launch = SimpleNamespace(args=[SimpleNamespace(name="s", kind=kind, constexpr=False, value=value)])
            self.assertIsNone(D.bind_call_scalars(["s"], [good], launch)[1])
            self.assertIsNotNone(D.bind_call_scalars(["s"], [bad], launch)[1])

    def setUp(self):
        self.frag = [dict(name="x", kind="placeholder", meta='Tensor "f32[2]"'),
                     dict(name="neg", kind="call", args=[{"ref": "x"}], kwargs={})]
        self.prov = {"params": {
            "in_ptr0": dict(role="input", buffer="arg0", origins=["x"]),
            "out_ptr0": dict(role="output", buffer="buf0", origins=["neg"])}}
        self.buffers = {"in_ptr0": "arg0", "out_ptr0": "buf0"}

    def bind(self, writes=("out_ptr0",)):
        return D.bind_from_provenance(self.prov, self.frag, self.buffers, written_params=set(writes))

    def test_complete_record(self):
        self.assertEqual(self.bind(), ({"x": "arg0"}, {"out_ptr0": "neg"}))

    def test_output_type_and_shape_need_explicit_correspondence(self):
        arg = SimpleNamespace(dtype="float32", shape=(2,))
        self.assertTrue(D.tensor_corresponds({"meta": 'Tensor "f32[2]"'}, arg))
        for meta in (None, 'Tensor "f32[1,2]"', 'Tensor "bf16[2]"', 'Tuple[Tensor, Tensor]'):
            with self.subTest(meta=meta):
                self.assertFalse(D.tensor_corresponds({"meta": meta}, arg))

    @unittest.skipUnless(importlib.util.find_spec("gmpy2"), "IR integration needs the tool environment")
    def test_ir_write_parameter_mapping(self):
        launch = SimpleNamespace(asm={"ttir": '''
module {
  tt.func public @copy(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>) {
    %0 = tt.load %arg0 : !tt.ptr<f32>
    tt.store %arg1, %0 : !tt.ptr<f32>
    tt.return
  }
}
'''}, ttir_params=lambda: [SimpleNamespace(name="input"), SimpleNamespace(name="output")])
        self.assertEqual(D.write_parameters(launch), {"output"})

    def test_missing_write_buffer_refused(self):
        del self.buffers["out_ptr0"]
        self.assertIsNone(self.bind()[0])

    def test_unrecorded_write_refused(self):
        self.buffers["out_ptr1"] = "buf1"
        self.assertIsNone(self.bind(("out_ptr0", "out_ptr1"))[0])

    def test_inplace_state_without_origin_refused(self):
        self.prov["params"]["in_out_ptr0"] = dict(role="inplace", buffer="state", origins=[])
        self.buffers["in_out_ptr0"] = "state"
        self.assertIsNone(self.bind(("out_ptr0", "in_out_ptr0"))[0])

    def test_missing_pre_state_refused(self):
        self.frag.append(dict(name="old_state", kind="placeholder", meta='Tensor "f32[2]"'))
        self.assertIsNone(self.bind()[0])

    def test_no_write_evidence_refused(self):
        self.assertIsNone(D.bind_from_provenance(self.prov, self.frag, self.buffers)[0])

    def test_no_outputs_is_not_a_complete_task(self):
        del self.prov["params"]["out_ptr0"]
        del self.buffers["out_ptr0"]
        self.assertIsNone(self.bind(())[0])

    def test_repeated_call_is_ambiguous(self):
        oc = SimpleNamespace(calls=[("kernel", ["a", "b"]), ("kernel", ["c", "d"])], params={"kernel": ["in_ptr0", "out_ptr0"]})
        rec = {"params": {"in_ptr0": {"buffer": "a"}, "out_ptr0": {"buffer": "b"}}}
        self.assertIsNone(D.select_call(oc, "kernel", [rec])[0])          # one record for two call sites
        other = {"params": {"in_ptr0": {"buffer": "c"}, "out_ptr0": {"buffer": "x"}}}
        self.assertIsNone(D.select_call(oc, "kernel", [rec, other])[0])   # second call site not recorded

    def test_repeated_call_with_recorded_order_binds_first_launch(self):
        # Inductor reuses one kernel for identical source; the capture holds the first launch
        oc = SimpleNamespace(calls=[("kernel", ["a", "b"]), ("kernel", ["c", "d"])], params={"kernel": ["in_ptr0", "out_ptr0"]})
        first = {"params": {"in_ptr0": {"buffer": "a"}, "out_ptr0": {"buffer": "b"}}}
        second = {"params": {"in_ptr0": {"buffer": "c"}, "out_ptr0": {"buffer": "d"}}}
        self.assertEqual(D.select_call(oc, "kernel", [first, second]), (["a", "b"], first))
        self.assertIsNone(D.select_call(oc, "kernel", [second, first])[0])  # order must agree

    def test_origin_node_and_inplace_alias_bind_exactly(self):
        self.prov["params"]["in_ptr0"] = dict(role="input", buffer="buf7", origins=["sdpa", "view_1"], origin_node="getitem_3")
        self.frag[0]["name"] = "getitem_3"
        self.frag[1]["args"] = [{"ref": "getitem_3"}]
        self.buffers["in_ptr0"] = "buf7"
        self.assertEqual(self.bind()[0], {"getitem_3": "buf7"})
        self.assertEqual(D.input_buffers_of("full_default", {"in_out_ptr0": dict(
            role="inplace", buffer="buf9", origins=["index_put_1"], other_names=["full_default", "buf9"])},
            {"in_out_ptr0": "buf9"}), {"buf9"})

    def test_fragment_cut_at_recorded_input_value(self):
        # the fragment recomputes mul from y, but the kernel receives mul's value (origin_node) in arg0
        frag = [dict(name="y", kind="placeholder", meta='Tensor "f32[2]"'),
                dict(name="mul", kind="call", args=[{"ref": "y"}, {"ref": "y"}], kwargs={}, meta='Tensor "f32[2]"'),
                dict(name="neg", kind="call", args=[{"ref": "mul"}], kwargs={})]
        prov = {"params": {"in_ptr0": dict(role="input", buffer="arg0", origins=["mul"], origin_node="mul"),
                           "out_ptr0": dict(role="output", buffer="buf0", origins=["neg"], origin_node="neg")}}
        self.assertEqual(D.bind_from_provenance(prov, frag, self.buffers, {"out_ptr0"}),
                         ({"mul": "arg0"}, {"out_ptr0": "neg"}))
        self.assertEqual([n["name"] for n in D.cut_fragment(frag, prov["params"], {"out_ptr0": "neg"})], ["mul", "neg"])

    def test_cut_keeps_nodes_with_another_path(self):
        # y also feeds the output directly, so it stays and must be bound; nothing records it here
        frag = [dict(name="y", kind="placeholder", meta='Tensor "f32[2]"'),
                dict(name="mul", kind="call", args=[{"ref": "y"}, {"ref": "y"}], kwargs={}),
                dict(name="add", kind="call", args=[{"ref": "mul"}, {"ref": "y"}], kwargs={})]
        prov = {"params": {"in_ptr0": dict(role="input", buffer="arg0", origins=["mul"], origin_node="mul"),
                           "out_ptr0": dict(role="output", buffer="buf0", origins=["add"], origin_node="add")}}
        self.assertIsNone(D.bind_from_provenance(prov, frag, self.buffers, {"out_ptr0"})[0])

    def test_placeholder_line_for_computed_node_is_internal(self):
        self.frag.insert(0, dict(name="neg", kind="placeholder", meta='Tensor "f32[2]"'))
        self.assertEqual(self.bind(), ({"x": "arg0"}, {"out_ptr0": "neg"}))

    def test_unique_shape_does_not_establish_input_origin(self):
        oc = object.__new__(D.OutputCode)
        oc.frag = {"kernel": copy.deepcopy(self.frag)}
        oc.params = {"kernel": ["in_ptr0", "out_ptr0"]}
        oc.bufshape = {"arg0": ("f32", (2,)), "buf0": ("f32", (2,))}
        self.assertEqual(oc.bind_inputs("kernel", ["arg0", "buf0"]), {"x": None})


if __name__ == "__main__":
    unittest.main()
