class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: "f32[1, 1, 2]", arg1_1: "f32[1, 1, 10, 10]"):
         # File: /data1/tzh/forkcert/theory_oracle/capture_fractional_maxpool_local_evidence_v0_1.py:42 in forward, code: return F.fractional_max_pool2d(
        fractional_max_pool2d = torch.ops.aten.fractional_max_pool2d.default(arg1_1, [1, 1], [5, 5], arg0_1);  arg1_1 = arg0_1 = None
        getitem: "f32[1, 1, 5, 5]" = fractional_max_pool2d[0];  fractional_max_pool2d = None
        return (getitem,)
        