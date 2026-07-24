class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: "f32[4, 257]"):
        # File: /data1/tzh/forkcert/theory_oracle/kernel_plumbing_calibration_v0_1.py:48 in program, code: return value.reshape(value.shape[0], -1).sum(dim=-1)
        sum_1: "f32[4]" = torch.ops.aten.sum.dim_IntList(arg0_1, [-1]);  arg0_1 = None
        return (sum_1,)
