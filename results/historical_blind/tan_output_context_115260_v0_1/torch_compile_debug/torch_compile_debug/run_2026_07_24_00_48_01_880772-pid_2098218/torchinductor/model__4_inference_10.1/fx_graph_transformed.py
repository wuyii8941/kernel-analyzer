class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: "f16[17, 5, 1, 7]", arg1_1: "f16[17, 5, 1, 7]", arg2_1: "f16[17, 5, 11, 7]", arg3_1: "f16[17, 5, 1, 7]", arg4_1: "f16[]", arg5_1: "f16[1]"):
        # File: /data1/tzh/forkcert/theory_oracle/capture_tan_context_inductor_provenance_v0_1.py:39, code: cat = torch.cat((values[3], values[2], values[1], values[0]), dim=2)
        cat: "f16[17, 5, 14, 7]" = torch.ops.aten.cat.default([arg3_1, arg2_1, arg1_1, arg0_1], 2);  arg3_1 = arg2_1 = arg1_1 = arg0_1 = None
        
        # File: /data1/tzh/forkcert/theory_oracle/capture_tan_context_inductor_provenance_v0_1.py:40, code: mul = torch.mul(cat, torch.max(values[4], p0))
        maximum: "f16[1]" = torch.ops.aten.maximum.default(arg4_1, arg5_1);  arg4_1 = arg5_1 = None
        mul: "f16[17, 5, 14, 7]" = torch.ops.aten.mul.Tensor(cat, maximum);  cat = maximum = None
        
        # File: /data1/tzh/forkcert/theory_oracle/capture_tan_context_inductor_provenance_v0_1.py:41, code: return mul, torch.tan(mul)
        tan: "f16[17, 5, 14, 7]" = torch.ops.aten.tan.default(mul)
        return (mul, tan)
        