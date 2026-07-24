class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: "f32[]", arg1_1: "f32[]"):
        # File: /data1/tzh/forkcert/theory_oracle/capture_fma_context_inductor_provenance_v0_1.py:30, code: max_scaled = x * scale
        mul: "f32[]" = torch.ops.aten.mul.Tensor(arg0_1, arg1_1)
        
        # File: /data1/tzh/forkcert/theory_oracle/capture_fma_context_inductor_provenance_v0_1.py:31, code: return torch.exp(max_scaled - x * scale)
        mul_1: "f32[]" = torch.ops.aten.mul.Tensor(arg0_1, arg1_1);  arg0_1 = arg1_1 = None
        sub: "f32[]" = torch.ops.aten.sub.Tensor(mul, mul_1);  mul = mul_1 = None
        exp: "f32[]" = torch.ops.aten.exp.default(sub);  sub = None
        return (exp,)
        