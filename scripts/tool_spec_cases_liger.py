"""Liger RoPE cases for tool_spec_check.py (environment: latest_train; Liger release or main via PYTHONPATH)."""

from __future__ import annotations

import numpy as np
import torch

from tool_spec_check import Case, f64, iv


class LigerRopeCase(Case):
    """Liger's RoPE kernel (``liger_rotary_pos_emb``, the function Liger patches into the HF model) on q."""

    B, HQ, HK, T = 1, 4, 2, 64

    def launch(self, inp):
        from liger_kernel.transformers.rope import liger_rotary_pos_emb

        q_out, _ = liger_rotary_pos_emb(inp["q"].clone(), inp["k"].clone(), inp["cos"], inp["sin"])
        return {"q": q_out}


class LigerRopeGptOss(LigerRopeCase):
    name = "liger_rope_gpt_oss"
    implementation = "liger_kernel.transformers.rope.liger_rotary_pos_emb (patched into modeling_gpt_oss)"
    specification = ("transformers.models.gpt_oss.modeling_gpt_oss.apply_rotary_pos_emb: cos/sin of width d/2; "
                     "out[:d/2] = x1 cos - x2 sin, out[d/2:] = x2 cos + x1 sin with x1 = x[:d/2], x2 = x[d/2:]")
    D = 64

    def inputs(self, seed):
        from transformers.models.gpt_oss.configuration_gpt_oss import GptOssConfig
        from transformers.models.gpt_oss.modeling_gpt_oss import GptOssRotaryEmbedding

        g = torch.Generator(device="cpu").manual_seed(seed)
        q = torch.randn(self.B, self.HQ, self.T, self.D, generator=g).cuda()
        k = torch.randn(self.B, self.HK, self.T, self.D, generator=g).cuda()
        start = int(torch.randint(0, 4096, (1,), generator=g))
        pos = torch.arange(start, start + self.T)[None].cuda()
        cos, sin = GptOssRotaryEmbedding(GptOssConfig(head_dim=self.D)).cuda()(q, pos)
        return {"q": q, "k": k, "cos": cos, "sin": sin}

    def spec(self, inp):
        x, c, s = f64(inp["q"]), f64(inp["cos"])[:, None], f64(inp["sin"])[:, None]
        h = x.shape[-1] // 2
        x1, x2 = x[..., :h], x[..., h:]
        a, b = x1 * c, x2 * s          # exact
        first = iv.add_bounds(a, -b)
        a, b = x2 * c, x1 * s
        second = iv.add_bounds(a, b)
        return {"q": (np.concatenate([first[0], second[0]], -1), np.concatenate([first[1], second[1]], -1))}


class LigerRopePartial(LigerRopeCase):
    name = "liger_rope_phi3_partial"
    implementation = "liger_kernel.transformers.rope.liger_rotary_pos_emb (patched into modeling_phi3)"
    specification = ("transformers.models.phi3.modeling_phi3.apply_rotary_pos_emb with partial_rotary_factor 0.75: "
                     "rotary_dim r = cos.shape[-1]; out[:r] = x_r cos + rotate_half(x_r) sin, out[r:] = x[r:]")
    D = 64

    def inputs(self, seed):
        from transformers.models.phi3.configuration_phi3 import Phi3Config
        from transformers.models.phi3.modeling_phi3 import Phi3RotaryEmbedding

        g = torch.Generator(device="cpu").manual_seed(seed)
        q = torch.randn(self.B, self.HQ, self.T, self.D, generator=g).cuda()
        k = torch.randn(self.B, self.HK, self.T, self.D, generator=g).cuda()
        start = int(torch.randint(0, 4096, (1,), generator=g))
        pos = torch.arange(start, start + self.T)[None].cuda()
        cfg = Phi3Config(hidden_size=self.D * self.HQ, num_attention_heads=self.HQ, num_key_value_heads=self.HK,
                         partial_rotary_factor=0.75)
        cos, sin = Phi3RotaryEmbedding(cfg).cuda()(q, pos)
        return {"q": q, "k": k, "cos": cos, "sin": sin}

    def spec(self, inp):
        x, c, s = f64(inp["q"]), f64(inp["cos"])[:, None], f64(inp["sin"])[:, None]
        r = c.shape[-1]
        xr, xp = x[..., :r], x[..., r:]
        rot = np.concatenate([-xr[..., r // 2:], xr[..., :r // 2]], -1)
        lo, hi = iv.add_bounds(xr * c, rot * s)
        return {"q": (np.concatenate([lo, xp], -1), np.concatenate([hi, xp], -1))}


CASES = [LigerRopeGptOss(), LigerRopePartial()]


