"""Call bindings for torch.sparse._triton_ops (hand-written Triton kernels shipped with torch; never integrated in this
project before the general-capability round).  Only legal calls and input construction -- no reference, no kernel
knowledge.  The block-sparse pattern is an input source (``block_sparse``)."""
import torch
import torch.sparse._triton_ops as T


def block_sparse(seed, shape, dtype, block=16, density=0.5):
    """values drawn per unit; the block pattern is structure (fixed by the shape), so every unit has the same
    coordinate frame."""
    g = torch.Generator().manual_seed(10_000 + seed)
    x = torch.randn(shape, generator=g, dtype=torch.float64)
    gp = torch.Generator().manual_seed(shape[0] * 1000 + shape[1])
    keep = torch.rand((shape[0] // block, shape[1] // block), generator=gp) < density
    keep[:, 0] = True                                     # every block row has a block (softmax rows are defined)
    mask = keep.repeat_interleave(block, 0).repeat_interleave(block, 1)
    return (x * mask).to(getattr(torch, dtype)).cuda()


def bsr_softmax(inp):
    xb = inp["x"].to_sparse_bsr(inp["block"])
    return {"values": T.bsr_softmax(xb).values()}


def bsr_dense_mm(inp):
    xb = inp["x"].to_sparse_bsr(inp["block"])
    return {"out": T.bsr_dense_mm(xb, inp["d"])}


def sampled_addmm(inp):
    mb = inp["m"].to_sparse_bsr(inp["block"])
    return {"values": T.sampled_addmm(mb, inp["a"], inp["b"], alpha=1.0, beta=0.0).values()}
