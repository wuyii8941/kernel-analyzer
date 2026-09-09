"""Independent same-input reference for a pure embedding lookup."""


def snapshot_pre_call(pointers, *, tokens, width, vocabulary_size):
    import torch

    if (type(tokens) is not int or type(width) is not int
            or type(vocabulary_size) is not int or min(tokens, width, vocabulary_size) <= 0):
        raise ValueError("Invalid embedding dimensions")
    layout = {
        "in_ptr0": (torch.int64, tokens),
        "in_ptr1": (torch.bfloat16, vocabulary_size * width),
        "out_ptr0": (torch.bfloat16, tokens * width),
    }
    if set(pointers) != set(layout):
        raise ValueError("Embedding pointer ABI differs")
    identities = []
    for name, (dtype, size) in layout.items():
        value = pointers[name]
        if (not isinstance(value, torch.Tensor) or value.dtype != dtype
                or value.numel() != size or not value.is_contiguous()):
            raise ValueError("Embedding storage differs: " + name)
        identities.append((value.device, value.untyped_storage().data_ptr()))
    if len({value.device for value in pointers.values()}) != 1:
        raise ValueError("Mixed embedding devices")
    if len(set(identities)) != len(identities):
        raise ValueError("Aliased embedding arguments require a separate contract")
    indices = pointers["in_ptr0"]
    if not bool(((indices >= 0) & (indices < vocabulary_size)).all()):
        raise ValueError("Embedding index outside the declared vocabulary")
    return {name: pointers[name].detach().clone() for name in ("in_ptr0", "in_ptr1")}


def select_output(pointers, candidate, *, tokens, width, vocabulary_size):
    import torch

    if set(pointers) != {"in_ptr0", "in_ptr1"}:
        raise ValueError("Expected preserved embedding inputs only")
    indices = pointers["in_ptr0"]
    weight = pointers["in_ptr1"]
    if (indices.dtype != torch.int64 or indices.numel() != tokens or not indices.is_contiguous()
            or weight.dtype != torch.bfloat16
            or weight.numel() != vocabulary_size * width or not weight.is_contiguous()):
        raise ValueError("Preserved embedding input layout differs")
    if not bool(((indices >= 0) & (indices < vocabulary_size)).all()):
        raise ValueError("Embedding index outside the declared vocabulary")
    expected = weight.reshape(vocabulary_size, width).index_select(0, indices.reshape(-1))
    if (not isinstance(candidate, torch.Tensor) or candidate.dtype != torch.bfloat16
            or candidate.device != expected.device or candidate.numel() != tokens * width
            or not candidate.is_contiguous()):
        raise ValueError("Embedding output storage differs")
    return expected.reshape(candidate.shape)
