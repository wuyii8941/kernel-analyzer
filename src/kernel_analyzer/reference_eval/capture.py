"""Capture real Triton launches: compiled artifacts, configuration and operands.

The recorder wraps the launcher returned by ``CompiledKernel.run``, which
every Triton launch goes through (``@triton.jit`` kernels and TorchInductor
kernels when Inductor's static CUDA launcher is disabled).  For each selected
launch it stores:

* the IR / configuration / final-artifact correspondence of *that*
  compilation (``asm['ttir']``, ``asm['ttgir']``, ``asm['ptx']``, a hash of
  the cubin, the compile metadata);
* the grid, the full argument list and which arguments are constexpr;
* copies of every tensor operand before and after the launch, together with
  the storage identity needed to rebuild aliasing in the reference memory.

Recompiling the TTIR is never used to establish this correspondence.
"""

from __future__ import annotations

import contextlib
import dataclasses
import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

ASM_KEYS = ("ttir", "ttgir", "llir", "ptx")


@dataclass
class CapturedArg:
    index: int
    name: str
    kind: str  # "tensor" | "int" | "float" | "bool" | "none" | "other"
    constexpr: bool
    signature_type: Optional[str]
    value: Any = None  # scalar value
    dtype: Optional[str] = None
    shape: Optional[tuple] = None
    stride: Optional[tuple] = None
    element_size: Optional[int] = None
    data_ptr: Optional[int] = None
    storage_ptr: Optional[int] = None
    storage_nbytes: Optional[int] = None
    # identity of the storage instance (StorageImpl), not just its address: a later tensor can reuse the address of a
    # freed one, so output binding compares this (detector 2.2)
    storage_id: Optional[int] = None
    before: Any = None  # CPU copy (uint8) of the storage, or of the window elements, before launch
    after: Any = None  # the same after launch
    window: Any = None  # storage-relative element indices of a windowed copy (None: whole storage)


@dataclass
class CapturedLaunch:
    index: int
    kernel_name: str
    kernel_hash: str
    grid: tuple
    args: list
    asm: dict
    cubin_sha256: Optional[str]
    metadata: dict
    libtriton_sha256: Optional[str]
    environment: dict = field(default_factory=dict)
    # tensors the kernel reaches only through raw addresses (pointer tables + tt.int_to_ptr), registered by the
    # workload with TritonLaunchRecorder.register_implicit; captured like tensor operands, bound to no parameter
    implicit: list = field(default_factory=list)

    def ttir_params(self) -> list:
        """Arguments that are parameters of the TTIR function, in order."""

        return [a for a in self.args if not a.constexpr]

    def storages(self) -> dict:
        """One entry per distinct device storage touched by tensor operands."""

        out = {}
        for arg in self.args:
            if arg.kind == "tensor" and arg.storage_ptr not in out:
                out[arg.storage_ptr] = arg
        return out


_LIBTRITON_DIGESTS: dict = {}


def _libtriton_sha256() -> Optional[str]:
    """SHA256 of the loaded libtriton; computed once per (path, size, mtime) and process -- hashing the library took
    ~1.3 s per recorder, which dominated the capture time of small kernels."""
    try:
        import glob

        import triton

        paths = sorted(glob.glob(os.path.join(os.path.dirname(triton.__file__), "_C", "libtriton*.so")))
        if not paths:
            return None
        st = os.stat(paths[0])
        key = (paths[0], st.st_size, st.st_mtime_ns)
        if key in _LIBTRITON_DIGESTS:
            return _LIBTRITON_DIGESTS[key]
        digest = hashlib.sha256()
        with open(paths[0], "rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                digest.update(chunk)
        _LIBTRITON_DIGESTS[key] = digest.hexdigest()
        return _LIBTRITON_DIGESTS[key]
    except Exception:  # pragma: no cover - recorded as unknown
        return None


def _jsonable(value):
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if dataclasses.is_dataclass(value):
        return _jsonable(dataclasses.asdict(value))
    return repr(value)


def _window_copy(tensor, indices, storage_relative=False):
    """Copy selected elements; returns (storage-relative indices, uint8 bytes)."""

    import numpy as np
    import torch

    if not tensor.is_contiguous():
        raise ValueError("windowed capture requires a contiguous tensor")
    size = tensor.element_size()
    base = (tensor.data_ptr() - tensor.untyped_storage().data_ptr()) // size
    idx = np.asarray(indices, dtype=np.int64)
    rel = idx if storage_relative else idx + base
    order = np.argsort(rel, kind="stable")
    rel = rel[order]
    flat = torch.empty(0, dtype=tensor.dtype, device=tensor.device).set_(
        tensor.untyped_storage(), 0, (tensor.untyped_storage().nbytes() // size,))
    picked = flat[torch.as_tensor(rel, device=tensor.device)].contiguous()
    raw = picked.view(torch.uint8).cpu()
    return rel, raw


def _storage_copy(tensor):
    import torch

    storage = tensor.untyped_storage()
    flat = torch.empty(storage.nbytes(), dtype=torch.uint8, device=tensor.device)
    flat.untyped_storage().copy_(storage)
    return flat.cpu()


_TL_DTYPE = {"uint8": "uint8", "uint16": "uint16", "uint32": "uint32", "uint64": "uint64", "int8": "int8",
             "int16": "int16", "int32": "int32", "int64": "int64", "fp16": "float16", "bf16": "bfloat16",
             "fp32": "float32", "fp64": "float64", "fp8e4nv": "float8_e4m3fn", "fp8e5": "float8_e5m2", "int1": "bool",
             "fp8e4b15": "uint8"}  # fp8e4b15 is an i8 in the IR (python/src/ir.cc get_fp8e4b15_ty)


def _unwrap(arg):
    """A torch tensor as is; a reinterpreting wrapper (triton.runtime.jit.TensorWrapper: ``.base`` torch tensor +
    triton ``.dtype``) as (its base tensor, the reinterpreted dtype name); otherwise (None, None).  DSL v2 increment 3:
    the official tests pass unsigned and fp8 operands this way."""
    import torch
    if isinstance(arg, torch.Tensor):
        return arg, str(arg.dtype).replace("torch.", "")
    base = getattr(arg, "base", None)
    dt = getattr(arg, "dtype", None)
    if isinstance(base, torch.Tensor) and dt is not None and hasattr(arg, "data_ptr"):
        if isinstance(dt, torch.dtype):  # triton.reinterpret(t, torch dtype) (official test_conversions)
            return base, str(dt).replace("torch.", "")
        name = _TL_DTYPE.get(getattr(dt, "name", str(dt)))
        if name is not None:
            return base, name
    return None, None


def _flatten_args(args, names, signature):
    """(name, value, signature type) per TTIR-level argument: tuple arguments are flattened the way the frontend
    flattens them (``shape`` -> ``shape.0``, ``shape.1``; nested tuples recursively), each element with its own
    signature entry, so an element specialized to a constexpr (an int equal to 1) is marked as such.  DSL v2
    increment 6: the official main build passes shapes and strides of tensor descriptors and lists of tensors as
    tuples (test_tensor_descriptor, test_cat_nd)."""
    def walk(name, value, sig):
        if isinstance(sig, str) and sig.startswith("tensordesc") and hasattr(value, "base") and hasattr(value, "strides"):
            # a host tensor descriptor, decomposed ABI (targets without TMA, e.g. sm_86; official
            # triton/backends/driver.py decompose_descriptor): base, shape, strides, padding == "nan",
            # round_f32_to_tf32, then shape (i32) and strides again
            shape, strides = [int(v) for v in value.shape], [int(v) for v in value.strides]
            yield name, value.base, "*desc"
            for k, v in enumerate(shape):
                yield f"{name}.shape.{k}", v, "i64"
            for k, v in enumerate(strides):
                yield f"{name}.stride.{k}", v, "i64"
            yield f"{name}.padding", bool(getattr(value, "padding", "zero") == "nan"), "i1"
            yield f"{name}.roundF32ToTF32", bool(getattr(value, "round_f32_to_tf32", False)), "i1"
            for k, v in enumerate(shape):
                yield f"{name}.shape.{k}", v, "i32"
            for k, v in enumerate(strides):
                yield f"{name}.stride.{k}", v, "i64"
            return
        # a tuple is flattened only when its signature entry is a tuple too (a constexpr tuple such as an output shape
        # has the single entry "constexpr" and stays one argument)
        if isinstance(value, (tuple, list)) and (isinstance(sig, (tuple, list)) or sig is None):  # includes torch.Size
            sigs = sig if isinstance(sig, (tuple, list)) else [None] * len(value)
            for k, v in enumerate(value):
                yield from walk(f"{name}.{k}", v, sigs[k] if k < len(sigs) else None)
        else:
            yield name, value, sig

    for i, arg in enumerate(args):
        name = names[i] if i < len(names) else f"arg{i}"
        yield from walk(name, arg, signature.get(name))


class TritonLaunchRecorder(contextlib.AbstractContextManager):
    """Record selected Triton launches while active.

    ``select(kernel_name, launch_index) -> bool`` chooses which launches are
    recorded with operand copies; ``max_launches`` bounds memory use.
    """

    def __init__(self, select: Callable[[str, int], bool] = lambda name, index: True,
                 copy_tensors: bool = True, max_launches: Optional[int] = None,
                 window: Optional[Callable] = None, keep_kernel: bool = True, keep_storages: bool = True):
        """``window(kernel_name, arg_name, tensor) -> element indices or None``.

        Returned indices are relative to the tensor's first element in its
        flattened (contiguous) layout; only those elements are copied.

        ``keep_storages``: hold a reference to every storage a recorded launch touches until the recorder exits, so the
        caching allocator cannot hand a freed intermediate's address to another tensor inside the recorded region
        (an address then identifies one storage instance; ``storage_id`` makes the identity explicit).
        """

        self.select = select
        self.copy_tensors = copy_tensors
        self.max_launches = max_launches
        self.window = window
        self.keep_kernel = keep_kernel
        self.launches: list[CapturedLaunch] = []
        self.implicit_tensors: list = []
        self.launch_count = 0
        self.keep_storages = keep_storages
        self._kept = []
        self._original = None
        self._inductor_static = None
        self._libtriton = _libtriton_sha256()

    _active = None  # the recorder that currently receives launches
    _hook_original = None  # CompiledKernel.run before the hook was installed

    @classmethod
    def install_hook(cls):
        """Install the launch hook once and keep it.  Inductor caches the launcher it gets from
        ``CompiledKernel.run`` at a kernel's first launch, so a hook installed only while a recorder
        is active misses kernels first launched earlier (e.g. an optimizer compiled at step 1).
        The installed hook forwards to whichever recorder is active, and costs one attribute
        check per launch otherwise.  Call it before the workload's first launch."""

        from triton.compiler.compiler import CompiledKernel

        if cls._hook_original is not None:
            return
        cls._hook_original = CompiledKernel.run
        original_fget = cls._hook_original.fget

        def patched(kernel):
            real = original_fget(kernel)

            def launcher(gx, gy, gz, stream, function, packed, launch_md, enter, exit_, *args):
                recorder = cls._active
                if recorder is None:
                    return real(gx, gy, gz, stream, function, packed, launch_md, enter, exit_, *args)
                index = recorder.launch_count
                recorder.launch_count += 1
                take = recorder.select(kernel.name, index) and (
                    recorder.max_launches is None or len(recorder.launches) < recorder.max_launches)
                record = recorder._before(kernel, (gx, gy, gz), args, index) if take else None
                result = real(gx, gy, gz, stream, function, packed, launch_md, enter, exit_, *args)
                if record is not None:
                    recorder._after(record, args)
                    recorder.launches.append(record)
                return result

            return launcher

        CompiledKernel.run = property(patched)

    @classmethod
    def register_implicit(cls, *tensors):
        """Tensors that the next launches read or write through raw addresses (e.g. a table of weight pointers
        that the kernel turns back into pointers with tt.int_to_ptr).  No-op without an active recorder."""
        if cls._active is not None:
            cls._active.implicit_tensors.extend(t for t in tensors if t is not None)

    @classmethod
    def remove_hook(cls):
        from triton.compiler.compiler import CompiledKernel

        if cls._hook_original is not None:
            CompiledKernel.run = cls._hook_original
            cls._hook_original = None

    def __enter__(self):
        try:
            import torch._inductor.config as inductor_config

            self._inductor_static = inductor_config.use_static_cuda_launcher
            inductor_config.use_static_cuda_launcher = False
        except Exception:  # pragma: no cover - inductor unavailable
            self._inductor_static = None
        self._installed_here = TritonLaunchRecorder._hook_original is None
        TritonLaunchRecorder.install_hook()
        TritonLaunchRecorder._active = self
        return self

    def __exit__(self, *exc):
        TritonLaunchRecorder._active = None
        self._kept = []  # identities are recorded as integers; tensors created inside the region are already bound
        if self._installed_here:  # a hook installed by install_hook() beforehand stays
            TritonLaunchRecorder.remove_hook()
        if self._inductor_static is not None:
            import torch._inductor.config as inductor_config

            inductor_config.use_static_cuda_launcher = self._inductor_static
        return False

    # ------------------------------------------------------------------

    def _before(self, kernel, grid, args, index) -> CapturedLaunch:
        import torch

        src = kernel.src
        names = list(getattr(getattr(src, "fn", None), "arg_names", []) or [])
        signature = dict(getattr(src, "signature", {}) or {})
        captured = []
        if self.copy_tensors:
            torch.cuda.synchronize()
        for i, (name, arg, sig) in enumerate(_flatten_args(args, names, signature)):
            constexpr = sig == "constexpr"
            tensor, dtype_name = _unwrap(arg)
            if tensor is not None:
                arg = tensor
                storage = arg.untyped_storage()
                item = CapturedArg(
                    i, name, "tensor", constexpr, sig, dtype=dtype_name,
                    shape=tuple(arg.shape), stride=tuple(arg.stride()), element_size=arg.element_size(),
                    data_ptr=arg.data_ptr(), storage_ptr=storage.data_ptr(),
                    storage_nbytes=storage.nbytes(), storage_id=storage._cdata)
                if self.keep_storages:
                    self._kept.append(storage)
                if self.copy_tensors:
                    indices = self.window(kernel.name, name, arg) if self.window is not None else None
                    if indices is None:
                        item.before = _storage_copy(arg)
                    else:
                        item.window, item.before = _window_copy(arg, indices)
            elif isinstance(arg, bool):
                item = CapturedArg(i, name, "bool", constexpr, sig, value=arg)
            elif isinstance(arg, int):
                item = CapturedArg(i, name, "int", constexpr, sig, value=arg)
            elif isinstance(arg, float):
                item = CapturedArg(i, name, "float", constexpr, sig, value=arg)
            elif arg is None:
                item = CapturedArg(i, name, "none", constexpr, sig)
            else:
                item = CapturedArg(i, name, "other", constexpr, sig, value=repr(arg))
            captured.append(item)
        asm = {key: kernel.asm[key] for key in ASM_KEYS if key in kernel.asm}
        cubin = kernel.asm.get("cubin")
        metadata = _jsonable(kernel.metadata._asdict())
        import triton

        record = CapturedLaunch(
            index=index, kernel_name=kernel.name, kernel_hash=kernel.hash, grid=tuple(grid),
            args=captured, asm=asm,
            cubin_sha256=hashlib.sha256(cubin).hexdigest() if isinstance(cubin, (bytes, bytearray)) else None,
            metadata=metadata, libtriton_sha256=self._libtriton,
            environment={"triton": triton.__version__, "torch": torch.__version__,
                         "device": torch.cuda.get_device_name(torch.cuda.current_device())})
        record._src = (names, signature)  # for _after: the same flattening of the arguments
        for j, t in enumerate(self.implicit_tensors):
            item = CapturedArg(-1 - j, f"implicit{j}", "tensor", False, None, dtype=str(t.dtype).replace("torch.", ""),
                               shape=tuple(t.shape), stride=tuple(t.stride()), element_size=t.element_size(),
                               data_ptr=t.data_ptr(), storage_ptr=t.untyped_storage().data_ptr(),
                               storage_nbytes=t.untyped_storage().nbytes(), storage_id=t.untyped_storage()._cdata)
            if self.keep_storages:
                self._kept.append(t.untyped_storage())
            if self.copy_tensors:
                item.before = _storage_copy(t)
            record.implicit.append(item)
        record.kernel = kernel if self.keep_kernel else None
        record.src_info = {
            "arg_names": names, "signature": signature,
            "constants": [[list(k) if isinstance(k, tuple) else k, _jsonable(v)]
                          for k, v in (getattr(src, "constants", {}) or {}).items()],
        }
        return record

    def storage_ids(self, storage_ptr: int) -> set:
        """Identities of the storage instances recorded at ``storage_ptr`` (None among them: not recorded)."""
        return {a.storage_id for launch in self.launches for a in list(launch.args) + list(launch.implicit)
                if a.kind == "tensor" and a.storage_ptr == storage_ptr}

    def _after(self, record: CapturedLaunch, args):
        import torch

        if not self.copy_tensors:
            return
        torch.cuda.synchronize()
        src = getattr(record, "_src", None)
        flat = [a for _, a, _ in _flatten_args(args, *(src or ([], {})))]
        for item, arg in zip(record.args, flat):
            if item.kind == "tensor":
                arg = _unwrap(arg)[0]
                if item.window is None:
                    item.after = _storage_copy(arg)
                else:
                    _, item.after = _window_copy(arg, item.window, storage_relative=True)
        for item, t in zip(record.implicit, self.implicit_tensors):
            item.after = _storage_copy(t)


# ----------------------------------------------------------------------
# Persistence
# ----------------------------------------------------------------------


def save_launch(launch: CapturedLaunch, directory: Path) -> Path:
    """Write one capture package: ``launch.json`` + ``ir/`` + ``operands.pt``."""

    import torch

    directory = Path(directory)
    (directory / "ir").mkdir(parents=True, exist_ok=True)
    for key, text in launch.asm.items():
        (directory / "ir" / f"kernel.{key}").write_text(text)
    operands = {}
    args_json = []
    for arg in launch.args:
        entry = {k: v for k, v in dataclasses.asdict(arg).items() if k not in ("before", "after", "window")}
        args_json.append(_jsonable(entry))
        if arg.before is not None:
            operands[f"{arg.index}.before"] = arg.before
            operands[f"{arg.index}.after"] = arg.after
        if arg.window is not None:
            operands[f"{arg.index}.window"] = torch.as_tensor(arg.window)
    torch.save(operands, directory / "operands.pt")
    manifest = {
        "schema": "kernel-analyzer-triton-capture-v1",
        "index": launch.index,
        "kernel_name": launch.kernel_name,
        "kernel_hash": launch.kernel_hash,
        "grid": list(launch.grid),
        "args": args_json,
        "ir_files": {key: f"ir/kernel.{key}" for key in launch.asm},
        "ir_sha256": {key: hashlib.sha256(text.encode()).hexdigest() for key, text in launch.asm.items()},
        "cubin_sha256": launch.cubin_sha256,
        "metadata": launch.metadata,
        "libtriton_sha256": launch.libtriton_sha256,
        "environment": launch.environment,
    }
    kernel = getattr(launch, "kernel", None)
    if kernel is not None:
        import shutil

        (directory / "compiled").mkdir(exist_ok=True)
        manifest["compiled_files"] = []
        for path in kernel.metadata_group.values():
            shutil.copy2(path, directory / "compiled" / Path(path).name)
            manifest["compiled_files"].append(Path(path).name)
    manifest["src_info"] = getattr(launch, "src_info", None)
    (directory / "launch.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return directory


def load_launch(directory: Path) -> CapturedLaunch:
    import torch

    directory = Path(directory)
    manifest = json.loads((directory / "launch.json").read_text())
    operands = torch.load(directory / "operands.pt", weights_only=True)
    args = []
    for entry in manifest["args"]:
        entry = dict(entry)
        for key in ("shape", "stride"):
            if entry.get(key) is not None:
                entry[key] = tuple(entry[key])
        arg = CapturedArg(**entry)
        arg.before = operands.get(f"{arg.index}.before")
        arg.after = operands.get(f"{arg.index}.after")
        window = operands.get(f"{arg.index}.window")
        arg.window = None if window is None else window.numpy()
        args.append(arg)
    asm = {key: (directory / path).read_text() for key, path in manifest["ir_files"].items()}
    for key, digest in manifest["ir_sha256"].items():
        if hashlib.sha256(asm[key].encode()).hexdigest() != digest:
            raise ValueError(f"{directory}: {key} does not match its recorded hash")
    launch = CapturedLaunch(
        index=manifest["index"], kernel_name=manifest["kernel_name"], kernel_hash=manifest["kernel_hash"],
        grid=tuple(manifest["grid"]), args=args, asm=asm, cubin_sha256=manifest["cubin_sha256"],
        metadata=manifest["metadata"], libtriton_sha256=manifest["libtriton_sha256"],
        environment=manifest.get("environment", {}))
    launch.directory = directory  # the package, for its compiled artifacts (cubin / SASS)
    return launch


# ----------------------------------------------------------------------
# Replay (step 3: same input, bitwise identical output)
# ----------------------------------------------------------------------


class _SourceShim:
    """The attributes of an ASTSource that Triton's launcher needs."""

    def __init__(self, info: dict):
        self.signature = dict(info["signature"])
        self.constants = {tuple(k) if isinstance(k, list) else k: v for k, v in info["constants"]}

        class _Fn:
            arg_names = list(info["arg_names"])
            launch_metadata = None

        self.fn = _Fn()


def load_compiled(directory: Path):
    """Rebuild the CompiledKernel saved in a capture package (no recompilation)."""

    from triton.compiler.compiler import CompiledKernel

    directory = Path(directory)
    manifest = json.loads((directory / "launch.json").read_text())
    group = {name: str(directory / "compiled" / name) for name in manifest["compiled_files"]}
    return CompiledKernel(_SourceShim(manifest["src_info"]), group, manifest["kernel_hash"])


def replay(launch: CapturedLaunch, kernel=None, device: str = "cuda") -> dict:
    """Re-launch the captured binary on the captured inputs; storage bytes after the launch."""

    import torch

    kernel = kernel if kernel is not None else getattr(launch, "kernel", None)
    if kernel is None:
        raise ValueError("no compiled kernel available for replay")
    storages = {}
    for arg in launch.args:
        if arg.kind == "tensor" and arg.storage_ptr not in storages:
            if arg.window is not None:
                raise ValueError("replay needs whole-storage captures")
            storages[arg.storage_ptr] = arg.before.to(device).clone()
    args = []
    for arg in launch.args:
        if arg.kind == "tensor":
            raw = storages[arg.storage_ptr]
            dtype = getattr(torch, arg.dtype)
            offset = (arg.data_ptr - arg.storage_ptr) // arg.element_size
            numel = raw.numel() // arg.element_size
            flat = torch.empty(0, dtype=dtype, device=device).set_(raw.untyped_storage(), 0, (numel,))
            args.append(flat.as_strided(arg.shape, arg.stride, offset))
        elif arg.kind in ("int", "float", "bool"):
            args.append(arg.value)
        elif arg.kind == "none":
            args.append(None)
        elif arg.constexpr:
            args.append(None)  # constexpr values are compiled into the binary; the launcher skips them
        else:
            raise ValueError(f"cannot replay argument kind {arg.kind}")
    gx, gy, gz = launch.grid
    stream = torch.cuda.current_stream().cuda_stream
    kernel.run(gx, gy, gz, stream, kernel.function, kernel.packed_metadata, None, None, None, *args)
    torch.cuda.synchronize()
    return {ptr: raw.cpu() for ptr, raw in storages.items()}


def check_replay(launch: CapturedLaunch, kernel=None, times: int = 2) -> dict:
    """Replay ``times`` times; compare every storage with the capture and across replays."""

    import torch

    expected = {arg.storage_ptr: arg.after for arg in launch.args if arg.kind == "tensor"}
    outputs = [replay(launch, kernel) for _ in range(times)]
    per_replay = [{str(ptr): bool(torch.equal(out[ptr], expected[ptr])) for ptr in out} for out in outputs]
    across = all(torch.equal(outputs[0][ptr], other[ptr]) for other in outputs[1:] for ptr in outputs[0])
    return {
        "replays": times,
        "identical_to_capture": per_replay,
        "replays_identical": bool(across),
        "bitwise_reproduced": bool(across and all(all(r.values()) for r in per_replay)),
        "correspondence": {
            "kernel_hash": launch.kernel_hash,
            "ir_sha256": {k: hashlib.sha256(v.encode()).hexdigest() for k, v in launch.asm.items()},
            "cubin_sha256": launch.cubin_sha256,
            "libtriton_sha256": launch.libtriton_sha256,
        },
    }
