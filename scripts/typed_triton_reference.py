"""Compile ABI-correct floating-storage variants of frozen Triton programs.

The generated Inductor wrapper freezes pointer element types in
``triton_meta.signature``.  A different allocation dtype therefore requires a
different compilation; passing FP32 storage to an already compiled ``*bf16``
binary only reinterprets bytes and is never a numerical reference.
"""

from __future__ import annotations

import ast
import hashlib
import token
import tokenize
from pathlib import Path
from typing import Any, Iterable, Mapping

from torch._inductor.async_compile import AsyncCompile


FLOAT_POINTER_ABIS = {"*bf16", "*fp16", "*fp32", "*fp64"}


def _function_name(node: ast.AST) -> str:
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if not isinstance(node, ast.Name):
        raise ValueError("call target is not a dotted name")
    parts.append(node.id)
    return ".".join(reversed(parts))


def embedded_triton_programs(path: Path) -> dict[str, str]:
    """Return every top-level ``async_compile.triton`` literal exactly.

    Generated Mamba wrappers can exceed 50 MiB.  Tokenize the file as a stream
    instead of constructing an AST for the whole wrapper; only each embedded
    kernel source is parsed later.
    """

    result: dict[str, str] = {}
    ignored = {tokenize.NL, tokenize.NEWLINE, tokenize.COMMENT,
               tokenize.INDENT, tokenize.DEDENT, tokenize.ENCODING}
    with tokenize.open(path) as handle:
        stream = (
            row for row in tokenize.generate_tokens(handle.readline)
            if row.type not in ignored
        )
        iterator = iter(stream)
        for first in iterator:
            if first.type != token.NAME or first.start[1] != 0:
                continue
            expected = [
                (token.OP, "="), (token.NAME, "async_compile"), (token.OP, "."),
                (token.NAME, "triton"), (token.OP, "("),
            ]
            consumed = []
            try:
                for kind, text in expected:
                    row = next(iterator)
                    consumed.append(row)
                    if row.type != kind or row.string != text:
                        break
                else:
                    name_literal = next(iterator)
                    comma = next(iterator)
                    source_literal = next(iterator)
                    if (
                        name_literal.type == token.STRING
                        and comma.type == token.OP and comma.string == ","
                        and source_literal.type == token.STRING
                    ):
                        symbol = first.string
                        declared = ast.literal_eval(name_literal.string)
                        source = ast.literal_eval(source_literal.string)
                        if declared != symbol or not isinstance(source, str):
                            raise ValueError(
                                f"{symbol} does not contain a matching literal Triton program"
                            )
                        if symbol in result:
                            raise ValueError(f"duplicate Triton symbol in wrapper: {symbol}")
                        result[symbol] = source
            except StopIteration:
                break
    return result


def _literal_mapping(node: ast.AST | None, *, label: str) -> dict[str, ast.AST]:
    if not isinstance(node, ast.Dict):
        raise ValueError(f"{label} is not a literal dictionary")
    result: dict[str, ast.AST] = {}
    for key, value in zip(node.keys, node.values):
        if (
            not isinstance(key, ast.Constant)
            or not isinstance(key.value, str)
        ):
            raise ValueError(f"{label} does not have literal string keys")
        result[key.value] = value
    return result


def _signature_literals(tree: ast.Module, symbol: str) -> dict[str, ast.Constant]:
    definitions = [
        node for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == symbol
    ]
    if len(definitions) != 1:
        raise ValueError(f"{symbol} Triton definition is not unique")
    decorators = definitions[0].decorator_list
    heuristic_calls = [
        node for node in decorators
        if isinstance(node, ast.Call)
        and _function_name(node.func).startswith("triton_heuristics.")
    ]
    if len(heuristic_calls) != 1:
        raise ValueError(f"{symbol} heuristic decorator is not unique")
    keywords = {row.arg: row.value for row in heuristic_calls[0].keywords if row.arg}
    meta = _literal_mapping(keywords.get("triton_meta"), label=f"{symbol}.triton_meta")
    signature = meta.get("signature")
    values = _literal_mapping(signature, label=f"{symbol}.signature")
    if any(not isinstance(value, ast.Constant) for value in values.values()):
        raise ValueError(f"{symbol}.signature values are not literals")
    return values  # type: ignore[return-value]


def fp32_pointer_program(source: str, symbol: str) -> tuple[str, dict[str, Any]]:
    """Rewrite only floating pointer ABI literals and preserve the program body.

    ``ast.unparse`` changes formatting, so both the frozen source and typed
    source digests are returned and later bound into the reference artifact.
    The parsed kernel body is checked to be structurally identical after
    removing only the changed signature constants.
    """

    tree = ast.parse(source)
    normalized_original = ast.parse(source)
    for literal in _signature_literals(normalized_original, symbol).values():
        if literal.value in FLOAT_POINTER_ABIS:
            literal.value = "*FLOAT_STORAGE"
    signature = _signature_literals(tree, symbol)
    original = {
        name: value.value for name, value in signature.items()
        if isinstance(value.value, str)
    }
    changed: dict[str, dict[str, str]] = {}
    for name, literal in signature.items():
        annotation = literal.value
        if annotation in FLOAT_POINTER_ABIS and annotation != "*fp32":
            changed[name] = {"from": str(annotation), "to": "*fp32"}
            literal.value = "*fp32"
    typed = ast.unparse(ast.fix_missing_locations(tree)) + "\n"
    reparsed = ast.parse(typed)
    observed = {
        name: value.value
        for name, value in _signature_literals(reparsed, symbol).items()
    }
    expected = {
        name: ("*fp32" if value in FLOAT_POINTER_ABIS else value)
        for name, value in original.items()
    }
    if observed != expected:
        raise RuntimeError(f"{symbol} typed signature rewrite is not exact")
    normalized_typed = ast.parse(typed)
    for literal in _signature_literals(normalized_typed, symbol).values():
        if literal.value in FLOAT_POINTER_ABIS:
            literal.value = "*FLOAT_STORAGE"
    if ast.dump(normalized_original, include_attributes=False) != ast.dump(
        normalized_typed, include_attributes=False
    ):
        raise RuntimeError(f"{symbol} rewrite changed more than floating pointer ABI literals")
    return typed, {
        "schema": "kernel-analyzer-typed-triton-program-v1",
        "symbol": symbol,
        "frozen_program_sha256": hashlib.sha256(source.encode()).hexdigest(),
        "typed_program_sha256": hashlib.sha256(typed.encode()).hexdigest(),
        "original_signature": original,
        "typed_signature": observed,
        "changed_float_pointers": changed,
        "only_pointer_abi_literals_changed": True,
    }


def collect_programs(modules: Iterable[Any]) -> dict[str, str]:
    result: dict[str, str] = {}
    for module in modules:
        for symbol, source in embedded_triton_programs(Path(module.__file__).resolve()).items():
            previous = result.setdefault(symbol, source)
            if previous != source:
                raise RuntimeError(f"runtime symbol has multiple programs: {symbol}")
    return result


def compile_fp32_pointer_kernels(
    modules: Iterable[Any],
    *,
    expected_program_sha256: Mapping[str, str],
    selected_symbols: set[str] | None = None,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """Compile independent typed kernels and wait for every requested symbol."""

    programs = collect_programs(modules)
    requested = set(expected_program_sha256)
    if selected_symbols is not None:
        unknown = selected_symbols - requested
        if unknown:
            raise ValueError(f"selected symbols are outside the frozen campaign: {sorted(unknown)}")
        requested &= selected_symbols
    missing = requested - programs.keys()
    if missing:
        raise RuntimeError(f"frozen Triton programs absent at runtime: {sorted(missing)}")
    compiler = AsyncCompile()
    kernels: dict[str, Any] = {}
    metadata: dict[str, dict[str, Any]] = {}
    for symbol in sorted(requested):
        source = programs[symbol]
        frozen_digest = hashlib.sha256(source.encode()).hexdigest()
        if frozen_digest != expected_program_sha256[symbol]:
            raise RuntimeError(f"runtime Triton program differs from frozen campaign: {symbol}")
        typed, row = fp32_pointer_program(source, symbol)
        kernels[symbol] = compiler.triton(symbol, typed, device_str="cuda")
        metadata[symbol] = row
    compiler.wait(kernels)
    return kernels, metadata
