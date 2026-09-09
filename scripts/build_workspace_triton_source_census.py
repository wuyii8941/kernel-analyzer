#!/usr/bin/env python3
"""Census saved compiled Triton sources across the workspace results tree.

This records observed source coverage only.  It does not infer mathematical
operator families, reference validity, runtime reach, or numerical outcomes.
"""
import argparse
import ast
from collections import Counter
import hashlib
import json
from pathlib import Path


def embedded_definitions(raw):
    source = raw.decode() if isinstance(raw, bytes) else raw
    tree = ast.parse(source)
    rows = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Assign) and isinstance(node.value, ast.Call)
                and isinstance(node.value.func, ast.Attribute)
                and node.value.func.attr == "triton"):
            continue
        symbols = [target.id for target in node.targets if isinstance(target, ast.Name)]
        if not symbols or len(node.value.args) < 2:
            continue
        program = node.value.args[1]
        if not isinstance(program, ast.Constant) or not isinstance(program.value, str):
            rows.extend({"symbol": symbol, "status": "NON_LITERAL_TRITON_PROGRAM"}
                        for symbol in symbols)
            continue
        program_tree = ast.parse(program.value)
        for symbol in symbols:
            functions = [item for item in program_tree.body
                         if isinstance(item, ast.FunctionDef) and item.name == symbol]
            if len(functions) != 1:
                rows.append({"symbol": symbol, "status": "AMBIGUOUS_OR_MISSING_FUNCTION"})
                continue
            function = functions[0]
            semantic = ast.FunctionDef(
                name=function.name, args=function.args, body=function.body,
                decorator_list=[], returns=function.returns,
                type_comment=function.type_comment,
            )
            rows.append({
                "symbol": symbol,
                "status": "TRITON_FUNCTION_INSPECTED",
                "function_semantic_ast_sha256": hashlib.sha256(
                    ast.dump(semantic).encode()).hexdigest(),
            })
    return rows


def nearest_runtime_package(source, results_root):
    for parent in source.parents:
        if parent == results_root.parent:
            break
        if parent.name == "runtime_release":
            return parent
        if parent.parent.name == "runtime_releases":
            return parent
    return None


def build(results_root, registered_releases):
    results_root = results_root.resolve()
    registered = {Path(path).resolve() for path in registered_releases}
    paths = sorted(results_root.rglob("output_code.py"))
    by_digest = {}
    path_rows = []
    definition_records = []
    for path in paths:
        raw = path.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        if digest not in by_digest:
            try:
                by_digest[digest] = embedded_definitions(raw)
            except (UnicodeDecodeError, SyntaxError) as exc:
                by_digest[digest] = [{"symbol": None, "status": "SOURCE_PARSE_FAILED",
                                      "reason": str(exc)}]
        package = nearest_runtime_package(path.resolve(), results_root)
        if package in registered:
            provenance = "REGISTERED_RELEASE"
        elif package is not None:
            provenance = "UNREGISTERED_RUNTIME_RELEASE"
        else:
            provenance = "OTHER_SAVED_COMPILED_SOURCE"
        path_rows.append({
            "source": str(path.resolve()), "source_sha256": digest,
            "runtime_package": str(package) if package is not None else None,
            "provenance_status": provenance,
            "definition_count": len(by_digest[digest]),
        })
        definition_records.extend({
            **row, "source": str(path.resolve()), "source_sha256": digest,
            "runtime_package": str(package) if package is not None else None,
            "provenance_status": provenance,
        } for row in by_digest[digest])
    unique_definitions = {
        row["function_semantic_ast_sha256"]
        for rows in by_digest.values() for row in rows
        if row.get("function_semantic_ast_sha256")
    }
    return {
        "schema": "workspace-triton-source-census-v1",
        "scope": "Every saved output_code.py below the declared results root",
        "selection_uses_numerical_results": False,
        "source_paths": len(paths),
        "unique_source_digests": len(by_digest),
        "release_qualified_definition_occurrences": len(definition_records),
        "unique_function_semantic_ast_digests": len(unique_definitions),
        "provenance_counts": dict(Counter(row["provenance_status"] for row in path_rows)),
        "source_records": path_rows,
        "definition_records": definition_records,
        "all_kernel_support_established": False,
        "warning": (
            "Source occurrences and semantic AST hashes are not mathematical operator "
            "families, reference support, valid measurements, or independent mechanisms."
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or not args.output.resolve().is_relative_to(Path("/data1/tzh")):
        parser.error("Choose a new output under /data1/tzh")
    inventory_raw = args.inventory.read_bytes()
    inventory = json.loads(inventory_raw)
    result = build(args.results_root, {row["release"] for row in inventory["records"]})
    result["input_sha256"] = {
        str(args.inventory.resolve()): hashlib.sha256(inventory_raw).hexdigest(),
        str(Path(__file__).resolve()): hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps({key: result[key] for key in (
        "source_paths", "unique_source_digests",
        "release_qualified_definition_occurrences",
        "unique_function_semantic_ast_digests", "provenance_counts",
    )}))


if __name__ == "__main__":
    main()
