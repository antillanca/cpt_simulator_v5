"""Simple physical DSL compiler.

The DSL is intentionally small and deterministic. It accepts YAML input and
compiles it into safe Lua plus metadata, generated tests, and documentation.
"""

from __future__ import annotations

import ast
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

SAFE_NODES = {
    ast.Expression,
    ast.Module,
    ast.Expr,
    ast.Assign,
    ast.Name,
    ast.Load,
    ast.Store,
    ast.BinOp,
    ast.UnaryOp,
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.Pow,
    ast.Mod,
    ast.USub,
    ast.UAdd,
    ast.Constant,
    ast.Call,
    ast.Attribute,
    ast.Subscript,
    ast.Index,
    ast.Tuple,
    ast.List,
    ast.Compare,
    ast.Gt,
    ast.GtE,
    ast.Lt,
    ast.LtE,
    ast.Eq,
    ast.NotEq,
    ast.IfExp,
}


class DSLCompilerError(ValueError):
    pass


def _load_dsl(dsl_input: str | dict[str, Any]) -> dict[str, Any]:
    if isinstance(dsl_input, dict):
        return dsl_input
    data = yaml.safe_load(dsl_input)
    if not isinstance(data, dict):
        raise DSLCompilerError("DSL must decode to a mapping.")
    return data


def _validate_expression(expression: str, allowed_names: set[str], lhs_name: str):
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise DSLCompilerError(f"Invalid equation expression: {expression}") from exc

    for node in ast.walk(tree):
        if type(node) not in SAFE_NODES:
            raise DSLCompilerError(f"Unsupported syntax in equation: {type(node).__name__}")
        if isinstance(node, ast.Name) and node.id not in allowed_names and node.id not in {lhs_name, "math"}:
            raise DSLCompilerError(f"Unknown symbol '{node.id}' in expression: {expression}")


def _to_lua_expression(expression: str) -> str:
    return expression.replace("True", "true").replace("False", "false")


def _generate_tests(equations: list[str], inputs: list[str]) -> list[dict[str, Any]]:
    tests = []
    base_values = {name: idx + 1 for idx, name in enumerate(inputs)}
    for idx, equation in enumerate(equations):
        tests.append(
            {
                "name": f"equation_{idx:02d}",
                "inputs": base_values.copy(),
                "expected_equation": equation,
            }
        )
    return tests


def _build_doc(metadata: dict[str, Any]) -> str:
    lines = [f"# {metadata['law']['name']}", "", "## Inputs"]
    for name in metadata.get("inputs", []):
        lines.append(f"- {name}")
    lines.extend(["", "## Equations"])
    for eq in metadata.get("equations", []):
        lines.append(f"- {eq}")
    lines.extend(["", "## Invariants"])
    for inv in metadata.get("invariants", []):
        lines.append(f"- {inv}")
    return "\n".join(lines)


def compile_dsl(dsl_input: str | dict[str, Any]) -> dict[str, Any]:
    """Compile verified YAML DSL into safe Lua and companion artifacts."""

    dsl = _load_dsl(dsl_input)
    law = dsl.get("law") or {}
    if not isinstance(law, dict) or not law.get("name"):
        raise DSLCompilerError("DSL requires law.name.")

    inputs = [item for item in dsl.get("inputs", []) if isinstance(item, str)]
    equations = [item for item in dsl.get("equations", []) if isinstance(item, str)]
    invariants = [item for item in dsl.get("invariants", []) if isinstance(item, str)]

    if not equations:
        raise DSLCompilerError("DSL requires at least one equation.")

    lhs_names = []
    for equation in equations:
        if "=" not in equation:
            raise DSLCompilerError(f"Equation must be an assignment: {equation}")
        lhs, _ = [part.strip() for part in equation.split("=", 1)]
        if not lhs:
            raise DSLCompilerError(f"Equation has empty left-hand side: {equation}")
        lhs_names.append(lhs)

    allowed_names = set(inputs) | set(lhs_names)
    lua_lines = [f"-- DSL law: {law['name']}", "return function(state)"]
    for name in dict.fromkeys(inputs + lhs_names):
        lua_lines.append(f"  local {name} = tonumber(state.{name}) or state.{name} or 0")
    for equation in equations:
        lhs, rhs = [part.strip() for part in equation.split("=", 1)]
        _validate_expression(rhs, set(allowed_names), lhs)
        lua_lines.append(f"  state.{lhs} = {_to_lua_expression(rhs)}")
        lua_lines.append(f"  {lhs} = state.{lhs}")
        allowed_names.add(lhs)
    lua_lines.append("  return state")
    lua_lines.append("end")

    metadata = {
        "law": law,
        "inputs": inputs,
        "equations": equations,
        "invariants": invariants,
    }
    tests = _generate_tests(equations, inputs)
    documentation = _build_doc(metadata)

    return {
        "lua_code": "\n".join(lua_lines),
        "metadata": metadata,
        "tests": tests,
        "documentation": documentation,
    }
