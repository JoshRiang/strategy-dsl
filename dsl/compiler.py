"""Compile AST to a Python callable: strategy(prices_df) -> weights Series."""
# Maintenance: last reviewed 2026-11-13 (WHEN..ELSE, and/or, scale, clipping)
from typing import Callable
import numpy as np
import pandas as pd
from .ast_nodes import (
    Node, Constant, Variable, Call, BinaryOp, UnaryOp, Assignment, When, Block,
)
from .stdlib import STDLIB


class CompilerError(Exception):
    pass


_ARITH = {"+": np.add, "-": np.subtract, "*": np.multiply, "/": np.divide}
KEYWORD_NAMES = {"when", "then", "else", "and", "or", "long", "short", "flat"}
_CMP = {
    ">": lambda a, b: a > b,
    "<": lambda a, b: a < b,
    ">=": lambda a, b: a >= b,
    "<=": lambda a, b: a <= b,
    "==": lambda a, b: a == b,
    "!=": lambda a, b: a != b,
}


class Compiler:
    def __init__(self):
        self.env: dict = {}

    def compile(self, block: Block) -> Callable:
        def strategy(df: pd.DataFrame) -> pd.Series:
            missing = [c for c in ("Open", "High", "Low", "Close", "Volume")
                       if c not in df.columns and c.lower() not in [str(x).lower() for x in df.columns]]
            if missing:
                raise CompilerError(f"missing price columns: {', '.join(missing)}")

            local = dict(self.env)
            seen_weight = False
            weights = pd.Series(0.0, index=df.index, name="weight")

            for stmt in block.statements:
                if isinstance(stmt, Assignment):
                    val = self._eval(stmt.value, df, local)
                    local[stmt.name] = val
                    if stmt.name == "weight":
                        seen_weight = True
                        weights = self._as_series(val, df)

                elif isinstance(stmt, When):
                    cond = self._eval(stmt.cond, df, local)
                    cond = self._as_bool_series(cond, df)

                    if stmt.action is not None:
                        if stmt.action == "long":
                            weights = weights.mask(cond, 1.0)
                        elif stmt.action == "short":
                            weights = weights.mask(cond, -1.0)
                        elif stmt.action == "flat":
                            weights = weights.mask(cond, 0.0)
                        else:
                            raise CompilerError(f"Unknown action: {stmt.action}")
                        seen_weight = True
                    else:
                        then_v = self._eval(stmt.then_expr, df, local) if stmt.then_expr is not None else 1.0
                        else_v = self._eval(stmt.else_expr, df, local) if stmt.else_expr is not None else 0.0

                        if isinstance(then_v, Assignment) or isinstance(else_v, Assignment):
                            raise CompilerError("nested assignment in WHEN is not supported")

                        then_s = self._as_series(then_v, df)
                        else_s = self._as_series(else_v, df)
                        weights = then_s.where(cond, else_s)
                        local["weight"] = weights
                        seen_weight = True
                else:
                    raise CompilerError(f"Unknown statement: {stmt}")

            if not seen_weight:
                raise CompilerError("strategy defines no weight")

            return weights.clip(-1.0, 1.0).fillna(0.0).rename("weight")

        return strategy

    # ---- helpers -------------------------------------------------------
    @staticmethod
    def _as_series(v, df: pd.DataFrame) -> pd.Series:
        if isinstance(v, pd.Series):
            return v.reindex(df.index)
        return pd.Series(v, index=df.index, dtype="float64")

    @staticmethod
    def _as_bool_series(v, df: pd.DataFrame) -> pd.Series:
        if isinstance(v, pd.Series):
            return v.reindex(df.index).fillna(False).astype(bool)
        return pd.Series(bool(v), index=df.index)

    def _eval(self, node: Node, df: pd.DataFrame, local: dict):
        if isinstance(node, Constant):
            return node.value

        if isinstance(node, Variable):
            if node.name in local:
                return local[node.name]
            if node.name in STDLIB:
                return STDLIB[node.name](df)
            # `Close` and `close` are the same field.
            lowered = node.name.lower()
            if lowered in STDLIB and node.name not in KEYWORD_NAMES:
                return STDLIB[lowered](df)
            raise CompilerError(f"unknown identifier: {node.name}")

        if isinstance(node, Call):
            if node.name not in STDLIB:
                raise CompilerError(f"unknown built-in: {node.name}")
            fn = STDLIB[node.name]
            if not callable(fn):
                raise CompilerError(f"unknown built-in: {node.name}")
            args = [self._eval(a, df, local) for a in node.args]
            try:
                return fn(*args)
            except KeyError as e:
                raise CompilerError(f"missing column or variable: {e}") from None
            except Exception as e:
                raise CompilerError(f"{node.name} failed: {e}") from None

        if isinstance(node, When):
            cond = self._as_bool_series(self._eval(node.cond, df, local), df)
            then_v = self._eval(node.then_expr, df, local) if node.then_expr is not None else 1.0
            else_v = self._eval(node.else_expr, df, local) if node.else_expr is not None else 0.0
            return self._as_series(then_v, df).where(cond, self._as_series(else_v, df))

        if isinstance(node, BinaryOp):
            left = self._eval(node.left, df, local)
            right = self._eval(node.right, df, local)

            if node.op in _CMP:
                return _CMP[node.op](left, right)
            if node.op == "and":
                return self._as_bool_series(left, df) & self._as_bool_series(right, df)
            if node.op == "or":
                return self._as_bool_series(left, df) | self._as_bool_series(right, df)
            if node.op in _ARITH:
                return _ARITH[node.op](left, right)
            raise CompilerError(f"unknown operator: {node.op}")

        if isinstance(node, UnaryOp):
            v = self._eval(node.operand, df, local)
            return -v

        raise CompilerError(f"unknown node: {node!r}")


def compile_src(src: str) -> Callable:
    from .parser import parse
    block = parse(src)
    return Compiler().compile(block)


# Public aliases used by the test suite and documented in the README.
compile_source = compile_src
DSLRuntimeError = CompilerError
