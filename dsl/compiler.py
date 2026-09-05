"""Compile AST to a Python callable: strategy(prices_df) -> weights Series."""
from typing import Callable
import pandas as pd
from .ast_nodes import (
    Node, Number, Variable, Call, BinaryOp, UnaryOp, Assignment, When, Block,
)
from .stdlib import STDLIB


class CompilerError(Exception):
    pass


class Compiler:
    def __init__(self):
        self.env: dict = {}

    def compile(self, block: Block) -> Callable:
        def strategy(df: pd.DataFrame) -> pd.Series:
            local = dict(self.env)
            weights = pd.Series(0.0, index=df.index)
            for stmt in block.statements:
                if isinstance(stmt, Assignment):
                    val = self._eval(stmt.expr, df, local)
                    local[stmt.name] = val
                    if stmt.name == "weight":
                        weights = val
                elif isinstance(stmt, When):
                    cond = self._eval(stmt.cond, df, local)
                    action = stmt.action
                    if action == "long":
                        weights = weights.where(~cond.astype(bool), 1.0)
                    elif action == "short":
                        weights = weights.where(~cond.astype(bool), -1.0)
                    elif action == "flat":
                        weights = weights.where(~cond.astype(bool), 0.0)
                    elif action.startswith("scale"):
                        weights = weights.where(~cond.astype(bool), cond)
                else:
                    raise CompilerError(f"Unknown statement: {stmt}")
            return weights
        return strategy

    def _eval(self, node: Node, df: pd.DataFrame, local: dict):
        if isinstance(node, Number):
            return node.value
        if isinstance(node, Variable):
            if node.name in local:
                return local[node.name]
            if node.name in STDLIB:
                v = STDLIB[node.name]
                if callable(v):
                    return v(df)
                return v
            raise CompilerError(f"Unknown variable: {node.name}")
        if isinstance(node, Call):
            if node.name not in STDLIB:
                raise CompilerError(f"Unknown function: {node.name}")
            fn = STDLIB[node.name]
            if not callable(fn):
                raise CompilerError(f"{node.name} is not callable")
            args = [self._eval(a, df, local) for a in node.args]
            return fn(*args)
        if isinstance(node, BinaryOp):
            l = self._eval(node.left, df, local)
            r = self._eval(node.right, df, local)
            if node.op == '+': return l + r
            if node.op == '-': return l - r
            if node.op == '*': return l * r
            if node.op == '/': return l / r
        if isinstance(node, UnaryOp):
            v = self._eval(node.operand, df, local)
            return -v
        raise CompilerError(f"Unknown node: {node}")


def compile_src(src: str) -> Callable:
    from .parser import parse
    block = parse(src)
    return Compiler().compile(block)
