
# Maintenance: last reviewed 2026-11-13
"""AST node types for the strategy DSL."""
from dataclasses import dataclass, field
from typing import Any, List, Optional


class Node:
    pass


@dataclass
class Constant(Node):
    value: Any


@dataclass
class Variable(Node):
    name: str


@dataclass
class Call(Node):
    name: str
    args: List[Node]


@dataclass
class BinaryOp(Node):
    op: str
    left: Node
    right: Node


@dataclass
class UnaryOp(Node):
    op: str
    operand: Node


@dataclass
class Assignment(Node):
    name: str
    value: Node

    @property
    def expr(self) -> Node:
        """Alias kept for callers that read `.expr`."""
        return self.value


@dataclass
class When(Node):
    """Conditional, in either of the two shapes the language allows.

    expression form  ``WHEN cond THEN a ELSE b``  -> then_expr / else_expr
    statement form   ``when cond: long``           -> action
    """

    cond: Node
    then_expr: Optional[Node] = None
    else_expr: Optional[Node] = None
    action: Optional[str] = None


@dataclass
class Block(Node):
    statements: List[Node] = field(default_factory=list)


# Public aliases: the tests and the README already use these names.
Number = Constant
Program = Block
