"""AST node types for the strategy DSL."""
from dataclasses import dataclass
from typing import Any, List, Union


class Node:
    pass


@dataclass
class Number(Node):
    value: float


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
    expr: Node


@dataclass
class When(Node):
    cond: Node
    action: str  # "long" / "short" / "flat" / "scale"


@dataclass
class Block(Node):
    statements: List[Node]
