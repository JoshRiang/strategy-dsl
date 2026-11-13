"""Recursive-descent parser for the strategy DSL.

Grammar (informal):
    program     := statement*
    statement   := assignment | when_statement
    assignment  := IDENT '=' expr
    when_stmt   := 'when' expr ':' action                 # statement form
                 | 'WHEN' expr 'THEN' assignment ('ELSE' assignment)?
    action      := 'long' | 'short' | 'flat' | 'scale' '(' expr ')'
    expr        := or_expr
    or_expr     := and_expr ('or' and_expr)*
    and_expr    := cmp_expr ('and' cmp_expr)*
    cmp_expr    := sum (('>'|'<'|'>='|'<='|'=='|'!=') sum)*
    sum         := product (('+'|'-') product)*
    product     := unary (('*'|'/') unary)*
    unary       := '-' unary | primary
    primary     := NUMBER | IDENT | call | '(' expr ')' | when_expr
    when_expr   := 'WHEN' expr 'THEN' expr ('ELSE' expr)?
    call        := IDENT '(' args? ')'
"""

# Maintenance: last reviewed 2026-11-13 (full grammar: and/or, WHEN..THEN..ELSE)
from typing import List, Optional
from .ast_nodes import (
    Node, Constant, Variable, Call, BinaryOp, UnaryOp, Assignment, When, Block,
)

KEYWORDS = {"when", "then", "else", "and", "or", "long", "short", "flat"}


class ParseError(Exception):
    pass


class Token:
    __slots__ = ("kind", "value", "pos", "line")

    def __init__(self, kind: str, value: str, pos: int, line: int = 1):
        self.kind = kind
        self.value = value
        self.pos = pos
        self.line = line

    def __repr__(self):
        return f"Token({self.kind}, {self.value!r}, pos={self.pos}, line={self.line})"


def tokenize(src: str) -> List[Token]:
    tokens: List[Token] = []
    i = 0
    line = 1
    n = len(src)
    while i < n:
        c = src[i]
        if c == "\n":
            line += 1
            i += 1
            continue
        if c.isspace():
            i += 1
            continue
        if c == "#":
            while i < n and src[i] != "\n":
                i += 1
            continue
        if c.isalpha() or c == "_":
            j = i
            while j < n and (src[j].isalnum() or src[j] == "_"):
                j += 1
            word = src[i:j]
            if word.lower() in KEYWORDS:
                tokens.append(Token(word.upper(), word.lower(), i, line))
            else:
                tokens.append(Token("IDENT", word, i, line))
            i = j
        elif c.isdigit() or (c == "." and i + 1 < n and src[i + 1].isdigit()):
            j = i
            while j < n and (src[j].isdigit() or src[j] == "."):
                j += 1
            tokens.append(Token("NUMBER", src[i:j], i, line))
            i = j
        elif c in "<>=!":
            two = src[i:i + 2]
            if two in (">=", "<=", "==", "!="):
                tokens.append(Token(two, two, i, line))
                i += 2
            elif c in "<>":
                tokens.append(Token(c, c, i, line))
                i += 1
            elif c == "=":
                tokens.append(Token("=", c, i, line))
                i += 1
            else:
                raise ParseError(f"Unexpected character {c!r} at {i}")
        elif c in "+-*/()=,:":
            tokens.append(Token(c, c, i, line))
            i += 1
        else:
            raise ParseError(f"Unexpected character {c!r} at {i}")
    return tokens


class Parser:
    def __init__(self, tokens: List[Token]):
        self.tokens = tokens
        self.pos = 0

    # ---- token helpers -------------------------------------------------
    def peek(self, offset: int = 0) -> Optional[Token]:
        idx = self.pos + offset
        return self.tokens[idx] if idx < len(self.tokens) else None

    def at(self, *kinds: str) -> bool:
        t = self.peek()
        return t is not None and t.kind in kinds

    def eat(self, kind: str) -> Token:
        t = self.peek()
        if t is None or t.kind != kind:
            actual = t.kind if t else "EOF"
            where = t.line if t else "end"
            raise ParseError(f"Expected {kind} but got {actual} at line {where}")
        self.pos += 1
        return t

    # ---- entry point ---------------------------------------------------
    def parse(self) -> Block:
        stmts: List[Node] = []
        while self.pos < len(self.tokens):
            stmts.append(self.parse_statement())
        return Block(stmts)

    def parse_statement(self) -> Node:
        t = self.peek()
        if t is None:
            raise ParseError("Unexpected end of input")
        if t.kind == "WHEN":
            return self.parse_when_statement()
        if t.kind == "IDENT" and self.peek(1) is not None and self.peek(1).kind == "=":
            return self.parse_assignment()
        raise ParseError(f"Unexpected token {t.kind} at line {t.line}")

    def parse_assignment(self) -> Assignment:
        name = self.eat("IDENT").value
        self.eat("=")
        return Assignment(name, self.parse_expr())

    def parse_when_statement(self) -> When:
        """`when cond: action`  or  `WHEN cond THEN assignment [ELSE assignment]`."""
        self.eat("WHEN")
        cond = self.parse_expr()

        if self.at(":"):
            self.eat(":")
            tok = self.peek()
            if tok is None:
                raise ParseError("Expected action after ':'")
            if tok.kind in ("LONG", "SHORT", "FLAT"):
                self.pos += 1
                return When(cond, action=tok.value)
            if tok.kind == "IDENT" and tok.value == "scale":
                self.pos += 1
                self.eat("(")
                arg = self.parse_expr()
                self.eat(")")
                return When(cond, action=f"scale({arg})")
            raise ParseError(f"Unknown action: {tok.kind}")

        if self.at("THEN"):
            self.eat("THEN")
            then = self.parse_assignment()
            els = None
            if self.at("ELSE"):
                self.eat("ELSE")
                els = self.parse_assignment()
            return When(cond, then_expr=then, else_expr=els)

        raise ParseError("Expected ':' or THEN after when-condition")

    # ---- expression precedence chain -----------------------------------
    def parse_expr(self) -> Node:
        return self.parse_or()

    def parse_or(self) -> Node:
        left = self.parse_and()
        while self.at("OR"):
            self.eat("OR")
            left = BinaryOp("or", left, self.parse_and())
        return left

    def parse_and(self) -> Node:
        left = self.parse_comparison()
        while self.at("AND"):
            self.eat("AND")
            left = BinaryOp("and", left, self.parse_comparison())
        return left

    def parse_comparison(self) -> Node:
        left = self.parse_add_sub()
        while self.at(">", "<", ">=", "<=", "==", "!="):
            op = self.peek().kind
            self.pos += 1
            left = BinaryOp(op, left, self.parse_add_sub())
        return left

    def parse_add_sub(self) -> Node:
        left = self.parse_mul_div()
        while self.at("+", "-"):
            op = self.peek().kind
            self.pos += 1
            left = BinaryOp(op, left, self.parse_mul_div())
        return left

    def parse_mul_div(self) -> Node:
        left = self.parse_unary()
        while self.at("*", "/"):
            op = self.peek().kind
            self.pos += 1
            left = BinaryOp(op, left, self.parse_unary())
        return left

    def parse_unary(self) -> Node:
        if self.at("-"):
            self.eat("-")
            return UnaryOp("-", self.parse_unary())
        return self.parse_primary()

    def parse_primary(self) -> Node:
        t = self.peek()
        if t is None:
            raise ParseError("Unexpected end of input")

        if t.kind == "NUMBER":
            self.pos += 1
            # Keep integer literals as int: pandas rejects float window args.
            return Constant(int(t.value) if "." not in t.value else float(t.value))

        if t.kind == "WHEN":
            # WHEN ... THEN ... ELSE ... used as an expression.
            self.eat("WHEN")
            cond = self.parse_expr()
            self.eat("THEN")
            then = self.parse_expr()
            els = None
            if self.at("ELSE"):
                self.eat("ELSE")
                els = self.parse_expr()
            return When(cond, then_expr=then, else_expr=els)

        if t.kind == "IDENT":
            if self.peek(1) is not None and self.peek(1).kind == "(":
                name = self.eat("IDENT").value
                self.eat("(")
                args: List[Node] = []
                if not self.at(")"):
                    args.append(self.parse_expr())
                    while self.at(","):
                        self.eat(",")
                        args.append(self.parse_expr())
                self.eat(")")
                return Call(name, args)
            self.pos += 1
            return Variable(t.value)

        if t.kind == "(":
            self.eat("(")
            e = self.parse_expr()
            self.eat(")")
            return e

        raise ParseError(f"Unexpected token {t.kind} at line {t.line}")


def parse(src: str) -> Block:
    return Parser(tokenize(src)).parse()


# Public aliases used by the test suite and documented in the README.
DSLSyntaxError = ParseError
