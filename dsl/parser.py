"""Recursive-descent parser for the strategy DSL.

Grammar (informal):
    program     := statement*
    statement   := assignment | when
    assignment  := IDENT '=' expr NEWLINE
    when        := 'when' expr ':' action
    action      := 'long' | 'short' | 'flat' | 'scale' '(' expr ')'
    expr        := term (('+'|'-') term)*
    term        := factor (('*'|'/') factor)*
    factor      := call | NUMBER | IDENT | '(' expr ')' | '-' factor
    call        := IDENT '(' args ')'
    args        := expr (',' expr)*
"""

# Maintenance: last reviewed 2026-09-12 (daily improvement cycle)
from typing import List, Optional
from .ast_nodes import (
    Node, Number, Variable, Call, BinaryOp, UnaryOp, Assignment, When, Block,
)


class ParseError(Exception):
    pass


class Token:
    def __init__(self, kind: str, value: str, pos: int):
        self.kind = kind
        self.value = value
        self.pos = pos
    def __repr__(self):
        return f"Token({self.kind}, {self.value!r}, {self.pos})"


def tokenize(src: str) -> List[Token]:
    tokens = []
    i = 0
    while i < len(src):
        c = src[i]
        if c.isspace():
            i += 1
            continue
        if c == '#':
            while i < len(src) and src[i] != '\n':
                i += 1
            continue
        if c.isalpha() or c == '_':
            j = i
            while j < len(src) and (src[j].isalnum() or src[j] == '_'):
                j += 1
            word = src[i:j]
            if word in ('when', 'long', 'short', 'flat', 'scale'):
                tokens.append(Token(word.upper(), word, i))
            else:
                tokens.append(Token('IDENT', word, i))
            i = j
        elif c.isdigit() or (c == '-' and i + 1 < len(src) and src[i+1].isdigit()):
            j = i + 1
            while j < len(src) and (src[j].isdigit() or src[j] == '.'):
                j += 1
            tokens.append(Token('NUMBER', src[i:j], i))
            i = j
        elif c in '+-*/()=,:':
            tokens.append(Token(c, c, i))
            i += 1
        else:
            raise ParseError(f"Unexpected character {c!r} at {i}")
    return tokens


class Parser:
    def __init__(self, tokens: List[Token]):
        self.tokens = tokens
        self.pos = 0

    def peek(self, offset: int = 0) -> Optional[Token]:
        idx = self.pos + offset
        return self.tokens[idx] if idx < len(self.tokens) else None

    def eat(self, kind: str) -> Token:
        t = self.peek()
        if t is None or t.kind != kind:
            actual = t.kind if t else 'EOF'
            raise ParseError(f"Expected {kind} but got {actual} at {t.pos if t else 'end'}")
        self.pos += 1
        return t

    def parse(self) -> Block:
        stmts = []
        while self.pos < len(self.tokens):
            stmts.append(self.parse_statement())
        return Block(stmts)

    def parse_statement(self) -> Node:
        t = self.peek()
        if t.kind == 'IDENT' and self.peek(1) and self.peek(1).kind == '=':
            name = self.eat('IDENT').value
            self.eat('=')
            expr = self.parse_expr()
            return Assignment(name, expr)
        if t.kind == 'WHEN':
            return self.parse_when()
        raise ParseError(f"Unexpected token {t.kind} at {t.pos}")

    def parse_when(self) -> When:
        self.eat('WHEN')
        cond = self.parse_expr()
        self.eat(':')
        action_tok = self.peek()
        if action_tok is None:
            raise ParseError("Expected action after ':'")
        if action_tok.kind in ('LONG', 'SHORT', 'FLAT'):
            self.pos += 1
            return When(cond, action_tok.value)
        if action_tok.kind == 'SCALE':
            self.eat('SCALE')
            self.eat('(')
            arg = self.parse_expr()
            self.eat(')')
            return When(cond, f"scale({arg})")
        raise ParseError(f"Unknown action: {action_tok.kind}")

    def parse_expr(self) -> Node:
        return self.parse_add_sub()

    def parse_add_sub(self) -> Node:
        left = self.parse_mul_div()
        while self.peek() and self.peek().kind in ('+', '-'):
            op = self.eat(self.peek().kind).value
            right = self.parse_mul_div()
            left = BinaryOp(op, left, right)
        return left

    def parse_mul_div(self) -> Node:
        left = self.parse_unary()
        while self.peek() and self.peek().kind in ('*', '/'):
            op = self.eat(self.peek().kind).value
            right = self.parse_unary()
            left = BinaryOp(op, left, right)
        return left

    def parse_unary(self) -> Node:
        if self.peek() and self.peek().kind == '-':
            self.eat('-')
            return UnaryOp('-', self.parse_unary())
        return self.parse_primary()

    def parse_primary(self) -> Node:
        t = self.peek()
        if t is None:
            raise ParseError("Unexpected end of input")
        if t.kind == 'NUMBER':
            self.pos += 1
            return Number(float(t.value))
        if t.kind == 'IDENT':
            if self.peek(1) and self.peek(1).kind == '(':
                name = self.eat('IDENT').value
                self.eat('(')
                args = []
                if not (self.peek() and self.peek().kind == ')'):
                    args.append(self.parse_expr())
                    while self.peek() and self.peek().kind == ',':
                        self.eat(',')
                        args.append(self.parse_expr())
                self.eat(')')
                return Call(name, args)
            self.pos += 1
            return Variable(t.value)
        if t.kind == '(':
            self.eat('(')
            e = self.parse_expr()
            self.eat(')')
            return e
        raise ParseError(f"Unexpected token {t.kind} at {t.pos}")


def parse(src: str) -> Block:
    return Parser(tokenize(src)).parse()
