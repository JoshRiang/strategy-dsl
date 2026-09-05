"""Tests for the DSL tokenizer + parser."""

import pytest

from dsl.ast_nodes import (
    Assignment,
    BinaryOp,
    Call,
    Constant,
    Program,
    UnaryOp,
    Variable,
    When,
)
from dsl.parser import DSLSyntaxError, parse, tokenize


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------


class TestTokenizer:
    def test_basic_number(self):
        tokens = tokenize("42")
        assert tokens[0].kind == "NUMBER"
        assert tokens[0].value == "42"

    def test_skips_whitespace_and_comments(self):
        tokens = tokenize("  1  # a comment\n 2 \n")
        numbers = [t for t in tokens if t.kind == "NUMBER"]
        assert [t.value for t in numbers] == ["1", "2"]

    def test_line_tracking_after_comment(self):
        tokens = tokenize("# one\nx\n")
        ident = next(t for t in tokens if t.kind == "IDENT")
        assert ident.line == 2

    def test_unknown_character_raises(self):
        with pytest.raises(DSLSyntaxError):
            tokenize("@")


# ---------------------------------------------------------------------------
# Parser — primitives
# ---------------------------------------------------------------------------


class TestParserPrimitives:
    def test_number(self):
        program = parse("weight = 1.5\n")
        assert isinstance(program, Program)
        stmt = program.statements[0]
        assert isinstance(stmt, Assignment)
        assert isinstance(stmt.value, Constant)
        assert stmt.value.value == 1.5

    def test_variable_and_signal(self):
        program = parse("weight = Close\n")
        stmt = program.statements[0]
        assert isinstance(stmt.value, Variable)
        assert stmt.value.name == "Close"

    def test_call_with_args(self):
        program = parse("weight = sma(Close, 20)\n")
        call = program.statements[0].value
        assert isinstance(call, Call)
        assert call.name == "sma"
        assert len(call.args) == 2
        assert isinstance(call.args[0], Variable)
        assert isinstance(call.args[1], Constant)

    def test_unary_minus(self):
        program = parse("weight = -Close\n")
        assert isinstance(program.statements[0].value, UnaryOp)
        assert program.statements[0].value.op == "-"


# ---------------------------------------------------------------------------
# Parser — operators / precedence
# ---------------------------------------------------------------------------


class TestParserOperators:
    def test_addition(self):
        program = parse("weight = Close + Open\n")
        expr = program.statements[0].value
        assert isinstance(expr, BinaryOp) and expr.op == "+"

    def test_precedence_mul_over_add(self):
        program = parse("weight = a + b * c\n")
        expr = program.statements[0].value
        assert expr.op == "+"
        assert expr.right.op == "*"

    def test_parentheses_override_precedence(self):
        program = parse("weight = (a + b) * c\n")
        expr = program.statements[0].value
        assert expr.op == "*"
        assert expr.left.op == "+"

    def test_comparison(self):
        program = parse("weight = Close > sma(Close, 20)\n")
        expr = program.statements[0].value
        assert isinstance(expr, BinaryOp) and expr.op == ">"

    def test_logical_and_or(self):
        program = parse("weight = (Close > Open) and (Volume > 0)\n")
        expr = program.statements[0].value
        assert expr.op == "and"


# ---------------------------------------------------------------------------
# Parser — WHEN
# ---------------------------------------------------------------------------


class TestParserWhen:
    def test_when_then_else(self):
        program = parse("weight = WHEN Close > 0 THEN 1 ELSE -1\n")
        expr = program.statements[0].value
        assert isinstance(expr, When)
        assert expr.else_expr is not None

    def test_when_no_else(self):
        program = parse("weight = WHEN Close > 0 THEN 1\n")
        expr = program.statements[0].value
        assert isinstance(expr, When)
        assert expr.else_expr is None

    def test_top_level_when_statement(self):
        program = parse("WHEN Close > Open THEN weight = 1 ELSE weight = -1\n")
        assert isinstance(program.statements[0], When)


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


class TestParserErrors:
    def test_unexpected_eof(self):
        with pytest.raises(DSLSyntaxError):
            parse("weight = sma(Close, ")

    def test_unknown_top_level(self):
        with pytest.raises(DSLSyntaxError):
            parse("garbage 1 + 2\n")


# ---------------------------------------------------------------------------
# Realistic multi-statement program
# ---------------------------------------------------------------------------


def test_realistic_momentum_program():
    src = (
        "mom = returns(Close, 252)\n"
        "signal = ts_rank(mom, 252)\n"
        "weight = scale(zscore(rank(signal)))\n"
    )
    program = parse(src)
    assert len(program.statements) == 3
    assert all(isinstance(s, Assignment) for s in program.statements)
    assert program.statements[2].name == "weight"
