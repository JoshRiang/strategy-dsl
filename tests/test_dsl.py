"""Tests for parser and compiler."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import numpy as np
import pandas as pd
from dsl.parser import parse, ParseError
from dsl.compiler import compile_src, CompilerError
from dsl.stdlib import sma, ema, zscore, ts_rank, rank, returns


def test_parse_simple_assignment():
    block = parse("x = 5")
    assert len(block.statements) == 1


def test_parse_binary():
    block = parse("x = 1 + 2 * 3")
    assert len(block.statements) == 1


def test_parse_function_call():
    block = parse("x = sma(close, 20)")
    assert len(block.statements) == 1


def test_parse_when_long():
    block = parse("when x > 0: long")
    assert len(block.statements) == 1


def test_parse_when_short():
    block = parse("when x < -0.5: short")
    assert len(block.statements) == 1


def test_parse_when_flat():
    block = parse("when abs(x) < 0.1: flat")
    assert len(block.statements) == 1


def test_parse_compound():
    src = """
signal = ts_rank(returns(close, 60), 252)
weight = rank(signal)
when weight > 0.5: long
when weight < 0.3: flat
"""
    block = parse(src)
    assert len(block.statements) == 3


def test_compile_runs():
    np.random.seed(0)
    n = 200
    df = pd.DataFrame({
        "Open": np.random.rand(n) + 100,
        "High": np.random.rand(n) + 100,
        "Low": np.random.rand(n) + 100,
        "Close": np.cumsum(np.random.randn(n)) + 100,
        "Volume": np.random.rand(n) * 1000,
    }, index=pd.date_range("2020-01-01", periods=n))
    strategy = compile_src("weight = 0.5\nwhen weight > 0: long")
    weights = strategy(df)
    assert len(weights) == n
    assert (weights == 1.0).all()


def test_stdlib_sma():
    s = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=float)
    out = sma(s, 3)
    assert abs(out.iloc[2] - 2.0) < 1e-9


def test_stdlib_zscore():
    s = pd.Series([1.0] * 60)
    out = zscore(s, 30)
    assert out.dropna().abs().max() < 1e-6  # constant = z = 0


if __name__ == "__main__":
    test_parse_simple_assignment()
    test_parse_binary()
    test_parse_function_call()
    test_parse_when_long()
    test_parse_when_short()
    test_parse_when_flat()
    test_parse_compound()
    test_compile_runs()
    test_stdlib_sma()
    test_stdlib_zscore()
    print("All tests passed")
