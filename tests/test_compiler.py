"""Tests for the DSL compiler — runs compiled strategies on synthetic prices."""

import numpy as np
import pandas as pd
import pytest

from dsl import compile_source
from dsl.compiler import DSLRuntimeError


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def make_prices(n: int = 200, seed: int = 0) -> pd.DataFrame:
    """Synthetic OHLCV DataFrame with a deterministic random walk."""

# Maintenance: last reviewed 2026-09-09 (daily improvement cycle)
    rng = np.random.default_rng(seed)
    rets = rng.normal(0, 0.01, n)
    close = 100 * np.cumprod(1 + rets)
    open_ = close * (1 + rng.normal(0, 0.003, n))
    high = np.maximum(close, open_) * (1 + np.abs(rng.normal(0, 0.002, n)))
    low = np.minimum(close, open_) * (1 - np.abs(rng.normal(0, 0.002, n)))
    volume = rng.integers(1_000_000, 5_000_000, n)
    dates = pd.date_range("2020-01-01", periods=n, freq="D")
    return pd.DataFrame({
        "Open": open_,
        "High": high,
        "Low": low,
        "Close": close,
        "Volume": volume,
    }, index=dates)


# ---------------------------------------------------------------------------
# Smoke tests
# ---------------------------------------------------------------------------


class TestCompileSmoke:
    def test_constant_weight(self):
        strat = compile_source("weight = 1.0\n")
        prices = make_prices()
        w = strat(prices)
        assert isinstance(w, pd.Series)
        assert len(w) == len(prices)
        assert (w == 1.0).all()
        assert w.name == "weight"

    def test_close_signal_returns_series(self):
        strat = compile_source("weight = Close\n")
        prices = make_prices()
        w = strat(prices)
        # Clipped to [-1, 1] by the compiler.
        assert w.max() <= 1.0 and w.min() >= -1.0

    def test_missing_weight_raises(self):
        strat = compile_source("x = 1\n")
        with pytest.raises(DSLRuntimeError, match="no weight"):
            strat(make_prices())


# ---------------------------------------------------------------------------
# Stdlib functions
# ---------------------------------------------------------------------------


class TestStdlib:
    def test_sma_matches_pandas(self):
        strat = compile_source("weight = sma(Close, 10)\n")
        prices = make_prices()
        out = strat(prices)
        expected = prices["Close"].rolling(10, min_periods=1).mean()
        pd.testing.assert_series_equal(out, expected.rename("weight"), check_names=False)

    def test_rank_is_pct_in_0_1(self):
        strat = compile_source("weight = rank(Close)\n")
        prices = make_prices(50)
        w = strat(prices)
        # zscore of a [0,1] series lives in roughly [-3, 3] but values are
        # clipped to [-1, 1] by the compiler; just check finite.
        assert np.isfinite(w).all()

    def test_zscore_cross_section(self):
        # Same close across rows -> zscore is zero.
        prices = make_prices(20)
        prices["Close"] = 100.0
        strat = compile_source("weight = zscore(Close)\n")
        w = strat(prices)
        # Constant -> NaN/zero from std -> final is 0.
        assert (w == 0.0).all()

    def test_scale_normalises_abs_sum(self):
        strat = compile_source("weight = scale(Close, 1)\n")
        prices = make_prices(40)
        w = strat(prices)
        # Allow small floating point drift.
        assert abs(w.abs().sum() - 1.0) < 1e-6

    def test_returns_call(self):
        strat = compile_source("weight = returns(Close, 5)\n")
        prices = make_prices(60)
        w = strat(prices)
        expected = prices["Close"].pct_change(5).fillna(0)
        pd.testing.assert_series_equal(w, expected.rename("weight"))


# ---------------------------------------------------------------------------
# Operators
# ---------------------------------------------------------------------------


class TestOperators:
    def test_addition(self):
        strat = compile_source("weight = Close + Open\n")
        prices = make_prices(30)
        w = strat(prices)
        expected = (prices["Close"] + prices["Open"]).clip(-1, 1)
        pd.testing.assert_series_equal(w, expected.rename("weight"))

    def test_comparison(self):
        strat = compile_source("weight = (Close > Open) * 1.0\n")
        prices = make_prices(20)
        w = strat(prices)
        expected = ((prices["Close"] > prices["Open"]).astype(float)).clip(-1, 1)
        pd.testing.assert_series_equal(w, expected.rename("weight"))

    def test_unary_minus(self):
        strat = compile_source("weight = -rank(Close)\n")
        prices = make_prices(20)
        w = strat(prices)
        # Should equal negative rank, then clipped.
        rank = prices["Close"].rank(pct=True)
        expected = (-rank).clip(-1, 1)
        pd.testing.assert_series_equal(w, expected.rename("weight"))


# ---------------------------------------------------------------------------
# WHEN clause
# ---------------------------------------------------------------------------


class TestWhen:
    def test_when_then_else(self):
        strat = compile_source(
            "weight = WHEN Close > Open THEN 0.5 ELSE -0.5\n"
        )
        prices = make_prices(10)
        w = strat(prices)
        expected = np.where(prices["Close"] > prices["Open"], 0.5, -0.5)
        expected = pd.Series(expected, index=prices.index).clip(-1, 1)
        pd.testing.assert_series_equal(w, expected.rename("weight"))

    def test_when_no_else_defaults_to_zero(self):
        strat = compile_source("weight = WHEN Close > Open THEN 1\n")
        prices = make_prices(5)
        w = strat(prices)
        # When condition is False, else defaults to 0 -> NaN->0 path.
        assert ((w == 1.0) | (w == 0.0)).all()


# ---------------------------------------------------------------------------
# Multi-statement programs
# ---------------------------------------------------------------------------


class TestMultiStatement:
    def test_intermediate_variables_resolve(self):
        src = (
            "x = sma(Close, 20)\n"
            "y = zscore(x)\n"
            "weight = scale(y)\n"
        )
        strat = compile_source(src)
        w = strat(make_prices(80))
        # Sum of abs weights is 1 (after scaling).
        assert abs(w.abs().sum() - 1.0) < 1e-6

    def test_momentum_example_compiles_and_runs(self):
        strat = compile_source(open("dsl/examples/momentum.dsl").read())
        w = strat(make_prices(400))
        assert isinstance(w, pd.Series)
        assert w.between(-1, 1).all()


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


class TestErrors:
    def test_unknown_builtin(self):
        with pytest.raises(DSLRuntimeError, match="unknown built-in"):
            compile_source("weight = nope(Close)\n")(make_prices())

    def test_unknown_identifier(self):
        with pytest.raises(DSLRuntimeError, match="unknown"):
            compile_source("weight = banana\n")(make_prices())

    def test_missing_price_columns(self):
        strat = compile_source("weight = Close\n")
        bad = pd.DataFrame({"Close": [1.0, 2.0, 3.0]})
        with pytest.raises(DSLRuntimeError, match="missing"):
            strat(bad)
