"""Lightweight backtest runner that pairs a compiled DSL strategy with
``yfinance`` price data.

The DSL emits weights ``w_t`` in ``[-1, 1]`` (long-short) or ``[0, 1]``
(long-only).  Position PnL is then

::

    r_{t+1} = w_t * (close_{t+1} - close_t) / close_t

which is a clean, transaction-cost-free approximation suitable for
illustrating the DSL in a CLI.  Real research-grade backtests should use
zipline / vectorbt / backtrader.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, List, Optional

import numpy as np
import pandas as pd


@dataclass
class BacktestResult:
    ticker: str
    n_periods: int
    total_return: float
    annualized_return: float
    annualized_vol: float
    sharpe: float
    max_drawdown: float
    n_observations: int

    def as_dict(self) -> dict:
        return {
            "ticker": self.ticker,
            "n_periods": self.n_periods,
            "n_observations": self.n_observations,
            "total_return": self.total_return,
            "annualized_return": self.annualized_return,
            "annualized_vol": self.annualized_vol,
            "sharpe": self.sharpe,
            "max_drawdown": self.max_drawdown,
        }


def _download(ticker: str, start: str, end: Optional[str]) -> pd.DataFrame:
    """Download OHLCV from yfinance. Lazy import so the CLI doesn't require
    yfinance for ``ast`` / ``compile`` / ``eval`` subcommands."""
    import yfinance as yf

    df = yf.download(
        ticker, start=start, end=end, progress=False, auto_adjust=True
    )
    if df.empty:
        raise RuntimeError(f"no data returned for {ticker}")
    # Flatten multi-index columns that yfinance now returns for single tickers.
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]
    return df.rename(columns=str.title)


def _strategy_returns(weights: pd.Series, close: pd.Series) -> pd.Series:
    """Apply weights to next-period returns."""
    rets = close.pct_change().shift(-1).fillna(0.0)
    # Align index.
    aligned_w = weights.reindex(close.index).fillna(0.0)
    return aligned_w * rets


def _summarize(ticker: str, rets: pd.Series) -> BacktestResult:
    rets = rets.dropna()
    if rets.empty:
        return BacktestResult(
            ticker=ticker, n_periods=0,
            total_return=0.0, annualized_return=0.0,
            annualized_vol=0.0, sharpe=0.0, max_drawdown=0.0,
            n_observations=0,
        )
    total = float((1.0 + rets).prod() - 1.0)
    days = max(len(rets), 1)
    cagr = float((1.0 + total) ** (252.0 / days) - 1.0) if days > 1 else 0.0
    vol = float(rets.std(ddof=0) * np.sqrt(252.0))
    sharpe = float(cagr / vol) if vol > 0 else 0.0
    cum = (1.0 + rets).cumprod()
    peak = cum.cummax()
    dd = float(((cum / peak) - 1.0).min())
    return BacktestResult(
        ticker=ticker,
        n_periods=days,
        n_observations=int(rets.ne(0).sum()),
        total_return=total,
        annualized_return=cagr,
        annualized_vol=vol,
        sharpe=sharpe,
        max_drawdown=dd,
    )


def run_backtest(
    strategy: Callable[[pd.DataFrame], pd.Series],
    tickers: Iterable[str],
    start: str,
    end: Optional[str] = None,
    long_short: bool = True,
) -> dict:
    """Run the strategy on each ticker and aggregate a JSON-friendly report."""
    results: List[dict] = []
    combined: List[pd.Series] = []
    for ticker in tickers:
        try:
            df = _download(ticker, start, end)
        except Exception as exc:
            results.append({"ticker": ticker, "error": str(exc)})
            continue
        weights = strategy(df)
        if not long_short:
            # Map [-1, 1] -> [0, 1] via (x + 1) / 2, then re-normalize.
            weights = ((weights + 1.0) / 2.0).clip(0.0, 1.0)
            weights = weights / weights.abs().sum().replace(0, 1) if weights.abs().sum() else weights
        rets = _strategy_returns(weights, df["Close"])
        combined.append(rets.rename(ticker))
        results.append(_summarize(ticker, rets).as_dict())

    # Aggregate equal-weighted portfolio if we have multiple tickers.
    if combined:
        aligned = pd.concat(combined, axis=1).fillna(0.0)
        port_rets = aligned.mean(axis=1)
        results.append({"portfolio": _summarize("PORTFOLIO", port_rets).as_dict()})
    return {"results": results}
