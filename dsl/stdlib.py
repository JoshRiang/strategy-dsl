"""Built-in functions for the strategy DSL.

Every function takes a pandas Series (or a DataFrame for the OHLCV field
accessors) and returns a Series aligned to the input index.
"""
# Maintenance: last reviewed 2026-11-13 (min_periods, cross-sectional zscore, scale)
import numpy as np
import pandas as pd


def sma(series: pd.Series, window) -> pd.Series:
    return series.rolling(int(window), min_periods=1).mean()


def ema(series: pd.Series, window) -> pd.Series:
    return series.ewm(span=int(window), adjust=False).mean()


def std(series: pd.Series, window) -> pd.Series:
    return series.rolling(int(window), min_periods=1).std()


def mean(series: pd.Series, window) -> pd.Series:
    return series.rolling(int(window), min_periods=1).mean()


def zscore(series: pd.Series, window=None) -> pd.Series:
    """Z-score.

    With a window: rolling standardisation.
    Without one: cross-sectional standardisation across the whole series.
    A constant series returns 0 rather than NaN.
    """
    if window is None:
        mu = series.mean()
        sd = series.std()
    else:
        mu = sma(series, window)
        sd = std(series, window)
    out = (series - mu) / sd
    return out.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def rank(series: pd.Series) -> pd.Series:
    """Percentile rank in [0, 1]."""
    if not isinstance(series.index, pd.MultiIndex):
        return series.rank(pct=True)
    return series.groupby(level=0).rank(pct=True)


def ts_rank(series: pd.Series, window) -> pd.Series:
    """Percentile rank of each value within its rolling window."""
    return series.rolling(int(window), min_periods=1).rank(pct=True)


def returns(series: pd.Series, period=1) -> pd.Series:
    return series.pct_change(periods=int(period)).fillna(0.0)


def delta(series: pd.Series, period=1) -> pd.Series:
    return series.diff(periods=int(period)).fillna(0.0)


def scale(series, target: float = 1.0) -> pd.Series:
    """Normalise so the sum of absolute values equals `target`."""
    s = series if isinstance(series, pd.Series) else pd.Series(series, dtype="float64")
    total = float(s.abs().sum())
    if not np.isfinite(total) or total == 0.0:
        return pd.Series(0.0, index=s.index)
    return s * (float(target) / total)


def abs_(series):
    return abs(series)


# ---- OHLCV field accessors ------------------------------------------------
def _field(df: pd.DataFrame, name: str) -> pd.Series:
    if name in df.columns:
        return df[name]
    lower = name.lower()
    for col in df.columns:
        if str(col).lower() == lower:
            return df[col]
    raise KeyError(f"missing price column {name!r}")


def open_field(df: pd.DataFrame) -> pd.Series:
    return _field(df, "Open")


def high_field(df: pd.DataFrame) -> pd.Series:
    return _field(df, "High")


def low_field(df: pd.DataFrame) -> pd.Series:
    return _field(df, "Low")


def close_field(df: pd.DataFrame) -> pd.Series:
    return _field(df, "Close")


def volume_field(df: pd.DataFrame) -> pd.Series:
    return _field(df, "Volume")


STDLIB = {
    # series functions
    "sma": sma, "ema": ema, "std": std, "mean": mean, "zscore": zscore,
    "rank": rank, "ts_rank": ts_rank, "returns": returns, "delta": delta,
    "scale": scale, "abs": abs_,
    # OHLCV fields
    "open": open_field, "high": high_field, "low": low_field,
    "close": close_field, "volume": volume_field,
}
