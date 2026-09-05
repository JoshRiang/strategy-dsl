"""Built-in functions for the strategy DSL."""
import pandas as pd
import numpy as np


def sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window).mean()


def ema(series: pd.Series, window: int) -> pd.Series:
    return series.ewm(span=window, adjust=False).mean()


def std(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window).std()


def mean(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window).mean()


def zscore(series: pd.Series, window: int) -> pd.Series:
    return (series - sma(series, window)) / std(series, window)


def rank(series: pd.Series) -> pd.Series:
    """Cross-sectional rank at each timestamp."""
    if not isinstance(series.index, pd.MultiIndex):
        return series.rank(pct=True)
    return series.groupby(level=0).rank(pct=True)


def ts_rank(series: pd.Series, window: int) -> pd.Series:
    """Time-series percentile rank over rolling window."""
    return series.rolling(window).rank(pct=True)


def returns(series: pd.Series, period: int = 1) -> pd.Series:
    return series.pct_change(periods=period)


def delta(series: pd.Series, period: int = 1) -> pd.Series:
    return series.diff(periods=period)


# Fields
def open_field(df: pd.DataFrame) -> pd.Series:
    return df["Open"]


def high_field(df: pd.DataFrame) -> pd.Series:
    return df["High"]


def low_field(df: pd.DataFrame) -> pd.Series:
    return df["Low"]


def close_field(df: pd.DataFrame) -> pd.Series:
    return df["Close"]


def volume_field(df: pd.DataFrame) -> pd.Series:
    return df["Volume"]


STDLIB = {
    # functions on series
    "sma": sma, "ema": ema, "std": std, "mean": mean, "zscore": zscore,
    "rank": rank, "ts_rank": ts_rank, "returns": returns, "delta": delta,
    # dataframe fields
    "open": open_field, "high": high_field, "low": low_field,
    "close": close_field, "volume": volume_field,
}
