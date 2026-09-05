# Project 9: Strategy DSL + Compiler

A small expression language for writing systematic trading strategies. Inspired by WorldQuant's alpha DSL and Zipline's pipeline API. Write strategies in plain text, compile them, and backtest on any OHLCV data.

## Why this exists
Strategy code in Python gets messy fast. A DSL keeps research clean: declare signals, combine them, output weights. No boilerplate. Strategy files become diffable, reviewable, and reusable.

## Example strategy
```dsl
# momentum.dsl
signal = ts_rank(returns(close, 20), 252)
weight = rank(signal) - 0.5
when weight > 0.1: long
when weight < -0.1: short
```

## Features
- Custom parser (recursive descent) → AST
- Stdlib: `sma`, `ema`, `rank`, `ts_rank`, `zscore`, `std`, `mean`, `returns`, `delta`
- Cross-sectional and time-series operations
- Compile to Python callable or backtest directly
- Sample strategies: momentum, mean reversion, multi-factor

## Quick start
```bash
pip install -r requirements.txt
python -m dsl --compile examples/momentum.dsl --backtest SPY 2010-01-01
```

## License
MIT
