# Strategy DSL

Domain-specific language for trading strategies: parse, compile, backtest.

A small expression language for defining trading strategies as weight signals. Grammar:

```
program     := statement*
statement   := assignment | when
assignment  := IDENT '=' expr NEWLINE
when        := 'when' expr ':' action
action      := 'long' | 'short' | 'flat' | 'scale' '(' expr ')'
expr        := term (('+' | '-') term)*
term        := factor (('*' | '/') factor)*
factor      := call | NUMBER | IDENT | '(' expr ')' | '-' factor
call        := IDENT '(' args ')'
```

Stdlib functions: `sma`, `ema`, `std`, `mean`, `zscore`, `rank`, `ts_rank`, `returns`, `delta`, plus OHLCV field accessors (`open`, `high`, `low`, `close`, `volume`). The compiler turns a `.dsl` source file into a Python callable `strategy(prices_df) -> weights Series` clipped to `[-1, 1]`.

```bash
git clone https://github.com/JoshRiang/strategy-dsl.git
cd strategy-dsl
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

```bash
# Compile a strategy file
python -m dsl --compile dsl/examples/momentum.dsl

# Compile and backtest on SPY from 2010
python -m dsl --compile dsl/examples/momentum.dsl --backtest SPY --start 2010-01-01

# Save weights to CSV
python -m dsl --compile dsl/examples/momentum.dsl --backtest SPY --out weights.csv
```

Example `.dsl` file (`dsl/examples/momentum.dsl`):

```dsl
# 12-1 month momentum strategy
signal = ts_rank(returns(close, 60), 252)
weight = rank(signal) - 0.5
when weight > 0.4: long
when weight < -0.4: short
```

```python
from dsl import compile_src

src = """
signal = ts_rank(returns(close, 60), 252)
weight = rank(signal) - 0.5
when weight > 0.4: long
when weight < -0.4: short
"""

strategy = compile_src(src)

import pandas as pd, numpy as np
dates = pd.date_range('2020-01-01', periods=400, freq='D')
close = 100 * np.exp(np.cumsum(np.random.default_rng(0).normal(0.0005, 0.02, 400)))
df = pd.DataFrame({'Open': close, 'High': close, 'Low': close, 'Close': close, 'Volume': 1_000_000}, index=dates)

weights = strategy(df)
print(f'Mean weight: {weights.mean():.3f}, min: {weights.min():.3f}, max: {weights.max():.3f}')
```

Project structure:

```
strategy-dsl/
├── dsl/
│   ├── __init__.py       # exports: Parser, ParseError, Compiler, CompilerError, STDLIB
│   ├── ast_nodes.py      # Node, Number, Variable, Call, BinaryOp, UnaryOp, Assignment, When, Block
│   ├── grammar.py        # informal grammar documentation
│   ├── parser.py         # recursive-descent Parser: tokenize + parse (precedence climbing)
│   ├── compiler.py       # Compiler: compile(block) -> strategy(df) -> weights; compile_src
│   ├── stdlib.py         # STDLIB: sma, ema, std, mean, zscore, rank, ts_rank, returns, delta + OHLCV
│   ├── backtest.py       # run_backtest: compiled strategy × yfinance -> JSON report
│   └── __main__.py       # CLI: --compile --backtest --start --out
├── dsl/examples/
│   ├── momentum.dsl
│   ├── mean_reversion.dsl
│   └── multi_factor.dsl
├── tests/
│   ├── conftest.py
│   ├── test_dsl.py       # tokenizer, parser precedence, when clauses, realistic momentum program
│   ├── test_compiler.py  # compiled strategies on synthetic prices: stdlib, operators, when, multi-statement
│   └── test_parser.py    # parser/compiler integration
├── requirements.txt      # pandas, numpy, yfinance
├── LICENSE
└── README.md
```

Testing:

```bash
pytest -v
```

License: MIT — see [LICENSE](LICENSE).

Author: Joshua Riangkamang — [github.com/JoshRiang](https://github.com/JoshRiang)
