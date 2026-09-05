"""CLI: python -m dsl --compile examples/momentum.dsl --backtest SPY 2010-01-01"""
import argparse
import sys
import pandas as pd
import yfinance as yf
from pathlib import Path
from .compiler import compile_src


EXAMPLES = Path(__file__).parent / "examples"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--compile", required=True, help="path to .dsl file")
    p.add_argument("--backtest", help="ticker to backtest on")
    p.add_argument("--start", default="2010-01-01")
    p.add_argument("--out", help="save weights to CSV")
    args = p.parse_args()

    src = Path(args.compile).read_text()
    print(f"=== Compiling {args.compile} ===")
    strategy = compile_src(src)
    print("OK: strategy compiled")

    if args.backtest:
        print(f"\n=== Backtesting on {args.backtest} from {args.start} ===")
        df = yf.download(args.backtest, start=args.start, progress=False)
        if df.empty:
            print("No data", file=sys.stderr)
            sys.exit(1)
        weights = strategy(df)
        print(f"weights: mean={weights.mean():.3f}, min={weights.min():.3f}, max={weights.max():.3f}")
        ret = (df["Close"].pct_change() * weights.shift(1)).dropna()
        if ret.std() > 0:
            sharpe = float(ret.mean() / ret.std() * (252 ** 0.5))
            print(f"  backtest Sharpe: {sharpe:.2f}")
            print(f"  total return: {(1 + ret).prod() - 1:.2%}")
        if args.out:
            weights.to_csv(args.out)
            print(f"  saved weights to {args.out}")


if __name__ == "__main__":
    main()
