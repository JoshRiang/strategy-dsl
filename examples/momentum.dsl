# 12-1 month momentum strategy
# Long when signal is in top decile, short when bottom
signal = ts_rank(returns(close, 60), 252)
weight = rank(signal) - 0.5
when weight > 0.4: long
when weight < -0.4: short
