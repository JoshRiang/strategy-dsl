# Multi-factor: combine momentum and value
# Value proxy: inverse of recent returns
mom = ts_rank(returns(close, 60), 252)
val = ts_rank(returns(close, 252), 504)
weight = rank(mom) + rank(val) - 1.0
when weight > 0.3: long
when weight < -0.3: short
