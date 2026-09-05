# Multi-factor: combine momentum and mean-reversion with a volatility gate.
mom   = returns(Close, 60)
rev   = -(Close - sma(Close, 20)) / std(Close, 20)
vol   = std(returns(Close), 60)
gate  = (vol < 0.03) * 1.0 + (vol >= 0.03) * 0.5
weight = scale(gate * (zscore(rank(mom)) + zscore(rank(rev))))
