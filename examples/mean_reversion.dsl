# Mean reversion: z-score on 20-day returns
# Long when oversold, short when overbought
signal = zscore(returns(close, 20), 60)
weight = -signal
when weight > 1.0: long
when weight < -1.0: short
