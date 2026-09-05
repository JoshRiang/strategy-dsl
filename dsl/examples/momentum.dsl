# Momentum strategy — classic 12-month return signal cross-sectionally ranked.
# Long top quintile, short bottom quintile (after rank -> zscore -> scale).
mom_12_1 = returns(Close, 252)
signal   = ts_rank(mom_12_1, 252)
weight   = scale(zscore(rank(signal)))
