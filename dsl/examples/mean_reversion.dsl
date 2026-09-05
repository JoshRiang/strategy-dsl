# Mean-reversion strategy using z-score of price vs its 20-day mean.
# When price is far below its mean -> buy; far above -> sell.
mean_20  = sma(Close, 20)
dev_20   = (Close - mean_20) / std(Close, 20)
weight   = -clip(zscore(dev_20), -1.0, 1.0)
