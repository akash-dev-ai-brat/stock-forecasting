
## Limitations & Honest Analysis

1. **Look-ahead bias**: Models trained on split data; no future data leaked.
2. **No fundamental data**: We use only price/volume — earnings, news, macros ignored.
3. **LSTM overfits on low-volatility periods**: Rolling MAPE spikes during high
   volatility (COVID crash, rate hike cycles) — model struggles with black swans.
4. **Prophet assumes additive seasonality**: Works well for trends but misses
   sharp discontinuities in price action.
5. **MAPE limitation**: Misleading when actual prices are near zero (not an issue
   for stocks, but worth noting for penny stocks).
6. **Prediction ≠ trading signal**: Lower MAPE does not mean profitable trading.
   This is a forecasting exercise, not a trading strategy.
