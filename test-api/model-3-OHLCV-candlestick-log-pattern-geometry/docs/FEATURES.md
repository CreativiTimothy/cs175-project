# General Pattern Feature Dictionary

The system does not hardcode double tops, wedges, head-and-shoulders, or other named patterns. Every 100-bar sequence is represented by reusable geometric primitives:

- previous-close log geometry;
- candle body, range, wick fractions, and close location;
- 2, 3, 5, 10, 20, 25, and 50-bar log returns;
- distance from and position inside 5, 10, 20, 25, 50, and 100-bar ranges;
- normalized trailing slopes of close, high, and low;
- trend-line convergence, momentum change, and slope acceleration;
- volatility, true range, and range compression/expansion;
- volume changes, volume ratios, z-score, and return-volume correlation.

All rolling calculations are trailing. No centered rolling window or future observation is used as an input.
