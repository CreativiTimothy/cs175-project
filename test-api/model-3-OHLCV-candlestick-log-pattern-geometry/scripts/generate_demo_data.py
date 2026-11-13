from pathlib import Path

import numpy as np
import pandas as pd


def main():
    """Create synthetic OHLC data only for pipeline testing, never for reported market results."""
    rng = np.random.default_rng(42)
    dates = pd.bdate_range("2015-01-01", periods=1800, tz="America/New_York")
    symbols = ["SPY", "QQQ", "AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL"]
    rows = []

    for symbol_index, symbol in enumerate(symbols):
        price = 70.0 + 15.0 * symbol_index
        for date in dates:
            overnight = rng.normal(0.0002, 0.006)
            intraday = rng.normal(0.0003, 0.012)
            open_price = max(1.0, price * (1.0 + overnight))
            close_price = max(1.0, open_price * (1.0 + intraday))
            high = max(open_price, close_price) * (1.0 + abs(rng.normal(0, 0.004)))
            low = min(open_price, close_price) * (1.0 - abs(rng.normal(0, 0.004)))
            volume = int(rng.lognormal(mean=15.0, sigma=0.35))
            rows.append([symbol, date, open_price, high, low, close_price, volume])
            price = close_price

    output = Path("data/raw/daily_bars.csv")
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=["symbol", "timestamp", "open", "high", "low", "close", "volume"]).to_csv(output, index=False)
    print(f"Saved synthetic demo data to {output}. Do not use it as real project evidence.")


if __name__ == "__main__":
    main()
