import argparse
import os
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from alpaca.data.enums import Adjustment, DataFeed
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit


def parse_timeframe(value):
    mapping = {
        "1Day": TimeFrame.Day,
        "1Week": TimeFrame.Week,
        "30Min": TimeFrame(30, TimeFrameUnit.Minute),
    }
    if value not in mapping:
        raise ValueError(f"Unsupported timeframe {value}. Choose from {sorted(mapping)}")
    return mapping[value]


def download_bars(symbols, start, end, timeframe, output):
    load_dotenv()
    api_key = os.getenv("ALPACA_API_KEY")
    secret_key = os.getenv("ALPACA_SECRET_KEY")
    if not api_key or not secret_key:
        raise RuntimeError("Set ALPACA_API_KEY and ALPACA_SECRET_KEY in .env before downloading data.")

    feed_name = os.getenv("ALPACA_DATA_FEED", "iex").lower()
    feed = DataFeed.IEX if feed_name == "iex" else DataFeed.SIP
    client = StockHistoricalDataClient(api_key, secret_key)
    request = StockBarsRequest(
        symbol_or_symbols=list(symbols),
        timeframe=parse_timeframe(timeframe),
        start=datetime.fromisoformat(start).replace(tzinfo=timezone.utc),
        end=datetime.fromisoformat(end).replace(tzinfo=timezone.utc),
        adjustment=Adjustment.ALL,
        feed=feed,
    )
    bars = client.get_stock_bars(request).df.reset_index()
    if bars.empty:
        raise RuntimeError("Alpaca returned no bars. Check symbols, dates, credentials, and feed access.")

    bars = bars.rename(columns={"symbol": "symbol", "timestamp": "timestamp"})
    wanted = ["symbol", "timestamp", "open", "high", "low", "close", "volume"]
    bars = bars[wanted].sort_values(["symbol", "timestamp"])
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    bars.to_csv(output, index=False)
    print(f"Saved {len(bars):,} rows for {bars['symbol'].nunique()} symbols to {output}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols", nargs="+", required=True)
    parser.add_argument("--start", required=True, help="YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="YYYY-MM-DD")
    parser.add_argument("--timeframe", choices=["1Day", "1Week", "30Min"], default="1Day")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    download_bars(args.symbols, args.start, args.end, args.timeframe, args.output)


if __name__ == "__main__":
    main()
