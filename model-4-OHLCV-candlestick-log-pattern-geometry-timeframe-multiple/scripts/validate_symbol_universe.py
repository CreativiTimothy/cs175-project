import argparse
import os
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

ASSETS_URL = "https://paper-api.alpaca.markets/v2/assets"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols-file", default="configs/symbols_100.txt")
    parser.add_argument("--output", default="outputs/symbol_validation.csv")
    args = parser.parse_args()

    load_dotenv()
    key = os.getenv("ALPACA_API_KEY")
    secret = os.getenv("ALPACA_SECRET_KEY")
    if not key or not secret:
        raise RuntimeError("Set ALPACA_API_KEY and ALPACA_SECRET_KEY in .env")

    requested = [
        line.strip().upper()
        for line in Path(args.symbols_file).read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    ]
    response = requests.get(
        ASSETS_URL,
        headers={"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret},
        params={"status": "active", "asset_class": "us_equity"},
        timeout=60,
    )
    response.raise_for_status()
    assets = {asset["symbol"]: asset for asset in response.json()}

    rows = []
    for symbol in requested:
        asset = assets.get(symbol, {})
        rows.append(
            {
                "symbol": symbol,
                "found": bool(asset),
                "active": asset.get("status") == "active",
                "tradable": bool(asset.get("tradable", False)),
                "fractionable": bool(asset.get("fractionable", False)),
                "exchange": asset.get("exchange"),
                "name": asset.get("name"),
            }
        )

    result = pd.DataFrame(rows)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)
    print(result.to_string(index=False))
    invalid = result[~(result["found"] & result["active"] & result["tradable"])]
    print(f"\nUsable: {len(result) - len(invalid)}/{len(result)}")
    if len(invalid):
        print("Replace or remove these symbols before downloading:")
        print(invalid[["symbol", "found", "active", "tradable"]].to_string(index=False))


if __name__ == "__main__":
    main()
