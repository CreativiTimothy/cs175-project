import argparse
import json
import os
import random
import time
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

DATA_URL = "https://data.alpaca.markets/v2/stocks/bars"
ASSETS_URL = "https://paper-api.alpaca.markets/v2/assets"


def api_headers():
    load_dotenv()
    key = os.getenv("ALPACA_API_KEY")
    secret = os.getenv("ALPACA_SECRET_KEY")
    if not key or not secret:
        raise RuntimeError("Set ALPACA_API_KEY and ALPACA_SECRET_KEY in .env")
    return {
        "APCA-API-KEY-ID": key,
        "APCA-API-SECRET-KEY": secret,
    }


def create_session():
    """Create one reusable HTTP session with connection pooling."""
    retry = Retry(
        total=0,
        connect=0,
        read=0,
        status=0,
        redirect=0,
        raise_on_status=False,
    )
    session = requests.Session()
    adapter = HTTPAdapter(max_retries=retry, pool_connections=20, pool_maxsize=20)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def load_symbols_from_file(path):
    symbols = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        symbol = line.strip().upper()
        if symbol and not symbol.startswith("#"):
            symbols.append(symbol)
    return sorted(set(symbols))


def get_active_symbols(session, headers, max_symbols=None):
    response = request_with_retry(
        session,
        ASSETS_URL,
        headers=headers,
        params={"status": "active", "asset_class": "us_equity"},
        connect_timeout=20,
        read_timeout=90,
        max_attempts=8,
    )
    exchanges = {"NASDAQ", "NYSE", "ARCA", "NYSEARCA", "BATS", "AMEX"}
    symbols = sorted(
        {
            asset["symbol"]
            for asset in response.json()
            if asset.get("tradable")
            and asset.get("status") == "active"
            and asset.get("exchange") in exchanges
        }
    )
    return symbols[:max_symbols] if max_symbols else symbols


def request_with_retry(
    session,
    url,
    headers,
    params,
    connect_timeout=30,
    read_timeout=180,
    max_attempts=10,
):
    """Retry HTTP errors, DNS/connect timeouts, and interrupted reads."""
    retryable_statuses = {408, 425, 429, 500, 502, 503, 504}

    for attempt in range(1, max_attempts + 1):
        try:
            response = session.get(
                url,
                headers=headers,
                params=params,
                timeout=(connect_timeout, read_timeout),
            )

            if response.status_code in retryable_statuses:
                retry_after = response.headers.get("Retry-After")
                if retry_after and retry_after.isdigit():
                    wait = float(retry_after)
                else:
                    wait = min(120.0, 2 ** (attempt - 1)) + random.uniform(0.0, 1.0)

                print(
                    f"HTTP {response.status_code} on attempt {attempt}/{max_attempts}; "
                    f"retrying in {wait:.1f}s"
                )
                time.sleep(wait)
                continue

            response.raise_for_status()
            return response

        except (
            requests.exceptions.ConnectTimeout,
            requests.exceptions.ReadTimeout,
            requests.exceptions.ConnectionError,
            requests.exceptions.ChunkedEncodingError,
        ) as error:
            if attempt == max_attempts:
                raise RuntimeError(
                    f"Alpaca request failed after {max_attempts} attempts: {error}"
                ) from error

            wait = min(120.0, 2 ** (attempt - 1)) + random.uniform(0.0, 1.0)
            print(
                f"Network error on attempt {attempt}/{max_attempts}: "
                f"{type(error).__name__}. Retrying in {wait:.1f}s"
            )
            time.sleep(wait)

    raise RuntimeError("Unexpected retry-loop exit")


def chunks(values, size):
    for start in range(0, len(values), size):
        yield values[start:start + size]


def load_checkpoint(path):
    if not path.exists():
        return {"completed_batches": []}
    return json.loads(path.read_text(encoding="utf-8"))


def save_checkpoint(path, checkpoint):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(checkpoint, indent=2), encoding="utf-8")
    temporary.replace(path)


def download_one_batch(
    session,
    headers,
    symbols,
    timeframe,
    start,
    end,
    feed,
    adjustment,
    page_limit,
    batch_file,
):
    """Download every page for one symbol batch, then atomically save it."""
    rows = []
    page_token = None
    page_number = 0

    while True:
        page_number += 1
        params = {
            "symbols": ",".join(symbols),
            "timeframe": timeframe,
            "start": start,
            "end": end,
            "limit": page_limit,
            "feed": feed,
            "adjustment": adjustment,
            "sort": "asc",
        }
        if page_token:
            params["page_token"] = page_token

        response = request_with_retry(
            session,
            DATA_URL,
            headers=headers,
            params=params,
            connect_timeout=30,
            read_timeout=300,
            max_attempts=10,
        )
        payload = response.json()

        page_rows = 0
        for symbol, bars in payload.get("bars", {}).items():
            for bar in bars:
                rows.append(
                    {
                        "symbol": symbol,
                        "timestamp": bar["t"],
                        "open": bar["o"],
                        "high": bar["h"],
                        "low": bar["l"],
                        "close": bar["c"],
                        "volume": bar["v"],
                    }
                )
                page_rows += 1

        print(
            f"  page {page_number}: {page_rows:,} bars "
            f"(batch total {len(rows):,})"
        )

        page_token = payload.get("next_page_token")
        if not page_token:
            break

    temporary = batch_file.with_suffix(batch_file.suffix + ".tmp")
    pd.DataFrame(rows).to_csv(temporary, index=False)
    temporary.replace(batch_file)
    return len(rows)


def combine_batches(batch_directory, output):
    files = sorted(batch_directory.glob("batch_*.csv"))
    if not files:
        raise RuntimeError("No completed batch files were found")

    temporary = output.with_suffix(output.suffix + ".tmp")
    wrote_header = False

    for batch_file in files:
        frame = pd.read_csv(batch_file)
        if frame.empty:
            continue
        frame.to_csv(
            temporary,
            mode="a",
            header=not wrote_header,
            index=False,
        )
        wrote_header = True

    if not wrote_header:
        raise RuntimeError("Completed batches did not contain any bars")

    frame = pd.read_csv(temporary, parse_dates=["timestamp"])
    frame = frame.sort_values(["symbol", "timestamp"]).drop_duplicates(
        ["symbol", "timestamp"]
    )
    frame.to_csv(output, index=False)
    temporary.unlink()
    return len(frame), frame["symbol"].nunique()


def download_bars(
    symbols,
    timeframe,
    start,
    end,
    output,
    feed="iex",
    adjustment="all",
    symbol_chunk_size=20,
    page_limit=10000,
    restart=False,
):
    """
    Download in resumable symbol batches.

    A completed symbol batch is never downloaded again unless --restart is
    supplied. If a page fails, only the current symbol batch is retried.
    """
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    work_directory = output.parent / f".{output.stem}_{timeframe}_download"
    batch_directory = work_directory / "batches"
    checkpoint_file = work_directory / "checkpoint.json"

    if restart and work_directory.exists():
        import shutil
        shutil.rmtree(work_directory)

    batch_directory.mkdir(parents=True, exist_ok=True)
    checkpoint = load_checkpoint(checkpoint_file)
    completed = set(checkpoint.get("completed_batches", []))

    session = create_session()
    headers = api_headers()
    symbol_batches = list(chunks(symbols, symbol_chunk_size))

    for batch_index, symbol_batch in enumerate(symbol_batches):
        batch_key = f"{batch_index:04d}"
        batch_file = batch_directory / f"batch_{batch_key}.csv"

        if batch_key in completed and batch_file.exists():
            print(
                f"{timeframe}: batch {batch_index + 1}/{len(symbol_batches)} "
                f"already complete; skipping"
            )
            continue

        print(
            f"{timeframe}: batch {batch_index + 1}/{len(symbol_batches)}, "
            f"{len(symbol_batch)} symbols"
        )

        row_count = download_one_batch(
            session=session,
            headers=headers,
            symbols=symbol_batch,
            timeframe=timeframe,
            start=start,
            end=end,
            feed=feed,
            adjustment=adjustment,
            page_limit=page_limit,
            batch_file=batch_file,
        )

        completed.add(batch_key)
        checkpoint = {
            "timeframe": timeframe,
            "start": start,
            "end": end,
            "feed": feed,
            "symbol_chunk_size": symbol_chunk_size,
            "completed_batches": sorted(completed),
            "last_completed_batch_rows": row_count,
        }
        save_checkpoint(checkpoint_file, checkpoint)

    total_rows, symbol_count = combine_batches(batch_directory, output)
    print(
        f"Saved {total_rows:,} {timeframe} bars for "
        f"{symbol_count:,} symbols to {output}"
    )
    print(f"Resume/checkpoint data remain in {work_directory}")


def main():
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--symbols-file")
    source.add_argument("--all-active", action="store_true")
    parser.add_argument("--max-symbols", type=int)
    parser.add_argument("--timeframe", required=True)
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--feed", default=os.getenv("ALPACA_DATA_FEED", "iex"))
    parser.add_argument("--adjustment", default="all")
    parser.add_argument("--symbol-chunk-size", type=int, default=20)
    parser.add_argument("--page-limit", type=int, default=10000)
    parser.add_argument("--restart", action="store_true")
    args = parser.parse_args()

    session = create_session()
    request_headers = api_headers()

    if args.symbols_file:
        symbols = load_symbols_from_file(args.symbols_file)
    else:
        symbols = get_active_symbols(
            session,
            request_headers,
            args.max_symbols,
        )

    if not symbols:
        raise RuntimeError("The symbol universe is empty")

    print(f"Universe: {len(symbols):,} symbols")
    download_bars(
        symbols=symbols,
        timeframe=args.timeframe,
        start=args.start,
        end=args.end,
        output=args.output,
        feed=args.feed,
        adjustment=args.adjustment,
        symbol_chunk_size=args.symbol_chunk_size,
        page_limit=args.page_limit,
        restart=args.restart,
    )


if __name__ == "__main__":
    main()
