# Stock Direction AI
# Model: Features are candlesticks (OHLC + volume) with logarithmic price change, with pattern detection (candle shape, returns, range location, trend geometry, curvature, compression, volatility, and volume) + multiple time frames to predict one time frame.

COMPSCI 175, Group 1

- Timothy Nguyen (`timotn14`, 32158000)
- Jonathan Yen Tran (`jonatyt1`, 45326114)
- Pratham Asher (`pbasher`, 12404157)

## Important API behavior

Alpaca's 10,000 limit applies to one response page across all symbols, not 10,000 bars per symbol. The downloader follows every `next_page_token` and divides large universes into symbol batches.

## Recommended staged universe

Do not begin with every listed asset and every one-minute bar. Storage and preprocessing grow extremely quickly. Use:

1. 100 liquid symbols for pipeline validation.
2. 500-1,000 liquid symbols for the main experiment.
3. Expand further only after measuring disk, RAM, missing bars, and training time.

A 4,000-symbol request with one-minute data can produce hundreds of millions of bars. The code supports pagination, but the local machine may not support that data volume.

## Download 100 symbols first

```cmd
python -m src.download_multitimeframe --config configs/multitimeframe_daily.yaml --all-active --max-symbols 100 --end 2026-11-18
```

## Train

```cmd
python -m src.train_multitimeframe --config configs/multitimeframe_daily.yaml
```

## Change prediction target

Change only:

```yaml
data:
  target_timeframe: 1day
```

Possible configured values include `1week`, `1day`, `4hour`, `2hour`, `1hour`, `30min`, `15min`, `5min`, and `1min`. The label is always the direction of the next bar in the selected target timeframe. Every context bar is aligned by `available_at`, which is bar start plus bar duration, so an unfinished or future context bar cannot leak into the prediction.

## Practical data-range warning

All configured timeframes must overlap. If one-minute data start in July 2026, daily training examples before July 2026 cannot contain one-minute context and are omitted. To retain a long daily history, either download a longer one-minute history permitted by the subscription or temporarily remove extremely short timeframes from the configuration for the first large-universe experiment.


# Curated 100-symbol starter universe

Copy:
- `symbols_100.txt` to your project as `configs/symbols_100.txt`
- `multitimeframe_daily_100.yaml` to `configs/multitimeframe_daily_100.yaml`
- `validate_symbol_universe.py` to `scripts/validate_symbol_universe.py`

Run from the project root:

```cmd
python scripts/validate_symbol_universe.py --symbols-file configs/symbols_100.txt
```

Then download the six-timeframe starter dataset:

```cmd
python -m src.download_multitimeframe --config configs/multitimeframe_daily_100.yaml --symbols-file configs/symbols_100.txt --end 2026-11-18
```

Then train:

```cmd
python -m src.train_multitimeframe --config configs/multitimeframe_daily_100.yaml
```

This starter config intentionally excludes 15-minute, 5-minute, and 1-minute bars. Add those later as separate ablation experiments after checking disk usage, common symbol coverage, and model performance.
