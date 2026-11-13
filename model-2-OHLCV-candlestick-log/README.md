# Stock Direction AI
# Model: Features are candlesticks (OHLC + volume) with logarithmic price change.

COMPSCI 175, Group 1

- Timothy Nguyen (`timotn14`, 32158000)
- Jonathan Yen Tran (`jonatyt1`, 45326114)
- Pratham Asher (`pbasher`, 12404157)

This project predicts the probability that a stock or ETF will close higher on the next trading day. It includes:

- a feedforward neural network (FNN), the primary course-aligned model;
- an always-up baseline and logistic-regression baseline;
- an optional LSTM sequence model;
- an optional next-day percentage-return regression model;
- daily, weekly, and 30-minute bar support;
- chronological 70/15/15 train/validation/test splitting;
- train-only normalization to prevent leakage;
- precision, recall, F1, ROC-AUC, Brier score, log loss, confusion matrices, and calibration plots;
- a historical top-k paper-trading backtest with SPY buy-and-hold comparison;
- a Streamlit dashboard for inspecting saved predictions;
- tests for model shapes, chronological splitting, and leakage safeguards.

The project does not place live orders. It is an educational research and paper-trading system.

## 1. Setup

### Docker (recommended)

```bash
docker build -t stock-direction-ai .
docker run --rm -it --env-file .env -v "${PWD}:/workspace" stock-direction-ai bash
```

On Windows PowerShell, `${PWD}` works. On Command Prompt, use `%CD%`.

### Local Python

Python 3.11 is recommended.

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

macOS/Linux:

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

Copy `.env.example` to `.env`. Add Alpaca credentials only if downloading real data.

## 2. Quick smoke test without API credentials

The demo generator creates clearly labeled synthetic data so the pipeline can be tested without pretending the data are real market observations.

```bash
python scripts/generate_demo_data.py
python -m src.train --config configs/daily_fnn.yaml
python -m src.evaluate --run-dir outputs/daily_fnn
python -m src.baselines --config configs/daily_fnn.yaml
python -m src.backtest --predictions outputs/daily_fnn/test_predictions.csv --data data/processed/model_rows.csv --output outputs/daily_fnn/backtest
```

## 3. Download real Alpaca data

Alpaca credentials are read from environment variables, never committed to source control.

```bash
python -m src.data.alpaca_downloader \
  --symbols SPY QQQ AAPL MSFT NVDA AMZN META GOOGL \
  --start 2016-01-01 \
  --end 2026-01-01 \
  --timeframe 1Day \
  --output data/raw/daily_bars.csv
```

The official `alpaca-py` client supports stock historical bars. The free IEX feed is selected by default in the downloader.

For weekly bars:

```bash
python -m src.data.alpaca_downloader --symbols SPY QQQ AAPL MSFT --start 2010-01-01 --end 2026-01-01 --timeframe 1Week --output data/raw/weekly_bars.csv
```

For 30-minute bars, use a shorter date range because intraday data are much larger:

```bash
python -m src.data.alpaca_downloader --symbols SPY QQQ AAPL MSFT --start 2024-01-01 --end 2026-01-01 --timeframe 30Min --output data/raw/30min_bars.csv
```

## 4. Train models

Primary FNN:

```bash
python -m src.train --config configs/daily_fnn.yaml
```

Optional LSTM:

```bash
python -m src.train --config configs/daily_lstm.yaml
```

Optional percentage-return regression:

```bash
python -m src.train --config configs/daily_regression.yaml
```

Weekly and 30-minute experiments:

```bash
python -m src.train --config configs/weekly_fnn.yaml
python -m src.train --config configs/intraday_30min_fnn.yaml
```

## 5. Evaluate and compare baselines

```bash
python -m src.evaluate --run-dir outputs/daily_fnn
python -m src.baselines --config configs/daily_fnn.yaml
```

The always-up baseline is essential because broad equity markets rise on more than half of many long historical samples. Logistic regression tests whether the neural network's extra complexity provides value over a simpler learned model.

## 6. Generate the latest probability ranking

After training, update the CSV with the newest available Alpaca bars and run:

```bash
python -m src.predict_latest --run-dir outputs/daily_fnn --config configs/daily_fnn.yaml
```

This creates `outputs/daily_fnn/latest_predictions.csv`, ranked from highest to lowest estimated probability of moving up during the next period.

## 7. Backtest

```bash
python -m src.backtest \
  --predictions outputs/daily_fnn/test_predictions.csv \
  --data data/processed/model_rows.csv \
  --output outputs/daily_fnn/backtest \
  --top-k 3 \
  --probability-threshold 0.55 \
  --transaction-cost-bps 5
```

The strategy ranks securities by predicted probability after each date and selects up to `top-k`. To avoid assuming execution at an already observed closing price, the executable return is next open to next close. SPY is evaluated over the same target dates.

## 8. Dashboard

```bash
streamlit run dashboard/app.py
```

## 9. Tests

```bash
pytest -q
```

## 10. Expected outputs

Each run directory contains:

```text
best_model.pt
config_used.yaml
history.csv
training_curve.png
test_predictions.csv
metrics.json
confusion_matrix.png
roc_curve.png
calibration.png
```

The backtest directory contains:

```text
trades.csv
portfolio_daily.csv
backtest_metrics.json
equity_curve.png
```

## 11. Evaluation cautions

- Test observations are never added to training during the primary evaluation.
- All split boundaries are chronological by target date.
- Scalers are fitted on training observations only.
- Hyperparameters are selected from validation performance, not test performance.
- A short live paper-trading demo must not be annualized and presented as reliable long-term performance.
- Results should be repeated with several random seeds before making strong claims.
