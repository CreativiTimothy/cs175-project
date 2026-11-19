# Project Status Report Guide

## 1. Project Summary

Describe the next-day stock/ETF direction task, output probability, and why always-up is a nontrivial baseline.

## 2. Approach/Algorithm

Document the pipeline:

1. Alpaca historical OHLC acquisition.
2. Scale-independent OHLC normalization relative to the previous close.
3. 100-bar windows.
4. Chronological 70/15/15 split.
5. FNN with ReLU, batch normalization, dropout, and sigmoid probability obtained from the output logit.
6. Binary cross-entropy training with AdamW and early stopping.
7. Optional LSTM and return-regression experiments.

Include the binary cross-entropy equation and a system diagram.

## 3. Schedule, Progress, Remaining Goals, and Challenges

Report completed data pipeline, baseline, FNN prototype, and tests. Remaining goals should include real-data experiments, hyperparameter selection, repeated seeds, LSTM comparison, backtesting, and presentation preparation.

## 4. Evaluation Plan

Quantitative: always-up, constant probability, logistic regression, FNN, and LSTM; accuracy, precision, recall, F1, ROC-AUC, Brier score, log loss; backtest total return, Sharpe ratio, volatility, maximum drawdown, and transaction costs.

Qualitative: training/validation loss, calibration plot, high-confidence correct and incorrect examples, ranked probability dashboard, and behavior in rising, falling, and volatile periods.

## 5. Resources Used

List PyTorch, scikit-learn, pandas, NumPy, Alpaca Market Data API, Streamlit, course slides, and every external code source actually used.
