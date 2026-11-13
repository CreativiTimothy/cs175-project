import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def performance_metrics(returns, periods_per_year=252):
    returns = pd.Series(returns).fillna(0.0)
    equity = (1.0 + returns).cumprod()
    total_return = float(equity.iloc[-1] - 1.0) if len(equity) else 0.0
    volatility = float(returns.std(ddof=1) * np.sqrt(periods_per_year)) if len(returns) > 1 else 0.0
    annualized_return = float(equity.iloc[-1] ** (periods_per_year / len(returns)) - 1.0) if len(returns) else 0.0
    sharpe = float((returns.mean() / returns.std(ddof=1)) * np.sqrt(periods_per_year)) if returns.std(ddof=1) > 0 else 0.0
    drawdown = equity / equity.cummax() - 1.0
    return {
        "total_return": total_return,
        "annualized_return": annualized_return,
        "annualized_volatility": volatility,
        "sharpe_ratio_no_risk_free_rate": sharpe,
        "maximum_drawdown": float(drawdown.min()) if len(drawdown) else 0.0,
        "positive_day_rate": float((returns > 0).mean()) if len(returns) else 0.0,
        "number_of_days": int(len(returns)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--data", required=False, help="Accepted for compatibility; metadata are already stored with predictions.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--probability-threshold", type=float, default=0.55)
    parser.add_argument("--transaction-cost-bps", type=float, default=5.0)
    args = parser.parse_args()

    predictions = pd.read_csv(args.predictions, parse_dates=["target_date"])
    required = {"symbol", "target_date", "probability_up", "executable_return"}
    missing = required.difference(predictions.columns)
    if missing:
        raise ValueError(f"Predictions are missing backtest columns: {sorted(missing)}")

    selected = predictions[predictions["probability_up"] >= args.probability_threshold].copy()
    selected = selected.sort_values(["target_date", "probability_up"], ascending=[True, False])
    selected = selected.groupby("target_date").head(args.top_k)
    one_way_cost = args.transaction_cost_bps / 10000.0
    selected["net_return"] = selected["executable_return"] - 2.0 * one_way_cost

    daily = selected.groupby("target_date")["net_return"].mean().rename("strategy_return").to_frame()
    all_dates = pd.Index(sorted(predictions["target_date"].unique()), name="target_date")
    daily = daily.reindex(all_dates, fill_value=0.0)

    spy = predictions[predictions["symbol"] == "SPY"].drop_duplicates("target_date").set_index("target_date")
    daily["spy_return"] = spy["target_return"].reindex(all_dates).fillna(0.0)
    daily["strategy_equity"] = (1.0 + daily["strategy_return"]).cumprod()
    daily["spy_equity"] = (1.0 + daily["spy_return"]).cumprod()

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    selected.to_csv(output / "trades.csv", index=False)
    daily.reset_index().to_csv(output / "portfolio_daily.csv", index=False)
    metrics = {
        "strategy": performance_metrics(daily["strategy_return"]),
        "spy_buy_and_hold": performance_metrics(daily["spy_return"]),
        "settings": vars(args),
    }
    with open(output / "backtest_metrics.json", "w", encoding="utf-8") as metrics_file:
        json.dump(metrics, metrics_file, indent=2)

    plt.figure(figsize=(10, 5))
    plt.plot(daily.index, daily["strategy_equity"], label="Model top-k strategy")
    plt.plot(daily.index, daily["spy_equity"], label="SPY buy and hold")
    plt.xlabel("Target date")
    plt.ylabel("Growth of $1")
    plt.title("Historical Paper-Trading Backtest")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output / "equity_curve.png", dpi=200)
    plt.close()
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
