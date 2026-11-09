from config import KEY, Secret, Endpoint

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockLatestTradeRequest

client = StockHistoricalDataClient(KEY, Secret)

# For Apple
request = StockLatestTradeRequest(symbol_or_symbols="AAPL")
latest_trades = client.get_stock_latest_trade(request)

apple = latest_trades["AAPL"]
print(apple)