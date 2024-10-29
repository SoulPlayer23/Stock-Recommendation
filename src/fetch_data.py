import yfinance as yf

def fetch_stock_data(stock_symbol, start_date="2020-01-01", end_date="2023-01-01"):
    stock_data = yf.download(stock_symbol, start=start_date, end=end_date)
    return stock_data