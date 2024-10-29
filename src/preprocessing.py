import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def compute_RSI(series, window=14):
    delta = series.diff(1)
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain/loss
    return 10 - (100 / (1 + rs))

def preprocess_data(stock_data):
    if not isinstance(stock_data, pd.DataFrame):
        raise ValueError("Input must be a pandas DataFrame")

    stock_data['SMA_20'] = stock_data['Close'].rolling(window=20).mean()
    stock_data['SMA_50'] = stock_data['Close'].rolling(window=50).mean()
    stock_data['EMA_20'] = stock_data['Close'].ewm(span=20, adjust=False).mean()
    stock_data['MACD'] = stock_data['Close'].ewm(span=20, adjust=False).mean() - stock_data['Close'].ewm(span=26, adjust=False).mean()
    stock_data['RSI'] = compute_RSI(stock_data['Close'], window=14)

    stock_data['Log_Returns'] = np.log(stock_data['Close'] / stock_data['Close'].shift(1))

    stock_data.dropna(inplace=True)

    return stock_data