import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split

def compute_RSI(series, window=14):
    delta = series.diff(1)
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain/loss
    return 10 - (100 / (1 + rs))

def preprocess_data(stock_data, test_size=0.2, random_state=42):
    if not isinstance(stock_data, pd.DataFrame):
        raise ValueError("Input must be a pandas DataFrame")

    stock_data['SMA_20'] = stock_data['Close'].rolling(window=20).mean()
    stock_data['SMA_50'] = stock_data['Close'].rolling(window=50).mean()
    stock_data['EMA_20'] = stock_data['Close'].ewm(span=20, adjust=False).mean()
    stock_data['MACD'] = stock_data['Close'].ewm(span=20, adjust=False).mean() - stock_data['Close'].ewm(span=26, adjust=False).mean()
    stock_data['RSI'] = compute_RSI(stock_data['Close'], window=14)

    features = ['Open', 'High', 'Low', 'Volume', 'SMA_20', 'SMA_50', 'EMA_20', 'MACD', 'RSI']
    x = stock_data[features]
    y = stock_data['Close']

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=random_state)

    imputer = KNNImputer(n_neighbors=5)
    x_train = imputer.fit_transform(x_train)
    x_test = imputer.transform(x_test)

    return x_train, x_test, y_train, y_test