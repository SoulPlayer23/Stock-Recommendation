from datetime import datetime, timedelta
import pandas as pd
from fetch_data import fetch_stock_data
from model import train_model
from recommendation import generate_recommendation

stock_file = "src/data/stocks.csv"
stocks = pd.read_csv(stock_file)['Symbol'].tolist()

for stock in stocks:
    print(f"\nProcessing Stock: {stock}")
    stock_data = fetch_stock_data(stock, start_date=(datetime.now()-timedelta(10*365)).strftime('%Y-%m-%d'), end_date=datetime.now().strftime('%Y-%m-%d'))

    model = train_model(stock_data, stock)

    latest_data = stock_data.iloc[-1:][['Open', 'High', 'Low', 'Volume', 'SMA_20', 'SMA_50', 'EMA_20', 'EMA_50', 'MACD', 'RSI', 'BB_Upper', 'BB_Lower']].values.reshape(1, -1)
    predicted_price = model.predict(latest_data)[0]
    current_price = stock_data['Close'].iloc[-1]

    recommendation = generate_recommendation(predicted_price, current_price)

    if isinstance(current_price, pd.Series):
        current_value = current_price.iloc[0]
    else:
        current_value = current_price[0]

    print(f"Prediction: {predicted_price:.2f}, Current Price: {current_value:.2f}, Recommendation: {recommendation}")