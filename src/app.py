from datetime import datetime
import pandas as pd
from fetch_data import fetch_stock_data
from preprocessing import preprocess_data
from model import train_model
from recommendation import generate_recommendation

stock_file = "src/data/stocks.csv"
stocks = pd.read_csv(stock_file)['Symbol'].tolist()

for stock in stocks:
    print(f"\nProcessing Stock: {stock}")
    stock_data = fetch_stock_data(stock, end_date=datetime.now().strftime('%Y-%m-%d'))
    stock_data = preprocess_data(stock_data)

    model = train_model(stock_data)

    latest_data = stock_data.iloc[-1:][['Open', 'High', 'Low', 'Volume', 'SMA_20', 'SMA_50', 'EMA_20', 'MACD', 'RSI']].values.reshape(1, -1)
    predicted_price = model.predict(latest_data)[0]
    current_price = stock_data['Close'].iloc[-1]

    recommendation = generate_recommendation(predicted_price, current_price)
    
    if isinstance(predicted_price, pd.Series):
        prediction_value = predicted_price.iloc[0]
    else:
        prediction_value = predicted_price[0]

    if isinstance(current_price, pd.Series):
        current_value = current_price.iloc[0]
    else:
        current_value = current_price[0]

    print(f"Prediction: {prediction_value:.2f}, Current Price: {current_value:.2f}, Recommendation: {recommendation}")