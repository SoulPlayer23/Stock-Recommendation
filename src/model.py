import pandas as pd
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import PolynomialFeatures

from preprocessing import preprocess_data

def plot_predictions(stock_data, model, stock_symbol): 
    plt.figure(figsize=(14, 7)) 
    plt.plot(stock_data.index, stock_data['Close'], label='Actual Prices', color='green', alpha=0.6) 

    future_stock_data = stock_data.iloc[-1:].copy() 
    future_stock_data = pd.concat([future_stock_data] * 30, ignore_index=True) 
    future_stock_data.index = pd.date_range(start=stock_data.index[-1], periods=30, freq='B') 
    features = ['Open', 'High', 'Low', 'Volume', 'SMA_20', 'SMA_50', 'EMA_20', 'EMA_50', 'MACD', 'RSI', 'BB_Upper', 'BB_Lower'] 
    future_stock_data[features] = future_stock_data[features].ffill().bfill() 
    future_stock_data['Prediction'] = model.predict(future_stock_data[features]) 
    plt.scatter(future_stock_data.index, future_stock_data['Prediction'], label='Prediction', color='blue', alpha=0.6) 

    # Plot the polynomial regression line for the training data 
    X = stock_data.index.map(pd.Timestamp.toordinal).values.reshape(-1, 1) 
    # Convert dates to ordinal numbers 
    y = stock_data['Close'] 
    # Fit polynomial regression model 
    poly_features = PolynomialFeatures(degree=2) 
    # Change degree as needed 
    X_poly = poly_features.fit_transform(X) 
    poly_model = ElasticNet() 
    poly_model.fit(X_poly, y) 
    regression_line = poly_model.predict(X_poly) 
    plt.plot(stock_data.index, regression_line, label='Regression Line', color='red', linewidth=2)
    
    plt.xlabel('Date') 
    plt.ylabel('Stock Price') 
    plt.title(f'Prediction for {stock_symbol}') 
    plt.legend() 
    plt.show()

def train_model(stock_data, stock_symbol):
    x_train, x_test, y_train, y_test = preprocess_data(stock_data, test_size=0.2, random_state=42)

    model = ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=100000)
    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = root_mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"Mean Absolute Error: {mae}")
    print(f"Root Mean Squared Error: {rmse}")
    print(f"R-squared: {r2}")
    
    plot_predictions(stock_data, model, stock_symbol)
    
    return model