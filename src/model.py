from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

def train_model(stock_data):
    features = ['Open', 'High', 'Low', 'Volume', 'SMA_20', 'SMA_50', 'EMA_20', 'MACD', 'RSI']
    x = stock_data[features]
    y = stock_data['Close']

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"Mean Squared Error: {mse}")
    print(f"R-squared: {r2}")

    plt.figure(figsize=(14,7))
    plt.plot(y_test.index.sort_values(), y_test, label="Actual Price", color="blue")
    plt.plot(y_test.index.sort_values(), y_pred, label="Predicted Price", color="red")
    plt.legend()
    plt.xlabel("Data")
    plt.ylabel("Stock Price")
    plt.title("Actual vs Predicted Price")
    plt.show()
    
    return model