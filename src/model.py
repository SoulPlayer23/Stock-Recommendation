from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
import matplotlib.pyplot as plt

from preprocessing import preprocess_data

def train_model(stock_data):
    x_train, x_test, y_train, y_test = preprocess_data(stock_data, test_size=0.2, random_state=42)

    model = ElasticNet(alpha=0.5, l1_ratio=0.5, max_iter=20000)
    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = root_mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"Mean Absolute Error: {mae}")
    print(f"Root Mean Squared Error: {rmse}")
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