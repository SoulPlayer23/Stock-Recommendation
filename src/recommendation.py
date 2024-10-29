def generate_recommendation(predicted_price, current_price):
    if (predicted_price > current_price).any():
        return "Buy"
    elif (predicted_price < current_price).any():
        return "Sell"
    else:
        return "Hold"