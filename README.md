# Stock Prediction

## Introduction

This application will be able to predict and recommend the buy/sell/hold of specific stocks based on the last few years data. Here we are using Linear Regression model to analyse the trend and predict the next outcome. 

## Local Setup

1. Create virtual environment:
    ```bash
    python -m venv venv
    ```
2. Activate virtual environment:
    ```bash
    venv\Scripts\activate
    ```
3. Install the dependencies:
    ```bash
    pip install -r requirements.txt
    ```
4. Run the application:
    ```bash
    python src/app.py
    ```
5. Deactivate virtual environment:
    ```bash
    deactivate
    ```
## Stock Symbols

We have to provide stocks symbols of specific stocks to get the stock data from yahoo finance. Please refer to yahoo finance to obtain the correct symbols. Here we store the symbols in data/stocks.csv file. You could also directly give it as a list or get user input to do the same according to your needs.