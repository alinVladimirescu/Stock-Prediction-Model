# XGBoost Stock Price Predictor

A Machine Learning pipeline that forecasts stock prices using **XGBoost** (Extreme Gradient Boosting). This tool fetches historical data from Yahoo Finance, engineers technical indicators as features, and predicts future price movements with a recursive forecasting strategy. Python 3.11.0 is preferred.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Library](https://img.shields.io/badge/XGBoost-Regressor-green)
![Data](https://img.shields.io/badge/Data-yfinance-orange)

##  Key Features

* **Automated Data Fetching:** Pulls live historical data using `yfinance`.
* **Advanced Feature Engineering:** Calculates technical indicators including:
    * RSI (Relative Strength Index)
    * MACD (Moving Average Convergence Divergence)
    * Moving Averages (5, 20, 50-day)
    * Volatility & Volume Ratios
* **Recursive Forecasting:** Predicts `N` days into the future by feeding predicted values back into the model as inputs for the next day.
* **Visualization:** Automatically generates and saves a plot comparing Test Data vs. Predictions and Future Forecasts.
* **Performance Metrics:** Calculates MAE (Mean Absolute Error), RMSE, and Directional Accuracy.

##  Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/alinVladimirescu/stock-predictor-xgboost.git](https://github.com/alinVladimirescu/stock-predictor-xgboost.git)
    cd stock-predictor-xgboost
    ```

2.  **Install the required packages:**
    ```bash
    pip install yfinance pandas numpy scikit-learn xgboost matplotlib
    ```

##  Usage

1.  Open the script (e.g., `main.py`).
2.  Modify the configuration variables at the bottom of the file if needed:
    ```python
    TICKER = "MA"       # Change to any stock symbol (e.g., AAPL, NVDA, TSLA)
    DAYS_AHEAD = 30     # How many days into the future to predict
    ```
3.  **Run the script:**
    ```bash
    python main.py
    ```

##  Methodology

### 1. Data Processing
The model uses `yfinance` to download 2 years of daily data. Missing values are dropped to ensure data integrity.

### 2. Feature Engineering
Raw prices are converted into meaningful signals. The model utilizes **Decision Trees** via XGBoost to interpret these signals.



The specific features generated include:
* **Trend Indicators:** SMA_5, SMA_20, SMA_50.
* **Momentum Indicators:** RSI (14-day), MACD.
* **Volatility:** Rolling standard deviation (10-day).
* **Lag Features:** Previous 5 days of Close prices and Volume to capture temporal dependencies.





### 3. Model Training
* **Algorithm:** `XGBRegressor`
* **Scaling:** `MinMaxScaler` is used to normalize features between 0 and 1.
* **Split:** Time-series split (last 20% of data used for testing) to prevent data leakage (using future data to predict the past).

##  Example Output

When you run the script, it will print evaluation metrics and save a visualization:

```text
Mean Absolute Error: $5.23
Root Mean Squared Error: $7.12
Directional Accuracy: 52.45%
...
Day 1: $450.20 (Change: $2.10, 0.47%)
Day 2: $452.15 (Change: $4.05, 0.90%)