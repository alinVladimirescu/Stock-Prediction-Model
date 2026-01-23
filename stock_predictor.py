import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import xgboost as xgb
import matplotlib.pyplot as plt
from datetime import timedelta

def get_stock_data(ticker, period="2y"):
    stock = yf.Ticker(ticker)
    df = stock.history(period=period)
    return df

def get_rsi(prices, window=14):
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0).rolling(window=window).mean())
    loss = (-delta.where(delta < 0, 0).rolling(window=window).mean())
    rsi = 100 - (100 / (1 + gain / loss))
    return rsi

def create_features(df):
    df['Returns'] = df['Close'].pct_change()
    df['High_Low'] = df['High'] - df['Low']

    df['MA_5'] = df['Close'].rolling(5).mean()
    df['MA_20'] = df['Close'].rolling(20).mean()
    df['MA_50'] = df['Close'].rolling(50).mean()

    df['RSI'] = get_rsi(df['Close'], 14)
    df['MACD'] = df['Close'].ewm(span=12).mean() - df['Close'].ewm(span=26).mean()

    df['Volatility'] = df['Close'].rolling(10).std()

    df['Volume_MA'] = df['Volume'].rolling(20).mean()
    df['Volume_Ratio'] = df['Volume'] / df['Volume_MA']

    for i in range(1, 6):
        df[f'Close_lag_{i}'] = df['Close'].shift(i)
        df[f'Volume_lag_{i}'] = df['Volume'].shift(i)
    
    df['Target'] = df['Close'].shift(-1)

    return df

def train_model(x_train, y_train):
    model = xgb.XGBRegressor(
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=7,
        min_child_weight=3,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    model.fit(x_train, y_train)
    return model

def evaluate_model(model, x_test, y_test):
    predictions = model.predict(x_test)
    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))


    actual_direction = y_test.values[1:] > y_test.values[:-1]
    pred_direction = (predictions[1:] > predictions[:-1])
    directional_accuracy = np.mean(actual_direction == pred_direction) * 100
    
    print(f"Mean Absolute Error: ${mae:.2f}")
    print(f"Root Mean Squared Error: ${rmse:.2f}")
    print(f"Directional Accuracy: {directional_accuracy:.2f}%")
    
    return predictions, mae, rmse, directional_accuracy

def predict_future(model, df, scaler, days=5):
    predictions = []
    future_df = df.copy()

    for day in range(days):
        df_with_features = create_features(future_df.copy())
        
        last_row = df_with_features.iloc[-1:]
        
        feature_cols = [col for col in df_with_features.columns 
                       if col not in ['Target', 'Dividends', 'Stock Splits']]
        
        current_features = last_row[feature_cols]
        
        current_features_scaled = scaler.transform(current_features)
        pred = model.predict(current_features_scaled)[0]
        predictions.append(pred)
        
        last_volume = future_df['Volume'].iloc[-1]
        new_date = future_df.index[-1] + timedelta(days=1)
        
        new_row = pd.DataFrame({
            'Open': [pred],
            'High': [pred * 1.005],  
            'Low': [pred * 0.995],   
            'Close': [pred],
            'Volume': [last_volume]  
        }, index=[new_date])
        
        future_df = pd.concat([future_df, new_row])
    
    return predictions


TICKER = "MA"
DAYS_AHEAD = 30

print(f"Stock predictor for {TICKER}")
print("Downloading data...")
df = get_stock_data(TICKER)
print(f"Downloaded {len(df)} days of data")

print("Creating features...")
df = create_features(df)
print(f"Total rows (including NaN): {len(df)}")

df_train = df.dropna().copy()
print(f"Training rows (after dropping NaN): {len(df_train)}")

feature_cols = [col for col in df_train.columns if col not in ["Target", "Dividends", "Stock Splits"]]
X = df_train[feature_cols]
Y = df_train['Target']

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, shuffle=False)

scaler = MinMaxScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

print("Training model...")
model = train_model(x_train_scaled, y_train)
print("Model trained!")

print("Model performance:")
predictions, mae, rmse, dir_acc = evaluate_model(model, x_test_scaled, y_test)

future_preds = predict_future(model, df, scaler, DAYS_AHEAD)

current_price = df['Close'].iloc[-1]
print(f"Current price ({df.index[-1].date()}): ${current_price:.2f}")

print("\nFuture Predictions:")
for i, pred in enumerate(future_preds, 1):
    change = pred - current_price
    change_pct = (change / current_price) * 100
    print(f"Day {i}: ${pred:.2f} (Change: ${change:.2f}, {change_pct:.2f}%)")

avg_pred = np.mean(future_preds)

if avg_pred > current_price:
    print("BUY BUY BUY NOW GET MONEY")
else:
    print("sell sell sell get out while you can")

plt.figure(figsize=(12, 6))

test_dates = df_train.index[-len(y_test):]
plt.plot(test_dates, y_test.values, label='Actual Price', color='blue', linewidth=2)
plt.plot(test_dates, predictions, label='Predicted Price', color='red', linestyle='--', linewidth=2)

last_date = test_dates[-1]
future_dates = [last_date + timedelta(days=i) for i in range(1, DAYS_AHEAD + 1)]
plt.plot(future_dates, future_preds, label='Future Forecast', color='green', marker='o', linewidth=2)

plt.title(f'{TICKER} - Price Prediction & Forecast')
plt.xlabel('Date')
plt.ylabel('Price ($)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f'{TICKER}_predictions.png', dpi=300)
print(f"\nChart saved as {TICKER}_predictions.png")



print("Top 10 Most Important Features:")
feature_importance = pd.DataFrame({
    'Feature': feature_cols,
    'Importance': model.feature_importances_
}).sort_values('Importance', ascending=False)

print(feature_importance.head(10).to_string(index=False))
plt.show()