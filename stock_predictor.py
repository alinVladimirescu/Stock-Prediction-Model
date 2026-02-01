import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import xgboost as xgb


def load_csv_data(filepath):
    df = pd.read_csv(filepath, parse_dates=['Date'])
    
    # Drop ID column if present
    if 'ID' in df.columns:
        df = df.drop(columns=['ID'])
    
    # Set Date as index
    df = df.set_index('Date')
    df = df.sort_index()
    
    # Keep only required columns and drop any rows with missing values
    required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    df = df[required_cols].dropna()
    
    return df


def load_test_csv(filepath):
    df = pd.read_csv(filepath, parse_dates=['Date'])
    
    if 'ID' in df.columns:
        df = df.drop(columns=['ID'])
    
    df = df.set_index('Date')
    df = df.sort_index()
    
    
    df = df.dropna()
    
    return df


def create_features(df):

    # Avoid division by zero
    eps = 1e-10
    
    # Basic Price Features
    df['High_Low_Pct'] = (df['High'] - df['Low']) / (df['Open'] + eps)
    df['High_Open_Pct'] = (df['High'] - df['Open']) / (df['Open'] + eps)
    df['Open_Low_Pct'] = (df['Open'] - df['Low']) / (df['Open'] + eps)
    df['Range'] = df['High'] - df['Low']
    df['Range_Pct'] = df['Range'] / (df['Open'] + eps)
    
    # Volume Features
    df['Volume_MA_5'] = df['Volume'].rolling(5).mean()
    df['Volume_MA_10'] = df['Volume'].rolling(10).mean()
    df['Volume_MA_20'] = df['Volume'].rolling(20).mean()
    df['Volume_Ratio'] = df['Volume'] / (df['Volume_MA_20'] + eps)
    df['Volume_Ratio_5'] = df['Volume'] / (df['Volume_MA_5'] + eps)
    df['Volume_Change'] = df['Volume'].pct_change(fill_method=None).fillna(0)
    df['Volume_Std_10'] = df['Volume'].rolling(10).std() / (df['Volume_MA_10'] + eps)
    
    # Moving Averages
    for window in [5, 10, 20, 50]:
        df[f'Open_MA_{window}'] = df['Open'].rolling(window).mean()
        df[f'Open_vs_MA{window}'] = df['Open'] / (df[f'Open_MA_{window}'] + eps)
    
    df['MA5_vs_MA20'] = df['Open_MA_5'] / (df['Open_MA_20'] + eps)
    df['MA10_vs_MA50'] = df['Open_MA_10'] / (df['Open_MA_50'] + eps)
    
    # Open Price Returns
    for period in [1, 2, 3, 5, 10, 20]:
        df[f'Open_Return_{period}'] = df['Open'].pct_change(period, fill_method=None).fillna(0)
    
    # RSI
    for period in [7, 14]:
        delta = df['Open'].diff()
        gain = delta.where(delta > 0, 0).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
        rs = gain / (loss + eps)
        df[f'RSI_{period}'] = 100 - (100 / (1 + rs))
    
    # MACD
    ema_12 = df['Open'].ewm(span=12, adjust=False).mean()
    ema_26 = df['Open'].ewm(span=26, adjust=False).mean()
    df['MACD'] = (ema_12 - ema_26) / (df['Open'] + eps)  # Normalized
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
    
    # Bollinger Bands
    bb_window = 20
    df['BB_Middle'] = df['Open'].rolling(bb_window).mean()
    bb_std = df['Open'].rolling(bb_window).std()
    df['BB_Upper'] = df['BB_Middle'] + 2 * bb_std
    df['BB_Lower'] = df['BB_Middle'] - 2 * bb_std
    df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / (df['BB_Middle'] + eps)
    df['BB_Position'] = (df['Open'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'] + eps)
    
    # Average True Range (ATR)
    high_low = df['High'] - df['Low']
    high_close_prev = abs(df['High'] - df['Open'].shift(1))
    low_close_prev = abs(df['Low'] - df['Open'].shift(1))
    true_range = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)
    df['ATR_14'] = true_range.rolling(14).mean() / (df['Open'] + eps)
    df['ATR_7'] = true_range.rolling(7).mean() / (df['Open'] + eps)
    # Stochastic Oscillator
    for period in [14]:
        lowest_low = df['Low'].rolling(period).min()
        highest_high = df['High'].rolling(period).max()
        df[f'Stoch_K_{period}'] = 100 * (df['Open'] - lowest_low) / (highest_high - lowest_low + eps)
        df[f'Stoch_D_{period}'] = df[f'Stoch_K_{period}'].rolling(3).mean()
    
    # Volatility Features
    df['Volatility_5'] = df['Open'].pct_change(fill_method=None).rolling(5).std()
    df['Volatility_10'] = df['Open'].pct_change(fill_method=None).rolling(10).std()
    df['Volatility_20'] = df['Open'].pct_change(fill_method=None).rolling(20).std()
    
    # Position of the Day
    df['Day_Position'] = (df['Open'] - df['Low']) / (df['High'] - df['Low'] + eps)
    
    # Lagged Features
    lag_features = ['High_Low_Pct', 'High_Open_Pct', 'Volume_Ratio', 'RSI_14', 'MACD_Hist']
    for feat in lag_features:
        if feat in df.columns:
            for i in range(1, 6):
                df[f'{feat}_lag_{i}'] = df[feat].shift(i)
    
    # Calendar Features
    if isinstance(df.index, pd.DatetimeIndex):
        df['DayOfWeek'] = df.index.dayofweek / 6  # Normalize 0-1
        df['Month'] = df.index.month / 12  # Normalize 0-1
        df['DayOfMonth'] = df.index.day / 31  # Normalize 0-1
        df['Quarter'] = df.index.quarter / 4  # Normalize 0-1
        # Is Monday (often has weekend gap effects)
        df['IsMonday'] = (df.index.dayofweek == 0).astype(int)
        # Is Friday (end of week behavior)
        df['IsFriday'] = (df.index.dayofweek == 4).astype(int)
    
    # Rolling Target Stats (historical Close/Open ratios)
    if 'Close' in df.columns:
        df['Close_Open_Ratio'] = df['Close'] / df['Open']
        df['Ratio_MA_5'] = df['Close_Open_Ratio'].shift(1).rolling(5).mean()
        df['Ratio_MA_20'] = df['Close_Open_Ratio'].shift(1).rolling(20).mean()
        df['Ratio_Std_10'] = df['Close_Open_Ratio'].shift(1).rolling(10).std()
    
    # Replace any remaining inf values with 0
    df = df.replace([np.inf, -np.inf], 0)
    
    return df


def train_model(x_train, y_train):
    
    
    xgb_model = xgb.XGBRegressor(
        n_estimators=2000,
        learning_rate=0.03,
        max_depth=6,
        min_child_weight=5,
        subsample=0.8,
        colsample_bytree=0.7,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        early_stopping_rounds=50,
        eval_metric='mae'
    )
    xgb_model.fit(
        x_train, y_train,
        eval_set=[(x_train, y_train)],
        verbose=False
    )
    return xgb_model


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

TRAIN_CSV = "data/train.csv"  
TEST_CSV  = "data/test.csv" 
print("=" * 60)
print("Stock Predictor for S&P 500 - Enhanced Version")
print("=" * 60)

print("\n[1/6] Loading training data from CSV...")
df_train_raw = load_csv_data(TRAIN_CSV)
print(f"      Loaded {len(df_train_raw)} rows from {TRAIN_CSV}")
print(f"      Date range: {df_train_raw.index.min()} to {df_train_raw.index.max()}")

print("\n[2/6] Loading test data from CSV...")
df_test_raw = load_test_csv(TEST_CSV)
print(f"      Loaded {len(df_test_raw)} rows from {TEST_CSV}")
print(f"      Date range: {df_test_raw.index.min()} to {df_test_raw.index.max()}")

# Create train features
print("\n[3/6] Creating features for training data...")
df_train_feat = create_features(df_train_raw.copy())
df_train_feat = df_train_feat.dropna().copy()
print(f"      Training rows after feature engineering: {len(df_train_feat)}")

# Clean up feature cols
exclude_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'Range', 
                'Open_MA_5', 'Open_MA_10', 'Open_MA_20', 'Open_MA_50',
                'Volume_MA_5', 'Volume_MA_10', 'Volume_MA_20',
                'BB_Middle', 'BB_Upper', 'BB_Lower', 'Close_Open_Ratio']

feature_cols = [col for col in df_train_feat.columns 
                if col not in exclude_cols 
                and df_train_feat[col].dtype in ['float64', 'int64', 'float32', 'int32']]

print(f"      Using {len(feature_cols)} features")

# Target is Close/Open RATIO
x_train = df_train_feat[feature_cols]
y_train = df_train_feat['Close'] / df_train_feat['Open']

scaler = RobustScaler()
x_train_scaled = scaler.fit_transform(x_train)

print("\n[4/6] Training ensemble model...")
model = train_model(x_train_scaled, y_train)
print("      Ensemble training complete!")

# Create test features
print("\n[5/6] Creating features for test data...")

# Need more history from training for rolling features (50 days for MA_50)
HISTORY_NEEDED = 60
history = df_train_raw[['Open', 'High', 'Low', 'Volume']].tail(HISTORY_NEEDED).copy()

# Combine history with test data for feature calculation
test_with_history = pd.concat([history, df_test_raw])
test_with_history = test_with_history.sort_index()

# Create features on combined data
test_feat = create_features(test_with_history.copy())

# Keep only test rows
test_feat = test_feat.loc[test_feat.index.isin(df_test_raw.index)].copy()
test_feat = test_feat.dropna()

# Handle missing features (some rolling stats from train that aren't in test)
missing_features = [f for f in feature_cols if f not in test_feat.columns]
for feat in missing_features:
    test_feat[feat] = 0

print(f"      Test rows with features: {len(test_feat)}")

# Predict Close/Open RATIO 
print("\n[6/6] Predicting Close prices...")
x_test = test_feat[feature_cols]
x_test_scaled = scaler.transform(x_test)
predicted_ratios = model.predict(x_test_scaled)

# Convert ratio to actual Close: Close = Open * ratio
test_open_prices = test_feat['Open'].values
test_predictions = test_open_prices * predicted_ratios

print(f"      Predicted {len(test_predictions)} Close values")
print(f"      Ratio range: {predicted_ratios.min():.4f} to {predicted_ratios.max():.4f}")
print(f"      Close range: ${test_predictions.min():.2f} to ${test_predictions.max():.2f}")


df_submission = pd.DataFrame({
    'ID': range(len(test_predictions)),
    'Close': test_predictions
})

# Save submission
OUTPUT_FILE = "data/predictions.csv"
df_submission.to_csv(OUTPUT_FILE, index=False)
print(f"\n✓ Predictions saved to {OUTPUT_FILE}")

print("\n" + "=" * 60)
# Feature Importance Display
print("Top 15 Most Important Features (XGBoost):")
print("=" * 60)
feature_importance = pd.DataFrame({
    'Feature': feature_cols,
    'Importance': model.feature_importances_
}).sort_values('Importance', ascending=False)

for i, row in feature_importance.head(15).iterrows():
    print(f"  {row['Feature']:30s} {row['Importance']:.4f}")

