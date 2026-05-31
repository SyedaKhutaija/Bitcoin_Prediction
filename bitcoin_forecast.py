import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('Agg') # Ensure compatibility in headless environments
from statsmodels.tsa.statespace.sarimax import SARIMAX
from pmdarima import auto_arima
from prophet import Prophet
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import warnings

warnings.filterwarnings('ignore')
import argparse
try:
    from live_data import fetch_live_data
except Exception:
    fetch_live_data = None

# --------------------------------------------------
# DATA PREPARATION
# --------------------------------------------------
parser = argparse.ArgumentParser(description='Bitcoin forecasting (optionally live)')
parser.add_argument('--live', action='store_true', help='Fetch live data from CoinGecko instead of local file')
args = parser.parse_args()

print("Loading data...")
if args.live and fetch_live_data is not None:
    df = fetch_live_data(days=365*3)  # fetch ~3 years of daily data
else:
    df = pd.read_excel('cleaned_bitcoin_data.xlsx')
if 'date' not in df.columns:
    df = df.reset_index()
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values('date')
df = df.set_index('date')
df = df.asfreq('D')

# Fill missing values if any (though instructions say 'cleaned')
df['close'] = df['close'].ffill()

target = df['close']
split_idx = int(len(target) * 0.8)
train, test = target.iloc[:split_idx], target.iloc[split_idx:]

results = {}

def evaluate(actual, pred, model_name):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    mape = mean_absolute_percentage_error(actual, pred) * 100
    results[model_name] = {'RMSE': rmse, 'MAE': mae, 'MAPE': mape}
    print(f"{model_name} - RMSE: {rmse:.2f}, MAE: {mae:.2f}, MAPE: {mape:.2f}%")
    return pred

# --------------------------------------------------
# MODEL 1: ARIMA
# --------------------------------------------------
print("\nTraining ARIMA...")
model_arima = auto_arima(train, seasonal=False, stepwise=True, suppress_warnings=True)
arima_pred = model_arima.predict(n_periods=len(test))
arima_pred.index = test.index
evaluate(test, arima_pred, 'ARIMA')

# --------------------------------------------------
# MODEL 2: SARIMA
# --------------------------------------------------
print("\nTraining SARIMA...")
model_sarima = auto_arima(train, seasonal=True, m=7, stepwise=True, suppress_warnings=True)
sarima_pred = model_sarima.predict(n_periods=len(test))
sarima_pred.index = test.index
evaluate(test, sarima_pred, 'SARIMA')

# --------------------------------------------------
# MODEL 3: FACEBOOK PROPHET
# --------------------------------------------------
print("\nTraining Prophet...")
df_prophet = train.reset_index().rename(columns={'date': 'ds', 'close': 'y'})
model_prophet = Prophet()
model_prophet.fit(df_prophet)
future = model_prophet.make_future_dataframe(periods=len(test))
forecast = model_prophet.predict(future)
prophet_pred = forecast['yhat'].iloc[-len(test):]
prophet_pred.index = test.index
evaluate(test, prophet_pred, 'Prophet')

# --------------------------------------------------
# MODEL 4: LSTM
# --------------------------------------------------
print("\nTraining LSTM...")
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(target.values.reshape(-1, 1))

def create_sequences(data, window):
    X, y = [], []
    for i in range(len(data) - window):
        X.append(data[i:i+window])
        y.append(data[i+window])
    return np.array(X), np.array(y)

window_size = 60
X, y = create_sequences(scaled_data, window_size)

# Split sequences
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

model_lstm = Sequential([
    LSTM(50, return_sequences=True, input_shape=(window_size, 1)),
    LSTM(50),
    Dense(1)
])
model_lstm.compile(optimizer='adam', loss='mse')
model_lstm.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)

lstm_pred_scaled = model_lstm.predict(X_test)
lstm_pred = scaler.inverse_transform(lstm_pred_scaled).flatten()

# Align LSTM predictions with test dates (last part of the data)
# Note: LSTM window affects the start date of predictions
test_actual_lstm = scaler.inverse_transform(y_test).flatten()
evaluate(test_actual_lstm, lstm_pred, 'LSTM')

# --------------------------------------------------
# EVALUATION SUMMARY
# --------------------------------------------------
comparison_df = pd.DataFrame(results).T
print("\nComparison Table:")
print(comparison_df)

# Identify best model
best_model = comparison_df['RMSE'].idxmin()
print(f"\nBest Performing Model: {best_model}")

# --------------------------------------------------
# VISUALIZATION
# --------------------------------------------------
plt.figure(figsize=(15, 10))

# Subplot for individual comparisons
models = [('ARIMA', arima_pred), ('SARIMA', sarima_pred), ('Prophet', prophet_pred)]
for i, (name, pred) in enumerate(models, 1):
    plt.subplot(3, 2, i)
    plt.plot(test.index, test.values, label='Actual', color='black')
    plt.plot(test.index, pred, label=f'Predicted ({name})', linestyle='--')
    plt.title(f'{name} Prediction')
    plt.legend()

# LSTM special case for indexing
plt.subplot(3, 2, 4)
plt.plot(test_actual_lstm, label='Actual', color='black')
plt.plot(lstm_pred, label='Predicted (LSTM)', linestyle='--')
plt.title('LSTM Prediction')
plt.legend()

# Combined Comparison
plt.subplot(3, 1, 3)
plt.plot(test.index, test.values, label='Actual', color='black', linewidth=2)
plt.plot(test.index, arima_pred, label='ARIMA', alpha=0.7)
plt.plot(test.index, sarima_pred, label='SARIMA', alpha=0.7)
plt.plot(test.index, prophet_pred, label='Prophet', alpha=0.7)
# For LSTM on combined plot, we need to align carefully. 
# For simplicity in the combined view, we'll plot the other three.
plt.title('Comparison of All Models (Classical & Prophet)')
plt.legend()

plt.tight_layout()
plt.savefig('forecast_results.png')
print("\nVisualization saved as 'forecast_results.png'")
