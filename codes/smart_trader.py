import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error

# Load data
nvda_data = pd.read_csv("/Users/harshinireddy/Desktop/ML_stockanalysis/data/NVDA.csv")
nvdq_data = pd.read_csv("/Users/harshinireddy/Desktop/ML_stockanalysis/data/NVDQ.csv")

# Convert Date column to datetime
nvda_data['Date'] = pd.to_datetime(nvda_data['Date'])
nvdq_data['Date'] = pd.to_datetime(nvdq_data['Date'])

# Sort data by Date
nvda_data = nvda_data.sort_values(by="Date")
nvdq_data = nvdq_data.sort_values(by="Date")

# Merge NVDA and NVDQ data on Date
combined_data = pd.merge(nvda_data, nvdq_data, on="Date", suffixes=("_nvda", "_nvdq"))

# Handle missing values (forward fill as an example)
combined_data.fillna(method='ffill', inplace=True)

# Ensure all numeric columns are properly formatted (e.g., remove commas, convert to float)
for col in combined_data.columns:
    if combined_data[col].dtype == 'object':  # Check for object type (strings)
        combined_data[col] = combined_data[col].str.replace(',', '').astype(float)

# Add moving averages as features
combined_data['MA_5_nvda'] = combined_data['Close_nvda'].rolling(window=5).mean()
combined_data['MA_10_nvda'] = combined_data['Close_nvda'].rolling(window=10).mean()

# Add lag features
combined_data['Lag_1_Close_nvda'] = combined_data['Close_nvda'].shift(1)
combined_data['Lag_2_Close_nvda'] = combined_data['Close_nvda'].shift(2)

# Add Relative Strength Index (RSI)
def calculate_rsi(data, window=14):
    delta = data.diff(1)
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    avg_gain = gain.rolling(window=window, min_periods=1).mean()
    avg_loss = loss.rolling(window=window, min_periods=1).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

combined_data['RSI_nvda'] = calculate_rsi(combined_data['Close_nvda'])

# Define prediction targets
combined_data['High_5d'] = combined_data['High_nvda'].shift(-5).rolling(window=5).max()
combined_data['Low_5d'] = combined_data['Low_nvda'].shift(-5).rolling(window=5).min()
combined_data['Avg_Close_5d'] = combined_data['Close_nvda'].shift(-5).rolling(window=5).mean()

# Drop rows with NaN values
combined_data = combined_data.dropna()

# Scale features
scaler = MinMaxScaler()
scaled_features = ['Open_nvda', 'High_nvda', 'Low_nvda', 'Close_nvda', 'Volume_nvda',
                   'MA_5_nvda', 'MA_10_nvda', 'Lag_1_Close_nvda', 'Lag_2_Close_nvda']
combined_data[scaled_features] = scaler.fit_transform(combined_data[scaled_features])

# Prepare data for training
X = combined_data.drop(['Date', 'High_5d', 'Low_5d', 'Avg_Close_5d'], axis=1)
y_high = combined_data['High_5d']
y_low = combined_data['Low_5d']
y_avg = combined_data['Avg_Close_5d']

# Split data into training and test sets
X_train, X_test, y_high_train, y_high_test = train_test_split(X, y_high, test_size=0.2, random_state=42)
_, _, y_low_train, y_low_test = train_test_split(X, y_low, test_size=0.2, random_state=42)
_, _, y_avg_train, y_avg_test = train_test_split(X, y_avg, test_size=0.2, random_state=42)

# Train models
high_model = RandomForestRegressor(random_state=42)
high_model.fit(X_train, y_high_train)

low_model = RandomForestRegressor(random_state=42)
low_model.fit(X_train, y_low_train)

avg_model = RandomForestRegressor(random_state=42)
avg_model.fit(X_train, y_avg_train)

# Evaluate models
y_high_pred = high_model.predict(X_test)
print("MAE for High Prices:", mean_absolute_error(y_high_test, y_high_pred))

y_low_pred = low_model.predict(X_test)
print("MAE for Low Prices:", mean_absolute_error(y_low_test, y_low_pred))

y_avg_pred = avg_model.predict(X_test)
print("MAE for Average Close Prices:", mean_absolute_error(y_avg_test, y_avg_pred))

# Simulate trading strategy
def trading_strategy(data, high_model, low_model, avg_model):
    predictions = pd.DataFrame()
    predictions['High_5d_pred'] = high_model.predict(data)
    predictions['Low_5d_pred'] = low_model.predict(data)
    predictions['Avg_Close_5d_pred'] = avg_model.predict(data)

    actions = []
    for index, row in predictions.iterrows():
        if row['High_5d_pred'] > data.iloc[index]['Close_nvda'] * 1.05:
            actions.append('BULLISH')  # Significant price increase
        elif row['Low_5d_pred'] < data.iloc[index]['Close_nvda'] * 0.95:
            actions.append('BEARISH')  # Significant price decrease
        else:
            actions.append('IDLE')  # Minor changes

    predictions['Action'] = actions
    return predictions

# Prepare the latest data for prediction
latest_data = X_test  # Replace with new data for live predictions
trading_actions = trading_strategy(latest_data, high_model, low_model, avg_model)

print(trading_actions)
import matplotlib.pyplot as plt

# Plot and save the predictions for High Prices
plt.figure(figsize=(10, 6))
plt.plot(y_high_test.values, label="Actual High Prices", linestyle='-', marker='o')
plt.plot(y_high_pred, label="Predicted High Prices", linestyle='--', marker='x')
plt.title("High Price Predictions")
plt.xlabel("Test Samples")
plt.ylabel("Price")
plt.legend()
plt.grid(True)
plt.savefig("/Users/harshinireddy/Desktop/ML_stockanalysis/output/high_price_predictions.png")
plt.close()

# Plot and save the predictions for Low Prices
plt.figure(figsize=(10, 6))
plt.plot(y_low_test.values, label="Actual Low Prices", linestyle='-', marker='o')
plt.plot(y_low_pred, label="Predicted Low Prices", linestyle='--', marker='x')
plt.title("Low Price Predictions")
plt.xlabel("Test Samples")
plt.ylabel("Price")
plt.legend()
plt.grid(True)
plt.savefig("/Users/harshinireddy/Desktop/ML_stockanalysis/output/low_price_predictions.png")
plt.close()

# Plot and save the predictions for Average Close Prices
plt.figure(figsize=(10, 6))
plt.plot(y_avg_test.values, label="Actual Avg Close Prices", linestyle='-', marker='o')
plt.plot(y_avg_pred, label="Predicted Avg Close Prices", linestyle='--', marker='x')
plt.title("Average Close Price Predictions")
plt.xlabel("Test Samples")
plt.ylabel("Price")
plt.legend()
plt.grid(True)
plt.savefig("/Users/harshinireddy/Desktop/ML_stockanalysis/output/avg_close_price_predictions.png")
plt.close()

print("Plots saved successfully in the output directory.")
