import pandas as pd
import numpy as np
import os
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import pickle

class StockPredictor:
    def __init__(self, data_path=None):
        # Determine the project base directory
        self.data_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # One level up from current script
        
        # # Use the provided data_path or set it to the 'data' directory within the project
        # self.data_path = base_dir
            
        self.lookback_period = 20
        self.nvda_model = None
        self.nvdq_model = None

    def load_data(self):
        """Load and preprocess stock data"""
        try:
            # Load data
            self.nvda_data = pd.read_csv(os.path.join(self.data_path, 'data', 'NVDA1Year.csv'))
            self.nvdq_data = pd.read_csv(os.path.join(self.data_path, 'data', 'NVDQ1Year.csv'))

            print("\nInitial data types:")
            print("NVDA:\n", self.nvda_data.dtypes)
            print("\nNVDQ:\n", self.nvdq_data.dtypes)

            # Convert dates
            self.nvda_data['Date'] = pd.to_datetime(self.nvda_data['Date'])
            self.nvdq_data['Date'] = pd.to_datetime(self.nvdq_data['Date'])

            # Function to convert price
            def convert_price(val):
                if isinstance(val, str):
                    return float(val.replace('$', '').replace(',', ''))
                return float(val)

            # Convert price columns
            price_cols = ['Close/Last', 'Open', 'High', 'Low']
            for col in price_cols:
                self.nvda_data[col] = self.nvda_data[col].apply(convert_price)
                self.nvdq_data[col] = self.nvdq_data[col].apply(convert_price)

            # Convert Volume
            def convert_volume(val):
                if isinstance(val, str):
                    return float(val.replace(',', ''))
                return float(val)

            self.nvda_data['Volume'] = self.nvda_data['Volume'].apply(convert_volume)
            self.nvdq_data['Volume'] = self.nvdq_data['Volume'].apply(convert_volume)

            # Sort by date
            self.nvda_data = self.nvda_data.sort_values('Date')
            self.nvdq_data = self.nvdq_data.sort_values('Date')

            print("\nData loaded successfully!")
            print(f"NVDA data range: {self.nvda_data['Date'].min()} to {self.nvda_data['Date'].max()}")
            print(f"NVDQ data range: {self.nvdq_data['Date'].min()} to {self.nvdq_data['Date'].max()}")

            return True

        except Exception as e:
            print(f"Error loading data: {e}")
            import traceback
            traceback.print_exc()
            return False

    def prepare_features_and_targets(self, data):
        """Prepare features and target variables."""
        try:
            df = data.copy()

            # Engineering technical indicators
            df['Returns'] = df['Close/Last'].pct_change()
            df['High_Low_Range'] = (df['High'] - df['Low']) / df['Close/Last']
            df['Close_Open_Range'] = (df['Close/Last'] - df['Open']) / df['Open']

            # Moving averages
            for window in [5, 10, 20]:
                df[f'MA_{window}'] = df['Close/Last'].rolling(window=window).mean()
                df[f'Volume_MA_{window}'] = df['Volume'].rolling(window=window).mean()

            # Momentum indicators
            df['ROC_5'] = df['Close/Last'].pct_change(periods=5)
            df['Volatility'] = df['Returns'].rolling(window=20).std()

            # Drop NaN values
            df = df.dropna()

            # Define features (X) and target (y)
            X = df[['High_Low_Range', 'Close_Open_Range', 'MA_5', 'MA_10', 'Volume_MA_5', 'ROC_5']].values
            y = df[['Open', 'High', 'Low', 'Close/Last']].values

            return X, y

        except Exception as e:
            print(f"Error in feature preparation: {e}")
            raise

    def train_model(self):
        """Train Linear Regression models for NVDA and NVDQ."""
        try:
            print("Training NVDA model...")
            X_nvda, y_nvda = self.prepare_features_and_targets(self.nvda_data)
            X_train, X_val, y_train, y_val = train_test_split(X_nvda, y_nvda, test_size=0.2, random_state=42)

            # Train NVDA model
            self.nvda_model = LinearRegression()
            self.nvda_model.fit(X_train, y_train)
            print("NVDA model training complete.")

            print("Training NVDQ model...")
            X_nvdq, y_nvdq = self.prepare_features_and_targets(self.nvdq_data)
            X_train, X_val, y_train, y_val = train_test_split(X_nvdq, y_nvdq, test_size=0.2, random_state=42)

            # Train NVDQ model
            self.nvdq_model = LinearRegression()
            self.nvdq_model.fit(X_train, y_train)
            print("NVDQ model training complete.")

            # Save the models
            self.save_models()

        except Exception as e:
            print(f"Error in training: {e}")
            raise

    def save_models(self):
        """Save trained models."""
        model_dir = os.path.join(self.data_path, 'models')
        os.makedirs(model_dir, exist_ok=True)

        # Save NVDA model
        with open(os.path.join(model_dir, 'nvda_model.pkl'), 'wb') as f:
            pickle.dump(self.nvda_model, f)
        print("NVDA model saved successfully.")

        # Save NVDQ model
        with open(os.path.join(model_dir, 'nvdq_model.pkl'), 'wb') as f:
            pickle.dump(self.nvdq_model, f)
        print("NVDQ model saved successfully.")

    def load_models(self):
        """Load trained models."""
        model_dir = os.path.join(self.data_path, 'models')
        nvda_model_path = os.path.join(model_dir, 'nvda_model.pkl')
        nvdq_model_path = os.path.join(model_dir, 'nvdq_model.pkl')

        if os.path.exists(nvda_model_path):
            with open(nvda_model_path, 'rb') as f:
                self.nvda_model = pickle.load(f)
            print("NVDA model loaded successfully.")
        else:
            print("NVDA model not found.")

        if os.path.exists(nvdq_model_path):
            with open(nvdq_model_path, 'rb') as f:
                self.nvdq_model = pickle.load(f)
            print("NVDQ model loaded successfully.")
        else:
            print("NVDQ model not found.")

    def predict_next_day(self, data, model):
        """Predict next day's prices."""
        try:
            X, _ = self.prepare_features_and_targets(data)
            return model.predict(X[-1].reshape(1, -1))

        except Exception as e:
            print(f"Error in prediction: {e}")
            raise

    def predict_next_days(self, data, model,start_date, num_days=5):
        """Predict the next 'num_days' prices sequentially."""
        # Filter data up to the given start_date
        if isinstance(start_date, str):
            start_date = pd.to_datetime(start_date)
            
        data['Date'] = pd.to_datetime(data['Date'])
        filtered_data = data[data['Date'] <= pd.to_datetime(start_date)].copy()

        if filtered_data.empty:
            raise ValueError("No data available for the given start_date or before it.")

        predictions = []
        current_date = start_date + pd.Timedelta(days=1)
        current_data = filtered_data.copy()

        for i in range(num_days):
            # Predict for the next day
            next_prediction = self.predict_next_day(current_data, model)
            
            # Extract predicted values
            open_price, high_price, low_price, close_price = next_prediction[0]
            
            # Append the prediction
            predictions.append({
                'date': current_date.strftime('%Y-%m-%d'),
                'open_price': open_price,
                'highest_price': high_price,
                'lowest_price': low_price,
                'close_price': close_price
            })
            
            # Add the prediction to the data for the next step
            new_row = {
                'Date': current_date.strftime('%Y-%m-%d'),
                'Open': open_price,
                'High': high_price,
                'Low': low_price,
                'Close/Last': close_price,
                'Volume': 0  # Placeholder for volume
            }
            current_data = pd.concat([current_data, pd.DataFrame([new_row])], ignore_index=True)
            current_date += pd.Timedelta(days=1)
        
        return predictions


if __name__ == "__main__":
    predictor = StockPredictor()

    if predictor.load_data():
        predictor.train_model()

        # Input start date from the user
        start_date = input("Enter the start date (YYYY-MM-DD): ")
        # Example prediction for NVDA
       
        nvda_prediction = predictor.predict_next_days(predictor.nvda_data, predictor.nvda_model,start_date, 5)
        print("NVDA Prediction (Next Day):", nvda_prediction)

        # Example prediction for NVDQ
        nvdq_prediction = predictor.predict_next_days(predictor.nvdq_data, predictor.nvdq_model,start_date, 5)
        print("NVDQ Prediction (Next Day):", nvdq_prediction)