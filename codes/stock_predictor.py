import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers.legacy import Adam  # Use legacy optimizer for M1/M2 Macs
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau  # Add ReduceLROnPlateau import
from sklearn.preprocessing import MinMaxScaler
import os

class StockPredictor:
    def __init__(self, data_path='/Users/harshinireddy/Desktop/ML_stockanalysis/data'):
        self.data_path = data_path
        self.lookback_period = 20
        self.nvda_scaler = MinMaxScaler()
        self.nvdq_scaler = MinMaxScaler()
        self.model_nvda = None
        self.model_nvdq = None
    def load_data(self):
        """Load and preprocess stock data"""
        try:
            # Load data
            self.nvda_data = pd.read_csv(os.path.join(self.data_path, 'NVDA.csv'))
            self.nvdq_data = pd.read_csv(os.path.join(self.data_path, 'NVDQ.csv'))

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

    def prepare_features(self, data):
        """Create enhanced technical indicators"""
        try:
            df = data.copy()

            # Price-based features
            df['Returns'] = df['Close/Last'].pct_change()
            df['High_Low_Range'] = (df['High'] - df['Low']) / df['Close/Last']
            df['Close_Open_Range'] = (df['Close/Last'] - df['Open']) / df['Open']

            # Moving averages and price relative to MA
            for window in [5, 10, 20]:
                df[f'MA_{window}'] = df['Close/Last'].rolling(window=window).mean()
                df[f'Price_to_MA_{window}'] = df['Close/Last'] / df[f'MA_{window}']
                df[f'Volume_MA_{window}'] = df['Volume'].rolling(window=window).mean()

            # Momentum indicators
            df['ROC_5'] = df['Close/Last'].pct_change(periods=5)
            df['ROC_10'] = df['Close/Last'].pct_change(periods=10)

            # Volatility
            df['Volatility'] = df['Returns'].rolling(window=20).std()

            # Price trend
            df['Up_Trend'] = (df['Close/Last'] > df['MA_20']).astype(float)
            df['Price_Momentum'] = df['Close/Last'].diff(5)

            # Normalize prices by last close
            last_close = df['Close/Last'].iloc[-1]
            df['High'] = df['High'] / last_close
            df['Low'] = df['Low'] / last_close
            df['Close/Last'] = df['Close/Last'] / last_close
            df['Open'] = df['Open'] / last_close

            # Drop NaN values
            df = df.dropna()

            return df

        except Exception as e:
            print(f"Error in prepare_features: {e}")
            raise
    def create_sequences(self, data, scaler):
        """Create sequences for LSTM"""
        try:
            # Prepare features
            features = self.prepare_features(data)

            # Create sequences
            X, y = [], []
            for i in range(len(features) - self.lookback_period - 5):
                # Input sequence
                seq = features[['Close/Last', 'High', 'Low']].iloc[i:(i + self.lookback_period)].values
                X.append(seq)

                # Target: next 5 days
                future = features.iloc[i + self.lookback_period:i + self.lookback_period + 5]
                y.append([
                    future['High'].max(),
                    future['Low'].min(),
                    future['Close/Last'].mean()
                ])

            return np.array(X), np.array(y)

        except Exception as e:
            print(f"Error in create_sequences: {e}")
            raise
    def build_model(self, input_shape, is_inverse_etf=False):
        """Build LSTM model with relative price prediction"""
        inputs = Input(shape=input_shape)

        x = LSTM(64, return_sequences=True)(inputs)
        x = BatchNormalization()(x)
        x = Dropout(0.2)(x)

        x = LSTM(32, return_sequences=True)(x)
        x = BatchNormalization()(x)
        x = Dropout(0.2)(x)

        x = LSTM(16)(x)
        x = BatchNormalization()(x)

        # Output layer predicts relative price changes
        outputs = Dense(3, activation='sigmoid')(x)

        model = Model(inputs=inputs, outputs=outputs)
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='huber',
            metrics=['mae']
        )
        return model

    def train(self):
        """Train models"""
        try:
            print("Training NVDA model...")
            X_nvda, y_nvda = self.create_sequences(self.nvda_data, self.nvda_scaler)
            self.model_nvda = self.build_model((self.lookback_period, 3), is_inverse_etf=False)

            self.model_nvda.fit(
                X_nvda, y_nvda,
                epochs=50,
                batch_size=32,
                validation_split=0.2,
                callbacks=[
                    EarlyStopping(patience=10, restore_best_weights=True),
                    ReduceLROnPlateau(factor=0.5, patience=5, min_lr=1e-6)
                ],
                verbose=1
            )

            print("\nTraining NVDQ model...")
            X_nvdq, y_nvdq = self.create_sequences(self.nvdq_data, self.nvdq_scaler)
            self.model_nvdq = self.build_model((self.lookback_period, 3), is_inverse_etf=True)

            self.model_nvdq.fit(
                X_nvdq, y_nvdq,
                epochs=50,
                batch_size=32,
                validation_split=0.2,
                callbacks=[
                    EarlyStopping(patience=10, restore_best_weights=True),
                    ReduceLROnPlateau(factor=0.5, patience=5, min_lr=1e-6)
                ],
                verbose=1
            )

            print("Training completed successfully!")
            self.save_model()  # Save both models after training
            return True

        except Exception as e:
            print(f"Error in training: {e}")
            import traceback
            traceback.print_exc()
            raise

    def save_model(self):
        """Save both NVDA and NVDQ models."""
        if self.model_nvda:
            self.model_nvda.save(os.path.join(self.data_path, 'model_nvda.h5'))
            print("NVDA model saved successfully.")
        if self.model_nvdq:
            self.model_nvdq.save(os.path.join(self.data_path, 'model_nvdq.h5'))
            print("NVDQ model saved successfully.")

    def load_saved_model(self):
        """Load both NVDA and NVDQ models."""
        from tensorflow.keras.models import load_model

        nvda_model_path = os.path.join(self.data_path, 'model_nvda.h5')
        nvdq_model_path = os.path.join(self.data_path, 'model_nvdq.h5')

        if os.path.exists(nvda_model_path):
            self.model_nvda = load_model(nvda_model_path)
            print("NVDA model loaded successfully.")
        else:
            print("Warning: NVDA model file not found.")

        if os.path.exists(nvdq_model_path):
            self.model_nvdq = load_model(nvdq_model_path)
            print("NVDQ model loaded successfully.")
        else:
            print("Warning: NVDQ model file not found.")

    import numpy as np

    def predict_next_period(self, target_date=None):
        """Predict prices for the next 5 business days using LSTM models."""
        try:
            if self.model_nvda is None or self.model_nvdq is None:
                raise ValueError("Models are not loaded. Call load_saved_model() first.")

            if target_date is None:
                target_date = pd.Timestamp.today().strftime('%Y-%m-%d')
            target_date = pd.to_datetime(target_date)
            print(f"\nPredicting starting from: {target_date}")

            # Filter historical data up to the target date
            nvda_filtered = self.nvda_data[self.nvda_data['Date'] <= target_date]
            nvdq_filtered = self.nvdq_data[self.nvdq_data['Date'] <= target_date]

            # Normalize data and create sequences
            nvda_features = self.prepare_features(nvda_filtered)
            nvdq_features = self.prepare_features(nvdq_filtered)

            last_nvda_seq = nvda_features[['Close/Last', 'High', 'Low']].values[-self.lookback_period:]
            last_nvdq_seq = nvdq_features[['Close/Last', 'High', 'Low']].values[-self.lookback_period:]

            # Reshape for LSTM model
            last_nvda_seq = np.expand_dims(last_nvda_seq, axis=0)
            last_nvdq_seq = np.expand_dims(last_nvdq_seq, axis=0)

            # Predict next 5 days
            nvda_predictions = self.model_nvda.predict(last_nvda_seq)[0]
            nvdq_predictions = self.model_nvdq.predict(last_nvdq_seq)[0]

            daily_predictions = []
            for i in range(5):
                daily_predictions.append({
                    'date': (target_date + pd.Timedelta(days=i + 1)).strftime('%Y-%m-%d'),
                    'nvda': {
                        'open_price': None,  # Open price not predicted by LSTM
                        'close_price': nvda_predictions[2],  # Average close price
                        'highest_price': nvda_predictions[0],
                        'lowest_price': nvda_predictions[1],
                        'average_price': nvda_predictions[2]
                    },
                    'nvdq': {
                        'open_price': None,
                        'close_price': nvdq_predictions[2],
                        'highest_price': nvdq_predictions[0],
                        'lowest_price': nvdq_predictions[1],
                        'average_price': nvdq_predictions[2]
                    }
                })

            return daily_predictions

        except Exception as e:
            print(f"Error in prediction: {e}")
            raise


if __name__ == "__main__":
    predictor = StockPredictor()

    print("Loading data...")
    if predictor.load_data():
        try:
            print("\nTraining models...")
            predictor.train()

            print("\nMaking predictions...")
            # Test with specific date
            test_date = '2024-03-15'
            predictions = predictor.predict_next_period(test_date)

            # Also test with default (today's) date
            print("\nMaking predictions for today...")
            today_predictions = predictor.predict_next_period()

        except Exception as e:
            print(f"Error: {e}")
            import traceback

            traceback.print_exc()
    else:
        print("Data loading failed!")