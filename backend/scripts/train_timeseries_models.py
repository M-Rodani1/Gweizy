"""
Advanced Time-Series Model Training for Gas Price Prediction

Implements LSTM (Long Short-Term Memory) neural networks and Prophet
for improved gas price forecasting on Base network.

This script trains models that better capture temporal patterns, seasonality,
and trends compared to the original RandomForest/GradientBoosting ensemble.

Usage:
    python scripts/train_timeseries_models.py --model lstm --horizon 1h
    python scripts/train_timeseries_models.py --model prophet --horizon 24h
    python scripts/train_timeseries_models.py --model all
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import json

# Check if tensorflow is available
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("Warning: TensorFlow not available. LSTM models will not be trainable.")

# Check if prophet is available
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False
    print("Warning: Prophet not available. Prophet models will not be trainable.")

from data.database import DatabaseManager


class TimeSeriesDataPreparator:
    """Prepares time-series data for LSTM and Prophet models"""

    def __init__(self, db_manager):
        self.db = db_manager
        self.scalers = {}

    def load_historical_data(self, hours=4320):  # 6 months default
        """Load historical gas prices from database"""
        print(f"Loading {hours} hours ({hours/24:.0f} days) of historical data...")

        session = self.db._get_session()
        try:
            from data.database import GasPrice
            from sqlalchemy import func

            cutoff = datetime.now() - timedelta(hours=hours)
            results = session.query(GasPrice).filter(
                GasPrice.timestamp >= cutoff
            ).order_by(GasPrice.timestamp.asc()).all()

            if len(results) == 0:
                raise ValueError("No historical data found in database!")

            # Convert to DataFrame
            df = pd.DataFrame([{
                'timestamp': r.timestamp,
                'gas_price': r.current_gas,
                'base_fee': r.base_fee,
                'priority_fee': r.priority_fee
            } for r in results])

            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values('timestamp')

            print(f"✓ Loaded {len(df):,} records")
            print(f"  Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
            print(f"  Average gas: {df['gas_price'].mean():.6f} Gwei")

            return df

        finally:
            session.close()

    def create_sequences(self, data, lookback=24, forecast_horizon=1):
        """
        Create sequences for LSTM training

        Args:
            data: Array of gas prices
            lookback: Number of historical time steps to use as input
            forecast_horizon: Number of time steps to predict ahead

        Returns:
            X: Input sequences (lookback timesteps)
            y: Target values (forecast_horizon ahead)
        """
        X, y = [], []

        for i in range(lookback, len(data) - forecast_horizon):
            X.append(data[i-lookback:i])
            y.append(data[i + forecast_horizon - 1])

        return np.array(X), np.array(y)

    def prepare_lstm_data(self, df, lookback=24, train_split=0.8):
        """
        Prepare data for LSTM training

        Args:
            df: DataFrame with timestamp and gas_price columns
            lookback: Number of historical hours to look back
            train_split: Fraction of data to use for training

        Returns:
            Dictionary containing train/test sets and scaler
        """
        print(f"\nPreparing LSTM data (lookback={lookback})...")

        # Extract gas prices
        prices = df['gas_price'].values.reshape(-1, 1)

        # Scale data to [0, 1] range (LSTM works better with scaled data)
        scaler = MinMaxScaler()
        scaled_prices = scaler.fit_transform(prices)

        # Create sequences
        X, y = self.create_sequences(scaled_prices, lookback=lookback, forecast_horizon=1)

        # Time-series split (no shuffling!)
        train_size = int(len(X) * train_split)

        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]

        print(f"✓ Created {len(X):,} sequences")
        print(f"  Training set: {len(X_train):,} sequences")
        print(f"  Test set: {len(X_test):,} sequences")

        # Reshape for LSTM input: [samples, timesteps, features]
        X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
        X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

        return {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'scaler': scaler,
            'lookback': lookback
        }

    def prepare_prophet_data(self, df):
        """
        Prepare data for Prophet model

        Prophet requires columns named 'ds' (datestamp) and 'y' (value)
        """
        print("\nPreparing Prophet data...")

        prophet_df = df[['timestamp', 'gas_price']].copy()
        prophet_df.columns = ['ds', 'y']

        print(f"✓ Prepared {len(prophet_df):,} records for Prophet")

        return prophet_df


class LSTMGasPredictor:
    """LSTM-based gas price predictor"""

    def __init__(self, lookback=24, units=50, layers=2, dropout=0.2):
        """
        Initialize LSTM model

        Args:
            lookback: Number of historical time steps
            units: Number of LSTM units per layer
            layers: Number of LSTM layers
            dropout: Dropout rate for regularization
        """
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow is required for LSTM models. Install with: pip install tensorflow")

        self.lookback = lookback
        self.units = units
        self.layers = layers
        self.dropout = dropout
        self.model = None
        self.scaler = None
        self.history = None

    def build_model(self):
        """Build LSTM architecture"""
        model = Sequential()

        # First LSTM layer
        if self.layers > 1:
            model.add(Bidirectional(LSTM(
                units=self.units,
                return_sequences=True,
                input_shape=(self.lookback, 1)
            )))
        else:
            model.add(LSTM(
                units=self.units,
                input_shape=(self.lookback, 1)
            ))

        model.add(Dropout(self.dropout))

        # Additional LSTM layers
        for i in range(1, self.layers):
            return_seq = (i < self.layers - 1)  # Only last layer doesn't return sequences
            model.add(Bidirectional(LSTM(
                units=self.units // (2 ** i),  # Decreasing units
                return_sequences=return_seq
            )))
            model.add(Dropout(self.dropout))

        # Output layer
        model.add(Dense(1))

        # Compile
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='huber',  # Robust to outliers
            metrics=['mae']
        )

        print(f"\n=== LSTM Architecture ===")
        model.summary()

        self.model = model
        return model

    def train(self, X_train, y_train, X_val, y_val, epochs=100, batch_size=32):
        """Train LSTM model with early stopping"""
        if self.model is None:
            self.build_model()

        print(f"\nTraining LSTM model...")
        print(f"  Epochs: {epochs}")
        print(f"  Batch size: {batch_size}")

        # Callbacks
        early_stop = EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        )

        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        )

        # Train
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stop, reduce_lr],
            verbose=1
        )

        print(f"\n✓ Training complete")
        return self.history

    def predict(self, X):
        """Make predictions"""
        return self.model.predict(X, verbose=0)

    def save(self, path):
        """Save model to disk"""
        self.model.save(f"{path}/lstm_model.h5")
        joblib.dump(self.scaler, f"{path}/lstm_scaler.pkl")
        print(f"✓ Saved LSTM model to {path}")

    @classmethod
    def load(cls, path):
        """Load model from disk"""
        model = cls()
        model.model = tf.keras.models.load_model(f"{path}/lstm_model.h5")
        model.scaler = joblib.load(f"{path}/lstm_scaler.pkl")
        return model


class ProphetGasPredictor:
    """Prophet-based gas price predictor"""

    def __init__(self, seasonality_mode='multiplicative'):
        """
        Initialize Prophet model

        Args:
            seasonality_mode: 'additive' or 'multiplicative'
        """
        if not PROPHET_AVAILABLE:
            raise ImportError("Prophet is required. Install with: pip install prophet")

        self.seasonality_mode = seasonality_mode
        self.model = None

    def build_model(self):
        """Build Prophet model with custom seasonalities"""
        model = Prophet(
            seasonality_mode=self.seasonality_mode,
            daily_seasonality=True,
            weekly_seasonality=True,
            yearly_seasonality=False,  # Not enough data yet
            changepoint_prior_scale=0.05,  # Flexibility of trend changes
            seasonality_prior_scale=10.0   # Strength of seasonality
        )

        # Add custom seasonalities
        model.add_seasonality(
            name='hourly',
            period=1,  # 1 day
            fourier_order=8  # Complexity of hourly pattern
        )

        self.model = model
        return model

    def train(self, df):
        """Train Prophet model"""
        if self.model is None:
            self.build_model()

        print(f"\nTraining Prophet model...")
        print(f"  Records: {len(df):,}")

        self.model.fit(df)

        print(f"✓ Training complete")
        return self.model

    def predict(self, periods=24):
        """
        Make future predictions

        Args:
            periods: Number of hours to forecast ahead
        """
        future = self.model.make_future_dataframe(periods=periods, freq='H')
        forecast = self.model.predict(future)
        return forecast

    def save(self, path):
        """Save model to disk"""
        joblib.dump(self.model, f"{path}/prophet_model.pkl")
        print(f"✓ Saved Prophet model to {path}")

    @classmethod
    def load(cls, path):
        """Load model from disk"""
        model = cls()
        model.model = joblib.load(f"{path}/prophet_model.pkl")
        return model


def calculate_directional_accuracy(y_true, y_pred):
    """Calculate percentage of correct directional predictions"""
    if len(y_true) < 2:
        return 0.0

    # Calculate actual direction (up/down from previous)
    actual_direction = np.diff(y_true) > 0

    # Calculate predicted direction
    predicted_direction = np.diff(y_pred) > 0

    # Compare
    correct = (actual_direction == predicted_direction).sum()
    total = len(actual_direction)

    return correct / total if total > 0 else 0.0


def evaluate_model(y_true, y_pred, model_name="Model"):
    """Evaluate model performance"""
    print(f"\n=== {model_name} Performance ===")

    # Flatten arrays if needed
    y_true = y_true.flatten() if len(y_true.shape) > 1 else y_true
    y_pred = y_pred.flatten() if len(y_pred.shape) > 1 else y_pred

    # Calculate metrics
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    dir_accuracy = calculate_directional_accuracy(y_true, y_pred)

    print(f"MAE:  {mae:.6f} Gwei")
    print(f"RMSE: {rmse:.6f} Gwei")
    print(f"R²:   {r2:.4f}")
    print(f"Directional Accuracy: {dir_accuracy:.2%}")

    return {
        'mae': float(mae),
        'rmse': float(rmse),
        'r2': float(r2),
        'directional_accuracy': float(dir_accuracy)
    }


def main():
    parser = argparse.ArgumentParser(description='Train advanced time-series models for gas prediction')

    parser.add_argument('--model', choices=['lstm', 'prophet', 'all'], default='all',
                       help='Model type to train')
    parser.add_argument('--horizon', choices=['1h', '4h', '24h', 'all'], default='all',
                       help='Forecast horizon')
    parser.add_argument('--lookback', type=int, default=24,
                       help='Number of hours to look back (LSTM only)')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Training epochs (LSTM only)')
    parser.add_argument('--save-dir', default='models/saved_models',
                       help='Directory to save trained models')

    args = parser.parse_args()

    # Initialize
    db = DatabaseManager()
    preparator = TimeSeriesDataPreparator(db)

    # Load data
    df = preparator.load_historical_data(hours=4320)  # 6 months

    if len(df) < 100:
        print("Error: Insufficient data for training. Need at least 100 records.")
        print("Run: python scripts/collect_historical_data.py --months 6")
        return

    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)

    results = {}

    # Train LSTM
    if args.model in ['lstm', 'all'] and TENSORFLOW_AVAILABLE:
        print("\n" + "="*60)
        print("Training LSTM Model")
        print("="*60)

        lstm_data = preparator.prepare_lstm_data(df, lookback=args.lookback)

        lstm_model = LSTMGasPredictor(
            lookback=args.lookback,
            units=64,
            layers=2,
            dropout=0.2
        )

        lstm_model.scaler = lstm_data['scaler']
        lstm_model.train(
            lstm_data['X_train'],
            lstm_data['y_train'],
            lstm_data['X_test'],
            lstm_data['y_test'],
            epochs=args.epochs,
            batch_size=32
        )

        # Evaluate
        y_pred_scaled = lstm_model.predict(lstm_data['X_test'])
        y_pred = lstm_data['scaler'].inverse_transform(y_pred_scaled)
        y_true = lstm_data['scaler'].inverse_transform(lstm_data['y_test'])

        results['lstm'] = evaluate_model(y_true, y_pred, "LSTM")

        # Save
        lstm_model.save(args.save_dir)

    # Train Prophet
    if args.model in ['prophet', 'all'] and PROPHET_AVAILABLE:
        print("\n" + "="*60)
        print("Training Prophet Model")
        print("="*60)

        prophet_data = preparator.prepare_prophet_data(df)

        # Split for evaluation
        train_size = int(len(prophet_data) * 0.8)
        train_df = prophet_data[:train_size]
        test_df = prophet_data[train_size:]

        prophet_model = ProphetGasPredictor()
        prophet_model.train(train_df)

        # Evaluate
        forecast = prophet_model.predict(periods=len(test_df))
        y_true = test_df['y'].values
        y_pred = forecast.tail(len(test_df))['yhat'].values

        results['prophet'] = evaluate_model(y_true, y_pred, "Prophet")

        # Save
        prophet_model.save(args.save_dir)

    # Save results
    results_path = f"{args.save_dir}/training_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n✓ Results saved to {results_path}")


if __name__ == "__main__":
    main()
