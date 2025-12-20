"""
Comprehensive Backtesting Framework for Gas Price Prediction Models

This script allows you to:
1. Test your current models against historical data
2. Compare different model versions
3. Generate detailed performance reports
4. Identify weaknesses in predictions
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import joblib
import json
from datetime import datetime, timedelta
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Import your existing modules
from data.database import get_db_connection
from models.feature_engineering import create_features
from models.advanced_features import create_advanced_features


class ComprehensiveBacktester:
    """
    Backtest gas price prediction models on historical data
    """

    def __init__(self, model_path=None, lookback_hours=168):
        """
        Args:
            model_path: Path to trained model (if None, loads latest)
            lookback_hours: How many hours of historical data to test on
        """
        self.lookback_hours = lookback_hours
        self.model_path = model_path or self._find_latest_model()
        self.results = {}

        # Load model
        try:
            self.model_data = joblib.load(self.model_path)
            self.model = self.model_data.get('model')
            self.scaler_X = self.model_data.get('scaler_X')
            self.scaler_y = self.model_data.get('scaler_y')
            print(f"✅ Loaded model from: {self.model_path}")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise

    def _find_latest_model(self):
        """Find the most recent model file"""
        model_dir = Path(__file__).parent.parent / 'models' / 'saved_models'
        model_files = list(model_dir.glob('gas_predictor_ensemble_*.joblib'))

        if not model_files:
            raise FileNotFoundError("No trained models found!")

        latest = max(model_files, key=lambda p: p.stat().st_mtime)
        return str(latest)

    def load_historical_data(self):
        """Load historical gas price data"""
        print(f"\n📊 Loading {self.lookback_hours} hours of historical data...")

        conn = get_db_connection()

        # Calculate time range
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(hours=self.lookback_hours)

        query = """
        SELECT timestamp, current_gas, base_fee, priority_fee, block_number
        FROM gas_prices
        WHERE timestamp >= ? AND timestamp <= ?
        ORDER BY timestamp ASC
        """

        df = pd.read_sql_query(
            query,
            conn,
            params=(start_time.isoformat(), end_time.isoformat()),
            parse_dates=['timestamp']
        )

        conn.close()

        print(f"✅ Loaded {len(df)} data points")
        print(f"   Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")

        return df

    def prepare_features(self, df):
        """Create features from historical data"""
        print("\n🔧 Engineering features...")

        # Create basic features
        df = create_features(df)

        # Create advanced features
        df = create_advanced_features(df)

        # Drop NaN rows (from rolling windows)
        df = df.dropna()

        print(f"✅ Created {len(df.columns)} features")
        print(f"   {len(df)} samples after removing NaN")

        return df

    def backtest_predictions(self, df, horizons=['1h', '4h', '24h']):
        """
        Backtest predictions for different time horizons

        Args:
            df: DataFrame with features
            horizons: List of prediction horizons to test

        Returns:
            Dictionary with results for each horizon
        """
        print("\n🚀 Starting backtest...")

        results = {}

        for horizon in horizons:
            print(f"\n📈 Testing {horizon} predictions...")

            # Create target variable (shifted future gas price)
            if horizon == '1h':
                shift = -12  # 5-min intervals, 12 = 1 hour
            elif horizon == '4h':
                shift = -48
            elif horizon == '24h':
                shift = -288
            else:
                continue

            # Create target
            df[f'target_{horizon}'] = df['current_gas'].shift(shift)

            # Remove rows where we don't have future data
            test_df = df[df[f'target_{horizon}'].notna()].copy()

            if len(test_df) == 0:
                print(f"⚠️  No data available for {horizon} horizon")
                continue

            # Prepare features
            feature_cols = [col for col in test_df.columns
                          if col not in ['timestamp', 'target_1h', 'target_4h', 'target_24h']]

            X = test_df[feature_cols].values
            y_true = test_df[f'target_{horizon}'].values

            # Scale features
            X_scaled = self.scaler_X.transform(X)

            # Make predictions
            y_pred_scaled = self.model.predict(X_scaled)

            # Inverse transform predictions
            y_pred = self.scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()

            # Calculate metrics
            metrics = self._calculate_metrics(y_true, y_pred, test_df['current_gas'].values)

            # Store results
            results[horizon] = {
                'y_true': y_true,
                'y_pred': y_pred,
                'timestamps': test_df['timestamp'].values,
                'current_gas': test_df['current_gas'].values,
                'metrics': metrics,
                'n_samples': len(y_true)
            }

            # Print metrics
            self._print_metrics(horizon, metrics)

        self.results = results
        return results

    def _calculate_metrics(self, y_true, y_pred, current_gas):
        """Calculate comprehensive performance metrics"""

        # Basic metrics
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)

        # MAPE (Mean Absolute Percentage Error)
        mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100

        # Directional Accuracy
        actual_direction = np.sign(np.diff(y_true))
        pred_direction = np.sign(np.diff(y_pred))
        directional_accuracy = np.mean(actual_direction == pred_direction)

        # Error distribution
        errors = y_pred - y_true
        error_std = np.std(errors)
        error_median = np.median(np.abs(errors))

        # Prediction vs current gas
        beat_naive = np.mean(np.abs(y_true - y_pred) < np.abs(y_true - current_gas[:-abs(len(y_true) - len(current_gas))]))

        # Spike detection performance
        spike_threshold = np.percentile(y_true, 90)  # Top 10% = spikes
        is_spike = y_true > spike_threshold

        if np.sum(is_spike) > 0:
            spike_mae = mean_absolute_error(y_true[is_spike], y_pred[is_spike])
            spike_detected = np.mean((y_pred > spike_threshold) & is_spike)
            spike_false_positive = np.mean((y_pred > spike_threshold) & ~is_spike)
        else:
            spike_mae = None
            spike_detected = None
            spike_false_positive = None

        return {
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'mape': mape,
            'directional_accuracy': directional_accuracy,
            'error_std': error_std,
            'error_median': error_median,
            'beat_naive_forecast': beat_naive,
            'spike_mae': spike_mae,
            'spike_detection_rate': spike_detected,
            'spike_false_positive_rate': spike_false_positive
        }

    def _print_metrics(self, horizon, metrics):
        """Pretty print metrics"""
        print(f"\n{'='*60}")
        print(f"  {horizon.upper()} PREDICTION METRICS")
        print(f"{'='*60}")

        # Overall accuracy
        print(f"\n📊 Overall Accuracy:")
        print(f"   MAE:        {metrics['mae']:.6f} Gwei")
        print(f"   RMSE:       {metrics['rmse']:.6f} Gwei")
        print(f"   R² Score:   {metrics['r2']:.4f} ({self._r2_rating(metrics['r2'])})")
        print(f"   MAPE:       {metrics['mape']:.2f}%")

        # Directional accuracy
        print(f"\n🎯 Directional Accuracy:")
        print(f"   {metrics['directional_accuracy']:.2%} ({self._dir_rating(metrics['directional_accuracy'])})")

        # Error distribution
        print(f"\n📉 Error Distribution:")
        print(f"   Std Dev:    {metrics['error_std']:.6f} Gwei")
        print(f"   Median:     {metrics['error_median']:.6f} Gwei")

        # Comparison to naive forecast
        print(f"\n🆚 vs Naive Forecast (predict = current):")
        print(f"   Better:     {metrics['beat_naive_forecast']:.2%} of the time")

        # Spike detection
        if metrics['spike_detection_rate'] is not None:
            print(f"\n⚡ Spike Detection:")
            print(f"   Detection:  {metrics['spike_detection_rate']:.2%}")
            print(f"   False Pos:  {metrics['spike_false_positive_rate']:.2%}")
            print(f"   Spike MAE:  {metrics['spike_mae']:.6f} Gwei")

    def _r2_rating(self, r2):
        """Get rating for R² score"""
        if r2 > 0.8:
            return "✅ EXCELLENT"
        elif r2 > 0.7:
            return "✅ GOOD"
        elif r2 > 0.5:
            return "⚠️  NEEDS IMPROVEMENT"
        else:
            return "❌ POOR"

    def _dir_rating(self, acc):
        """Get rating for directional accuracy"""
        if acc > 0.8:
            return "✅ EXCELLENT"
        elif acc > 0.7:
            return "✅ GOOD"
        elif acc > 0.6:
            return "⚠️  NEEDS IMPROVEMENT"
        else:
            return "❌ POOR"

    def plot_results(self, save_path='backtest_results.png'):
        """Generate visualizations of backtest results"""
        print(f"\n📊 Generating plots...")

        n_horizons = len(self.results)
        fig, axes = plt.subplots(n_horizons, 3, figsize=(18, 5*n_horizons))

        if n_horizons == 1:
            axes = axes.reshape(1, -1)

        for idx, (horizon, data) in enumerate(self.results.items()):
            # Plot 1: Predictions vs Actual
            ax1 = axes[idx, 0]
            ax1.plot(data['timestamps'], data['y_true'], label='Actual', alpha=0.7, linewidth=1)
            ax1.plot(data['timestamps'], data['y_pred'], label='Predicted', alpha=0.7, linewidth=1)
            ax1.set_title(f'{horizon.upper()} - Predictions vs Actual')
            ax1.set_xlabel('Time')
            ax1.set_ylabel('Gas Price (Gwei)')
            ax1.legend()
            ax1.grid(True, alpha=0.3)

            # Plot 2: Error distribution
            ax2 = axes[idx, 1]
            errors = data['y_pred'] - data['y_true']
            ax2.hist(errors, bins=50, edgecolor='black', alpha=0.7)
            ax2.axvline(0, color='red', linestyle='--', linewidth=2, label='Zero Error')
            ax2.set_title(f'{horizon.upper()} - Error Distribution')
            ax2.set_xlabel('Prediction Error (Gwei)')
            ax2.set_ylabel('Frequency')
            ax2.legend()
            ax2.grid(True, alpha=0.3)

            # Plot 3: Scatter plot
            ax3 = axes[idx, 2]
            ax3.scatter(data['y_true'], data['y_pred'], alpha=0.5, s=10)
            ax3.plot([data['y_true'].min(), data['y_true'].max()],
                    [data['y_true'].min(), data['y_true'].max()],
                    'r--', linewidth=2, label='Perfect Prediction')
            ax3.set_title(f'{horizon.upper()} - Predicted vs Actual')
            ax3.set_xlabel('Actual Gas Price (Gwei)')
            ax3.set_ylabel('Predicted Gas Price (Gwei)')
            ax3.legend()
            ax3.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✅ Plots saved to: {save_path}")

        return fig

    def generate_report(self, save_path='backtest_report.json'):
        """Generate comprehensive JSON report"""
        print(f"\n📄 Generating report...")

        report = {
            'model_path': self.model_path,
            'test_date': datetime.utcnow().isoformat(),
            'lookback_hours': self.lookback_hours,
            'horizons': {}
        }

        for horizon, data in self.results.items():
            report['horizons'][horizon] = {
                'n_samples': data['n_samples'],
                'metrics': {k: float(v) if v is not None else None
                           for k, v in data['metrics'].items()}
            }

        # Save to file
        with open(save_path, 'w') as f:
            json.dump(report, f, indent=2)

        print(f"✅ Report saved to: {save_path}")

        return report

    def compare_with_baseline(self):
        """Compare model performance with simple baselines"""
        print(f"\n🆚 COMPARISON WITH BASELINE MODELS")
        print(f"{'='*60}")

        for horizon, data in self.results.items():
            print(f"\n{horizon.upper()} Predictions:")

            y_true = data['y_true']
            y_pred = data['y_pred']
            current = data['current_gas'][:len(y_true)]

            # Baseline 1: Naive forecast (predict = current)
            naive_mae = mean_absolute_error(y_true, current)

            # Baseline 2: Moving average
            ma_pred = pd.Series(current).rolling(12).mean().fillna(current).values
            ma_mae = mean_absolute_error(y_true, ma_pred)

            # Baseline 3: Linear trend
            from scipy import stats
            slope, intercept = stats.linregress(range(len(current)), current)[:2]
            trend_pred = slope * np.arange(len(y_true)) + intercept
            trend_mae = mean_absolute_error(y_true, trend_pred)

            # Our model
            model_mae = data['metrics']['mae']

            print(f"\n  MAE Comparison:")
            print(f"    Naive (current):      {naive_mae:.6f} Gwei")
            print(f"    Moving Average:       {ma_mae:.6f} Gwei")
            print(f"    Linear Trend:         {trend_mae:.6f} Gwei")
            print(f"    Our Model:            {model_mae:.6f} Gwei ✅")

            # Improvement percentages
            naive_improvement = ((naive_mae - model_mae) / naive_mae) * 100
            ma_improvement = ((ma_mae - model_mae) / ma_mae) * 100

            print(f"\n  Improvement vs Baselines:")
            print(f"    vs Naive:             {naive_improvement:+.1f}%")
            print(f"    vs Moving Average:    {ma_improvement:+.1f}%")


def main():
    """Run comprehensive backtest"""
    print("="*60)
    print("  GAS PRICE PREDICTION MODEL - BACKTEST")
    print("="*60)

    # Create backtester
    backtester = ComprehensiveBacktester(lookback_hours=168)  # Test on 1 week

    # Load data
    df = backtester.load_historical_data()

    # Prepare features
    df = backtester.prepare_features(df)

    # Run backtest
    results = backtester.backtest_predictions(df, horizons=['1h', '4h', '24h'])

    # Compare with baselines
    backtester.compare_with_baseline()

    # Generate visualizations
    backtester.plot_results('backtest_results.png')

    # Generate report
    report = backtester.generate_report('backtest_report.json')

    print("\n" + "="*60)
    print("  ✅ BACKTEST COMPLETE!")
    print("="*60)
    print("\nGenerated files:")
    print("  📊 backtest_results.png - Visualization of predictions")
    print("  📄 backtest_report.json - Detailed metrics report")


if __name__ == '__main__':
    main()
