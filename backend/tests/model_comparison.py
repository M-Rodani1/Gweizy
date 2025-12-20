"""
Model Comparison Tool

Compare different models side-by-side to see which performs better
Useful before/after implementing improvements
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import joblib
from datetime import datetime, timedelta
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
from pathlib import Path

from data.database import get_db_connection
from models.feature_engineering import create_features
from models.advanced_features import create_advanced_features


class ModelComparison:
    """
    Compare multiple models on the same test set
    """

    def __init__(self, test_hours=168):
        """
        Args:
            test_hours: Hours of data to test on
        """
        self.test_hours = test_hours
        self.models = {}
        self.results = {}

    def load_model(self, name, model_path):
        """
        Load a model for comparison

        Args:
            name: Friendly name for the model (e.g., "Current Production", "New Transformer")
            model_path: Path to model file
        """
        try:
            model_data = joblib.load(model_path)
            self.models[name] = {
                'model': model_data.get('model'),
                'scaler_X': model_data.get('scaler_X'),
                'scaler_y': model_data.get('scaler_y'),
                'path': model_path,
                'metadata': model_data.get('metadata', {})
            }
            print(f"✅ Loaded model: {name}")
        except Exception as e:
            print(f"❌ Error loading {name}: {e}")

    def load_test_data(self):
        """Load test data"""
        print(f"\n📊 Loading {self.test_hours} hours of test data...")

        conn = get_db_connection()

        end_time = datetime.utcnow()
        start_time = end_time - timedelta(hours=self.test_hours)

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

        # Create features
        print("🔧 Engineering features...")
        df = create_features(df)
        df = create_advanced_features(df)
        df = df.dropna()

        print(f"✅ {len(df)} samples after feature engineering")

        return df

    def compare_models(self, df, horizon='1h'):
        """
        Compare all loaded models on same test set

        Args:
            df: Test dataframe
            horizon: Prediction horizon
        """

        if len(self.models) == 0:
            print("❌ No models loaded! Use load_model() first")
            return

        print(f"\n{'='*80}")
        print(f"  COMPARING {len(self.models)} MODELS - {horizon.upper()} PREDICTIONS")
        print(f"{'='*80}")

        # Create target
        if horizon == '1h':
            shift = -12
        elif horizon == '4h':
            shift = -48
        elif horizon == '24h':
            shift = -288
        else:
            return

        df[f'target_{horizon}'] = df['current_gas'].shift(shift)
        test_df = df[df[f'target_{horizon}'].notna()].copy()

        # Prepare features
        feature_cols = [col for col in test_df.columns
                       if col not in ['timestamp', 'target_1h', 'target_4h', 'target_24h']]

        X = test_df[feature_cols].values
        y_true = test_df[f'target_{horizon}'].values

        # Test each model
        results = {}

        for model_name, model_data in self.models.items():
            print(f"\n🔍 Testing: {model_name}")

            try:
                # Scale features
                X_scaled = model_data['scaler_X'].transform(X)

                # Predict
                y_pred_scaled = model_data['model'].predict(X_scaled)

                # Inverse transform
                y_pred = model_data['scaler_y'].inverse_transform(
                    y_pred_scaled.reshape(-1, 1)
                ).flatten()

                # Calculate metrics
                mae = mean_absolute_error(y_true, y_pred)
                rmse = np.sqrt(mean_squared_error(y_true, y_pred))
                r2 = r2_score(y_true, y_pred)
                mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100

                # Directional accuracy
                actual_dir = np.sign(np.diff(y_true))
                pred_dir = np.sign(np.diff(y_pred))
                dir_acc = np.mean(actual_dir == pred_dir)

                results[model_name] = {
                    'mae': mae,
                    'rmse': rmse,
                    'r2': r2,
                    'mape': mape,
                    'directional_accuracy': dir_acc,
                    'y_pred': y_pred
                }

                print(f"   MAE: {mae:.6f}, R²: {r2:.4f}, Dir Acc: {dir_acc:.2%}")

            except Exception as e:
                print(f"   ❌ Error: {e}")
                results[model_name] = None

        self.results[horizon] = {
            'y_true': y_true,
            'timestamps': test_df['timestamp'].values,
            'model_results': results
        }

        # Print comparison table
        self._print_comparison_table(horizon)

        return results

    def _print_comparison_table(self, horizon):
        """Print formatted comparison table"""

        print(f"\n{'='*80}")
        print(f"  COMPARISON TABLE - {horizon.upper()}")
        print(f"{'='*80}\n")

        results = self.results[horizon]['model_results']

        # Create comparison dataframe
        comparison_data = []

        for model_name, metrics in results.items():
            if metrics:
                comparison_data.append({
                    'Model': model_name,
                    'MAE': metrics['mae'],
                    'RMSE': metrics['rmse'],
                    'R²': metrics['r2'],
                    'MAPE': metrics['mape'],
                    'Dir Acc': metrics['directional_accuracy']
                })

        df_comp = pd.DataFrame(comparison_data)

        # Find best for each metric
        best_mae = df_comp['MAE'].min()
        best_rmse = df_comp['RMSE'].min()
        best_r2 = df_comp['R²'].max()
        best_mape = df_comp['MAPE'].min()
        best_dir_acc = df_comp['Dir Acc'].max()

        # Print table
        print(f"{'Model':<25} {'MAE':>12} {'RMSE':>12} {'R²':>8} {'MAPE':>10} {'Dir Acc':>10}")
        print("-" * 80)

        for _, row in df_comp.iterrows():
            mae_marker = "✅" if row['MAE'] == best_mae else "  "
            rmse_marker = "✅" if row['RMSE'] == best_rmse else "  "
            r2_marker = "✅" if row['R²'] == best_r2 else "  "
            mape_marker = "✅" if row['MAPE'] == best_mape else "  "
            dir_acc_marker = "✅" if row['Dir Acc'] == best_dir_acc else "  "

            print(f"{row['Model']:<25} "
                  f"{mae_marker} {row['MAE']:>10.6f} "
                  f"{rmse_marker} {row['RMSE']:>10.6f} "
                  f"{r2_marker} {row['R²']:>6.4f} "
                  f"{mape_marker} {row['MAPE']:>8.2f}% "
                  f"{dir_acc_marker} {row['Dir Acc']:>8.2%}")

        # Show improvement percentages
        if len(df_comp) > 1:
            print(f"\n📊 Improvement Analysis:")
            print(f"   (vs worst performing model)")

            worst_mae = df_comp['MAE'].max()
            worst_r2 = df_comp['R²'].min()

            mae_improvement = ((worst_mae - best_mae) / worst_mae) * 100
            r2_improvement = ((best_r2 - worst_r2) / abs(worst_r2)) * 100 if worst_r2 != 0 else 0

            print(f"   Best MAE improvement:  {mae_improvement:.1f}%")
            print(f"   Best R² improvement:   {r2_improvement:.1f}%")

    def plot_comparison(self, horizon='1h', save_path=None):
        """Generate comparison plots"""

        if horizon not in self.results:
            print(f"❌ No results for {horizon}")
            return

        data = self.results[horizon]
        y_true = data['y_true']
        timestamps = data['timestamps']
        models = data['model_results']

        n_models = len(models)

        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'Model Comparison - {horizon.upper()} Predictions', fontsize=16)

        # Plot 1: Predictions over time
        ax1 = axes[0, 0]
        ax1.plot(timestamps, y_true, label='Actual', linewidth=2, color='black', alpha=0.7)

        colors = plt.cm.tab10(np.linspace(0, 1, n_models))
        for (model_name, metrics), color in zip(models.items(), colors):
            if metrics:
                ax1.plot(timestamps, metrics['y_pred'], label=model_name, alpha=0.7, color=color)

        ax1.set_xlabel('Time')
        ax1.set_ylabel('Gas Price (Gwei)')
        ax1.set_title('Predictions Over Time')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot 2: MAE comparison
        ax2 = axes[0, 1]
        model_names = [name for name, metrics in models.items() if metrics]
        maes = [metrics['mae'] for name, metrics in models.items() if metrics]

        bars = ax2.bar(model_names, maes, color=colors[:len(model_names)])
        ax2.set_ylabel('MAE (Gwei)')
        ax2.set_title('Mean Absolute Error')
        ax2.tick_params(axis='x', rotation=45)

        # Highlight best
        min_idx = np.argmin(maes)
        bars[min_idx].set_color('green')
        bars[min_idx].set_alpha(0.8)

        # Plot 3: R² comparison
        ax3 = axes[1, 0]
        r2_scores = [metrics['r2'] for name, metrics in models.items() if metrics]

        bars = ax3.bar(model_names, r2_scores, color=colors[:len(model_names)])
        ax3.set_ylabel('R² Score')
        ax3.set_title('R² Score Comparison')
        ax3.tick_params(axis='x', rotation=45)
        ax3.axhline(y=0.7, color='r', linestyle='--', label='Target (0.7)')
        ax3.legend()

        # Highlight best
        max_idx = np.argmax(r2_scores)
        bars[max_idx].set_color('green')
        bars[max_idx].set_alpha(0.8)

        # Plot 4: Directional accuracy
        ax4 = axes[1, 1]
        dir_accs = [metrics['directional_accuracy'] for name, metrics in models.items() if metrics]

        bars = ax4.bar(model_names, [acc*100 for acc in dir_accs], color=colors[:len(model_names)])
        ax4.set_ylabel('Directional Accuracy (%)')
        ax4.set_title('Directional Accuracy')
        ax4.tick_params(axis='x', rotation=45)
        ax4.axhline(y=75, color='r', linestyle='--', label='Target (75%)')
        ax4.legend()

        # Highlight best
        max_idx = np.argmax(dir_accs)
        bars[max_idx].set_color('green')
        bars[max_idx].set_alpha(0.8)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✅ Plot saved to: {save_path}")

        return fig


def example_usage():
    """Example: Compare current model with a hypothetical improved model"""

    print("="*80)
    print("  MODEL COMPARISON TOOL")
    print("="*80)

    # Create comparison
    comparison = ModelComparison(test_hours=168)  # Test on 1 week

    # Load models to compare
    # Find all models in saved_models directory
    model_dir = Path(__file__).parent.parent / 'models' / 'saved_models'
    model_files = sorted(model_dir.glob('gas_predictor_ensemble_*.joblib'))

    if len(model_files) == 0:
        print("❌ No models found to compare!")
        return

    # Load latest model
    comparison.load_model("Current Model", str(model_files[-1]))

    # If there are previous versions, load them too
    if len(model_files) > 1:
        comparison.load_model("Previous Model", str(model_files[-2]))

    # Load test data
    df = comparison.load_test_data()

    # Compare models
    for horizon in ['1h', '4h', '24h']:
        comparison.compare_models(df, horizon)

    # Generate plots
    for horizon in ['1h', '4h', '24h']:
        comparison.plot_comparison(horizon, f'model_comparison_{horizon}.png')

    print("\n" + "="*80)
    print("  ✅ COMPARISON COMPLETE!")
    print("="*80)


if __name__ == '__main__':
    example_usage()
