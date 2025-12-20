"""
Live Performance Monitoring for Gas Price Predictions

Continuously monitors prediction accuracy in real-time
Tracks drift, alerts on performance degradation
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from data.database import get_db_connection
import time
import json
from pathlib import Path


class LivePerformanceMonitor:
    """
    Monitor live prediction performance
    Compare predictions to actual values as time passes
    """

    def __init__(self, check_interval=300):
        """
        Args:
            check_interval: How often to check (seconds)
        """
        self.check_interval = check_interval
        self.metrics_history = []

    def get_recent_predictions(self, hours=24):
        """Get predictions made in last N hours"""
        conn = get_db_connection()

        query = """
        SELECT
            p.timestamp as prediction_time,
            p.horizon,
            p.predicted_gas,
            p.actual_gas,
            p.model_version,
            g.current_gas as gas_at_prediction
        FROM predictions p
        LEFT JOIN gas_prices g ON p.timestamp = g.timestamp
        WHERE p.timestamp >= ?
        AND p.actual_gas IS NOT NULL
        ORDER BY p.timestamp DESC
        """

        cutoff_time = datetime.utcnow() - timedelta(hours=hours)

        df = pd.read_sql_query(
            query,
            conn,
            params=(cutoff_time.isoformat(),),
            parse_dates=['prediction_time']
        )

        conn.close()

        return df

    def calculate_live_metrics(self, df, horizon='1h'):
        """Calculate metrics for validated predictions"""

        # Filter by horizon
        df_horizon = df[df['horizon'] == horizon].copy()

        if len(df_horizon) == 0:
            return None

        # Calculate metrics
        y_true = df_horizon['actual_gas'].values
        y_pred = df_horizon['predicted_gas'].values

        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)

        # MAPE
        mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100

        # Directional accuracy
        actual_direction = np.sign(np.diff(y_true))
        pred_direction = np.sign(np.diff(y_pred))
        directional_accuracy = np.mean(actual_direction == pred_direction)

        # Error percentiles
        errors = np.abs(y_pred - y_true)
        p50_error = np.median(errors)
        p90_error = np.percentile(errors, 90)
        p95_error = np.percentile(errors, 95)

        return {
            'horizon': horizon,
            'n_predictions': len(df_horizon),
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'mape': mape,
            'directional_accuracy': directional_accuracy,
            'p50_error': p50_error,
            'p90_error': p90_error,
            'p95_error': p95_error,
            'timestamp': datetime.utcnow().isoformat()
        }

    def check_performance_drift(self, current_metrics, window_hours=72):
        """
        Detect if performance is degrading
        Compare current metrics to historical baseline
        """

        # Load historical metrics
        metrics_file = Path(__file__).parent / 'metrics_history.json'

        if metrics_file.exists():
            with open(metrics_file, 'r') as f:
                history = json.load(f)
        else:
            history = []

        # Add current metrics
        history.append(current_metrics)

        # Keep only recent history
        cutoff_time = datetime.utcnow() - timedelta(hours=window_hours)
        history = [m for m in history
                  if datetime.fromisoformat(m['timestamp']) > cutoff_time]

        # Save updated history
        with open(metrics_file, 'w') as f:
            json.dump(history, f, indent=2)

        # Calculate baseline (mean of historical metrics)
        if len(history) < 10:
            return None  # Not enough data

        historical_mae = [m['mae'] for m in history[:-1]]
        historical_r2 = [m['r2'] for m in history[:-1]]
        historical_dir_acc = [m['directional_accuracy'] for m in history[:-1]]

        baseline_mae = np.mean(historical_mae)
        baseline_r2 = np.mean(historical_r2)
        baseline_dir_acc = np.mean(historical_dir_acc)

        # Compare current to baseline
        current_mae = current_metrics['mae']
        current_r2 = current_metrics['r2']
        current_dir_acc = current_metrics['directional_accuracy']

        # Calculate drift
        mae_drift_pct = ((current_mae - baseline_mae) / baseline_mae) * 100
        r2_drift_pct = ((current_r2 - baseline_r2) / baseline_r2) * 100
        dir_acc_drift_pct = ((current_dir_acc - baseline_dir_acc) / baseline_dir_acc) * 100

        # Detect alerts
        alerts = []

        if mae_drift_pct > 20:  # MAE increased by >20%
            alerts.append({
                'type': 'WARNING',
                'metric': 'MAE',
                'message': f'MAE increased by {mae_drift_pct:.1f}%',
                'severity': 'high' if mae_drift_pct > 50 else 'medium'
            })

        if r2_drift_pct < -10:  # R² decreased by >10%
            alerts.append({
                'type': 'WARNING',
                'metric': 'R²',
                'message': f'R² decreased by {abs(r2_drift_pct):.1f}%',
                'severity': 'high' if r2_drift_pct < -20 else 'medium'
            })

        if dir_acc_drift_pct < -5:  # Directional accuracy decreased
            alerts.append({
                'type': 'WARNING',
                'metric': 'Directional Accuracy',
                'message': f'Direction accuracy decreased by {abs(dir_acc_drift_pct):.1f}%',
                'severity': 'medium'
            })

        return {
            'baseline': {
                'mae': baseline_mae,
                'r2': baseline_r2,
                'directional_accuracy': baseline_dir_acc
            },
            'current': {
                'mae': current_mae,
                'r2': current_r2,
                'directional_accuracy': current_dir_acc
            },
            'drift': {
                'mae_pct': mae_drift_pct,
                'r2_pct': r2_drift_pct,
                'dir_acc_pct': dir_acc_drift_pct
            },
            'alerts': alerts
        }

    def print_live_metrics(self, metrics, drift_analysis=None):
        """Pretty print live performance metrics"""

        print("\n" + "="*70)
        print(f"  LIVE PERFORMANCE MONITOR - {metrics['horizon'].upper()}")
        print(f"  {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}")
        print("="*70)

        print(f"\n📊 Validated Predictions: {metrics['n_predictions']}")

        print(f"\n🎯 Current Performance:")
        print(f"   MAE:                  {metrics['mae']:.6f} Gwei")
        print(f"   RMSE:                 {metrics['rmse']:.6f} Gwei")
        print(f"   R²:                   {metrics['r2']:.4f}")
        print(f"   MAPE:                 {metrics['mape']:.2f}%")
        print(f"   Directional Accuracy: {metrics['directional_accuracy']:.2%}")

        print(f"\n📈 Error Distribution:")
        print(f"   Median Error (P50):   {metrics['p50_error']:.6f} Gwei")
        print(f"   P90 Error:            {metrics['p90_error']:.6f} Gwei")
        print(f"   P95 Error:            {metrics['p95_error']:.6f} Gwei")

        if drift_analysis and drift_analysis['alerts']:
            print(f"\n⚠️  PERFORMANCE ALERTS:")
            for alert in drift_analysis['alerts']:
                severity_icon = "🔴" if alert['severity'] == 'high' else "🟡"
                print(f"   {severity_icon} {alert['message']}")

            print(f"\n📊 Drift Analysis:")
            print(f"   MAE drift:      {drift_analysis['drift']['mae_pct']:+.1f}%")
            print(f"   R² drift:       {drift_analysis['drift']['r2_pct']:+.1f}%")
            print(f"   Dir Acc drift:  {drift_analysis['drift']['dir_acc_pct']:+.1f}%")
        else:
            print(f"\n✅ No performance degradation detected")

        print("\n" + "="*70)

    def monitor_continuously(self):
        """Run continuous monitoring loop"""

        print("🔍 Starting live performance monitor...")
        print(f"   Check interval: {self.check_interval} seconds")
        print("   Press Ctrl+C to stop\n")

        try:
            while True:
                # Get recent predictions
                df = self.get_recent_predictions(hours=24)

                if len(df) > 0:
                    # Check each horizon
                    for horizon in ['1h', '4h', '24h']:
                        metrics = self.calculate_live_metrics(df, horizon)

                        if metrics:
                            # Check for drift
                            drift = self.check_performance_drift(metrics)

                            # Print results
                            self.print_live_metrics(metrics, drift)

                            # Save to history
                            self.metrics_history.append(metrics)

                else:
                    print(f"⏳ No validated predictions yet. Waiting...")

                # Wait before next check
                time.sleep(self.check_interval)

        except KeyboardInterrupt:
            print("\n\n✅ Monitoring stopped by user")

            # Save final metrics
            if self.metrics_history:
                report_path = f"live_metrics_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
                with open(report_path, 'w') as f:
                    json.dump(self.metrics_history, f, indent=2)
                print(f"📄 Metrics saved to: {report_path}")


def run_single_check():
    """Run a single performance check (for testing)"""

    monitor = LivePerformanceMonitor()

    print("🔍 Running performance check...")

    df = monitor.get_recent_predictions(hours=24)

    if len(df) == 0:
        print("⚠️  No validated predictions found in last 24 hours")
        print("   Predictions need time to be validated after the horizon passes.")
        return

    print(f"\n✅ Found {len(df)} validated predictions")

    for horizon in ['1h', '4h', '24h']:
        metrics = monitor.calculate_live_metrics(df, horizon)

        if metrics:
            drift = monitor.check_performance_drift(metrics)
            monitor.print_live_metrics(metrics, drift)


if __name__ == '__main__':
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == '--continuous':
        # Continuous monitoring
        monitor = LivePerformanceMonitor(check_interval=300)  # Check every 5 min
        monitor.monitor_continuously()
    else:
        # Single check
        run_single_check()
