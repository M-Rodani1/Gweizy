import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from data.database import DatabaseManager, Prediction


class ModelEvaluator:
    def __init__(self):
        self.db = DatabaseManager()
    
    def backtest_model(self, model_data, hours_back=168):
        """
        Backtest model on recent data
        Compare predictions vs actual values
        """
        from models.feature_engineering import GasFeatureEngineer
        
        engineer = GasFeatureEngineer()
        df = engineer.prepare_training_data(hours_back=hours_back)
        
        feature_cols = engineer.get_feature_columns(df)
        X = df[feature_cols]
        
        # Get model horizon
        horizon_map = {'1h': 'target_1h', '4h': 'target_4h', '24h': 'target_24h'}
        
        results = []
        
        for horizon, target_col in horizon_map.items():
            if horizon in model_data:
                model = model_data[horizon]['model']
                y_true = df[target_col].dropna()
                X_valid = X.loc[y_true.index]
                
                y_pred = model.predict(X_valid)
                
                mae = np.mean(np.abs(y_true - y_pred))
                mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
                
                results.append({
                    'horizon': horizon,
                    'mae': mae,
                    'mape': mape,
                    'samples': len(y_true)
                })
        
        return pd.DataFrame(results)
    
    def calculate_accuracy_over_time(self):
        """
        Calculate how accurate past predictions were
        Fetch predictions from database and compare to actuals
        """
        # Get predictions from last 7 days
        cutoff = datetime.now() - timedelta(days=7)
        predictions = self.db.session.query(Prediction).filter(
            Prediction.timestamp >= cutoff,
            Prediction.actual_gas.isnot(None)
        ).all()
        
        if not predictions:
            return {'message': 'No predictions with actuals yet'}
        
        df = pd.DataFrame([{
            'horizon': p.horizon,
            'predicted': p.predicted_gas,
            'actual': p.actual_gas,
            'error': abs(p.predicted_gas - p.actual_gas),
            'timestamp': p.timestamp
        } for p in predictions])
        
        # Group by horizon
        accuracy = df.groupby('horizon').agg({
            'error': ['mean', 'std', 'count'],
            'predicted': 'mean',
            'actual': 'mean'
        }).round(6)
        
        return accuracy.to_dict()


# Test evaluator
if __name__ == "__main__":
    evaluator = ModelEvaluator()
    
    # Load models
    from models.model_trainer import GasModelTrainer
    
    models = {}
    for horizon in ['1h', '4h', '24h']:
        try:
            models[horizon] = GasModelTrainer.load_model(horizon)
        except FileNotFoundError:
            print(f"Model for {horizon} not found. Train models first.")
    
    if models:
        results = evaluator.backtest_model(models, hours_back=168)
        print("\n📊 Backtest Results:")
        print(results)

