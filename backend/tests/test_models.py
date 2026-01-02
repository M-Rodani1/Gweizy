#!/usr/bin/env python3
"""
Test ML models
Run: python tests/test_models.py
"""

import sys
sys.path.append('.')

import os
from models.model_trainer import GasModelTrainer

class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    BLUE = '\033[94m'
    YELLOW = '\033[93m'
    END = '\033[0m'

def print_test(name, passed, message=""):
    status = f"{Colors.GREEN}✅ PASS{Colors.END}" if passed else f"{Colors.RED}❌ FAIL{Colors.END}"
    print(f"{status} - {name}")
    if message:
        print(f"     {message}")

def test_model_files_exist():
    """Test if model files exist"""
    horizons = ['1h', '4h', '24h']
    all_exist = True
    
    for horizon in horizons:
        filepath = f'models/saved_models/model_{horizon}.pkl'
        exists = os.path.exists(filepath)
        
        if not exists:
            all_exist = False
            print(f"     {Colors.YELLOW}⚠️  Missing: {filepath}{Colors.END}")
    
    print_test('Model Files Exist', all_exist)
    return all_exist

def test_models_loadable():
    """Test if models can be loaded"""
    horizons = ['1h', '4h', '24h']
    all_loaded = True
    
    for horizon in horizons:
        try:
            model_data = GasModelTrainer.load_model(horizon)
            has_model = 'model' in model_data
            has_metrics = 'metrics' in model_data
            
            if not (has_model and has_metrics):
                all_loaded = False
                print(f"     {Colors.YELLOW}⚠️  Invalid structure: {horizon}{Colors.END}")
        except Exception as e:
            all_loaded = False
            print(f"     {Colors.YELLOW}⚠️  Failed to load {horizon}: {str(e)}{Colors.END}")
    
    print_test('Models Loadable', all_loaded)
    return all_loaded

def test_model_metrics():
    """Test model performance metrics"""
    try:
        model_data = GasModelTrainer.load_model('1h')
        metrics = model_data.get('metrics', {})
        
        # Check metrics exist
        has_mae = 'mae' in metrics
        has_r2 = 'r2' in metrics
        
        # Check metrics are reasonable
        mae = metrics.get('mae', float('inf'))
        r2 = metrics.get('r2', -1)
        
        # MAE should be small (< 1 gwei for good model)
        # R² should be positive
        good_mae = mae < 1.0
        good_r2 = r2 > 0
        
        passed = has_mae and has_r2 and good_mae and good_r2
        
        print_test(
            'Model Metrics', 
            passed, 
            f"MAE: {mae:.6f}, R²: {r2:.4f}"
        )
        return passed
        
    except Exception as e:
        print_test('Model Metrics', False, f"Error: {str(e)}")
        return False

def test_model_predictions():
    """Test if models can make predictions"""
    try:
        import numpy as np
        from models.feature_engineering import GasFeatureEngineer
        
        # Get some dummy data
        engineer = GasFeatureEngineer()
        df = engineer.prepare_training_data(hours_back=168)
        
        if len(df) < 10:
            print_test('Model Predictions', False, "Not enough data")
            return False
        
        feature_cols = engineer.get_feature_columns(df)
        X = df[feature_cols].iloc[-1:].values
        
        # Try prediction with 1h model
        model_data = GasModelTrainer.load_model('1h')
        model = model_data['model']
        
        prediction = model.predict(X)[0]
        
        # Check prediction is reasonable (0 < pred < 1000)
        is_reasonable = 0 < prediction < 1000
        
        print_test('Model Predictions', is_reasonable, f"Sample prediction: {prediction:.6f} gwei")
        return is_reasonable
        
    except Exception as e:
        print_test('Model Predictions', False, f"Error: {str(e)}")
        return False

def main():
    print(f"\n{Colors.BLUE}{'='*60}{Colors.END}")
    print(f"{Colors.BLUE}🤖 Testing ML Models{Colors.END}")
    print(f"{Colors.BLUE}{'='*60}{Colors.END}\n")
    
    tests = [
        test_model_files_exist,
        test_models_loadable,
        test_model_metrics,
        test_model_predictions
    ]
    
    results = []
    for test in tests:
        result = test()
        results.append(result)
        print()
    
    passed = sum(results)
    total = len(results)
    
    print(f"{Colors.BLUE}{'='*60}{Colors.END}")
    if passed == total:
        print(f"{Colors.GREEN}✅ All model tests passed! ({passed}/{total}){Colors.END}")
        sys.exit(0)
    else:
        print(f"{Colors.RED}❌ {total - passed} test(s) failed ({passed}/{total}){Colors.END}")
        print(f"\n{Colors.YELLOW}💡 Tip: Run 'python scripts/train_model.py' to train models{Colors.END}")
        sys.exit(1)

if __name__ == "__main__":
    main()

