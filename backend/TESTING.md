# Testing Checklist

## Pre-Training Checks

### 1. Check Data Availability
```bash
cd backend
python -c "from data.database import DatabaseManager; db = DatabaseManager(); data = db.get_historical_data(hours=720); print(f'Records: {len(data)}')"
```

**Expected**: At least 100 records (ideally 500+ for good models)

### 2. Test Data Collection
```bash
python -c "from data.collector import BaseGasCollector; c = BaseGasCollector(); print(c.get_current_gas())"
```

**Expected**: Returns current gas price data

## Training Models

### 3. Train Models
```bash
python scripts/train_model.py
```

**Expected Output**:
- ✅ Prepared training samples
- Training progress for each model type
- Best model selection for each horizon
- Model files saved to `models/saved_models/`

### 4. Verify Saved Models
```bash
ls -lh models/saved_models/
```

**Expected**: Three model files:
- `model_1h.pkl`
- `model_4h.pkl`
- `model_24h.pkl`

## API Testing

### 5. Start the Server
```bash
python app.py
```

**Expected**: Server starts on `http://localhost:5000`

### 6. Test Health Endpoint
```bash
curl http://localhost:5000/api/health | python -m json.tool
```

**Expected**: `{"status": "success", ...}`

### 7. Test Current Gas
```bash
curl http://localhost:5000/api/current | python -m json.tool
```

**Expected**: Current gas price data

### 8. Test Predictions (with ML models)
```bash
curl http://localhost:5000/api/predictions | python -m json.tool
```

**Expected**:
- `predictions` object with 1h, 4h, 24h values
- `model_info` with model details
- `historical` data for graphs
- `predicted_points` for visualization

### 9. Test Specific Horizon
```bash
curl http://localhost:5000/api/predictions/1h | python -m json.tool
```

**Expected**: Prediction for 1h horizon with model info

### 10. Test Accuracy Endpoint
```bash
curl http://localhost:5000/api/accuracy | python -m json.tool
```

**Expected**: Accuracy metrics (may return message if no predictions with actuals yet)

## Model Evaluation

### 11. Backtest Models
```bash
python -c "
from models.model_evaluator import ModelEvaluator
from models.model_trainer import GasModelTrainer

evaluator = ModelEvaluator()
models = {}
for h in ['1h', '4h', '24h']:
    models[h] = GasModelTrainer.load_model(h)

results = evaluator.backtest_model(models, hours_back=168)
print(results)
"
```

**Expected**: Backtest results with MAE and MAPE for each horizon

## Troubleshooting

### Issue: "Not enough data"
**Solution**: Run data collection for at least 2-3 days before training

### Issue: "Model not found"
**Solution**: Run `python scripts/train_model.py` first

### Issue: Import errors
**Solution**: Make sure all dependencies are installed: `pip install -r requirements.txt`

### Issue: Database errors
**Solution**: Check DATABASE_URL in `.env` file

