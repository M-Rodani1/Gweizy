# 🧪 Testing Suite for Gas Price Prediction Models

Comprehensive testing tools to measure and improve your ML model performance.

---

## 📁 Files in This Directory

| File | Purpose | When to Use |
|------|---------|-------------|
| `run_first_test.py` | **Quick sanity check** | First time setup, quick validation |
| `comprehensive_backtester.py` | **Full historical testing** | Before/after improvements, baseline measurement |
| `live_performance_monitor.py` | **Real-time monitoring** | Production monitoring, drift detection |
| `model_comparison.py` | **Compare model versions** | A/B testing, choosing best model |

---

## 🚀 Quick Start

### 1. First Time? Start Here

```bash
python run_first_test.py
```

This will guide you through your first test and check if everything is set up correctly.

**Time**: 5 minutes
**What you get**: Quick baseline performance metrics

---

### 2. Full Backtest

```bash
python comprehensive_backtester.py
```

Tests your model on 1 week of historical data.

**Time**: 2-5 minutes
**Output**:
- `backtest_results.png` - Visualization
- `backtest_report.json` - Detailed metrics

**What it tests**:
- All horizons: 1h, 4h, 24h
- MAE, RMSE, R², MAPE, directional accuracy
- Comparison vs naive baselines
- Spike detection performance
- Error distribution

---

### 3. Live Performance Monitoring

```bash
# Single check
python live_performance_monitor.py

# Continuous monitoring (every 5 min)
python live_performance_monitor.py --continuous
```

Tracks how your predictions perform in real-time.

**Time**: Instant check or continuous
**Output**:
- Real-time accuracy metrics
- Performance drift alerts
- `metrics_history.json` - Historical tracking

**Use for**:
- Production monitoring
- Detecting model degradation
- Validating predictions as they age

---

### 4. Model Comparison

```bash
python model_comparison.py
```

Compare different model versions side-by-side.

**Time**: 2-3 minutes
**Output**:
- Side-by-side comparison table
- `model_comparison_*.png` - Comparison charts

**Use for**:
- Before/after improvements
- A/B testing
- Choosing which model to deploy

---

## 📊 Understanding Test Results

### Key Metrics

**MAE (Mean Absolute Error)**
- Lower = Better
- Target: < 0.001 Gwei
- Meaning: "Predictions are off by X Gwei on average"

**R² Score**
- Higher = Better (0 to 1)
- Target: > 0.70
- Meaning: "Model explains X% of gas price variance"

**Directional Accuracy**
- Higher = Better (0% to 100%)
- Target: > 75%
- Meaning: "Correctly predict UP/DOWN X% of time"

### Good Results Example

```
1H PREDICTIONS
══════════════════════════════
MAE:        0.000800 Gwei ✅
R²:         0.7500 ✅
Directional: 78% ✅
vs Naive:   85% better ✅
```

→ **Model is performing well!**

### Needs Improvement Example

```
1H PREDICTIONS
══════════════════════════════
MAE:        0.002000 Gwei ⚠️
R²:         0.5500 ⚠️
Directional: 62% ⚠️
vs Naive:   55% better ⚠️
```

→ **See ML_IMPROVEMENT_PLAN.md for fixes**

---

## 🔬 Testing Workflow

### Before Implementing Improvements

```bash
# 1. Run baseline test
python comprehensive_backtester.py

# 2. Save results
cp backtest_report.json baseline_before.json
cp backtest_results.png baseline_before.png

# 3. Document metrics
# MAE: 0.001000, R²: 0.7000, Dir: 75%
```

### After Implementing Improvements

```bash
# 1. Retrain model
cd ../scripts
python train_ensemble_final.py

# 2. Re-test
cd ../testing
python comprehensive_backtester.py

# 3. Save results
cp backtest_report.json baseline_after.json

# 4. Compare
python model_comparison.py

# 5. Deploy if better!
```

---

## 🎯 Testing Checklist

Before deploying to production:

- [ ] Run `run_first_test.py` - sanity check
- [ ] Run `comprehensive_backtester.py` - full evaluation
- [ ] MAE < 0.001 Gwei ✅
- [ ] R² > 0.70 ✅
- [ ] Directional accuracy > 75% ✅
- [ ] Compare with previous model
- [ ] Test on multiple time periods
- [ ] Monitor live for 24h
- [ ] Save baseline metrics
- [ ] Document changes

---

## 🛠️ Troubleshooting

### "No trained models found!"

```bash
cd ../scripts
python train_ensemble_final.py
```

### "Not enough data"

Need 1,000+ records. Check:

```bash
cd ../data
sqlite3 gas_prices.db "SELECT COUNT(*) FROM gas_prices;"
```

If low, let data collector run for 24+ hours.

### R² is negative

Model is performing worse than predicting the mean.

**Fixes**:
1. More training data needed (10,000+ points)
2. Retrain from scratch
3. Check for NaN/Inf in features
4. Review data quality

### Module not found

```bash
pip install -r ../../requirements.txt
```

---

## 📚 Documentation

- **Quick Start**: `../../HOW_TO_TEST_YOUR_SYSTEM.md`
- **Full Guide**: `../../TESTING_GUIDE.md`
- **Improvement Plan**: `../../ML_IMPROVEMENT_PLAN.md`

---

## 💡 Tips

1. **Test regularly** - Run backtest weekly to catch drift
2. **Save baselines** - Keep historical metrics for comparison
3. **Monitor live** - Run continuous monitoring in production
4. **Test before deploy** - Always validate improvements
5. **Compare models** - Use model_comparison.py for A/B testing

---

## 🎯 Expected Performance

### Current (Baseline)
- R²: 70%
- Directional: 75%
- MAE: 0.001 Gwei

### After Improvements
- R²: 85% (+15%)
- Directional: 90% (+15%)
- MAE: 0.0005 Gwei (-50%)

**Timeline**: 1-2 months with systematic improvements

---

## 🚀 Ready to Test!

Start with:
```bash
python run_first_test.py
```

Good luck! 🎉
