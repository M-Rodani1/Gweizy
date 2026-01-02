# Model Retraining Guide

## Current Status

The ML prediction models need to be retrained with the current data structure. The app currently uses fallback predictions (simple percentage increases) until models are retrained.

## Option 1: Retrain Locally (Recommended)

You've already done this! The models were successfully retrained locally with:
- 113 features (matching production)
- 27,000+ training samples
- MAE around 0.01-0.02 gwei

The trained models are saved in `backend/models/saved_models/` but are gitignored (too large for git).

### To deploy local models to Railway:

1. **Option A: Use Railway CLI**
   ```bash
   # Install Railway CLI if needed
   npm i -g @railway/cli

   # Login to Railway
   railway login

   # Link to your project
   railway link

   # Copy models to Railway volume
   railway run python3 backend/scripts/retrain_models_simple.py
   ```

2. **Option B: Manual upload via Railway dashboard**
   - Go to Railway dashboard
   - Open your service
   - Navigate to the `/data` volume
   - Upload the model files from `backend/models/saved_models/`:
     - model_1h.pkl (7.8MB)
     - model_4h.pkl (7.4MB)
     - model_24h.pkl (7.6MB)
     - scaler_1h.pkl (3.8KB)
     - scaler_4h.pkl (3.8KB)
     - scaler_24h.pkl (3.8KB)

## Option 2: Retrain on Railway

SSH into Railway and run the retraining script:

```bash
# Using Railway CLI
railway run bash

# Then inside the container
python3 backend/scripts/retrain_models_simple.py
```

This will:
1. Fetch all historical data from the database (currently ~27K records)
2. Create 113 features matching production
3. Train Random Forest models for each horizon
4. Save models to `backend/models/saved_models/`

## Option 3: Use the Retraining API

Trigger retraining via the API endpoint:

```bash
curl -X POST https://basegasfeesml-production.up.railway.app/api/retraining/trigger \
  -H "Content-Type: application/json" \
  -d '{"model_type": "all", "force": true}'
```

Note: The existing retraining API may need updates to work with the simple script.

## Verifying Models are Loaded

After retraining, check the logs or test the predictions endpoint:

```bash
curl https://basegasfeesml-production.up.railway.app/api/predictions
```

If successful, you should see:
- Real ML predictions (not fallback)
- Confidence intervals
- No warning about "Using fallback predictions"

## Model Performance

Expected metrics after retraining with current data:
- **1h horizon**: MAE ~0.017 gwei, Directional Accuracy ~58%
- **4h horizon**: MAE ~0.011 gwei, Directional Accuracy ~56%
- **24h horizon**: MAE ~0.009 gwei, Directional Accuracy ~55%

## Automatic Retraining

The system includes automatic retraining triggers:
- When prediction accuracy drops below thresholds
- When significant new data is collected (10K+ records)
- Weekly scheduled retraining

These are managed by the `ModelRetrainer` class in `utils/model_retrainer.py`.

## Troubleshooting

### Models still using fallback after retraining

Check that model files exist and are readable:
```bash
ls -lh backend/models/saved_models/model_*.pkl
```

### Feature mismatch errors

Ensure you're using the latest version of `advanced_features.py` which creates 113 features.

### Insufficient data for training

The retraining script requires at least 100 valid samples per horizon. Check collection status:
```bash
curl https://basegasfeesml-production.up.railway.app/api/analytics/collection-stats?hours=720
```

---

## Next Steps

1. Deploy local models to Railway (Option 1A or 1B above)
2. Verify predictions endpoint is working with ML models
3. Monitor performance via `/api/analytics/model-health`
4. Set up automatic retraining schedule
