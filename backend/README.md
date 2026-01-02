# Base Gas Price Prediction Backend

A professional Flask backend for predicting Base network gas prices using machine learning.

## Project Structure

```
backend/
├── app.py                      # Main Flask app
├── config.py                   # Configuration
├── requirements.txt            # Dependencies
├── .env.example               # Environment template
├── data/
│   ├── collector.py           # Fetch Base gas data
│   ├── database.py            # Database operations
│   └── schema.sql             # DB schema
├── models/
│   ├── feature_engineering.py # Feature creation
│   ├── model_trainer.py       # Train ML models
│   ├── model_evaluator.py     # Evaluate model performance
│   └── saved_models/          # Trained model files
├── scripts/
│   └── train_model.py         # Training script
├── api/
│   ├── routes.py              # API endpoints
│   └── utils.py               # Helper functions
└── tests/
    └── test_api.py            # Unit tests
```

## Setup

### 1. Install Dependencies

```bash
cd backend
pip install -r requirements.txt
```

### 2. Environment Configuration

Create a `.env` file in the `backend/` directory (use `.env.example` as a template):

```bash
# Flask Configuration
DEBUG=True
PORT=5000

# Base Network
BASE_RPC_URL=https://mainnet.base.org
BASE_CHAIN_ID=8453
BASESCAN_API_KEY=

# APIs
OWLRACLE_API_KEY=

# Database
DATABASE_URL=sqlite:///gas_data.db
```

### 3. Run the Application

```bash
python app.py
```

The API will be available at `http://localhost:5000`

## API Endpoints

### Health Check
- `GET /api/health` - Check API health status

### Current Gas Price
- `GET /api/current` - Get current Base gas price

### Historical Data
- `GET /api/historical?hours=24` - Get historical gas prices (default: 720 hours)

### Predictions
- `GET /api/predictions` - Get predictions for all horizons (1h, 4h, 24h)
- `GET /api/predictions/<horizon>` - Get prediction for specific horizon (1h, 4h, or 24h)

### Statistics
- `GET /api/stats?hours=24` - Get gas price statistics

### Model Accuracy
- `GET /api/accuracy` - Get model accuracy metrics

### Transactions
- `GET /api/transactions?limit=10` - Get recent Base transactions

### Configuration
- `GET /api/config` - Get platform configuration

### Cache Management
- `POST /api/cache/clear` - Clear API cache (admin)

## Database

The application uses SQLAlchemy and supports both SQLite (default) and PostgreSQL.

### SQLite (Default)
No additional setup required. Database file will be created automatically.

### PostgreSQL
Update `DATABASE_URL` in `.env`:
```
DATABASE_URL=postgresql://user:password@localhost:5432/gas_data
```

## Testing

Run unit tests:

```bash
python -m pytest tests/
```

Or using unittest:

```bash
python -m unittest tests.test_api
```

## Machine Learning Models

The system uses production-quality ML models to predict gas prices at 1h, 4h, and 24h horizons.

### Training Models

Before using ML predictions, you need to train the models:

1. **Collect Data First**: Make sure you have at least 2-3 days of historical data
   ```bash
   # Check data count
   python -c "from data.database import DatabaseManager; print(len(DatabaseManager().get_historical_data(hours=720)))"
   ```

2. **Train Models**:
   ```bash
   python scripts/train_model.py
   ```

   This will:
   - Prepare features from historical data
   - Train multiple model architectures (Random Forest, Gradient Boosting, Ridge)
   - Select the best model for each horizon
   - Save models to `models/saved_models/`

3. **Expected Output**:
   ```
   🎯 Training models for 1h prediction horizon
   📊 Training Random Forest...
   📊 Training Gradient Boosting...
   📊 Training Ridge Regression...
   ✅ Best model for 1h: RandomForest
      MAE: 0.002341
      RMSE: 0.003156
      R²: 0.8234
   ```

### Model Features

The models use engineered features including:
- **Time features**: Hour, day of week, cyclical encodings
- **Lag features**: Gas prices from 1h, 3h, 6h, 12h, 24h ago
- **Rolling statistics**: Mean, std, min, max over various windows
- **Rate of change**: Percentage changes over time

### Model Evaluation

Models are evaluated using:
- **MAE** (Mean Absolute Error): Average prediction error in Gwei
- **RMSE** (Root Mean Squared Error): Penalizes larger errors
- **R² Score**: Coefficient of determination
- **Directional Accuracy**: Percentage of correct up/down predictions

### Retraining

Models should be retrained periodically (e.g., daily) to adapt to changing patterns:
```bash
python scripts/train_model.py
```

## Production Features

### Caching
- In-memory TTL cache for all endpoints
- Configurable cache durations per endpoint
- Cache hit/miss logging
- Manual cache clearing via `/api/cache/clear`

### Logging
- Structured logging to console and daily log files
- Request/response logging with timing
- Error traceback logging
- Logs stored in `logs/` directory

### Rate Limiting
- 200 requests/day per IP
- 50 requests/hour per IP
- 429 error responses for exceeded limits

### Error Handling
- Comprehensive error handlers (404, 500, 429)
- Detailed error logging
- User-friendly error messages

### CORS Configuration
- Configured for common frontend origins
- Supports localhost and Vercel deployments
- Configurable via `FRONTEND_URL` environment variable

### Base Transaction Scanner
- Fetches recent transactions from Base network
- Decodes common method signatures
- Formatted for frontend consumption

See [PRODUCTION_FEATURES.md](PRODUCTION_FEATURES.md) for detailed documentation.

## Development Notes

- The collector uses Base RPC as primary source with Owlracle API as fallback
- Data is automatically saved to database when fetched via `/api/current`
- Models are loaded automatically when the API starts
- If models aren't trained, the API falls back to simple heuristics
- All endpoints are cached for performance
- Request logging is enabled for monitoring

## Testing

### Quick Test
```bash
# Test all endpoints
./test_endpoints.sh
```

### Manual Testing
```bash
# Health check
curl http://localhost:5000/api/health

# Current gas
curl http://localhost:5000/api/current

# Predictions
curl http://localhost:5000/api/predictions

# Transactions
curl http://localhost:5000/api/transactions?limit=5
```

## Next Steps

1. Set up scheduled data collection (every 5 minutes)
2. Set up automated model retraining (daily)
3. Add model versioning and A/B testing
4. Implement prediction confidence intervals
5. Add more sophisticated features (network congestion, transaction volume, etc.)
6. Deploy to production with Redis for distributed caching

