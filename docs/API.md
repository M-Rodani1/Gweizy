# Base Gas Optimiser API Documentation

**Base URL:** `https://basegasfeesml-production.up.railway.app`

**Version:** 1.0

## Overview

The Base Gas Optimiser API provides ML-powered gas price predictions, real-time network data, and transaction optimization features for the Base L2 network.

## Authentication

Currently, the API is publicly accessible. Rate limiting is applied to prevent abuse.

## Rate Limits

- **Default:** 100 requests/minute per IP
- **Predictions:** 60 requests/minute
- **Heavy endpoints:** 30 requests/minute

---

## Core Endpoints

### Health Check

Check API and system health status.

```http
GET /api/health
```

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2024-01-17T12:00:00Z",
  "components": {
    "database": "ok",
    "models": "ok",
    "cache": "ok"
  }
}
```

---

### Current Gas Price

Get the current Base network gas price.

```http
GET /api/current
```

**Response:**
```json
{
  "gas_price_gwei": 0.00115,
  "gas_price_wei": 1150000,
  "timestamp": "2024-01-17T12:00:00Z",
  "source": "base_rpc",
  "chain_id": 8453
}
```

**Cache:** 30 seconds

---

### Gas Price Predictions

Get ML-powered gas price predictions for different time horizons.

```http
GET /api/predictions
```

**Query Parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `horizon` | string | all | `1h`, `4h`, `24h`, or `all` |
| `explain` | boolean | false | Include prediction explanation |

**Response:**
```json
{
  "predictions": {
    "1h": {
      "predicted_gwei": 0.00118,
      "confidence": 0.85,
      "direction": "up",
      "change_percent": 2.6
    },
    "4h": {
      "predicted_gwei": 0.00112,
      "confidence": 0.78,
      "direction": "down",
      "change_percent": -2.6
    },
    "24h": {
      "predicted_gwei": 0.00105,
      "confidence": 0.72,
      "direction": "down",
      "change_percent": -8.7
    }
  },
  "current_gas": 0.00115,
  "timestamp": "2024-01-17T12:00:00Z",
  "model_version": "v2.1.0"
}
```

---

### Prediction Explanation

Get detailed explanation for a specific prediction.

```http
GET /api/explain/{horizon}
```

**Path Parameters:**
| Parameter | Type | Description |
|-----------|------|-------------|
| `horizon` | string | `1h`, `4h`, or `24h` |

**Response:**
```json
{
  "horizon": "1h",
  "prediction": 0.00118,
  "confidence": 0.85,
  "factors": [
    {
      "name": "hour_of_day",
      "importance": 0.32,
      "value": 14,
      "impact": "positive"
    },
    {
      "name": "network_congestion",
      "importance": 0.28,
      "value": 0.65,
      "impact": "positive"
    }
  ],
  "model_type": "hybrid_ensemble"
}
```

---

### Historical Gas Prices

Get historical gas price data.

```http
GET /api/historical
```

**Query Parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `hours` | integer | 168 | Hours of history (max 720) |
| `timeframe` | string | hourly | `hourly` or `daily` |

**Response:**
```json
{
  "data": [
    {
      "timestamp": "2024-01-17T11:00:00Z",
      "gas_price_gwei": 0.00112,
      "block_number": 12345678
    }
  ],
  "timeframe": "hourly",
  "count": 168
}
```

---

### Gas Price Patterns

Get hourly and daily gas price patterns for heatmap visualization.

```http
GET /api/gas/patterns
```

**Response:**
```json
{
  "hourly_patterns": {
    "0": { "avg": 0.00095, "min": 0.0008, "max": 0.0012 },
    "1": { "avg": 0.00092, "min": 0.0008, "max": 0.0011 }
  },
  "daily_patterns": {
    "monday": { "avg": 0.00105, "peak_hour": 14 },
    "tuesday": { "avg": 0.00108, "peak_hour": 15 }
  },
  "best_times": {
    "cheapest_hour": 4,
    "cheapest_day": "sunday",
    "expensive_hour": 14,
    "expensive_day": "tuesday"
  }
}
```

---

### Pattern Matching

Get historical pattern matching for gas predictions.

```http
GET /api/patterns
```

**Response:**
```json
{
  "current_pattern": {
    "type": "ascending",
    "strength": 0.75
  },
  "similar_patterns": [
    {
      "date": "2024-01-10",
      "similarity": 0.92,
      "outcome": "spike",
      "change_percent": 15.2
    }
  ],
  "prediction": {
    "direction": "up",
    "confidence": 0.82
  }
}
```

---

## Analytics Endpoints

### Analytics Dashboard

Get comprehensive analytics dashboard data.

```http
GET /api/analytics/dashboard
```

**Response:**
```json
{
  "performance": {
    "accuracy_1h": 0.87,
    "accuracy_4h": 0.82,
    "accuracy_24h": 0.75
  },
  "predictions_today": 1245,
  "active_alerts": 23,
  "data_quality": {
    "completeness": 0.98,
    "freshness": "2m ago"
  }
}
```

---

### Volatility Analysis

Get gas price volatility index (GVI).

```http
GET /api/analytics/volatility
```

**Response:**
```json
{
  "available": true,
  "volatility_index": 45,
  "level": "moderate",
  "description": "Normal gas price fluctuations",
  "color": "yellow",
  "trend": "stable",
  "metrics": {
    "current_price": 0.00115,
    "avg_price": 0.00110,
    "std_dev": 0.00012
  }
}
```

**Volatility Levels:**
- 0-20: Very Low (green)
- 20-40: Low (green)
- 40-60: Moderate (yellow)
- 60-80: High (orange)
- 80-100: Extreme (red)

---

### Whale Activity

Monitor large transactions that may impact gas prices.

```http
GET /api/analytics/whales
```

**Response:**
```json
{
  "available": true,
  "current": {
    "whale_count": 3,
    "activity_level": "moderate",
    "description": "Moderate whale activity detected",
    "estimated_price_impact_pct": 6,
    "impact": "moderate"
  },
  "recent_whales": [
    {
      "tx_hash": "0x...",
      "gas_used": 850000,
      "timestamp": "2024-01-17T11:55:00Z"
    }
  ]
}
```

---

### Anomaly Detection

Get anomaly detection results for unusual price movements.

```http
GET /api/analytics/anomalies
```

**Response:**
```json
{
  "available": true,
  "status": "normal",
  "status_color": "green",
  "anomaly_count": 0,
  "anomalies": [],
  "current_analysis": {
    "price": 0.00115,
    "z_score": 0.5,
    "vs_average_pct": 5
  }
}
```

---

### Model Ensemble Status

Get ML model ensemble health and status.

```http
GET /api/analytics/ensemble
```

**Response:**
```json
{
  "available": true,
  "health": {
    "status": "healthy",
    "color": "green",
    "loaded_models": 4,
    "total_models": 5,
    "health_pct": 80
  },
  "primary_model": "hybrid_predictor",
  "prediction_mode": "ML-powered predictions active",
  "models": [
    { "name": "hybrid_predictor", "type": "primary", "loaded": true },
    { "name": "spike_detector", "type": "classifier", "loaded": true }
  ]
}
```

---

## Network & On-Chain Endpoints

### Network State

Get current network state with on-chain features.

```http
GET /api/onchain/network-state
```

**Response:**
```json
{
  "block_number": 12345678,
  "timestamp": "2024-01-17T12:00:00Z",
  "gas_price_gwei": 0.00115,
  "base_fee_gwei": 0.001,
  "block_utilization": 0.65,
  "pending_txs": 145,
  "congestion_level": "moderate"
}
```

---

### Mempool Status

Get current mempool status and metrics.

```http
GET /api/mempool/status
```

**Response:**
```json
{
  "status": "active",
  "metrics": {
    "pending_count": 45,
    "avg_gas_price": 0.00115,
    "median_gas_price": 0.001,
    "arrival_rate": 12.5
  },
  "signals": {
    "is_congested": false,
    "congestion_level": "low",
    "count_momentum": 0.02,
    "gas_momentum": -0.03
  },
  "interpretation": {
    "trend": "stable",
    "recommendation": "Good time to transact"
  }
}
```

---

## Alerts Endpoints

### Create Alert

Create a new gas price alert.

```http
POST /api/alerts
```

**Request Body:**
```json
{
  "user_id": "user123",
  "alert_type": "below",
  "threshold_gwei": 0.001,
  "notification_method": "webhook",
  "notification_target": "https://your-webhook.com/gas-alert",
  "cooldown_minutes": 60
}
```

**Alert Types:** `below`, `above`, `spike`, `drop`

**Response:**
```json
{
  "id": "alert_abc123",
  "created_at": "2024-01-17T12:00:00Z",
  "status": "active"
}
```

---

### Get User Alerts

```http
GET /api/alerts/{user_id}
```

**Response:**
```json
{
  "alerts": [
    {
      "id": "alert_abc123",
      "alert_type": "below",
      "threshold_gwei": 0.001,
      "is_active": true,
      "last_triggered": null
    }
  ]
}
```

---

### Delete Alert

```http
DELETE /api/alerts/{alert_id}?user_id={user_id}
```

---

## Multi-Chain Endpoints

### Get All Chain Gas Prices

```http
GET /api/multichain/gas
```

**Response:**
```json
{
  "chains": {
    "8453": { "name": "Base", "gas_gwei": 0.00115 },
    "1": { "name": "Ethereum", "gas_gwei": 25.5 },
    "42161": { "name": "Arbitrum", "gas_gwei": 0.1 }
  },
  "timestamp": "2024-01-17T12:00:00Z"
}
```

---

### Compare Chains

```http
GET /api/multichain/compare?gas_units=21000&tx_type=transfer
```

**Response:**
```json
{
  "comparison": [
    { "chain": "Base", "chain_id": 8453, "cost_usd": 0.0024 },
    { "chain": "Arbitrum", "chain_id": 42161, "cost_usd": 0.021 },
    { "chain": "Ethereum", "chain_id": 1, "cost_usd": 1.07 }
  ],
  "cheapest": "Base",
  "gas_units": 21000
}
```

---

## Accuracy & Validation Endpoints

### Accuracy Metrics

Get current accuracy metrics for all prediction horizons.

```http
GET /api/accuracy/metrics
```

**Response:**
```json
{
  "1h": {
    "mae": 0.00008,
    "rmse": 0.00012,
    "mape": 6.5,
    "directional_accuracy": 0.87,
    "sample_count": 1250
  },
  "4h": {
    "mae": 0.00015,
    "rmse": 0.00022,
    "mape": 12.3,
    "directional_accuracy": 0.82
  },
  "24h": {
    "mae": 0.00028,
    "rmse": 0.00035,
    "mape": 18.7,
    "directional_accuracy": 0.75
  }
}
```

---

### Validation Summary

```http
GET /api/validation/summary
```

**Response:**
```json
{
  "total_predictions": 15420,
  "validated_predictions": 14890,
  "pending_validation": 530,
  "validation_rate": 0.966,
  "last_validation": "2024-01-17T11:55:00Z"
}
```

---

## Monitoring Endpoints

### System Health

```http
GET /api/monitoring/health
```

**Response:**
```json
{
  "overall_status": "healthy",
  "components": {
    "api": { "status": "healthy", "latency_ms": 45 },
    "database": { "status": "healthy", "connections": 12 },
    "models": { "status": "healthy", "loaded": 5 },
    "cache": { "status": "healthy", "hit_rate": 0.89 }
  },
  "uptime_seconds": 86400
}
```

---

### API Performance

```http
GET /api/monitoring/api-performance
```

**Response:**
```json
{
  "requests_total": 125000,
  "requests_per_minute": 85,
  "avg_latency_ms": 52,
  "p95_latency_ms": 150,
  "error_rate": 0.002,
  "top_endpoints": [
    { "path": "/api/predictions", "requests": 45000 },
    { "path": "/api/current", "requests": 32000 }
  ]
}
```

---

### Cache Statistics

```http
GET /api/monitoring/cache-stats
```

**Response:**
```json
{
  "hit_rate": 0.89,
  "total_requests": 125000,
  "cache_hits": 111250,
  "cache_misses": 13750,
  "size_bytes": 52428800,
  "entries": 1250
}
```

---

## Personalization Endpoints

### Get Recommendations

Get personalized recommendations for a wallet address.

```http
GET /api/personalization/recommendations/{user_address}
```

**Response:**
```json
{
  "recommendations": [
    {
      "type": "optimal_time",
      "message": "Based on your history, you save 15% by transacting between 2-4 AM UTC",
      "confidence": 0.85
    },
    {
      "type": "gas_alert",
      "message": "Gas is currently 20% below your average transaction price",
      "action": "transact_now"
    }
  ],
  "user_stats": {
    "total_transactions": 45,
    "avg_gas_paid": 0.00125,
    "potential_savings_pct": 18
  }
}
```

---

### Track Transaction

Record a transaction to improve recommendations.

```http
POST /api/personalization/track-transaction
```

**Request Body:**
```json
{
  "user_address": "0x...",
  "tx_hash": "0x...",
  "gas_price_gwei": 0.00115,
  "gas_used": 21000,
  "chain_id": 8453
}
```

---

## Agent Endpoints

### Get AI Recommendation

Get AI-powered transaction timing recommendation.

```http
POST /api/agent/recommend
```

**Request Body:**
```json
{
  "tx_type": "swap",
  "gas_amount": 150000,
  "urgency": "medium",
  "current_gas": 0.00115,
  "chain_id": 8453
}
```

**Urgency Levels:** `low`, `medium`, `high`, `critical`

**Response:**
```json
{
  "recommendation": "wait",
  "wait_time_minutes": 45,
  "expected_savings_pct": 12,
  "confidence": 0.78,
  "reasoning": "Gas prices typically drop 12% in the next hour based on current patterns"
}
```

---

## Error Responses

All endpoints return errors in a consistent format:

```json
{
  "error": true,
  "code": "RATE_LIMITED",
  "message": "Too many requests. Please try again in 60 seconds.",
  "retry_after": 60
}
```

**Common Error Codes:**
| Code | HTTP Status | Description |
|------|-------------|-------------|
| `RATE_LIMITED` | 429 | Rate limit exceeded |
| `INVALID_PARAMS` | 400 | Invalid request parameters |
| `NOT_FOUND` | 404 | Resource not found |
| `SERVER_ERROR` | 500 | Internal server error |
| `MODEL_UNAVAILABLE` | 503 | ML models temporarily unavailable |

---

## WebSocket Events

Connect to real-time updates via WebSocket:

```javascript
const socket = io('https://basegasfeesml-production.up.railway.app');

// Subscribe to gas price updates
socket.on('gas_update', (data) => {
  console.log('New gas price:', data.gas_price_gwei);
});

// Subscribe to prediction updates
socket.on('prediction_update', (data) => {
  console.log('New prediction:', data);
});
```

---

## SDK Examples

### JavaScript/TypeScript

```typescript
// Fetch current gas price
const response = await fetch('https://basegasfeesml-production.up.railway.app/api/current');
const data = await response.json();
console.log(`Current gas: ${data.gas_price_gwei} gwei`);

// Get predictions
const predictions = await fetch('/api/predictions?horizon=1h&explain=true');
const predData = await predictions.json();
```

### Python

```python
import requests

# Get current gas price
response = requests.get('https://basegasfeesml-production.up.railway.app/api/current')
data = response.json()
print(f"Current gas: {data['gas_price_gwei']} gwei")

# Get predictions with explanation
predictions = requests.get('/api/predictions', params={'horizon': '1h', 'explain': True})
```

### cURL

```bash
# Get current gas price
curl https://basegasfeesml-production.up.railway.app/api/current

# Get predictions
curl "https://basegasfeesml-production.up.railway.app/api/predictions?horizon=1h"

# Create alert
curl -X POST https://basegasfeesml-production.up.railway.app/api/alerts \
  -H "Content-Type: application/json" \
  -d '{"user_id": "user123", "alert_type": "below", "threshold_gwei": 0.001}'
```

---

## Changelog

### v1.0 (January 2024)
- Initial API release
- Core gas price endpoints
- ML prediction endpoints
- Analytics dashboard
- Alert system
- Multi-chain support
