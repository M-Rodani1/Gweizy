# Base Gas Optimiser - Project Pitch
**10-Minute Presentation**

---

## 1. PROBLEM STATEMENT & MOTIVATION

Every day, millions of transactions happen on Base network. Users pay gas fees without knowing if they're overpaying.

**The Problem:**

Gas prices on Base fluctuate wildly - sometimes 2-3x higher depending on the time of day. Most users don't know when to transact, so they pay whatever the current price is.

**The Pain Point:**
```
Same transaction:
├── 9 PM (peak):    0.0043 gwei  💸 High cost
└── 3 AM (off-peak): 0.0017 gwei  💰 60% cheaper

Problem: Users have no way to know this
```

**Why This Matters:**

For users making 10 transactions per week, timing their transactions during off-peak hours could save **up to 60%** on gas fees monthly. That's significant savings just by knowing WHEN to transact.

---

## 2. SOLUTION & KEY FEATURES

**Base Gas Optimiser** - An AI-powered platform that predicts Base network gas prices and tells users the BEST time to transact.

### Three Core Features:

**1. Traffic Light Gas Indicator**
```
🟢 Green:  Gas 30% below average → Transact NOW
🟡 Yellow: Gas near average     → Typical pricing
🔴 Red:    Gas 50% above average → WAIT if possible
```

Real-time visual feedback. Users instantly know if NOW is a good time to transact.

**2. Best Times Widget**
```
┌─────────────────────────────────────┐
│  CHEAPEST HOURS  │ EXPENSIVE HOURS  │
├──────────────────┼──────────────────┤
│ 🟢 3am - 0.0017  │ 🔴 9pm - 0.0043  │
│ 🟢 2am - 0.0018  │ 🔴 8pm - 0.0038  │
│ 🟢 4am - 0.0019  │ 🔴 10pm - 0.0039 │
└──────────────────┴──────────────────┘
```

Historical pattern analysis. We analysed thousands of Base transactions to identify the cheapest and most expensive hours.

**3. ML-Powered Predictions**
```
1 Hour:  0.00225 gwei ↑ 4.2%  (High Confidence)
4 Hours: 0.00238 gwei ↑ 10.1% (Medium Confidence)
24 Hours: 0.00245 gwei ↑ 13.4% (High Confidence)

Recommendation: Gas expected to rise. Transact now to save.
```

Machine learning predicts gas prices 1, 4, and 24 hours ahead with confidence levels.

---

## 3. HOW AI IS USED IN THE PROJECT

**AI/ML Powers Our Core Predictions**

### Machine Learning Pipeline:

**Data Collection:**
- Live gas prices from Base blockchain via RPC every 5 minutes
- Historical transaction data spanning multiple weeks
- Base fee + priority fee analysis

**Feature Engineering (20+ Features):**
```python
# Engineered features from raw gas data
- Lag features: gas price 1h, 4h, 24h ago
- Rolling statistics: moving averages, volatility
- Time encoding: cyclical hour/day patterns
- Momentum indicators: RSI, MACD, Bollinger Bands
```

**Model Architecture:**
```
Model: Ensemble (RandomForest + GradientBoosting)
├── RandomForest: 100 trees, max depth 15
├── GradientBoosting: 100 trees, learning rate 0.1
└── Cross-validation: Time-series split (prevents data leakage)
```

**Performance Metrics:**
```
✅ 59.83% directional accuracy
   (Predicts UP/DOWN correctly 60% of the time)
✅ R² Score: 7.09%
✅ MAE: 0.000275 gwei
✅ Sub-200ms API response time
```

**Why This Works:**

Gas prices are HIGHLY time-dependent. There are strong patterns:
- Weekends are cheaper
- Certain hours are consistently expensive
- The ML model learned these patterns from Base network data

**AI-Powered Explanations:**

We also use Claude AI to generate natural language explanations of predictions, making complex ML insights understandable for everyday users.

---

## 4. TARGET USERS & EXPECTED IMPACT

### Who Benefits?

**Primary Users:**
- **DeFi traders** - Making multiple swaps daily
- **NFT collectors** - Timing mints to save on gas
- **DAOs** - Batching transactions during cheap hours
- **Regular Base users** - Anyone making transactions

### Expected Impact:

**Cost Savings:**
```
Example: User making 10 transactions/week

Option A: Peak times (9pm average)
└── Gas: 0.0043 gwei

Option B: Optimised timing (3am average)
└── Gas: 0.0017 gwei

💰 Savings: Up to 60% on gas fees monthly
```

**Network Efficiency:**

When users transact at optimal times, the entire Base network benefits:
- Less congestion during peak hours
- More distributed transaction load
- Better overall network performance

**Accessibility:**

Mobile-first design makes gas optimisation accessible to everyone:
- Works perfectly on phones, tablets, desktop
- Real-time updates every 30 seconds
- No technical knowledge required

---

## 5. DEMO & WALKTHROUGH

**Live Dashboard:** https://basegasfeesml.netlify.app

### Key Interactions:

**1. Landing Page**
- Clear value proposition
- Live gas price indicator
- Call-to-action to view dashboard

**2. Traffic Light System**
- Real-time colour-coded indicator (Green/Yellow/Red)
- Shows current gas vs. hourly and 24h averages
- Updates every 5 minutes

**3. Best Times Display**
- Side-by-side comparison: cheapest vs. most expensive hours
- Based on 168 hours of historical data
- Helps users plan transactions in advance

**4. Prediction Cards**
- Three time horizons: 1h, 4h, 24h
- Each shows:
  - Predicted gas price
  - Confidence level (High/Medium/Low)
  - Percentage change vs. current
  - Visual range slider (best case → worst case)
  - Actionable recommendation

**5. MetaMask Integration**
- One-click wallet connection
- Automatic Base network detection
- Auto-switch to Base if on wrong network
- Seamless user experience

**6. Technical Features**
- Direct Base RPC integration (no third-party APIs)
- Automatic RPC failover to backup endpoints
- Aggressive caching for <200ms response times
- Mobile-responsive design (44px touch targets)

### API Endpoints:

**Live API:** https://basegasfeesml.onrender.com/api

```
/api/current      → Current Base gas price
/api/predictions  → ML predictions (1h, 4h, 24h)
/api/explain      → AI-generated explanations
/api/accuracy     → Model performance metrics
```

**Example API Response:**
```json
{
  "current": {
    "current_gas": 0.002160,
    "base_fee": 0.001944,
    "priority_fee": 0.000216
  },
  "predictions": {
    "1h": {
      "predictedGwei": 0.002245,
      "confidence": 0.75,
      "direction": "up"
    }
  }
}
```

---

## TECHNICAL STACK

**Frontend:**
- React 19 + TypeScript
- Vite (fast builds, HMR)
- Tailwind CSS (mobile-first)
- Recharts (data visualisation)
- Deployed on **Netlify**

**Backend:**
- Python 3.11 + Flask
- scikit-learn (ML models)
- Web3 (Base RPC integration)
- PostgreSQL (data storage)
- Deployed on **Render**

**Blockchain:**
- Base Network (Chain ID: 8453)
- Direct RPC integration
- MetaMask wallet support

**Open Source:**
- Repository: github.com/M-Rodani1/basegasfeesML
- MIT License
- All code publicly available

---

## WHAT WE BUILT IN 4 DAYS

**Day 1:** Data collection pipeline + RPC rate limiting solutions

**Day 2-3:** Feature engineering + ML model training (23% → 59.83% accuracy)

**Day 4:** API development + Frontend integration + Deployment

**Key Achievement:** Built a production-ready gas prediction platform in under 96 hours.

---

## CLOSING

**Base Gas Optimiser** makes Base network transactions cheaper for everyone.

**Live Now:**
- 🌐 Dashboard: https://basegasfeesml.netlify.app
- 🔌 API: https://basegasfeesml.onrender.com/api
- 💻 GitHub: github.com/M-Rodani1/basegasfeesML

**Our Vision:** Every Base user should know when to transact, what the best times are, and what future prices will be - making the entire network more efficient.

**Thank you!**
