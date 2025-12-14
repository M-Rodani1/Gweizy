# Base Gas Optimizer - Hackathon Submission

## Project Overview

**Base Gas Optimizer** is an AI-powered gas price prediction tool for the Base network that helps users save 30-65% on transaction fees by timing their trades optimally.

## Live URLs

- **Frontend**: https://basegasfeesoptimiser.netlify.app
- **Backend API**: https://basegasfeesprediction.onrender.com
- **Config Endpoint**: https://basegasfeesprediction.onrender.com/config.json
- **GitHub Repository**: https://github.com/M-Rodani1/basegasfeesML

## Hackathon Requirements Met

### 1. Web App Built
- **Frontend**: React + TypeScript (Vite)
- **Backend**: Flask (Python)
- **Deployed**: Netlify (frontend) + Render (backend)

### 2. Base Integration via Client-Side JavaScript
- **File**: `src/utils/baseIntegration.ts`
- **Functions**:
  - `connectToBase()` - Connect wallet to Base network
  - `getBaseGasPrice()` - Get real-time gas prices from Base RPC
  - `getBaseNetworkInfo()` - Get network information
  - `estimateTransactionCost()` - Estimate transaction costs
- **Base Chain ID**: 8453 (Base Mainnet)
- **RPC**: https://mainnet.base.org

### 3. JSON Configuration Route
- **Endpoint**: https://basegasfeesprediction.onrender.com/config.json
- **Returns**: JSON configuration with:
  - App metadata (name, description, version)
  - Base network configuration (chain_id: 8453)
  - API endpoints
  - Farcaster frame configuration

### 4. Farcaster Integration
- **Farcaster Manifest**: `/.well-known/farcaster.json`
- **Account Association**: Verified via Base Build
- **Mini App SDK**: Integrated with `@farcaster/miniapp-sdk`
- **Meta Tags**: `fc:miniapp` tags in HTML

### 5. Base Developer Platform
- **Verification Meta Tag**: `<meta name="base:app_id" content="693e249fd19763ca26ddc2a8" />`
- **Base Build**: Verified and tested
- **Account Ownership**: Confirmed via account association

## Technical Stack

### Frontend
- **Framework**: React 19 + TypeScript
- **Styling**: Tailwind CSS
- **Routing**: React Router
- **Charts**: Recharts
- **Animation**: Framer Motion
- **Build**: Vite
- **Hosting**: Netlify

### Backend
- **Framework**: Flask (Python)
- **ML Models**: Scikit-learn ensemble models
- **Database**: PostgreSQL
- **API**: RESTful with CORS
- **Hosting**: Render

### Base Integration
- **Wallet Connection**: MetaMask/Coinbase Wallet via Web3
- **Network**: Base Mainnet (Chain ID: 8453)
- **RPC Calls**: Direct to Base RPC endpoint
- **Mini App**: Farcaster SDK integration

## Features

### Core Features
1. **Real-time Gas Predictions**
   - 5-minute, 15-minute, 30-minute, 1-hour forecasts
   - ML-powered ensemble models
   - 87%+ directional accuracy

2. **Base Wallet Connection**
   - Connect to Base network with one click
   - Auto-switch to Base if on wrong network
   - Display wallet address and network info

3. **Transaction Cost Estimator**
   - Estimate costs for different transaction types
   - Compare current vs. predicted prices
   - Calculate potential savings

4. **Model Transparency**
   - Display model accuracy metrics
   - Show confidence intervals
   - Explain predictions

### Base-Specific Features
1. **Base Mini App**
   - Runs inside Base app/Warpcast
   - Optimized for mobile
   - Native-like experience

2. **Base RPC Integration**
   - Direct blockchain queries
   - Real-time gas price fetching
   - Network status monitoring

3. **Farcaster Frame**
   - Shareable on Farcaster
   - Interactive buttons
   - Preview image

## API Endpoints

### Backend Endpoints
- `GET /api/health` - Health check
- `GET /api/current` - Current gas price
- `GET /api/predictions` - Future predictions
- `GET /api/transactions` - Recent transactions
- `GET /api/accuracy` - Model accuracy metrics
- `GET /config.json` - Base platform configuration

### Base RPC Endpoints (Client-Side)
- `eth_requestAccounts` - Wallet connection
- `eth_gasPrice` - Current gas price
- `eth_blockNumber` - Latest block
- `eth_chainId` - Network ID
- `eth_estimateGas` - Gas estimation

## Security

- **No Private Keys**: All wallet operations in browser
- **CORS Configured**: Only whitelisted origins
- **Rate Limiting**: 100 requests/minute
- **Input Validation**: All API inputs validated
- **HTTPS**: Both deployments use HTTPS

## Performance

- **API Response Time**: <500ms average
- **Caching**: 30-second response cache
- **Auto-Refresh**: 30-second data updates
- **Lazy Loading**: Components load on demand
- **CDN**: Static assets via Netlify CDN

## Testing

### Manual Testing Completed
- Frontend loads correctly
- Wallet connection to Base works
- Gas predictions display
- API endpoints respond
- Meta tags present
- Farcaster manifest validated
- Base Build verification passed
- Mobile responsive

## Deployment Process

1. **Backend**: Deployed to Render with `gunicorn`
2. **Frontend**: Deployed to Netlify with auto-builds
3. **Verification**: Base app_id meta tag added
4. **Manifest**: Farcaster account association generated
5. **Testing**: Validated via Base Build preview tool

## Hackathon Achievements

- Built a complete web application
- Integrated with Base blockchain
- Created JSON configuration endpoint
- Implemented Farcaster frame
- Verified via Base developer platform
- Deployed to production
- Ready for users

## Team

- Mohamed Rodani - Full Stack Development

## Contact

- **Email**: contact@basegasoptimizer.com
- **GitHub**: https://github.com/M-Rodani1

## Future Improvements

1. Gas price alerts via notifications
2. Historical data analysis
3. Multi-chain support (Optimism, Arbitrum)
4. Advanced ML models (LSTM, Transformer)
5. User preference learning
6. Portfolio gas optimization

---

**Thank you for reviewing our submission!**

This project demonstrates full integration with the Base ecosystem, including blockchain connectivity, developer platform verification, and Farcaster social integration.
