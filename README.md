# Base Gas Optimiser

AI-powered gas price predictions for Base network. Save up to 40% on transaction fees with pattern-based insights and live blockchain data.

**🏆 Winner of the QMUL AI Society x Coinbase Hackathon** - demonstrating how machine learning can optimise blockchain transaction costs on Base L2.

## Live URLs

- **Frontend**: https://basegasfeesml.pages.dev
- **Backend API**: https://basegasfeesml-production.up.railway.app/api
- **API Documentation**: [docs/API.md](./docs/API.md)

## Features

- **Real-Time Gas Indicator** - Traffic light system shows if NOW is a good time to transact
- **Best Times Widget** - Shows cheapest/most expensive hours based on historical patterns
- **24-Hour Predictions** - ML-powered forecasts for 1h, 4h, and 24h horizons
- **Savings Calculator** - Estimate cost savings by waiting for optimal gas prices
- **Wallet Integration** - MetaMask support for Base network (Chain ID: 8453)
- **Live Gas Data** - Direct Base RPC integration for real-time pricing
- **Mobile-First Design** - Fully responsive, works on all devices
- **Real-Time Updates** - WebSocket integration for live data collection progress
- **Data Collection Dashboard** - Visual progress tracking with detailed milestones
- **Model Performance Metrics** - Live validation metrics and accuracy tracking
- **Network Intelligence Panel** - Advanced ML-driven insights
- **Gas Price Alerts** - Custom threshold notifications
- **Transaction Cost Calculator** - Real-time cost estimation for common operations

## Project Structure

```
Gweizy-main/
├── backend/              # Python Flask ML Backend
│   ├── api/              # API endpoints
│   ├── models/           # ML models and training
│   ├── services/         # Business logic services
│   ├── data/             # Data collection and database
│   └── utils/            # Utility functions
│
├── frontend/             # React + TypeScript Frontend
│   ├── src/              # Source code
│   ├── pages/            # Page components
│   └── public/           # Static assets
│
├── docs/                 # Documentation
│   ├── guides/           # How-to guides and tutorials
│   ├── planning/         # Roadmaps and planning docs
│   ├── deployment/       # Deployment guides
│   └── API.md            # API documentation
│
└── scripts/              # Utility scripts
```

For detailed structure information, see:
- [Organization Guide](./docs/ORGANIZATION.md) - Complete codebase organization
- [Project Structure](./docs/guides/PROJECT_STRUCTURE.md) - Detailed structure breakdown

## Quick Start

### Frontend Development
```bash
cd frontend
npm install
npm run dev
```
Open http://localhost:5173

### Backend Development
```bash
cd backend
pip install -r requirements.txt
python app.py
```
API runs on http://localhost:5000

## Tech Stack

**Frontend:**
- React 19.2.3 + TypeScript
- Vite 6.4.1 (build tool)
- Tailwind CSS (styling)
- Recharts (data visualization)
- Socket.IO Client (real-time updates)
- Service Worker (offline caching)
- Deployed on **Cloudflare Pages**

**Backend:**
- Python 3.x + Flask
- Scikit-learn (ML models)
- PostgreSQL (data storage)
- Socket.IO (WebSocket server)
- Deployed on **Railway**

**Blockchain:**
- Base Network (Chain ID: 8453)
- Direct RPC integration
- Live gas price fetching

## Performance Optimizations

The application is heavily optimized for fast loading and smooth user experience:

**Bundle Optimization:**
- Aggressive vendor code splitting (React, Charts, UI libraries separated)
- Terser minification with 2-pass compression
- CSS code splitting and minification
- Bundle size reduced from 347KB to ~200KB (42% reduction)

**Lazy Loading:**
- Intersection Observer-based component lazy loading
- Progressive loading with configurable viewport margins
- Saves 329KB on initial page load
- Custom `LazySection` wrapper for automatic optimization

**Caching Strategy:**
- Service Worker with dual caching strategy
- Cache-first for static assets (instant repeat visits)
- Network-first with stale-while-revalidate for API calls
- 5-minute API response cache

**Real-Time Updates:**
- WebSocket connections replace polling
- Live data collection progress updates
- Reduced server load and bandwidth usage

**Resource Hints:**
- DNS prefetch for Railway API and Base RPC
- Preconnect to critical origins
- Data preloading before app bundle loads

## Base Network Integration

- Live gas prices from Base RPC endpoints
- Pattern analysis of Base network transactions
- Optimized for Base L2 gas fee structure
- Real-time blockchain data via `eth_getBlockByNumber`


## Mobile Optimisation

The application is fully optimised for mobile devices with:
- Responsive hamburger navigation menu
- Touch-friendly buttons (minimum 44x44px touch targets)
- Adaptive typography and spacing
- Mobile-first responsive charts and graphs
- Horizontal scroll prevention

## License

MIT
