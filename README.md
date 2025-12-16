# Base Gas Optimizer

AI-powered gas price predictions for Base network. Save up to 40% on transaction fees with pattern-based insights and live blockchain data.

**Built for the QMUL AI Society x Coinbase Hackathon** - demonstrating how machine learning can optimise blockchain transaction costs on Base L2.

## Live URLs

- **Frontend**: https://basegasfeesml.netlify.app
- **Backend API**: https://basegasfeesml.onrender.com/api

## Features

- **Real-Time Gas Indicator** - Traffic light system shows if NOW is a good time to transact
- **Best Times Widget** - Shows cheapest/most expensive hours based on historical patterns
- **24-Hour Predictions** - ML-powered forecasts for 1h, 4h, and 24h horizons
- **Savings Calculator** - Estimate cost savings by waiting for optimal gas prices
- **Wallet Integration** - MetaMask support for Base network (Chain ID: 8453)
- **Live Gas Data** - Direct Base RPC integration for real-time pricing
- **Mobile-First Design** - Fully responsive, works on all devices

## Project Structure

```
gasFeesPrediction-main/
├── frontend/              # React + TypeScript Frontend
│   ├── components/        # React components
│   ├── pages/            # Landing & Dashboard pages
│   ├── src/              # Utilities & API clients
│   └── public/           # Static assets
│
├── backend/              # Python Flask ML Backend
│   ├── api/              # API endpoints
│   ├── models/           # ML models
│   └── data/             # Data processing
│
└── Documentation/        # Project docs
```

See [PROJECT_STRUCTURE.md](./PROJECT_STRUCTURE.md) for detailed structure information.

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
- Deployed on **Netlify**

**Backend:**
- Python 3.x + Flask
- Scikit-learn (ML models)
- PostgreSQL (data storage)
- Deployed on **Render**

**Blockchain:**
- Base Network (Chain ID: 8453)
- Direct RPC integration
- Live gas price fetching

## Base Network Integration

- Live gas prices from Base RPC endpoints
- Pattern analysis of Base network transactions
- Optimized for Base L2 gas fee structure
- Real-time blockchain data via `eth_getBlockByNumber`

## Documentation

See [HACKATHON_SUBMISSION.md](HACKATHON_SUBMISSION.md) for complete hackathon submission details.

## Mobile Optimisation

The application is fully optimised for mobile devices with:
- Responsive hamburger navigation menu
- Touch-friendly buttons (minimum 44x44px touch targets)
- Adaptive typography and spacing
- Mobile-first responsive charts and graphs
- Horizontal scroll prevention

## License

MIT
