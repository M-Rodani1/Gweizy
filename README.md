# Base Gas Optimiser

AI-powered gas price predictions for Base network. Save up to 65% on transaction fees with ML-driven forecasts.

## Live URLs

- **Frontend**: https://basegasfeesml.netlify.app
- **Backend API**: https://basegasfeesml.onrender.com
- **Config Endpoint**: https://basegasfeesml.onrender.com/config.json

## Features

- Real-time gas predictions (5-min, 15-min, 30-min, 1-hour forecasts)
- Base wallet connection with MetaMask/Coinbase Wallet
- Transaction cost calculator
- ML-powered ensemble models with 87%+ accuracy
- Farcaster Mini App integration
- Mobile-optimised responsive design

## Tech Stack

**Frontend**: React 19 + TypeScript + Tailwind CSS + Vite
**Backend**: Flask + Scikit-learn + PostgreSQL
**Blockchain**: Base Mainnet (Chain ID: 8453)

## Local Development

### Backend
```bash
cd backend
pip install -r requirements.txt
python app.py
```

### Frontend
```bash
npm install
npm run dev
```

## Deployment

- Frontend: Netlify (auto-deploy from main branch)
- Backend: Render (gunicorn WSGI server)

## Base Integration

- Client-side Base RPC integration in [src/utils/baseIntegration.ts](src/utils/baseIntegration.ts)
- Farcaster manifest at [public/.well-known/farcaster.json](public/.well-known/farcaster.json)
- Base verification meta tag in [index.html](index.html)

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
# Deployment Sun Dec 14 19:08:01 GMT 2025
