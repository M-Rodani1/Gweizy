# Base Gas Optimizer - Project Structure

## 📁 Directory Organization

```
gasFeesPrediction-main/
├── frontend/              # React + TypeScript Frontend
│   ├── components/        # React components
│   ├── pages/            # Page components (Landing, Dashboard)
│   ├── src/              # Source utilities and API clients
│   ├── public/           # Static assets
│   ├── package.json      # Frontend dependencies
│   ├── vite.config.ts    # Vite build configuration
│   └── tsconfig.json     # TypeScript configuration
│
├── backend/              # Python Flask ML Backend
│   ├── api/              # API routes and endpoints
│   ├── models/           # ML models and training scripts
│   ├── data/             # Data storage and processing
│   ├── utils/            # Backend utilities
│   ├── requirements.txt  # Python dependencies
│   └── app.py            # Flask application entry point
│
├── scripts/              # Build and deployment scripts
├── .gitignore           # Git ignore rules
├── README.md            # Main project README
└── Documentation/       # Project documentation
    ├── DEPLOYMENT_GUIDE.md
    ├── HACKATHON_SUBMISSION.md
    ├── MODEL_ANALYSIS.md
    └── WEEK1_IMPROVEMENTS.md
```

## 🚀 Quick Start

### Frontend Development
```bash
cd frontend
npm install
npm run dev
```

### Backend Development
```bash
cd backend
pip install -r requirements.txt
python app.py
```

## 🔗 Related Documentation

- **[README.md](./README.md)** - Main project overview
- **[DEPLOYMENT_GUIDE.md](./DEPLOYMENT_GUIDE.md)** - Deployment instructions
- **[HACKATHON_SUBMISSION.md](./HACKATHON_SUBMISSION.md)** - Hackathon details
- **[MODEL_ANALYSIS.md](./MODEL_ANALYSIS.md)** - ML model documentation
- **[WEEK1_IMPROVEMENTS.md](./WEEK1_IMPROVEMENTS.md)** - Feature improvements log

## 📦 Tech Stack

**Frontend:**
- React 19.2.3 + TypeScript
- Vite 6.4.1
- Tailwind CSS
- Recharts for data visualization
- Deployed on Netlify

**Backend:**
- Python 3.x + Flask
- Machine Learning predictions
- Historical gas data analysis
- Deployed on Render

## 🌐 Live Demo

- **Frontend:** https://basegasfeesml.netlify.app/
- **Backend API:** https://basegasfeesml.onrender.com/api

## 📝 License

MIT License - See LICENSE file for details
