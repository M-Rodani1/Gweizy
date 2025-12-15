# Base Gas Optimizer - Frontend Video Script
**Duration:** 2:30 minutes
**Presenter:** [Frontend Developer Name]

---

## [0:00 - 0:15] INTRODUCTION

**VISUAL:** Show GitHub repo frontend folder structure or VS Code with frontend/ folder open

**SCRIPT:**
> Hi! I'm [Name], and I built the frontend for Base Gas Optimizer. I'm going to show you how we created a real-time, mobile-first dashboard that helps Base users save money on gas fees.

**SHOW:** `frontend/` folder structure with subdirectories visible:
- `frontend/components/` - React components
- `frontend/pages/` - Landing & Dashboard
- `frontend/src/` - API client and utilities
- `frontend/public/` - Static assets

**SCRIPT:**
> This is a React 19 + TypeScript frontend built with Vite, deployed on Netlify. We focused on three things: real-time updates, mobile-first design, and making complex ML predictions easy to understand.

---

## [0:15 - 0:50] THE TRAFFIC LIGHT SYSTEM

**VISUAL:** Show `frontend/components/RelativePriceIndicator.tsx` in VS Code

**CODE LOCATION:** `frontend/components/RelativePriceIndicator.tsx` lines 67-111

**SCRIPT:**
> The centerpiece is our traffic light gas indicator. It compares current gas prices to historical averages and tells you if NOW is a good time to transact.

**SHOW CODE:**
```typescript
const getPriceLevel = (current: number, avg: number): PriceLevel => {
  const ratio = current / avg;

  if (ratio < 0.7) {
    return {
      level: 'Excellent',
      color: 'green',
      emoji: '🟢',
      description: 'Gas is significantly below average',
      action: 'Perfect time to transact!'
    };
  } else if (ratio < 0.9) {
    return {
      level: 'Good',
      color: 'cyan',
      emoji: '🔵',
      description: 'Gas is below average',
      action: 'Good time to transact'
    };
  } else if (ratio < 1.15) {
    return {
      level: 'Average',
      color: 'yellow',
      emoji: '🟡',
      description: 'Gas is near average',
      action: 'Typical price for this time'
    };
  } else if (ratio < 1.5) {
    return {
      level: 'High',
      color: 'orange',
      emoji: '🟠',
      description: 'Gas is above average',
      action: 'Consider waiting if not urgent'
    };
  } else {
    return {
      level: 'Very High',
      color: 'red',
      emoji: '🔴',
      description: 'Gas is significantly above average',
      action: 'Wait unless transaction is urgent'
    };
  }
};
```

**VISUAL:** Show live dashboard with different color states

**SCRIPT:**
> Five levels: Green means gas is 30% below average - transact now! Red means it's 50% above average - wait if you can. Simple visual feedback that saves users money.

**VISUAL:** Show the real-time update code

**CODE LOCATION:** `frontend/components/RelativePriceIndicator.tsx` lines 24-65

**SHOW CODE:**
```typescript
useEffect(() => {
  const fetchAverages = async () => {
    try {
      const response = await fetch(`${import.meta.env.VITE_API_URL}/historical?hours=24`);
      const data = await response.json();

      if (data.historical && data.historical.length > 0) {
        const currentHour = new Date().getUTCHours();

        // Calculate current hour average
        const hourData = data.historical.filter((point: any) => {
          const timestamp = new Date(point.timestamp);
          return timestamp.getUTCHours() === currentHour;
        });

        if (hourData.length > 0) {
          const hourAvg = hourData.reduce((sum: number, p: any) =>
            sum + (p.gwei || p.current_gas || 0), 0) / hourData.length;
          setHourlyAverage(hourAvg);
        }

        // Calculate 24h average
        const dayAvg = data.historical.reduce((sum: number, p: any) =>
          sum + (p.gwei || p.current_gas || 0), 0) / data.historical.length;
        setDayAverage(dayAvg);
      }
    } catch (error) {
      console.error('Failed to fetch averages:', error);
      // Fallback to reasonable defaults
      setHourlyAverage(0.0024);
      setDayAverage(0.0024);
    } finally {
      setLoading(false);
    }
  };

  fetchAverages();

  // Refresh every 5 minutes
  const interval = setInterval(fetchAverages, 5 * 60 * 1000);
  return () => clearInterval(interval);
}, []);
```

**SCRIPT:**
> It updates every 5 minutes, comparing current gas to both the hourly average and 24-hour average. Users get instant, actionable advice.

---

## [0:50 - 1:25] BEST TIME WIDGET & PREDICTIONS

**VISUAL:** Show `frontend/components/BestTimeWidget.tsx` code

**CODE LOCATION:** `frontend/components/BestTimeWidget.tsx` lines 26-41

**SCRIPT:**
> We also show the best and worst times to transact based on Base network patterns.

**SHOW CODE:**
```typescript
useEffect(() => {
  const fetchData = async () => {
    try {
      const response = await fetch(`${API_URL}/historical?hours=168`);
      const data = await response.json();

      if (data.historical && data.historical.length > 100) {
        // Group by UTC hour
        const hourlyData: { [hour: number]: number[] } = {};

        data.historical.forEach((point: any) => {
          const timestamp = new Date(point.timestamp);
          const hour = timestamp.getUTCHours();

          if (!hourlyData[hour]) {
            hourlyData[hour] = [];
          }
          hourlyData[hour].push(point.gwei || point.current_gas || 0);
        });

        // Calculate averages for each hour
        const hourlyAverages: { hour: number; avgGas: number }[] = [];
        for (let hour = 0; hour < 24; hour++) {
          if (hourlyData[hour] && hourlyData[hour].length > 0) {
            const avg = hourlyData[hour].reduce((a, b) => a + b, 0) / hourlyData[hour].length;
            hourlyAverages.push({ hour, avgGas: avg });
          }
        }
```

**VISUAL:** Show the Best Time Widget UI on dashboard

**SCRIPT:**
> We analyze 168 hours of historical data, group by UTC hour, and show the cheapest 3 hours on the left and most expensive 3 on the right. Users can plan their transactions around these patterns.

**VISUAL:** Show `frontend/components/PredictionCards.tsx`

**CODE LOCATION:** `frontend/components/PredictionCards.tsx` lines 80-103

**SCRIPT:**
> But the real power is in the ML predictions - 1 hour, 4 hours, and 24 hours ahead.

**SHOW CODE:**
```typescript
// Determine recommendation based on prediction vs current
if (predicted < current * 0.9) {
  // Gas dropping significantly
  color = 'red';
  if (confidenceLevel === 'high') {
    recommendation = `Gas expected to drop significantly. Wait ${horizon} for potential ${Math.abs(percentChange).toFixed(1)}% savings.`;
  } else if (confidenceLevel === 'medium') {
    recommendation = `Gas likely to drop, but prediction has moderate uncertainty. Consider waiting if transaction isn't urgent.`;
  } else {
    recommendation = `Prediction suggests gas may drop, but confidence is low. Proceed with caution.`;
  }
} else if (predicted > current * 1.1) {
  // Gas rising significantly
  color = 'green';
  if (confidenceLevel === 'high') {
    recommendation = `Gas expected to rise by ${percentChange.toFixed(1)}%. Transact now to save money.`;
  } else if (confidenceLevel === 'medium') {
    recommendation = `Gas likely to rise, but with moderate confidence. Consider transacting soon.`;
  } else {
    recommendation = `Prediction suggests gas may rise, but confidence is low. Monitor closely.`;
  }
} else {
  // Gas relatively stable
  color = 'yellow';
  recommendation = `Gas expected to remain stable. Transaction timing is flexible.`;
}
```

**VISUAL:** Show prediction cards with confidence levels

**SCRIPT:**
> Each prediction shows confidence level - High, Medium, or Low - based on the ML model's certainty. We also show a visual range slider indicating best-case to worst-case scenarios, so users understand the uncertainty.

**CODE LOCATION:** `frontend/components/PredictionCards.tsx` lines 227-270

**SHOW CODE:**
```typescript
{/* Visual Range Slider */}
<div className="mt-3">
  <div className="flex justify-between text-xs text-gray-400 mb-1">
    <span>Best case</span>
    <span>Worst case</span>
  </div>
  <div className="relative h-2 bg-gray-700 rounded-full overflow-hidden">
    {/* Range bar */}
    <div
      className="absolute h-full bg-gradient-to-r from-green-500/30 to-red-500/30"
      style={{
        left: `${((lowerBound - minGas) / (maxGas - minGas)) * 100}%`,
        width: `${((upperBound - lowerBound) / (maxGas - minGas)) * 100}%`
      }}
    />
    {/* Predicted point */}
    <div
      className="absolute top-1/2 -translate-y-1/2 w-3 h-3 bg-cyan-400 rounded-full border-2 border-gray-900"
      style={{
        left: `${((predicted - minGas) / (maxGas - minGas)) * 100}%`,
        marginLeft: '-6px'
      }}
    />
  </div>
  <div className="flex justify-between text-xs text-gray-500 mt-1">
    <span>{(lowerBound * 1000).toFixed(3)} gwei</span>
    <span>{(upperBound * 1000).toFixed(3)} gwei</span>
  </div>
</div>
```

**SCRIPT:**
> The range slider makes ML predictions understandable. Users see the predicted value and the possible range in one visual.

---

## [1:25 - 2:00] WALLET INTEGRATION & REAL-TIME UPDATES

**VISUAL:** Show `frontend/src/utils/wallet.ts`

**CODE LOCATION:** `frontend/src/utils/wallet.ts` lines 64-109

**SCRIPT:**
> We integrated MetaMask for wallet connection, with automatic Base network detection and switching.

**SHOW CODE:**
```typescript
export async function connectWallet(): Promise<string> {
  if (!window.ethereum) {
    throw new Error('MetaMask not installed');
  }

  try {
    // Request account access
    const accounts = await window.ethereum.request({
      method: 'eth_requestAccounts'
    }) as string[];

    if (!accounts || accounts.length === 0) {
      throw new Error('No accounts found');
    }

    // Check if we're on Base network
    const currentChainId = await getCurrentChainId();

    if (currentChainId !== BASE_CHAIN_ID_DECIMAL) {
      // Try to switch to Base
      try {
        await window.ethereum.request({
          method: 'wallet_switchEthereumChain',
          params: [{ chainId: BASE_CHAIN_ID }]
        });
      } catch (switchError: any) {
        // Network not added yet, add it
        if (switchError.code === 4902) {
          await window.ethereum.request({
            method: 'wallet_addEthereumChain',
            params: [{
              chainId: BASE_CHAIN_ID,
              chainName: 'Base',
              nativeCurrency: {
                name: 'Ethereum',
                symbol: 'ETH',
                decimals: 18
              },
              rpcUrls: ['https://mainnet.base.org'],
              blockExplorerUrls: ['https://basescan.org']
            }]
          });
        } else {
          throw switchError;
        }
      }
    }

    return accounts[0];
  } catch (error) {
    console.error('Wallet connection error:', error);
    throw error;
  }
}
```

**VISUAL:** Show wallet connection button and MetaMask popup

**SCRIPT:**
> When users connect, we check if they're on Base network. If not, we automatically prompt them to switch. If they don't have Base added, we add it for them with the correct RPC and block explorer.

**VISUAL:** Show `frontend/src/utils/baseRpc.ts`

**CODE LOCATION:** `frontend/src/utils/baseRpc.ts` lines 39-71

**SCRIPT:**
> For live gas prices, we fetch directly from the Base blockchain using JSON-RPC calls.

**SHOW CODE:**
```typescript
export async function fetchLiveBaseGas(): Promise<number> {
  const rpcUrl = BASE_RPC_URLS[currentRpcIndex];

  try {
    const response = await fetch(rpcUrl, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        jsonrpc: '2.0',
        method: 'eth_getBlockByNumber',
        params: ['latest', false],
        id: 1
      })
    });

    const data = await response.json();

    if (data.error) {
      throw new Error(data.error.message || 'RPC error');
    }

    if (!data.result || !data.result.baseFeePerGas) {
      throw new Error('Invalid response from RPC');
    }

    // Convert hex to gwei
    const baseFeeGwei = hexToGwei(data.result.baseFeePerGas);

    return baseFeeGwei;

  } catch (error) {
    console.error(`Error fetching from ${rpcUrl}:`, error);

    // Rotate to next RPC URL
    currentRpcIndex = (currentRpcIndex + 1) % BASE_RPC_URLS.length;

    // Retry with next RPC
    if (retryCount < 2) {
      return fetchLiveBaseGas();
    }

    throw error;
  }
}
```

**SCRIPT:**
> We call `eth_getBlockByNumber` to get the latest block's base fee, convert from hex to gwei. If one RPC is rate-limited, we automatically rotate to a backup. This ensures the dashboard always has live data.

**VISUAL:** Show the 30-second refresh in Dashboard

**CODE LOCATION:** `frontend/pages/Dashboard.tsx` lines 82-91

**SHOW CODE:**
```typescript
// Auto-refresh data every 30 seconds
useEffect(() => {
  if (apiStatus === 'online') {
    const interval = setInterval(() => {
      loadData();
    }, 30000);

    return () => clearInterval(interval);
  }
}, [apiStatus]);
```

**SCRIPT:**
> Everything updates every 30 seconds - predictions, graphs, current gas prices. Users get real-time data without refreshing the page.

---

## [2:00 - 2:30] MOBILE-FIRST & DEPLOYMENT

**VISUAL:** Show responsive design code in components

**CODE LOCATION:** `frontend/components/RelativePriceIndicator.tsx` lines 166-178

**SCRIPT:**
> We built mobile-first. Every component adapts to screen size.

**SHOW CODE:**
```typescript
<div className={`bg-gradient-to-br from-gray-800 to-gray-900 p-4 sm:p-6 rounded-lg shadow-lg border-2 ${colors.border} ${className}`}>
  {/* Status Indicator */}
  <div className="text-center mb-4">
    <div className="text-6xl sm:text-7xl mb-3 animate-pulse">
      {status.emoji}
    </div>
    <div className={`text-xl sm:text-2xl font-bold ${colors.text} mb-2`}>
      {status.level} Time to Transact
    </div>
    <div className="text-sm sm:text-base text-gray-400">
      {status.description}
    </div>
  </div>
```

**VISUAL:** Show browser resize demo

**SCRIPT:**
> Text scales from `text-sm` on mobile to `text-2xl` on desktop. Padding adjusts. Grids collapse to single column. All buttons meet the 44px minimum touch target for accessibility.

**VISUAL:** Show `frontend/vite.config.ts` and `frontend/package.json`

**CODE LOCATION:** `frontend/vite.config.ts` lines 1-23

**SHOW CODE:**
```typescript
export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, '.', '');
  return {
    server: {
      port: 3000,
      host: '0.0.0.0',
    },
    plugins: [react()],
    define: {
      'process.env.API_KEY': JSON.stringify(env.GEMINI_API_KEY),
      'process.env.GEMINI_API_KEY': JSON.stringify(env.GEMINI_API_KEY)
    },
    resolve: {
      alias: {
        '@': path.resolve(__dirname, '.'),
      }
    }
  };
});
```

**CODE LOCATION:** `frontend/package.json` lines 6-9

**SHOW CODE:**
```json
{
  "scripts": {
    "dev": "vite",
    "build": "vite build && cp public/manifest.json dist/manifest.json 2>/dev/null || cp manifest.json dist/manifest.json 2>/dev/null || true && cp public/_redirects dist/_redirects 2>/dev/null || true",
    "preview": "vite preview"
  }
}
```

**SCRIPT:**
> Built with Vite for instant hot reload during development. Production build optimizes chunks, tree-shakes unused code, and copies the PWA manifest for offline support.

**VISUAL:** Show Netlify deployment dashboard or vercel.json

**CODE LOCATION:** `frontend/vercel.json` lines 1-7

**SHOW CODE:**
```json
{
  "buildCommand": "npm run build",
  "outputDirectory": "dist",
  "devCommand": "npm run dev",
  "installCommand": "npm install",
  "framework": "vite"
}
```

**SCRIPT:**
> Deployed on Netlify with automatic builds on every git push. The dashboard is live at basegasfeesml.netlify.app.

**VISUAL:** Show live dashboard one final time

**SCRIPT:**
> The result? A fast, mobile-friendly dashboard that makes complex ML predictions easy to understand. Users see color-coded traffic lights, hourly patterns, and real-time blockchain data - all updating automatically every 30 seconds.

**VISUAL:** Show final stats or feature highlights

**SCRIPT:**
> Built with React 19, TypeScript, Tailwind CSS, and Recharts. All open source on GitHub.

> Thanks for watching!

---

## PRODUCTION CHECKLIST

### Screen Recordings to Include:
- [ ] VS Code: `frontend/` folder structure overview
- [ ] `frontend/components/RelativePriceIndicator.tsx` - Traffic light logic
- [ ] Live dashboard showing color state changes
- [ ] `frontend/components/BestTimeWidget.tsx` - Hourly patterns
- [ ] `frontend/components/PredictionCards.tsx` - ML prediction cards with confidence
- [ ] Prediction range slider visualization
- [ ] `frontend/src/utils/wallet.ts` - MetaMask connection flow
- [ ] MetaMask popup and network switching
- [ ] `frontend/src/utils/baseRpc.ts` - RPC integration
- [ ] `frontend/pages/Dashboard.tsx` - Auto-refresh code
- [ ] Browser resize demo (desktop → tablet → mobile)
- [ ] `frontend/vite.config.ts` - Build configuration
- [ ] `frontend/package.json` - Dependencies
- [ ] Netlify deployment dashboard
- [ ] Live site at basegasfeesml.netlify.app

### Code Snippets to Highlight:
- [ ] Traffic light price level logic (`RelativePriceIndicator.tsx` lines 67-111)
- [ ] Real-time update intervals (`RelativePriceIndicator.tsx` lines 24-65)
- [ ] Best time hourly grouping (`BestTimeWidget.tsx` lines 26-41)
- [ ] Prediction recommendation logic (`PredictionCards.tsx` lines 80-103)
- [ ] Visual range slider (`PredictionCards.tsx` lines 227-270)
- [ ] MetaMask connection with Base network auto-add (`wallet.ts` lines 64-109)
- [ ] Base RPC JSON-RPC call (`baseRpc.ts` lines 39-71)
- [ ] Dashboard 30-second refresh (`Dashboard.tsx` lines 82-91)
- [ ] Responsive Tailwind classes (`RelativePriceIndicator.tsx` lines 166-178)
- [ ] Vite configuration (`vite.config.ts` lines 1-23)

### Key Features to Demonstrate:
- [ ] **Traffic Light System**: 5 color levels (Green/Cyan/Yellow/Orange/Red)
- [ ] **Best Time Widget**: Cheapest vs. most expensive hours
- [ ] **ML Predictions**: 1h/4h/24h with confidence levels
- [ ] **Range Slider**: Visual uncertainty representation
- [ ] **Wallet Integration**: MetaMask connection with Base network auto-add
- [ ] **Real-time Updates**: 30-second refresh, 5-minute refresh for averages
- [ ] **Live RPC Data**: Direct blockchain integration
- [ ] **Responsive Design**: Mobile-first with Tailwind breakpoints
- [ ] **Fast Build**: Vite with instant HMR
- [ ] **Deployment**: Netlify with automatic builds

### Live Demo URLs:
- **Frontend:** https://basegasfeesml.netlify.app
- **Dashboard:** https://basegasfeesml.netlify.app/app
- **API (backend):** https://basegasfeesml.onrender.com/api

---

## FILE REFERENCE QUICK MAP

| Timestamp | Topic | Primary Files | Lines |
|-----------|-------|---------------|-------|
| 0:00-0:15 | Intro | `frontend/` folder | - |
| 0:15-0:35 | Traffic Light Logic | `frontend/components/RelativePriceIndicator.tsx` | 67-111 |
| 0:35-0:50 | Real-time Updates | `frontend/components/RelativePriceIndicator.tsx` | 24-65 |
| 0:50-1:10 | Best Time Widget | `frontend/components/BestTimeWidget.tsx` | 26-41 |
| 1:10-1:25 | Prediction Cards | `frontend/components/PredictionCards.tsx` | 80-103, 227-270 |
| 1:25-1:45 | Wallet Integration | `frontend/src/utils/wallet.ts` | 64-109 |
| 1:45-2:00 | Base RPC Integration | `frontend/src/utils/baseRpc.ts` | 39-71 |
| 2:00-2:15 | Auto-refresh | `frontend/pages/Dashboard.tsx` | 82-91 |
| 2:15-2:30 | Mobile & Deployment | `frontend/components/RelativePriceIndicator.tsx`<br>`frontend/vite.config.ts`<br>`frontend/vercel.json` | 166-178<br>1-23<br>1-7 |

---

## TECHNICAL SPECIFICATIONS

**Frontend Stack:**
- React 19.2.3 (Concurrent features)
- TypeScript ~5.8.2
- Vite 6.2.0 (Build tool)
- Tailwind CSS (Styling)
- Recharts 3.5.1 (Data visualization)
- Framer Motion 12.23.26 (Animations)
- React Router DOM 7.10.1 (Client-side routing)

**Key Features:**
- **Traffic Light System:** 5 levels (Excellent/Good/Average/High/Very High)
- **Real-time Updates:** 30-second predictions, 5-minute averages
- **ML Predictions:** 1h (59.83% accuracy), 4h, 24h horizons
- **Wallet Integration:** MetaMask + Coinbase Wallet support
- **Base Network:** Chain ID 8453 (0x2105)
- **RPC Endpoints:** 3 fallback RPC URLs with auto-rotation
- **Responsive:** Mobile-first with sm/md/lg breakpoints
- **PWA Support:** manifest.json for offline capability

**Performance:**
- First Load: < 2s (Vite optimized chunks)
- API Response: < 200ms (backend caching)
- Update Interval: 30s (predictions), 5min (averages)
- RPC Calls: Auto-retry with fallback URLs

**Deployment:**
- Platform: Netlify
- URL: https://basegasfeesml.netlify.app
- Framework: Vite
- Build: Automatic on git push
- SPA Routing: Client-side with react-router-dom

---

**END OF SCRIPT**
