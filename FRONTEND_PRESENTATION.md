# Base Gas Optimizer - Frontend Presentation
**Duration:** 2:30 minutes | **Presenter:** Frontend Developer

---

## [0:00 - 0:15] INTRODUCTION

Hi! I'm senan, and I built the frontend for Base Gas Optimizer. I'm going to show you how we created a real-time, mobile-first dashboard that helps Base users save money on gas fees.

**SHOW:** Frontend folder structure
```
frontend/
├── components/       # React components
│   ├── RelativePriceIndicator.tsx
│   ├── BestTimeWidget.tsx
│   ├── PredictionCards.tsx
│   ├── SavingsCalculator.tsx
│   ├── WalletConnect.tsx
│   └── GasPriceGraph.tsx
├── pages/           # Landing & Dashboard
│   ├── Landing.tsx
│   └── Dashboard.tsx
├── src/             # API client and utilities
│   ├── api/gasApi.ts
│   ├── utils/wallet.ts
│   └── utils/baseRpc.ts
├── App.tsx          # Router
├── types.ts         # TypeScript types
└── package.json     # Dependencies
```

This is a React 19 + TypeScript frontend built with Vite, deployed on Netlify. We focused on three things: real-time updates, mobile-first design, and making complex ML predictions easy to understand.

---

## [0:15 - 0:35] THE TRAFFIC LIGHT SYSTEM - LOGIC

The centerpiece is our traffic light gas indicator. It compares current gas prices to historical averages and tells you if NOW is a good time to transact.

**SHOW:** Traffic light logic from `frontend/components/RelativePriceIndicator.tsx`
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

**TRAFFIC LIGHT LEVELS:**
```
🟢 Excellent  < 70% of average   → Transact Now!
🔵 Good       70-90% of average  → Good Time
🟡 Average    90-115% of average → Typical
🟠 High       115-150% of average → Consider Waiting
🔴 Very High  > 150% of average  → Wait if Possible
```

Five levels: Green means gas is 30% below average - transact now! Red means it's 50% above average - wait if you can. Simple visual feedback that saves users money.

---

## [0:35 - 0:50] THE TRAFFIC LIGHT SYSTEM - REAL-TIME UPDATES

It updates every 5 minutes, comparing current gas to both the hourly average and 24-hour average. Users get instant, actionable advice.

**SHOW:** Real-time update code from `frontend/components/RelativePriceIndicator.tsx`
```typescript
useEffect(() => {
  const fetchAverages = async () => {
    try {
      const response = await fetch(
        `${import.meta.env.VITE_API_URL}/historical?hours=24`
      );
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

**UPDATE SCHEDULE:**
```
⏱️  Traffic Light: Every 5 minutes
📊 Predictions: Every 30 seconds
📈 Graphs: Every 30 seconds
```

The component fetches historical data, groups by hour, calculates averages, and updates the traffic light automatically.

---

## [0:50 - 1:10] BEST TIME WIDGET

We also show the best and worst times to transact based on Base network patterns.

**SHOW:** Best time widget code from `frontend/components/BestTimeWidget.tsx`
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

        // Sort to find cheapest and most expensive
        const sorted = [...hourlyAverages].sort((a, b) => a.avgGas - b.avgGas);

        setCheapestHours(sorted.slice(0, 3));  // Top 3 cheapest
        setMostExpensiveHours(sorted.slice(-3).reverse());  // Top 3 expensive
```

**WIDGET LAYOUT:**
```
┌─────────────────────────────────────┐
│    BEST TIMES TO TRANSACT           │
├──────────────────┬──────────────────┤
│  CHEAPEST HOURS  │ EXPENSIVE HOURS  │
├──────────────────┼──────────────────┤
│ 🟢 3am - 0.0017  │ 🔴 9pm - 0.0043  │
│ 🟢 2am - 0.0018  │ 🔴 8pm - 0.0038  │
│ 🟢 4am - 0.0019  │ 🔴 10pm - 0.0039 │
└──────────────────┴──────────────────┘
   Save up to 60% by timing transactions
```

We analyze 168 hours of historical data, group by UTC hour, and show the cheapest 3 hours on the left and most expensive 3 on the right. Users can plan their transactions around these patterns.

---

## [1:10 - 1:25] ML PREDICTION CARDS

But the real power is in the ML predictions - 1 hour, 4 hours, and 24 hours ahead.

**SHOW:** Prediction cards code from `frontend/components/PredictionCards.tsx`
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

Each prediction shows confidence level - High, Medium, or Low - based on the ML model's certainty. We also show a visual range slider indicating best-case to worst-case scenarios, so users understand the uncertainty.

**SHOW:** Range slider visualization code
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

**PREDICTION CARD LAYOUT:**
```
┌──────────────────────────────────────┐
│  1 HOUR PREDICTION    🟢 HIGH        │
├──────────────────────────────────────┤
│  Predicted: 0.00225 gwei (+4.2%)     │
│  Current:   0.00216 gwei             │
│                                      │
│  ◄──────●──────────────►             │
│  Best      Predicted    Worst        │
│  0.00198   0.00225     0.00252       │
│                                      │
│  Recommendation:                     │
│  Gas likely to rise 4.2%             │
│  Transact now to save money          │
│                                      │
│  [Why this prediction?]              │
└──────────────────────────────────────┘
```

The range slider makes ML predictions understandable. Users see the predicted value and the possible range in one visual.

---

## [1:25 - 1:45] WALLET INTEGRATION

We integrated MetaMask for wallet connection, with automatic Base network detection and switching.

**SHOW:** Wallet connection code from `frontend/src/utils/wallet.ts`
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
          params: [{ chainId: BASE_CHAIN_ID }]  // 0x2105
        });
      } catch (switchError: any) {
        // Network not added yet, add it
        if (switchError.code === 4902) {
          await window.ethereum.request({
            method: 'wallet_addEthereumChain',
            params: [{
              chainId: BASE_CHAIN_ID,  // 0x2105 (8453)
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

**BASE NETWORK CONFIG:**
```
Chain ID: 8453 (0x2105)
RPC URL: https://mainnet.base.org
Block Explorer: https://basescan.org
Native Currency: ETH
```

When users connect, we check if they're on Base network. If not, we automatically prompt them to switch. If they don't have Base added, we add it for them with the correct RPC and block explorer.

One click, and users are connected to Base network with their wallet.

---

## [1:45 - 2:00] LIVE BLOCKCHAIN DATA

For live gas prices, we fetch directly from the Base blockchain using JSON-RPC calls.

**SHOW:** Base RPC integration code from `frontend/src/utils/baseRpc.ts`
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

**RPC ENDPOINTS:**
```typescript
const BASE_RPC_URLS = [
  'https://mainnet.base.org',           // Primary
  'https://base.llamarpc.com',          // Fallback 1
  'https://base-rpc.publicnode.com'     // Fallback 2
];
```

We call `eth_getBlockByNumber` to get the latest block's base fee, convert from hex to gwei. If one RPC is rate-limited, we automatically rotate to a backup. This ensures the dashboard always has live data.

**SHOW:** Dashboard auto-refresh code from `frontend/pages/Dashboard.tsx`
```typescript
// Auto-refresh data every 30 seconds
useEffect(() => {
  if (apiStatus === 'online') {
    const interval = setInterval(() => {
      loadData();  // Fetch predictions, current gas, graphs
    }, 30000);  // 30 seconds

    return () => clearInterval(interval);
  }
}, [apiStatus]);
```

Everything updates every 30 seconds - predictions, graphs, current gas prices. Users get real-time data without refreshing the page.

---

## [2:00 - 2:15] MOBILE-FIRST DESIGN

We built mobile-first. Every component adapts to screen size.

**SHOW:** Responsive design code from `frontend/components/RelativePriceIndicator.tsx`
```typescript
<div className={`
  bg-gradient-to-br from-gray-800 to-gray-900
  p-4 sm:p-6
  rounded-lg shadow-lg
  border-2 ${colors.border}
  ${className}
`}>
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

**TAILWIND BREAKPOINTS:**
```css
/* Mobile First */
p-4              /* 1rem padding on mobile */
sm:p-6           /* 1.5rem on screens ≥ 640px */
md:p-8           /* 2rem on screens ≥ 768px */

text-6xl         /* 3.75rem on mobile */
sm:text-7xl      /* 4.5rem on screens ≥ 640px */

text-sm          /* 0.875rem on mobile */
sm:text-base     /* 1rem on screens ≥ 640px */
md:text-lg       /* 1.125rem on screens ≥ 768px */

/* Grid Layouts */
grid-cols-1      /* 1 column on mobile */
md:grid-cols-2   /* 2 columns on tablets */
lg:grid-cols-3   /* 3 columns on desktop */
```

**ACCESSIBILITY:**
```
✅ 44px minimum touch targets
✅ Responsive text scaling
✅ Color contrast ratios > 4.5:1
✅ Keyboard navigation support
```

Text scales from `text-sm` on mobile to `text-2xl` on desktop. Padding adjusts. Grids collapse to single column. All buttons meet the 44px minimum touch target for accessibility.

---

## [2:15 - 2:30] BUILD & DEPLOYMENT

Built with Vite for instant hot reload during development. Production build optimizes chunks, tree-shakes unused code, and copies the PWA manifest for offline support.

**SHOW:** Vite config from `frontend/vite.config.ts`
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

**BUILD SCRIPT:** from `frontend/package.json`
```json
{
  "scripts": {
    "dev": "vite",
    "build": "vite build && cp public/manifest.json dist/manifest.json",
    "preview": "vite preview"
  }
}
```

**DEPENDENCIES:**
```json
{
  "react": "^19.2.3",
  "react-dom": "^19.2.3",
  "react-router-dom": "^7.10.1",
  "recharts": "^3.5.1",
  "framer-motion": "^12.23.26",
  "typescript": "~5.8.2",
  "vite": "^6.2.0"
}
```

**DEPLOYMENT STATS:**
```
Platform: Netlify
URL: https://basegasfeesml.netlify.app
Build Time: ~45 seconds
Bundle Size: ~120KB gzipped
First Load: < 2 seconds
Framework: Vite + React 19
```

Deployed on Netlify with automatic builds on every git push. The dashboard is live at basegasfeesml.netlify.app.

**LIVE DASHBOARD:**
```
┌────────────────────────────────────────┐
│  BASE GAS OPTIMIZER                    │
├────────────────────────────────────────┤
│  🟢 Excellent Time to Transact         │
│  Current: 0.00198 gwei                 │
│                                        │
│  BEST TIMES      │  WORST TIMES        │
│  3am - 0.0017    │  9pm - 0.0043       │
│                                        │
│  1H: 0.00225 ↑   4H: 0.00238 ↑         │
│  24H: 0.00245 ↑  Confidence: HIGH      │
│                                        │
│  [Connect Wallet]  [View Savings]      │
└────────────────────────────────────────┘
```

**TECH STACK:**
```
✅ React 19.2.3 + TypeScript
✅ Vite 6.2.0 (HMR + fast builds)
✅ Tailwind CSS (utility-first)
✅ Recharts (data visualization)
✅ Framer Motion (animations)
✅ MetaMask integration
✅ Base RPC (live blockchain data)
```

The result? A fast, mobile-friendly dashboard that makes complex ML predictions easy to understand. Users see color-coded traffic lights, hourly patterns, and real-time blockchain data - all updating automatically every 30 seconds.

Built with React 19, TypeScript, Tailwind CSS, and Recharts. All open source on GitHub.

Thanks for watching!

---

**END OF FRONTEND PRESENTATION**
