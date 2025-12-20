# Cloudflare Setup Guide - Base Gas Optimizer

This guide will help you deploy your app to Cloudflare Pages with Workers for instant global performance.

## Benefits
- ⚡ **< 50ms response times** worldwide
- 🌍 **Global edge caching** in 275+ cities
- 💰 **100,000 requests/day FREE**
- 🚀 **Zero cold starts**
- 📦 **Automatic scaling**

---

## Step 1: Create Cloudflare Account

1. Go to https://dash.cloudflare.com/sign-up
2. Sign up for a free account
3. Verify your email

---

## Step 2: Install Wrangler CLI

```bash
npm install -g wrangler
wrangler login
```

This will open a browser window to authorize Wrangler.

---

## Step 3: Create KV Namespace

```bash
# Create production KV namespace
wrangler kv:namespace create "GAS_CACHE"
```

**Output will look like:**
```
⛅️ wrangler 3.x.x
--------------------
🌀 Creating namespace with title "gas-fees-api-GAS_CACHE"
✨ Success!
Add the following to your wrangler.toml:
[[kv_namespaces]]
binding = "GAS_CACHE"
id = "abc123def456..."
```

**Copy the `id` value** and update `wrangler.toml`:

```toml
[[kv_namespaces]]
binding = "GAS_CACHE"
id = "abc123def456..."  # Replace with your actual ID
```

---

## Step 4: Deploy Worker

```bash
# Deploy the API caching worker
wrangler deploy
```

**Output:**
```
✨ Success! Uploaded 1 file
🌍 Published gas-fees-api
   https://gas-fees-api.YOUR_USERNAME.workers.dev
```

**Copy your worker URL** - you'll need it for the frontend.

---

## Step 5: Create Cloudflare Pages Project

```bash
# In the frontend directory
cd frontend

# Build the project
npm run build

# Deploy to Cloudflare Pages
npx wrangler pages deploy dist --project-name=base-gas-optimizer
```

**First time setup:**
```
✔ Enter the project name: base-gas-optimizer
✔ Enter the production branch: main
```

---

## Step 6: Update Frontend Environment

Update `frontend/.env.production`:

```env
VITE_API_URL=https://gas-fees-api.YOUR_USERNAME.workers.dev/api
```

Replace `YOUR_USERNAME` with your actual Cloudflare username.

---

## Step 7: Set Up GitHub Actions (Optional)

For automatic deployments on git push:

1. Get your Cloudflare API Token:
   - Go to https://dash.cloudflare.com/profile/api-tokens
   - Click "Create Token"
   - Use "Edit Cloudflare Workers" template
   - Copy the token

2. Get your Account ID:
   - Go to https://dash.cloudflare.com
   - Click on Workers & Pages
   - Copy your Account ID from the right sidebar

3. Add GitHub Secrets:
   - Go to your GitHub repo → Settings → Secrets and variables → Actions
   - Add `CLOUDFLARE_API_TOKEN` with your API token
   - Add `CLOUDFLARE_ACCOUNT_ID` with your account ID

4. Update `.github/workflows/deploy-cloudflare.yml`:
   ```yaml
   env:
     VITE_API_URL: https://gas-fees-api.YOUR_USERNAME.workers.dev/api
   ```

5. Push to GitHub - automatic deployment will start!

---

## Step 8: Test Your Deployment

1. Visit your Cloudflare Pages URL (shown after deployment)
2. Open browser DevTools → Network tab
3. Refresh the page
4. Look for API requests - they should have:
   - `X-Cache: HIT` (on subsequent requests)
   - Response times < 100ms globally

---

## How It Works

### Architecture

```
User Request
    ↓
Cloudflare Pages (HTML/JS/CSS) - served from edge
    ↓
Cloudflare Worker (/api/*) - edge compute
    ↓
Check KV Cache?
    ├─ HIT → Return cached data (< 10ms)
    └─ MISS → Fetch from Render backend
               ↓
        Cache in KV with TTL
               ↓
        Return data to user
```

### Cache Strategy

- **Current gas prices**: 10 seconds cache
- **Predictions**: 30 seconds cache
- **Accuracy metrics**: 1 minute cache
- **Historical data**: 5 minutes cache
- **Validation**: 2 minutes cache
- **Network state**: 15 seconds cache

This means:
- First user hits Render (slow, 500-2000ms)
- Next 10-300 seconds of users hit edge cache (fast, < 50ms)
- Cache expires → next user refreshes from Render
- Process repeats

---

## Cost Breakdown (Free Tier)

### Cloudflare Workers
- **100,000 requests/day FREE**
- **10 ms CPU time per request**
- **Beyond free**: $0.50 per million requests

### Cloudflare Pages
- **500 builds/month FREE**
- **Unlimited bandwidth**
- **Beyond free**: $0.25 per build

### KV Storage
- **1 GB storage FREE**
- **100,000 reads/day FREE**
- **1,000 writes/day FREE**

**Your usage (estimated):**
- 1,000 users/day × 10 requests/user = 10,000 requests/day
- **Cost: $0/month** (well within free tier)

---

## Monitoring

View real-time analytics:

```bash
# Worker analytics
wrangler tail

# Or via dashboard
https://dash.cloudflare.com → Workers & Pages → gas-fees-api → Metrics
```

---

## Troubleshooting

### Worker not caching
```bash
# Check KV namespace binding
wrangler kv:namespace list

# View cached keys
wrangler kv:key list --namespace-id=YOUR_KV_ID
```

### CORS errors
- Check worker.js has correct CORS headers
- Verify `Access-Control-Allow-Origin: *`

### Backend timing out
- Worker will return cached data even if Render is sleeping
- First request may be slow (cold start)
- Subsequent requests served from KV cache

---

## Next Steps

Once deployed, you'll have:
✅ Global CDN for your frontend
✅ Edge-cached API responses
✅ < 50ms load times worldwide
✅ Zero cold starts for users
✅ Automatic HTTPS

Your app will feel **instant** compared to Netlify + Render!
