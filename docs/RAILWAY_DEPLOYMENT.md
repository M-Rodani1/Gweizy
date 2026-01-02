# Railway Deployment Guide

## Why Railway?

Railway is **significantly better** than Render for this app:

✅ **True background workers** (your `worker.py` will actually run!)
✅ **Persistent storage** (database won't reset on deployments)
✅ **Better free tier** ($5 credit/month - enough for this app)
✅ **Faster deployments** (2-3 min vs 5-10 min)
✅ **Better DX** (cleaner UI, easier debugging)

## Cost Estimate

**Expected monthly cost:** $5-8

- Web service: ~$3-4/month
- Worker service: ~$2-3/month
- Database storage: Included (SQLite on persistent volume)

**Free credit:** $5/month on Hobby plan

## Setup Steps

### 1. Create Railway Account

1. Go to [railway.app](https://railway.app)
2. Sign in with GitHub
3. Verify email
4. Select **Hobby Plan** ($5/month credit)

### 2. Create New Project

1. Click **"New Project"**
2. Select **"Deploy from GitHub repo"**
3. Connect to your GitHub account
4. Select `M-Rodani1/basegasfeesML` repository
5. Railway will auto-detect Python and start deploying

### 3. Configure Services

Railway will create one service by default. You need to add a worker service.

#### Add Worker Service

1. In your project dashboard, click **"+ New"**
2. Select **"Service"**
3. Choose **"GitHub Repo"** → Same repository
4. Name it **"data-collector-worker"**

#### Configure Web Service

1. Click on the **web service**
2. Go to **"Variables"** tab
3. Add these environment variables:

```bash
USE_WORKER_PROCESS=true
BASE_RPC_URL=https://mainnet.base.org
BASE_CHAIN_ID=8453
PORT=5001
DEBUG=false
```

4. Go to **"Settings"** tab
5. Set **Start Command** (if not auto-detected):
   ```bash
   cd backend && gunicorn app:app --bind 0.0.0.0:$PORT --workers 2 --threads 2 --timeout 120
   ```

6. Under **"Networking"**, click **"Generate Domain"** to get public URL

#### Configure Worker Service

1. Click on the **worker service**
2. Go to **"Variables"** tab
3. Add these environment variables:

```bash
ENABLE_DATA_COLLECTION=true
COLLECTION_INTERVAL=60
BASE_RPC_URL=https://mainnet.base.org
BASE_CHAIN_ID=8453
```

4. Go to **"Settings"** tab
5. Set **Start Command**:
   ```bash
   cd backend && python3 worker.py
   ```

6. Set **Service Type** to **"Worker"** (not Web)

### 4. Add Persistent Volume (Critical!)

This ensures your database persists across deployments.

1. In the **web service**, go to **"Data"** tab
2. Click **"+ New Volume"**
3. Set:
   - **Mount Path:** `/app/backend`
   - **Size:** 1GB (free tier)
4. Click **"Add Volume"**

Do the same for the **worker service** so both share the database.

### 5. Deploy

1. Railway auto-deploys on push to `main`
2. Both services will build and start
3. Check logs in each service to verify:
   - **Web:** `✓ On-chain features collection started`
   - **Worker:** `DATA COLLECTION WORKER STARTING`

### 6. Update Frontend

Update your frontend to point to the new Railway URL:

1. Get your Railway URL from the web service (e.g., `https://basegasfeesml-production.up.railway.app`)
2. Update frontend API calls to use this URL

## Verification

### Check Web Service

```bash
curl https://your-railway-url.railway.app/api/health
```

Should return:
```json
{
  "status": "ok",
  "database_connected": true,
  "models_loaded": true
}
```

### Check Worker Status

```bash
curl https://your-railway-url.railway.app/api/cron/status
```

Should return:
```json
{
  "data_collection": {
    "collection_active": true,
    "gas_prices": 100+,
    "onchain_features": 100+
  }
}
```

### Check Model Performance

After 24 hours of data collection:

```bash
curl https://your-railway-url.railway.app/api/model-stats
```

Should show real metrics instead of zeros!

## Architecture on Railway

```
┌─────────────────────────────────────────────┐
│           Railway Project                   │
│                                             │
│  ┌───────────────┐    ┌─────────────────┐  │
│  │  Web Service  │    │ Worker Service  │  │
│  │   (Gunicorn)  │    │   (worker.py)   │  │
│  │               │    │                 │  │
│  │ • Flask API   │    │ • Gas collector │  │
│  │ • Predictions │    │ • OnChain       │  │
│  │ • Validation  │    │   collector     │  │
│  │               │    │ • Every 60s     │  │
│  │ Public URL ✓  │    │ Background ✓    │  │
│  └───────┬───────┘    └────────┬────────┘  │
│          │                     │            │
│          └──────────┬──────────┘            │
│                     │                       │
│            ┌────────▼────────┐              │
│            │ Persistent Vol  │              │
│            │  (1GB SQLite)   │              │
│            │                 │              │
│            │ ✓ Survives      │              │
│            │   deployments   │              │
│            └─────────────────┘              │
└─────────────────────────────────────────────┘
```

## Files Created

1. **`railway.toml`** - Railway configuration
2. **`railway.json`** - Project settings
3. **`nixpacks.toml`** - Build configuration
4. **`backend/Procfile`** - Process definitions (web + worker)
5. **`backend/worker.py`** - Background worker script

## Monitoring

### View Logs

**Web Service:**
1. Click on web service
2. Go to **"Deployments"** tab
3. Click latest deployment
4. View **"Build Logs"** and **"Deploy Logs"**

**Worker Service:**
1. Click on worker service
2. Same process
3. Look for: `✓ Collection #N: Block XXXXX`

### Check Metrics

Railway dashboard shows:
- CPU usage
- Memory usage
- Network egress
- Estimated monthly cost

## Troubleshooting

### Worker Not Starting

**Check:**
1. Start command is set to `cd backend && python3 worker.py`
2. Service type is **"Worker"** not "Web"
3. Environment variable `ENABLE_DATA_COLLECTION=true`

**Fix:** Redeploy worker service

### Database Empty After Deployment

**Issue:** Persistent volume not configured

**Fix:**
1. Add volume to web service: `/app/backend`
2. Add volume to worker service: `/app/backend`
3. Redeploy both services

### Worker Collecting but Web Shows Zero

**Issue:** Services using different databases

**Fix:** Ensure both services have volume mounted at **same path**

### Running Out of Free Credit

**Options:**
1. Reduce worker interval: `COLLECTION_INTERVAL=120` (2 min)
2. Use fewer Gunicorn workers: `--workers 1`
3. Upgrade to Developer plan ($5/mo base + usage)

## Migration from Render

### Update GitHub Secrets (if using)

Railway uses different environment variables. Update any CI/CD:

**Old (Render):**
```bash
RENDER_API_KEY=xxx
```

**New (Railway):**
```bash
RAILWAY_TOKEN=xxx
```

### Update Frontend URLs

**Search and replace in frontend:**

```bash
# Find all instances
grep -r "basegasfeesml.onrender.com" frontend/

# Replace with Railway URL
# e.g., basegasfeesml-production.up.railway.app
```

### DNS Configuration (Optional)

Railway supports custom domains:

1. Go to web service **"Settings"**
2. Scroll to **"Domains"**
3. Click **"+ Custom Domain"**
4. Follow DNS setup instructions

## Rollback Plan

If something goes wrong:

1. Railway keeps deployment history
2. Click any previous deployment
3. Click **"Redeploy"**
4. Instant rollback!

## Cost Optimization

**To stay within $5/month free credit:**

- Use 1 Gunicorn worker: `--workers 1`
- Increase collection interval: `COLLECTION_INTERVAL=120`
- Monitor usage in Railway dashboard
- Set spending limit in account settings

**Current config should cost ~$5-8/month** (slightly over free tier but worth it)

## Next Steps After Deployment

1. ✅ Verify both services are running
2. ✅ Check logs for data collection
3. ⏳ Wait 1 hour, verify database has data
4. ⏳ Wait 24 hours, check Model Performance widget
5. 🎉 Enjoy working predictions!

## Support

**Railway Community:**
- [Discord](https://discord.gg/railway)
- [Docs](https://docs.railway.app)
- [Community Forum](https://help.railway.app)

**Project Issues:**
- Check Railway logs first
- Verify environment variables
- Ensure volumes are mounted correctly

---

**Ready to deploy?** Follow the steps above and your app will be running on Railway in ~10 minutes!

**Estimated timeline:**
- Setup: 5 min
- First deployment: 3-5 min
- Data collection starts: Immediately
- Full metrics available: 24 hours
