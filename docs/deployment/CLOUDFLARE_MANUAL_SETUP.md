# Manual Cloudflare Setup (Web Dashboard)

Since wrangler CLI has installation issues, use this manual setup guide via the Cloudflare web dashboard.

---

## Step 1: Create Cloudflare Account

1. Go to https://dash.cloudflare.com/sign-up
2. Sign up with your email
3. Verify your email
4. You'll land on the dashboard

---

## Step 2: Deploy the Worker (API Cache)

### 2.1 Create Worker

1. In the Cloudflare dashboard, click **Workers & Pages** in the left sidebar
2. Click **Create Application**
3. Click **Create Worker**
4. Name it: `gas-fees-api`
5. Click **Deploy**

### 2.2 Edit Worker Code

1. After deployment, click **Edit Code**
2. Delete the default code
3. Copy and paste the code from `cloudflare-worker/worker.js`
4. Click **Save and Deploy**

### 2.3 Create KV Namespace

1. Go back to **Workers & Pages**
2. Click **KV** in the top tabs
3. Click **Create a namespace**
4. Name it: `GAS_CACHE`
5. Click **Add**
6. **Copy the Namespace ID** (you'll need this)

### 2.4 Bind KV to Worker

1. Go back to **Workers & Pages**
2. Click on your `gas-fees-api` worker
3. Click **Settings** tab
4. Scroll to **Variables and Secrets**
5. Under **KV Namespace Bindings**, click **Add binding**
   - Variable name: `GAS_CACHE`
   - KV namespace: Select `GAS_CACHE` from dropdown
6. Click **Save**

**Your Worker is now live!** Copy the worker URL (e.g., `https://gas-fees-api.YOUR_USERNAME.workers.dev`)

---

## Step 3: Deploy Frontend to Cloudflare Pages

### 3.1 Build the Frontend

```bash
cd /Users/rodan/Documents/gasFeesPrediction-main/frontend

# Create production env file
echo "VITE_API_URL=https://gas-fees-api.YOUR_USERNAME.workers.dev/api" > .env.production
# Replace YOUR_USERNAME with your actual Cloudflare username from the worker URL

# Build
npm run build
```

### 3.2 Deploy to Pages

**Option A: Direct Upload (Easiest)**

1. In Cloudflare dashboard, go to **Workers & Pages**
2. Click **Create Application**
3. Click **Pages** tab
4. Click **Upload assets**
5. Name your project: `base-gas-optimizer`
6. Click **Create project**
7. Drag and drop the entire `frontend/dist` folder
8. Click **Deploy site**

**Option B: Connect to GitHub (Auto-deploy on push)**

1. In Cloudflare dashboard, go to **Workers & Pages**
2. Click **Create Application**
3. Click **Pages** tab
4. Click **Connect to Git**
5. Authorize GitHub
6. Select repository: `basegasfeesML`
7. Configure build settings:
   - **Production branch**: `main`
   - **Build command**: `cd frontend && npm install && npm run build`
   - **Build output directory**: `frontend/dist`
   - **Environment variables**: Click **Add variable**
     - Variable name: `VITE_API_URL`
     - Value: `https://gas-fees-api.YOUR_USERNAME.workers.dev/api`
8. Click **Save and Deploy**

**Your site is now live!** Copy the Pages URL (e.g., `https://base-gas-optimizer.pages.dev`)

---

## Step 4: Test Your Deployment

1. Visit your Cloudflare Pages URL
2. Open browser DevTools (F12) → Network tab
3. Refresh the page
4. Look at the API requests:
   - First request: `X-Cache: MISS` (slow, ~500-2000ms)
   - Refresh again: `X-Cache: HIT` (fast, < 50ms!)

---

## Step 5: Update GitHub Repo (Optional)

Update your environment files with the real URLs:

```bash
# Update .env.production
echo "VITE_API_URL=https://gas-fees-api.YOUR_USERNAME.workers.dev/api" > frontend/.env.production

# Commit
git add frontend/.env.production
git commit -m "Update API URL to Cloudflare Worker endpoint"
git push
```

If you used GitHub integration, Cloudflare will auto-deploy on every push!

---

## Troubleshooting

### Worker returns 500 errors
1. Go to Workers & Pages → `gas-fees-api` → **Logs** tab
2. Check for errors
3. Make sure KV namespace is bound correctly

### Pages not loading
1. Check build output directory is `frontend/dist`
2. Verify `VITE_API_URL` environment variable is set
3. Check Pages → **Functions** tab for errors

### Still seeing slow loads
1. Hard refresh: Ctrl+Shift+R (Windows) or Cmd+Shift+R (Mac)
2. Check Network tab for `X-Cache: HIT` on API calls
3. First load will always be slow (cache miss), subsequent loads should be fast

---

## Performance Comparison

**Before (Netlify + Render):**
- First load: 2-60 seconds (cold start)
- Subsequent: 500-2000ms

**After (Cloudflare):**
- First request per endpoint: 500-2000ms (cache miss → fetches from Render)
- Next 10-300 seconds: < 50ms (cache hit!)
- After cache expires: Next user refreshes cache, others get cached data

**Result:** 95%+ of users get instant < 50ms loads!

---

## Next Steps

1. Monitor usage: **Workers & Pages** → **Analytics**
2. View logs: **Workers & Pages** → `gas-fees-api` → **Logs**
3. Custom domain: **Workers & Pages** → `base-gas-optimizer` → **Custom domains**

Your app is now globally distributed and blazing fast! 🚀
