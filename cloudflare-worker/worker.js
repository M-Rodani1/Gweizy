/**
 * Cloudflare Worker - Gas Fees API Proxy with Intelligent Caching
 * Provides instant global access with optimized cache strategy and keep-alive
 */

const BACKEND_API = 'https://basegasfeesml.onrender.com/api';

// Optimized cache durations - longer times to reduce Render hits
const CACHE_DURATIONS = {
  current: 30,        // 30 seconds - current gas (balance freshness vs load)
  predictions: 300,   // 5 minutes - predictions don't change often
  accuracy: 600,      // 10 minutes - accuracy metrics are stable
  historical: 900,    // 15 minutes - historical data rarely changes
  validation: 300,    // 5 minutes - validation metrics
  network: 60,        // 1 minute - network state changes moderately
  retraining: 300,    // 5 minutes - model status
  leaderboard: 180,   // 3 minutes - leaderboard updates
  stats: 300,         // 5 minutes - global stats
  health: 60,         // 1 minute - health checks
  default: 120        // 2 minutes - fallback for unknown endpoints
};

// Keep-alive configuration
const KEEP_ALIVE_INTERVAL = 600000; // 10 minutes (in milliseconds)
let lastKeepAlive = 0;

export default {
  async fetch(request, env, ctx) {
    const url = new URL(request.url);

    // CORS headers
    const corsHeaders = {
      'Access-Control-Allow-Origin': '*',
      'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
      'Access-Control-Allow-Headers': 'Content-Type',
    };

    // Handle CORS preflight
    if (request.method === 'OPTIONS') {
      return new Response(null, { headers: corsHeaders });
    }

    // Extract endpoint path
    const path = url.pathname.replace('/api/', '').replace(/^\/+/, '');

    try {
      // Trigger keep-alive in background (non-blocking)
      ctx.waitUntil(keepRenderAlive(env));

      // Check KV cache first
      const cacheKey = `${path}${url.search}`;
      const cached = await env.GAS_CACHE.get(cacheKey, 'json');

      if (cached) {
        return new Response(JSON.stringify(cached), {
          headers: {
            ...corsHeaders,
            'Content-Type': 'application/json',
            'X-Cache': 'HIT',
            'X-Cache-Age': getCacheAge(cached),
            'Cache-Control': 'public, max-age=30'
          }
        });
      }

      // Fetch from backend with timeout
      const backendUrl = `${BACKEND_API}/${path}${url.search}`;
      const controller = new AbortController();
      const timeout = setTimeout(() => controller.abort(), 30000); // 30 second timeout

      const backendResponse = await fetch(backendUrl, {
        headers: {
          'Content-Type': 'application/json',
          'User-Agent': 'Cloudflare-Worker-Cache/1.0'
        },
        signal: controller.signal
      });

      clearTimeout(timeout);

      if (!backendResponse.ok) {
        // Return cached data if available (stale-while-revalidate pattern)
        const staleCache = await env.GAS_CACHE.get(cacheKey + ':stale', 'json');
        if (staleCache) {
          return new Response(JSON.stringify(staleCache), {
            status: 200,
            headers: {
              ...corsHeaders,
              'Content-Type': 'application/json',
              'X-Cache': 'STALE',
              'X-Warning': 'Serving stale data due to backend error'
            }
          });
        }

        return new Response(JSON.stringify({
          error: 'Backend temporarily unavailable',
          status: backendResponse.status
        }), {
          status: backendResponse.status,
          headers: { ...corsHeaders, 'Content-Type': 'application/json' }
        });
      }

      const data = await backendResponse.json();

      // Add metadata
      data._cached_at = Date.now();
      data._cache_key = cacheKey;

      // Determine cache duration based on endpoint
      const cacheDuration = getCacheDuration(path);

      // Store in KV with expiration
      ctx.waitUntil(
        Promise.all([
          // Fresh cache
          env.GAS_CACHE.put(cacheKey, JSON.stringify(data), {
            expirationTtl: cacheDuration
          }),
          // Stale cache (backup for errors, 10x longer TTL)
          env.GAS_CACHE.put(cacheKey + ':stale', JSON.stringify(data), {
            expirationTtl: cacheDuration * 10
          })
        ])
      );

      // Log performance metrics
      ctx.waitUntil(
        logMetrics(env, {
          endpoint: path,
          cache_hit: false,
          response_time: Date.now(),
          backend_status: backendResponse.status
        })
      );

      return new Response(JSON.stringify(data), {
        headers: {
          ...corsHeaders,
          'Content-Type': 'application/json',
          'X-Cache': 'MISS',
          'X-Cache-Duration': `${cacheDuration}s`,
          'Cache-Control': `public, max-age=${cacheDuration}`
        }
      });

    } catch (error) {
      console.error('Worker error:', error);

      // Try to serve stale cache on error
      const cacheKey = `${path}${url.search}`;
      const staleCache = await env.GAS_CACHE.get(cacheKey + ':stale', 'json');

      if (staleCache) {
        return new Response(JSON.stringify(staleCache), {
          status: 200,
          headers: {
            ...corsHeaders,
            'Content-Type': 'application/json',
            'X-Cache': 'STALE',
            'X-Warning': 'Serving stale data due to worker error'
          }
        });
      }

      return new Response(JSON.stringify({
        error: error.message,
        type: error.name
      }), {
        status: 500,
        headers: { ...corsHeaders, 'Content-Type': 'application/json' }
      });
    }
  },

  // Scheduled event handler for automated maintenance
  async scheduled(event, env, ctx) {
    const cronTime = new Date(event.scheduledTime);
    const hour = cronTime.getUTCHours();
    const minute = cronTime.getUTCMinutes();
    const dayOfWeek = cronTime.getUTCDay();

    console.log(`[CRON] Triggered at ${cronTime.toISOString()}`);

    // Every 10 minutes: Keep Render backend alive
    if (minute % 10 === 0) {
      console.log('[CRON] Running keep-alive ping');
      ctx.waitUntil(keepRenderAlive(env, true));
    }

    // Sunday 2 AM: Retrain models
    if (dayOfWeek === 0 && hour === 2 && minute === 0) {
      console.log('[CRON] Triggering weekly model retraining');
      ctx.waitUntil(triggerModelRetraining(env));
    }

    // Every 6 hours: Health check
    if (hour % 6 === 0 && minute === 0) {
      console.log('[CRON] Running health check');
      ctx.waitUntil(performHealthCheck(env));
    }
  }
};

/**
 * Keep Render backend alive by pinging health endpoint
 */
async function keepRenderAlive(env, force = false) {
  const now = Date.now();

  // Only ping every 10 minutes
  if (!force && now - lastKeepAlive < KEEP_ALIVE_INTERVAL) {
    return;
  }

  try {
    const response = await fetch(`${BACKEND_API}/health`, {
      method: 'GET',
      headers: {
        'User-Agent': 'Cloudflare-Worker-KeepAlive/1.0'
      }
    });

    if (response.ok) {
      lastKeepAlive = now;
      console.log('Keep-alive ping successful');

      // Store last ping time in KV
      await env.GAS_CACHE.put('last_keepalive', now.toString(), {
        expirationTtl: 3600 // 1 hour
      });
    }
  } catch (error) {
    console.error('Keep-alive ping failed:', error);
  }
}

/**
 * Determine cache duration for endpoint
 */
function getCacheDuration(path) {
  if (path.includes('current')) return CACHE_DURATIONS.current;
  if (path.includes('prediction')) return CACHE_DURATIONS.predictions;
  if (path.includes('accuracy')) return CACHE_DURATIONS.accuracy;
  if (path.includes('historical')) return CACHE_DURATIONS.historical;
  if (path.includes('validation')) return CACHE_DURATIONS.validation;
  if (path.includes('network') || path.includes('onchain')) return CACHE_DURATIONS.network;
  if (path.includes('retraining')) return CACHE_DURATIONS.retraining;
  if (path.includes('leaderboard')) return CACHE_DURATIONS.leaderboard;
  if (path.includes('stats')) return CACHE_DURATIONS.stats;
  if (path.includes('health')) return CACHE_DURATIONS.health;
  return CACHE_DURATIONS.default;
}

/**
 * Calculate cache age from metadata
 */
function getCacheAge(data) {
  if (data._cached_at) {
    const age = Math.floor((Date.now() - data._cached_at) / 1000);
    return `${age}s`;
  }
  return 'unknown';
}

/**
 * Log performance metrics (could expand to analytics)
 */
async function logMetrics(env, metrics) {
  try {
    const key = `metrics:${Date.now()}`;
    await env.GAS_CACHE.put(key, JSON.stringify(metrics), {
      expirationTtl: 86400 // Keep for 24 hours
    });
  } catch (error) {
    // Fail silently - don't break the request
    console.error('Failed to log metrics:', error);
  }
}

/**
 * Trigger model retraining on backend
 */
async function triggerModelRetraining(env) {
  try {
    console.log('[RETRAIN] Calling backend retraining endpoint...');

    const response = await fetch(`${BACKEND_API}/cron/retrain`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'User-Agent': 'Cloudflare-Worker-Cron/1.0',
        'X-Cron-Trigger': 'weekly-retrain'
      },
      body: JSON.stringify({
        trigger: 'scheduled',
        timestamp: new Date().toISOString()
      })
    });

    const result = await response.json();

    if (response.ok) {
      console.log('[RETRAIN] ✅ Success:', result);

      // Store retraining result in KV
      await env.GAS_CACHE.put('last_retrain_result', JSON.stringify({
        timestamp: new Date().toISOString(),
        success: true,
        result
      }), {
        expirationTtl: 604800 // Keep for 7 days
      });
    } else {
      console.error('[RETRAIN] ❌ Failed:', result);

      // Store failure for monitoring
      await env.GAS_CACHE.put('last_retrain_result', JSON.stringify({
        timestamp: new Date().toISOString(),
        success: false,
        error: result
      }), {
        expirationTtl: 604800 // Keep for 7 days
      });
    }

    return result;
  } catch (error) {
    console.error('[RETRAIN] Error triggering retraining:', error);
    return { success: false, error: error.message };
  }
}

/**
 * Perform health check on models
 */
async function performHealthCheck(env) {
  try {
    console.log('[HEALTH] Running model health check...');

    const response = await fetch(`${BACKEND_API}/cron/health-check`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'User-Agent': 'Cloudflare-Worker-Cron/1.0'
      }
    });

    const health = await response.json();

    console.log('[HEALTH] Result:', health.healthy ? '✅ Healthy' : '⚠️ Degraded');

    if (!health.healthy) {
      console.warn('[HEALTH] Alerts:', health.alerts);

      // TODO: Send notification (email/Discord/Slack)
      // For now, just log to KV for monitoring
      await env.GAS_CACHE.put('health_alert', JSON.stringify({
        timestamp: new Date().toISOString(),
        alerts: health.alerts
      }), {
        expirationTtl: 86400 // Keep for 24 hours
      });
    }

    // Store latest health check
    await env.GAS_CACHE.put('last_health_check', JSON.stringify({
      timestamp: new Date().toISOString(),
      health
    }), {
      expirationTtl: 21600 // Keep for 6 hours
    });

    return health;
  } catch (error) {
    console.error('[HEALTH] Error during health check:', error);
    return { healthy: false, error: error.message };
  }
}
