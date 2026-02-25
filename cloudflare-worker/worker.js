/**
 * Cloudflare Worker - Gas Fees API Proxy with Intelligent Caching
 * Provides instant global access with optimized cache strategy and keep-alive
 */

const BACKEND_API = 'https://basegasfeesml-production.up.railway.app/api';

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

    console.log(`[CRON] Triggered at ${cronTime.toISOString()}`);

    // Every 5 minutes: Collect gas data (also keeps backend alive)
    if (minute % 5 === 0) {
      console.log('[CRON] Collecting gas data');
      ctx.waitUntil(collectGasData(env));
    }

    // Every 6 hours: Health check
    if (hour % 6 === 0 && minute === 0) {
      console.log('[CRON] Running health check');
      ctx.waitUntil(performHealthCheck(env));
    }

    // Hourly at :15: Update prediction accuracy
    if (minute === 15) {
      console.log('[CRON] Updating prediction accuracy');
      ctx.waitUntil(updatePredictionAccuracy(env));
    }

    // Hourly at :30: Validate predictions against actuals
    if (minute === 30) {
      console.log('[CRON] Validating predictions');
      ctx.waitUntil(validatePredictions(env));
    }
  }
};

/**
 * Collect gas data by calling /current endpoint (also keeps backend alive)
 * This stores the current gas price in Railway's database for validation
 */
async function collectGasData(env) {
  try {
    const response = await fetch(`${BACKEND_API}/current?chain=base`, {
      method: 'GET',
      headers: {
        'User-Agent': 'Cloudflare-Worker-Collector/1.0'
      }
    });

    if (response.ok) {
      const data = await response.json();
      console.log(`[COLLECT] ✅ Gas price: ${data.current_gas?.toFixed(4)} gwei`);

      // Store last collection time in KV
      await env.GAS_CACHE.put('last_collection', JSON.stringify({
        timestamp: new Date().toISOString(),
        gas_price: data.current_gas
      }), {
        expirationTtl: 3600 // 1 hour
      });
    } else {
      console.error('[COLLECT] ❌ Failed:', response.status);
    }
  } catch (error) {
    console.error('[COLLECT] Error:', error);
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
 * Send Discord webhook notification
 */
async function sendDiscordNotification(env, { title, description, color, fields = [] }) {
  // Check if webhook URL is configured
  const webhookUrl = env.DISCORD_WEBHOOK_URL;
  if (!webhookUrl) {
    console.log('[DISCORD] No webhook URL configured, skipping notification');
    return;
  }

  try {
    const embed = {
      title,
      description,
      color, // Discord color as integer (e.g., 0xFF0000 for red)
      timestamp: new Date().toISOString(),
      footer: {
        text: 'Gweizy Gas Predictor'
      },
      fields
    };

    const response = await fetch(webhookUrl, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        username: 'Gweizy Bot',
        avatar_url: 'https://gweizy.xyz/logo.png',
        embeds: [embed]
      })
    });

    if (response.ok) {
      console.log('[DISCORD] ✅ Notification sent:', title);
    } else {
      console.error('[DISCORD] ❌ Failed to send:', response.status);
    }
  } catch (error) {
    console.error('[DISCORD] Error sending notification:', error);
  }
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
 * Update prediction accuracy by comparing predictions to actuals
 */
async function updatePredictionAccuracy(env) {
  try {
    console.log('[ACCURACY] Updating prediction accuracy...');

    const response = await fetch(`${BACKEND_API}/cron/update-accuracy`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'User-Agent': 'Cloudflare-Worker-Cron/1.0',
        'X-Cron-Trigger': 'hourly-accuracy'
      }
    });

    const result = await response.json();

    if (response.ok && result.success) {
      console.log('[ACCURACY] ✅ Updated:', result);

      // Store last accuracy update in KV
      await env.GAS_CACHE.put('last_accuracy_update', JSON.stringify({
        timestamp: new Date().toISOString(),
        success: true,
        actual_gas: result.actual_gas,
        drift_status: result.drift_status,
        should_retrain: result.should_retrain
      }), {
        expirationTtl: 7200 // Keep for 2 hours
      });

      // If drift is detected and retrain is recommended, send notification
      if (result.should_retrain) {
        console.warn('[ACCURACY] ⚠️ Retrain recommended:', result.retrain_reasons);
        await env.GAS_CACHE.put('retrain_recommended', JSON.stringify({
          timestamp: new Date().toISOString(),
          reasons: result.retrain_reasons,
          drift_status: result.drift_status
        }), {
          expirationTtl: 86400 // Keep for 24 hours
        });

        // Send Discord alert
        await sendDiscordNotification(env, {
          title: '⚠️ Model Retrain Recommended',
          description: 'Prediction accuracy has degraded significantly. Consider triggering a model retrain.',
          color: 0xFFA500, // Orange
          fields: [
            {
              name: 'Reasons',
              value: result.retrain_reasons?.join('\n') || 'Accuracy below threshold',
              inline: false
            },
            {
              name: 'Current Gas',
              value: `${result.actual_gas?.toFixed(4) || 'N/A'} gwei`,
              inline: true
            },
            {
              name: 'Drift Status',
              value: Object.entries(result.drift_status || {})
                .map(([h, d]) => `${h}: ${d.is_drifting ? '🔴' : '🟢'}`)
                .join(' | '),
              inline: true
            }
          ]
        });
      }
    } else {
      console.error('[ACCURACY] ❌ Failed:', result);
    }

    return result;
  } catch (error) {
    console.error('[ACCURACY] Error updating accuracy:', error);
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

      // Store alert in KV for monitoring
      await env.GAS_CACHE.put('health_alert', JSON.stringify({
        timestamp: new Date().toISOString(),
        alerts: health.alerts
      }), {
        expirationTtl: 86400 // Keep for 24 hours
      });

      // Send Discord notification
      await sendDiscordNotification(env, {
        title: '🔴 Model Health Degraded',
        description: 'The prediction model is showing signs of degraded performance.',
        color: 0xFF0000, // Red
        fields: (health.alerts || []).map(alert => ({
          name: alert.severity?.toUpperCase() || 'ALERT',
          value: alert.message || 'Unknown issue',
          inline: false
        }))
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

/**
 * Validate pending predictions against actual gas prices
 */
async function validatePredictions(env) {
  try {
    console.log('[VALIDATION] Running prediction validation...');

    const response = await fetch(`${BACKEND_API}/validation/validate`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'User-Agent': 'Cloudflare-Worker-Cron/1.0',
        'X-Cron-Trigger': 'hourly-validation'
      }
    });

    const result = await response.json();

    if (response.ok) {
      console.log('[VALIDATION] ✅ Validated:', result.validated, 'predictions');

      // Store validation result in KV
      await env.GAS_CACHE.put('last_validation', JSON.stringify({
        timestamp: new Date().toISOString(),
        validated: result.validated,
        pending: result.pending
      }), {
        expirationTtl: 7200 // Keep for 2 hours
      });
    } else {
      console.error('[VALIDATION] ❌ Failed:', result);
    }

    return result;
  } catch (error) {
    console.error('[VALIDATION] Error:', error);
    return { success: false, error: error.message };
  }
}
