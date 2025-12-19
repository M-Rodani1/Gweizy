/**
 * Cloudflare Worker - Gas Fees API Proxy with KV Caching
 * Provides instant global access to gas price data with edge caching
 */

const BACKEND_API = 'https://basegasfeesml.onrender.com/api';
const CACHE_DURATIONS = {
  current: 10,        // 10 seconds - very fresh
  predictions: 30,    // 30 seconds
  accuracy: 60,       // 1 minute
  historical: 300,    // 5 minutes
  validation: 120,    // 2 minutes
  network: 15,        // 15 seconds
  retraining: 120     // 2 minutes
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

    // Extract endpoint path - remove /api prefix if present
    let path = url.pathname;
    if (path.startsWith('/api/')) {
      path = path.substring(5); // Remove '/api/'
    } else if (path.startsWith('/api')) {
      path = path.substring(4); // Remove '/api'
    }

    // Remove leading slash if present
    if (path.startsWith('/')) {
      path = path.substring(1);
    }

    // If no path, return error
    if (!path || path === '') {
      return new Response(JSON.stringify({
        error: 'No endpoint specified',
        usage: 'Use /api/current, /api/predictions, etc.'
      }), {
        status: 400,
        headers: { ...corsHeaders, 'Content-Type': 'application/json' }
      });
    }

    try {
      // Check KV cache first
      const cacheKey = `${path}${url.search}`;
      const cached = await env.GAS_CACHE.get(cacheKey, 'json');

      if (cached) {
        console.log(`Cache HIT: ${cacheKey}`);
        return new Response(JSON.stringify(cached), {
          headers: {
            ...corsHeaders,
            'Content-Type': 'application/json',
            'X-Cache': 'HIT',
            'Cache-Control': 'public, max-age=10'
          }
        });
      }

      console.log(`Cache MISS: ${cacheKey}`);

      // Fetch from backend
      const backendUrl = `${BACKEND_API}/${path}${url.search}`;
      console.log(`Fetching: ${backendUrl}`);

      const backendResponse = await fetch(backendUrl, {
        headers: {
          'Content-Type': 'application/json'
        }
      });

      if (!backendResponse.ok) {
        return new Response(JSON.stringify({
          error: 'Backend error',
          status: backendResponse.status,
          endpoint: path
        }), {
          status: backendResponse.status,
          headers: { ...corsHeaders, 'Content-Type': 'application/json' }
        });
      }

      const data = await backendResponse.json();

      // Determine cache duration based on endpoint
      let cacheDuration = 30; // default
      if (path.startsWith('current')) cacheDuration = CACHE_DURATIONS.current;
      else if (path.startsWith('predictions')) cacheDuration = CACHE_DURATIONS.predictions;
      else if (path.startsWith('accuracy')) cacheDuration = CACHE_DURATIONS.accuracy;
      else if (path.startsWith('historical')) cacheDuration = CACHE_DURATIONS.historical;
      else if (path.startsWith('validation')) cacheDuration = CACHE_DURATIONS.validation;
      else if (path.startsWith('network')) cacheDuration = CACHE_DURATIONS.network;
      else if (path.startsWith('retraining')) cacheDuration = CACHE_DURATIONS.retraining;

      // Store in KV with expiration
      ctx.waitUntil(
        env.GAS_CACHE.put(cacheKey, JSON.stringify(data), {
          expirationTtl: cacheDuration
        })
      );

      return new Response(JSON.stringify(data), {
        headers: {
          ...corsHeaders,
          'Content-Type': 'application/json',
          'X-Cache': 'MISS',
          'Cache-Control': `public, max-age=${cacheDuration}`
        }
      });

    } catch (error) {
      console.error('Worker error:', error);
      return new Response(JSON.stringify({
        error: error.message,
        stack: error.stack
      }), {
        status: 500,
        headers: { ...corsHeaders, 'Content-Type': 'application/json' }
      });
    }
  }
};
