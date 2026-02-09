/**
 * Service Worker for Gweizy
 *
 * Provides offline caching for static assets and API responses.
 * Uses a cache-first strategy for static assets and network-first for API calls.
 */

// Version-based cache name - update this on each deployment to force cache refresh
const CACHE_VERSION = 'v4-20260209';
const CACHE_NAME = `base-gas-optimizer-${CACHE_VERSION}`;
const API_CACHE_NAME = `api-cache-${CACHE_VERSION}`;

// Only cache truly static assets (images, fonts, etc.)
const STATIC_ASSET_EXTENSIONS = ['.png', '.jpg', '.jpeg', '.svg', '.gif', '.webp', '.woff', '.woff2', '.ttf', '.eot', '.ico'];

// API endpoints that should be cached for offline use
const CACHEABLE_API_PATHS = ['/current', '/predictions', '/health', '/hybrid'];

// Maximum age for cached API responses (5 minutes)
const API_CACHE_MAX_AGE = 5 * 60 * 1000;

// Install event - clear old caches immediately
self.addEventListener('install', (event) => {
  console.log('[SW] Installing service worker, version:', CACHE_VERSION);
  // Force activation of new service worker
  self.skipWaiting();
});

// Activate event - clean up ALL old caches
self.addEventListener('activate', (event) => {
  console.log('[SW] Activating service worker, clearing old caches');
  event.waitUntil(
    caches.keys().then((cacheNames) => {
      return Promise.all(
        cacheNames.map((cacheName) => {
          // Delete ALL old caches (including old versions)
          if (cacheName !== CACHE_NAME) {
            console.log('[SW] Deleting old cache:', cacheName);
            return caches.delete(cacheName);
          }
        })
      );
    }).then(() => {
      // Take control of all pages immediately
      return self.clients.claim();
    })
  );
});

// Fetch event - network-first for HTML/JS, cache-only for static assets
self.addEventListener('fetch', (event) => {
  const { request } = event;
  const url = new URL(request.url);

  // Skip non-GET requests
  if (request.method !== 'GET') {
    return;
  }

  // API requests - network-first with offline cache fallback
  const isApiRequest = url.pathname.includes('/api/') || url.hostname.includes('railway.app');
  const isCacheableApi = isApiRequest && CACHEABLE_API_PATHS.some(path => url.pathname.includes(path));

  if (isCacheableApi) {
    event.respondWith(networkFirstWithCache(request));
    return;
  }

  // Non-cacheable API requests - let browser handle directly
  if (isApiRequest || url.hostname.includes('cloudflare')) {
    return;
  }

  // Check if this is a static asset (image, font, etc.)
  const isStaticAsset = STATIC_ASSET_EXTENSIONS.some(ext => url.pathname.toLowerCase().endsWith(ext));

  if (isStaticAsset) {
    // Static assets: Cache-first strategy (images, fonts can be cached)
    event.respondWith(
      caches.match(request).then((cachedResponse) => {
        if (cachedResponse) {
          return cachedResponse;
        }

        return fetch(request).then((response) => {
          // Don't cache non-successful responses
          if (!response || response.status !== 200 || response.type === 'error') {
            return response;
          }

          // Clone the response
          const responseToCache = response.clone();

          caches.open(CACHE_NAME).then((cache) => {
            cache.put(request, responseToCache);
          });

          return response;
        });
      })
    );
  } else {
    // HTML, JS, CSS: Network-first strategy (always get fresh content)
    event.respondWith(
      fetch(request)
        .then((response) => {
          // Don't cache HTML/JS/CSS - always get fresh from network
          return response;
        })
        .catch((error) => {
          // If network fails, try cache as fallback (offline support)
          return caches.match(request).then((cachedResponse) => {
            if (cachedResponse) {
              return cachedResponse;
            }
            throw error;
          });
        })
    );
  }
});

/**
 * Network-first strategy with cache fallback for API requests
 * Caches successful responses for offline use
 */
async function networkFirstWithCache(request) {
  try {
    const networkResponse = await fetch(request);

    // Cache successful responses
    if (networkResponse.ok) {
      const cache = await caches.open(API_CACHE_NAME);
      const responseToCache = networkResponse.clone();

      // Add cache timestamp header
      const headers = new Headers(responseToCache.headers);
      headers.set('sw-cached-at', Date.now().toString());

      const cachedResponse = new Response(await responseToCache.blob(), {
        status: responseToCache.status,
        statusText: responseToCache.statusText,
        headers: headers,
      });

      cache.put(request, cachedResponse);
    }

    return networkResponse;
  } catch (error) {
    // Network failed, try cache
    const cachedResponse = await caches.match(request);

    if (cachedResponse) {
      const cachedAt = cachedResponse.headers.get('sw-cached-at');
      const age = cachedAt ? Date.now() - parseInt(cachedAt) : 0;

      console.log('[SW] Serving cached API response:', request.url, `(age: ${Math.round(age / 1000)}s)`);

      // Return cached response even if stale (offline scenario)
      return cachedResponse;
    }

    // No cache available, return offline error
    return new Response(
      JSON.stringify({
        error: 'Offline',
        message: 'No cached data available',
        offline: true,
      }),
      {
        status: 503,
        headers: { 'Content-Type': 'application/json' },
      }
    );
  }
}

/**
 * Handle messages from main thread
 */
self.addEventListener('message', (event) => {
  const { type } = event.data || {};

  switch (type) {
    case 'SKIP_WAITING':
      self.skipWaiting();
      break;

    case 'CLEAR_API_CACHE':
      caches.delete(API_CACHE_NAME);
      console.log('[SW] API cache cleared');
      break;

    case 'GET_CACHE_INFO':
      getCacheInfo().then((info) => {
        if (event.ports && event.ports[0]) {
          event.ports[0].postMessage({ type: 'CACHE_INFO', payload: info });
        }
      });
      break;
  }
});

/**
 * Get cache information
 */
async function getCacheInfo() {
  const cacheNames = await caches.keys();
  let totalSize = 0;
  let entryCount = 0;

  for (const name of cacheNames) {
    const cache = await caches.open(name);
    const keys = await cache.keys();
    entryCount += keys.length;

    for (const request of keys) {
      const response = await cache.match(request);
      if (response) {
        const blob = await response.blob();
        totalSize += blob.size;
      }
    }
  }

  return {
    version: CACHE_VERSION,
    caches: cacheNames,
    totalSize,
    entryCount,
  };
}

console.log('[SW] Service Worker loaded, version:', CACHE_VERSION);
