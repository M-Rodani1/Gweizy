// Version-based cache name - update this on each deployment to force cache refresh
const CACHE_VERSION = 'v3-20260102';
const CACHE_NAME = `base-gas-optimizer-${CACHE_VERSION}`;

// Only cache truly static assets (images, fonts, etc.)
const STATIC_ASSET_EXTENSIONS = ['.png', '.jpg', '.jpeg', '.svg', '.gif', '.webp', '.woff', '.woff2', '.ttf', '.eot', '.ico'];

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

  // API requests - always go to network, no caching (to avoid CORS issues)
  if (url.pathname.includes('/api/') || url.hostname.includes('railway.app') || url.hostname.includes('cloudflare')) {
    // Don't intercept - let the browser handle it directly
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
