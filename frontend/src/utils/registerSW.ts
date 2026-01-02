export function registerServiceWorker() {
  if ('serviceWorker' in navigator && import.meta.env.PROD) {
    window.addEventListener('load', () => {
      navigator.serviceWorker
        .register('/sw.js', { updateViaCache: 'none' }) // Always check for new service worker
        .then((registration) => {
          console.log('✓ Service Worker registered successfully:', registration.scope);

          // Check for updates immediately and on focus
          const checkForUpdates = () => {
            registration.update();
          };

          // Check for updates on page load
          checkForUpdates();

          // Check for updates when page regains focus (user returns to tab)
          document.addEventListener('visibilitychange', () => {
            if (!document.hidden) {
              checkForUpdates();
            }
          });

          // Check for updates periodically (every 5 minutes)
          setInterval(checkForUpdates, 5 * 60 * 1000);

          // Listen for service worker updates
          registration.addEventListener('updatefound', () => {
            const newWorker = registration.installing;
            if (newWorker) {
              newWorker.addEventListener('statechange', () => {
                if (newWorker.state === 'activated' && navigator.serviceWorker.controller) {
                  // New service worker activated - reload page to get new version
                  console.log('🔄 New service worker activated, reloading page...');
                  window.location.reload();
                }
              });
            }
          });

          // Handle controller change (service worker takeover)
          navigator.serviceWorker.addEventListener('controllerchange', () => {
            console.log('🔄 Service worker controller changed, reloading page...');
            window.location.reload();
          });
        })
        .catch((error) => {
          console.warn('Service Worker registration failed:', error);
        });
    });
  }
}
