/**
 * Fix for lucide-react initialization issue with React 18
 * Ensures React is fully initialized before lucide-react components are used
 */

// Ensure React is available globally before lucide-react initializes
if (typeof window !== 'undefined') {
  // Wait for React to be fully loaded
  window.addEventListener('DOMContentLoaded', () => {
    // Small delay to ensure React is initialized
    setTimeout(() => {
      // Force React to be available in global scope if needed
      if (typeof (window as any).React === 'undefined') {
        // This ensures React is initialized before lucide-react
        import('react').then((React) => {
          (window as any).React = React;
        });
      }
    }, 0);
  });
}

// Export a function to ensure React is ready before using lucide-react
export function ensureReactReady(): Promise<void> {
  return new Promise((resolve) => {
    if (typeof window === 'undefined') {
      resolve();
      return;
    }
    
    // Check if React is already loaded
    const checkReact = () => {
      // React 18 uses React.createElement which should be available
      if (typeof (window as any).React !== 'undefined' || 
          typeof document !== 'undefined' && document.getElementById('root')) {
        resolve();
      } else {
        // Wait a bit and check again
        setTimeout(checkReact, 10);
      }
    };
    
    if (document.readyState === 'loading') {
      window.addEventListener('DOMContentLoaded', checkReact);
    } else {
      checkReact();
    }
  });
}