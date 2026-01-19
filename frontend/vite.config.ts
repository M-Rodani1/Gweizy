import path from 'path';
import { defineConfig, loadEnv } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig(({ mode }) => {
    const env = loadEnv(mode, '.', '');
    return {
      server: {
        port: 3000,
        host: '0.0.0.0',
      },
      plugins: [
        react({
          // Enable automatic JSX runtime
          jsxRuntime: 'automatic',
          babel: {
            plugins: [
              // Ensure React is properly transformed
            ],
          },
        })
      ],
      define: {
        'process.env.API_KEY': JSON.stringify(env.GEMINI_API_KEY),
        'process.env.GEMINI_API_KEY': JSON.stringify(env.GEMINI_API_KEY)
      },
      resolve: {
        alias: {
          '@': path.resolve(__dirname, '.'),
        },
        dedupe: ['react', 'react-dom', 'react-router', 'react-router-dom'], // Ensure single instance of React and router
      },
      optimizeDeps: {
        include: ['lucide-react', 'react', 'react-dom', 'react-router-dom'], // Force pre-bundling to avoid initialization issues
        esbuildOptions: {
          target: 'es2020',
        },
      },
      build: {
        rollupOptions: {
          output: {
            // Code splitting strategy that prevents React initialization errors:
            // 1. React and ALL React-dependent packages stay in MAIN ENTRY (return undefined)
            // 2. Only truly independent packages get split into separate chunks
            manualChunks: (id) => {
              if (id.includes('node_modules')) {
                // KEEP IN MAIN ENTRY (return undefined) - ensures React loads first
                // React core and DOM - MUST stay in entry
                if (id.includes('/react/') || id.includes('/react-dom/')) {
                  return undefined; // Stays in main entry bundle
                }
                
                // ALL packages that import React - MUST stay in entry
                if (id.includes('react-router') ||
                    id.includes('react-is') ||
                    id.includes('react-countup') ||
                    id.includes('react-hot-toast') ||
                    id.includes('lucide-react') ||
                    id.includes('@tanstack/react-query') ||
                    id.includes('@tanstack/react-virtual') ||
                    id.includes('@sentry/react') ||
                    id.includes('framer-motion') ||
                    id.includes('@farcaster') ||
                    id.includes('recharts')) {
                  return undefined; // Stays in main entry bundle
                }
                
                // SAFE TO SPLIT - packages that DON'T depend on React
                // These load in parallel but don't need React to initialize
                if (id.includes('socket.io')) {
                  return 'vendor-socket';
                }
                
                // D3 and other chart utilities (used by recharts internally but no React dep)
                if (id.includes('d3-') || id.includes('victory-vendor')) {
                  return 'vendor-charts';
                }
                
                // Other truly independent utilities
                if (id.includes('clsx') || 
                    id.includes('tailwind-merge') ||
                    id.includes('class-variance-authority')) {
                  return 'vendor-utils';
                }
                
                // Everything else stays in main entry to be safe
                return undefined;
              }
            }
          }
        },
        minify: 'terser',
        terserOptions: {
          compress: {
            drop_console: true,
            drop_debugger: true,
            pure_funcs: ['console.log', 'console.info', 'console.debug', 'console.trace'],
            passes: 2,
            // Disable eval() to avoid CSP violations
            unsafe: false,
            unsafe_comps: false,
            unsafe_math: false,
            unsafe_methods: false,
            unsafe_proto: false,
            unsafe_regexp: false,
            unsafe_undefined: false,
            // Preserve React's internal structure
            keep_classnames: false,
            keep_fnames: false,
          },
          mangle: {
            safari10: true,
            // Don't mangle React internals
            reserved: ['React', 'ReactDOM']
          },
          // Disable eval() entirely
          format: {
            comments: false,
            preserve_annotations: false
          }
        },
        sourcemap: false, // Disabled for production to reduce bundle size
        chunkSizeWarningLimit: 600, // Increased slightly to reduce warnings
        cssCodeSplit: true,
        assetsInlineLimit: 4096,
        cssMinify: true,
        // Target modern browsers for smaller bundles
        target: 'es2020',
        // Enable module preload for faster loading
        modulePreload: {
          polyfill: true
        }
      },
      css: {
        devSourcemap: false
      },
      // Experimental: Enable faster builds
      esbuild: {
        // Remove console.log in production
        drop: mode === 'production' ? ['console', 'debugger'] : [],
        // Target modern browsers
        target: 'es2020',
        // Minify syntax
        minifyIdentifiers: true,
        minifySyntax: true,
        minifyWhitespace: true
      }
    };
});
