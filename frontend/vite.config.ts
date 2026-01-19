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
        })
      ],
      define: {
        'process.env.API_KEY': JSON.stringify(env.GEMINI_API_KEY),
        'process.env.GEMINI_API_KEY': JSON.stringify(env.GEMINI_API_KEY)
      },
      resolve: {
        alias: {
          '@': path.resolve(__dirname, '.'),
        }
      },
      optimizeDeps: {
        include: ['lucide-react'], // Force pre-bundling of lucide-react to avoid initialization issues
      },
      build: {
        rollupOptions: {
          output: {
            manualChunks: (id) => {
              // Separate node_modules into more granular chunks
              if (id.includes('node_modules')) {
                // React core - most critical, load first
                if (id.includes('react') && !id.includes('react-dom') && !id.includes('react-router')) {
                  return 'vendor-react-core';
                }
                if (id.includes('react-dom')) {
                  return 'vendor-react-dom';
                }
                // Router - load with React
                if (id.includes('react-router')) {
                  return 'vendor-router';
                }
                // Recharts is heavy - separate it completely
                if (id.includes('recharts')) {
                  return 'vendor-charts';
                }
                // UI libraries (except lucide-react)
                if (id.includes('framer-motion')) {
                  return 'vendor-ui';
                }
                // Query library
                if (id.includes('@tanstack/react-query')) {
                  return 'vendor-query';
                }
                // Monitoring
                if (id.includes('@sentry')) {
                  return 'vendor-sentry';
                }
                // WebSocket
                if (id.includes('socket.io')) {
                  return 'vendor-socket';
                }
                // Toast notifications
                if (id.includes('react-hot-toast')) {
                  return 'vendor-toast';
                }
                // Farcaster SDK
                if (id.includes('@farcaster')) {
                  return 'vendor-farcaster';
                }
                // Zustand
                if (id.includes('zustand')) {
                  return 'vendor-state';
                }
                // React Virtual
                if (id.includes('@tanstack/react-virtual')) {
                  return 'vendor-virtual';
                }
                // Other vendor code
                return 'vendor-misc';
              }
              // Separate large feature components
              if (id.includes('/components/') && (id.includes('Chart') || id.includes('Graph'))) {
                return 'chunk-charts';
              }
            }
          },
          // Enable tree-shaking optimizations
          // Note: Preserve side effects for libraries that need them
          treeshake: {
            moduleSideEffects: (id) => {
              // Preserve side effects for lucide-react and other icon libraries
              if (id.includes('lucide-react') || id.includes('lucide')) {
                return true;
              }
              return false;
            },
            propertyReadSideEffects: false,
            tryCatchDeoptimization: false
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
            unsafe_undefined: false
          },
          mangle: {
            safari10: true
          },
          // Disable eval() entirely
          format: {
            comments: false
          }
        },
        sourcemap: false,
        chunkSizeWarningLimit: 500,
        cssCodeSplit: true,
        assetsInlineLimit: 4096,
        cssMinify: true
      },
      css: {
        devSourcemap: false
      }
    };
});
