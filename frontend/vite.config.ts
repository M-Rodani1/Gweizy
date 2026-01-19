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
            manualChunks: (id) => {
              // CRITICAL: React MUST load first, before ANY other code
              // Put React in main entry bundle to ensure it loads synchronously
              // Don't split React into vendor chunks to avoid initialization race conditions
              
              // Separate node_modules into more granular chunks
              if (id.includes('node_modules')) {
                // React core MUST be in main bundle (don't split it)
                // This ensures React is initialized before any vendor chunks load
                if ((id.includes('/react/') || id.includes('\\react\\')) && 
                    !id.includes('react-dom') && 
                    !id.includes('react-router')) {
                  // React core stays in main bundle - DO NOT split
                  return undefined;
                }
                
                // React DOM should also be in main bundle to ensure proper initialization
                if (id.includes('react-dom')) {
                  // Keep react-dom with main bundle for now
                  return undefined;
                }
                
                // ALL React-dependent packages MUST be together in vendor-react-core
                // This ensures they load AFTER React is initialized
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
                  return 'vendor-react-core';
                }
                
                // WebSocket - no React dependency, can be separate
                if (id.includes('socket.io')) {
                  return 'vendor-socket';
                }
                
                // Zustand - state management, separate (doesn't depend on React directly)
                if (id.includes('zustand')) {
                  return 'vendor-state';
                }
                
                // Everything else goes to vendor-misc (only truly non-React dependencies)
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
