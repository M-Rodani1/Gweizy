/**
 * Vitest configuration for unit testing
 */

import { defineConfig } from 'vitest/config';
import react from '@vitejs/plugin-react';
import path from 'path';

export default defineConfig({
  plugins: [react()],
  test: {
    environment: 'jsdom',
    globals: true,
    setupFiles: ['./src/__tests__/setup.ts'],
    coverage: {
      // Coverage provider
      provider: 'v8',

      // Reporters for output
      reporter: ['text', 'text-summary', 'html', 'lcov', 'json'],

      // Output directory
      reportsDirectory: './coverage',

      // Files to include in coverage
      include: [
        'src/**/*.{ts,tsx}',
      ],

      // Files to exclude from coverage
      exclude: [
        'src/**/*.d.ts',
        'src/**/*.test.{ts,tsx}',
        'src/**/*.spec.{ts,tsx}',
        'src/**/*.stories.{ts,tsx}',
        'src/__tests__/**',
        'src/main.tsx',
        'src/vite-env.d.ts',
        'src/**/__mocks__/**',
        'src/types/**',
      ],

      // Thresholds for coverage enforcement
      // Start with baseline thresholds and increase over time
      thresholds: {
        // Global thresholds (realistic starting point)
        lines: 30,
        functions: 25,
        branches: 20,
        statements: 30,

        // Per-file thresholds (optional, stricter for critical files)
        perFile: false,

        // Auto-update thresholds (set to true to automatically update)
        autoUpdate: false,
      },

      // Clean coverage results before running
      clean: true,

      // Skip files with no statements
      skipFull: false,

      // All files should be included, not just tested ones
      all: true,
    },
  },
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
    },
  },
});
