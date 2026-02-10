/**
 * Stryker Mutation Testing Configuration
 *
 * Run mutation tests with: npm run test:mutation
 * View report: Open reports/mutation/html/index.html
 *
 * @type {import('@stryker-mutator/api/core').PartialStrykerOptions}
 */
export default {
  // Package manager
  packageManager: 'npm',

  // Test runner - use Vitest
  testRunner: 'vitest',
  vitest: {
    configFile: 'vitest.config.ts',
  },

  // TypeScript checker
  checkers: ['typescript'],
  tsconfigFile: 'tsconfig.json',

  // Files to mutate
  mutate: [
    'src/**/*.ts',
    'src/**/*.tsx',
    // Exclude test files
    '!src/**/*.test.ts',
    '!src/**/*.test.tsx',
    '!src/**/*.spec.ts',
    '!src/**/*.spec.tsx',
    '!src/**/*.stories.tsx',
    // Exclude type-only files
    '!src/**/*.d.ts',
    '!src/types/**/*',
    // Exclude configuration files
    '!src/config/**/*',
    // Focus on utility functions and hooks for best mutation testing value
    // 'src/utils/**/*.ts',
    // 'src/hooks/**/*.ts',
  ],

  // Ignore patterns
  ignorePatterns: [
    'node_modules',
    'dist',
    'coverage',
    'storybook-static',
    'e2e',
    '.storybook',
  ],

  // Mutation operators to use
  mutators: {
    excludedMutations: [
      // Exclude mutations that often create equivalent mutants
      'StringLiteral', // "text" -> "" creates many equivalent mutants
    ],
  },

  // Concurrency settings
  concurrency: 4,

  // Timeout settings
  timeoutMS: 60000,
  timeoutFactor: 2.5,

  // Thresholds for mutation score
  thresholds: {
    high: 80,
    low: 60,
    break: null, // Don't fail the build on low score (yet)
  },

  // Reporters
  reporters: ['html', 'clear-text', 'progress', 'dashboard'],
  htmlReporter: {
    fileName: 'reports/mutation/html/index.html',
  },
  clearTextReporter: {
    allowColor: true,
    logTests: false,
    maxTestsToLog: 3,
  },

  // Dashboard reporter (for CI/CD integration)
  dashboard: {
    project: 'github.com/M-Rodani1/Gweizy',
    version: 'main',
    // Set STRYKER_DASHBOARD_API_KEY in CI environment
  },

  // Incremental mode - only test changed files
  incremental: true,
  incrementalFile: '.stryker-cache/incremental.json',

  // Disable coverage analysis for faster runs (optional)
  coverageAnalysis: 'perTest',

  // Log level
  logLevel: 'info',

  // Dry run mode for testing configuration
  // dryRunOnly: true,
};
