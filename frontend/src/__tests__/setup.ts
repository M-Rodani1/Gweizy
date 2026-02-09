/**
 * Test setup file
 * Configures test environment with React Testing Library
 *
 * Note: MSW is opt-in per test file. See src/__tests__/mocks/msw.example.test.ts
 * for usage examples. Import and start the server in your test file:
 *
 *   import { server } from './mocks/server';
 *   beforeAll(() => server.listen());
 *   afterEach(() => server.resetHandlers());
 *   afterAll(() => server.close());
 */

import { afterEach } from 'vitest';
import { cleanup } from '@testing-library/react';
import '@testing-library/jest-dom';

// Cleanup after each test
afterEach(() => {
  cleanup();
});
