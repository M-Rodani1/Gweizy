/**
 * MSW Server Setup
 *
 * Creates a mock server for intercepting API requests in tests.
 * Use setupServer for Node.js environments (Vitest, Jest).
 */

import { setupServer } from 'msw/node';
import { handlers } from './handlers';

// Create the server instance with default handlers
export const server = setupServer(...handlers);

// Export for test setup
export { handlers };
