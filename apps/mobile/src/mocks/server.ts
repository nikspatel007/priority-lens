/**
 * MSW Server Setup for Testing
 *
 * Import this in tests that need API mocking.
 */

import { setupServer } from 'msw/node';
import { handlers } from './handlers';

// Create the mock server with default handlers
export const server = setupServer(...handlers);
