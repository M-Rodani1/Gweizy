/**
 * API Contract Tests
 *
 * These tests verify that API responses match expected contracts/schemas.
 * They ensure the frontend and backend maintain compatible data structures.
 *
 * Uses Zod schemas for contract validation.
 */

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { z } from 'zod';

// ============================================================================
// API Response Contracts (Schemas)
// ============================================================================

/**
 * Contract for /api/gas endpoint
 */
const GasResponseContract = z.object({
  success: z.boolean(),
  current_gas: z.number().positive().optional(),
  gas_price: z.number().positive().optional(),
  timestamp: z.string().optional(),
  network: z.string().optional(),
  source: z.string().optional(),
}).passthrough(); // Allow additional fields

/**
 * Contract for /api/predictions endpoint
 */
const PredictionsResponseContract = z.object({
  success: z.boolean(),
  predictions: z.object({
    '1h': z.array(z.object({
      time: z.string(),
      predictedGwei: z.number().optional(),
      gwei: z.number().optional(),
      confidence: z.number().min(0).max(1).optional(),
    })).optional(),
    '4h': z.array(z.any()).optional(),
    '24h': z.array(z.any()).optional(),
    historical: z.array(z.any()).optional(),
  }).optional(),
  model_version: z.string().optional(),
  generated_at: z.string().optional(),
}).passthrough();

/**
 * Contract for /api/accuracy endpoint
 */
const AccuracyResponseContract = z.object({
  success: z.boolean(),
  metrics: z.object({
    mae: z.number().nonnegative().optional(),
    rmse: z.number().nonnegative().optional(),
    r2: z.number().optional(),
    directional_accuracy: z.number().min(0).max(1).optional(),
    mape: z.number().nonnegative().optional(),
  }).optional(),
  horizon: z.string().optional(),
  period: z.string().optional(),
}).passthrough();

/**
 * Contract for /api/health endpoint
 */
const HealthResponseContract = z.object({
  status: z.string().optional(), // Can be various status strings
  success: z.boolean().optional(),
  components: z.record(
    z.string(), // key type
    z.object({
      status: z.string(),
      latency_ms: z.number().optional(),
    })
  ).optional(),
  version: z.string().optional(),
  uptime: z.number().optional(),
}).passthrough();

/**
 * Contract for /api/chains endpoint
 */
const ChainsResponseContract = z.object({
  success: z.boolean(),
  chains: z.array(z.object({
    id: z.number(),
    name: z.string(),
    symbol: z.string().optional(),
    gasPrice: z.number().optional(),
    gas_price: z.number().optional(),
  })).optional(),
}).passthrough();

/**
 * Contract for WebSocket gas price update
 */
const WebSocketGasUpdateContract = z.object({
  type: z.literal('gas_update').optional(),
  event: z.literal('gas_price').optional(),
  data: z.object({
    gas_price: z.number().positive().optional(),
    gasPrice: z.number().positive().optional(),
    timestamp: z.string().or(z.number()).optional(),
    chain_id: z.number().optional(),
  }).optional(),
}).passthrough();

/**
 * Contract for error responses
 */
const ErrorResponseContract = z.object({
  success: z.literal(false).optional(),
  error: z.string().optional(),
  message: z.string().optional(),
  code: z.string().or(z.number()).optional(),
  details: z.any().optional(),
}).passthrough();

// ============================================================================
// Contract Validation Helper
// ============================================================================

interface ContractValidationResult {
  valid: boolean;
  errors?: z.ZodError;
  data?: unknown;
}

function validateContract<T extends z.ZodSchema>(
  schema: T,
  data: unknown
): ContractValidationResult {
  const result = schema.safeParse(data);
  if (result.success) {
    return { valid: true, data: result.data };
  }
  return { valid: false, errors: result.error };
}

// ============================================================================
// Mock API Responses (representing actual backend responses)
// ============================================================================

const mockResponses = {
  gas: {
    valid: {
      success: true,
      current_gas: 25.5,
      gas_price: 25.5,
      timestamp: '2024-01-15T12:00:00Z',
      network: 'mainnet',
    },
    invalid: {
      success: 'yes', // Should be boolean
      current_gas: -5, // Should be positive
    },
  },
  predictions: {
    valid: {
      success: true,
      predictions: {
        '1h': [
          { time: '13:00', predictedGwei: 24.5, confidence: 0.85 },
          { time: '14:00', predictedGwei: 23.8, confidence: 0.82 },
        ],
        '4h': [],
        '24h': [],
      },
      model_version: '2.1.0',
      generated_at: '2024-01-15T12:00:00Z',
    },
    invalid: {
      success: true,
      predictions: 'not an object', // Should be object
    },
  },
  accuracy: {
    valid: {
      success: true,
      metrics: {
        mae: 0.0003,
        rmse: 0.0004,
        r2: 0.85,
        directional_accuracy: 0.72,
        mape: 2.5,
      },
      horizon: '1h',
      period: '7d',
    },
    invalid: {
      success: true,
      metrics: {
        mae: -0.5, // Should be non-negative
        directional_accuracy: 1.5, // Should be 0-1
      },
    },
  },
  health: {
    valid: {
      status: 'healthy',
      success: true,
      components: {
        database: { status: 'up', latency_ms: 5 },
        cache: { status: 'up', latency_ms: 1 },
      },
      version: '1.0.0',
      uptime: 86400,
    },
    invalid: {
      status: 'unknown', // Not a valid enum value
    },
  },
  chains: {
    valid: {
      success: true,
      chains: [
        { id: 1, name: 'Ethereum', symbol: 'ETH', gasPrice: 25.5 },
        { id: 8453, name: 'Base', symbol: 'ETH', gasPrice: 0.5 },
      ],
    },
    invalid: {
      success: true,
      chains: [{ name: 'Missing ID' }], // Missing required id
    },
  },
  websocket: {
    valid: {
      type: 'gas_update',
      data: {
        gas_price: 25.5,
        timestamp: '2024-01-15T12:00:00Z',
        chain_id: 1,
      },
    },
    invalid: {
      type: 'unknown_type',
      data: { gas_price: 'not a number' },
    },
  },
  error: {
    valid: {
      success: false,
      error: 'Rate limit exceeded',
      code: 'RATE_LIMIT',
      details: { retry_after: 60 },
    },
  },
};

// ============================================================================
// Contract Tests
// ============================================================================

describe('API Contract Tests', () => {
  describe('Gas API Contract', () => {
    it('should validate correct gas response', () => {
      const result = validateContract(GasResponseContract, mockResponses.gas.valid);
      expect(result.valid).toBe(true);
    });

    it('should reject invalid gas response', () => {
      const result = validateContract(GasResponseContract, mockResponses.gas.invalid);
      expect(result.valid).toBe(false);
      expect(result.errors).toBeDefined();
    });

    it('should handle empty response gracefully', () => {
      const result = validateContract(GasResponseContract, {});
      // Empty object should fail because success is required
      expect(result.valid).toBe(false);
    });

    it('should allow additional fields (forward compatibility)', () => {
      const responseWithExtra = {
        ...mockResponses.gas.valid,
        new_field: 'some value',
        another_field: 123,
      };
      const result = validateContract(GasResponseContract, responseWithExtra);
      expect(result.valid).toBe(true);
    });
  });

  describe('Predictions API Contract', () => {
    it('should validate correct predictions response', () => {
      const result = validateContract(
        PredictionsResponseContract,
        mockResponses.predictions.valid
      );
      expect(result.valid).toBe(true);
    });

    it('should reject invalid predictions structure', () => {
      const result = validateContract(
        PredictionsResponseContract,
        mockResponses.predictions.invalid
      );
      expect(result.valid).toBe(false);
    });

    it('should validate prediction array items', () => {
      const response = {
        success: true,
        predictions: {
          '1h': [
            { time: '13:00', predictedGwei: 24.5, confidence: 0.85 },
          ],
        },
      };
      const result = validateContract(PredictionsResponseContract, response);
      expect(result.valid).toBe(true);
    });
  });

  describe('Accuracy API Contract', () => {
    it('should validate correct accuracy response', () => {
      const result = validateContract(
        AccuracyResponseContract,
        mockResponses.accuracy.valid
      );
      expect(result.valid).toBe(true);
    });

    it('should reject out-of-range values', () => {
      const result = validateContract(
        AccuracyResponseContract,
        mockResponses.accuracy.invalid
      );
      expect(result.valid).toBe(false);
    });
  });

  describe('Health API Contract', () => {
    it('should validate correct health response', () => {
      const result = validateContract(
        HealthResponseContract,
        mockResponses.health.valid
      );
      expect(result.valid).toBe(true);
    });

    it('should accept various status values', () => {
      // Health API can return various status strings
      const result = validateContract(
        HealthResponseContract,
        mockResponses.health.invalid
      );
      // With flexible status, this should pass
      expect(result.valid).toBe(true);
    });

    it('should reject non-string status values', () => {
      const invalidStatus = {
        status: 123, // Should be string
        success: true,
      };
      const result = validateContract(HealthResponseContract, invalidStatus);
      expect(result.valid).toBe(false);
    });
  });

  describe('Chains API Contract', () => {
    it('should validate correct chains response', () => {
      const result = validateContract(
        ChainsResponseContract,
        mockResponses.chains.valid
      );
      expect(result.valid).toBe(true);
    });

    it('should reject chains with missing required fields', () => {
      const result = validateContract(
        ChainsResponseContract,
        mockResponses.chains.invalid
      );
      expect(result.valid).toBe(false);
    });
  });

  describe('WebSocket Message Contract', () => {
    it('should validate correct WebSocket gas update', () => {
      const result = validateContract(
        WebSocketGasUpdateContract,
        mockResponses.websocket.valid
      );
      expect(result.valid).toBe(true);
    });

    it('should handle various message formats', () => {
      const alternativeFormat = {
        event: 'gas_price',
        data: {
          gasPrice: 25.5,
          timestamp: 1705320000,
        },
      };
      const result = validateContract(WebSocketGasUpdateContract, alternativeFormat);
      expect(result.valid).toBe(true);
    });
  });

  describe('Error Response Contract', () => {
    it('should validate error responses', () => {
      const result = validateContract(
        ErrorResponseContract,
        mockResponses.error.valid
      );
      expect(result.valid).toBe(true);
    });

    it('should accept various error formats', () => {
      const formats = [
        { success: false, error: 'Error message' },
        { success: false, message: 'Error message' },
        { success: false, error: 'Error', code: 500 },
        { success: false, error: 'Error', code: 'ERR_CODE' },
      ];

      formats.forEach((format) => {
        const result = validateContract(ErrorResponseContract, format);
        expect(result.valid).toBe(true);
      });
    });
  });

  describe('Contract Versioning', () => {
    it('should handle backward compatible changes', () => {
      // Old response format (fewer fields)
      const oldFormat = {
        success: true,
        current_gas: 25.5,
      };
      const result = validateContract(GasResponseContract, oldFormat);
      expect(result.valid).toBe(true);
    });

    it('should handle forward compatible changes', () => {
      // New response format (more fields)
      const newFormat = {
        success: true,
        current_gas: 25.5,
        gas_price: 25.5,
        timestamp: '2024-01-15T12:00:00Z',
        // New fields that old client doesn't know about
        priority_fee: 1.5,
        max_fee: 30.0,
        base_fee: 24.0,
      };
      const result = validateContract(GasResponseContract, newFormat);
      expect(result.valid).toBe(true);
    });
  });

  describe('Edge Cases', () => {
    it('should handle null values', () => {
      const withNull = {
        success: true,
        current_gas: null,
      };
      const result = validateContract(GasResponseContract, withNull);
      // null is not a positive number
      expect(result.valid).toBe(false);
    });

    it('should handle undefined values', () => {
      const withUndefined = {
        success: true,
        current_gas: undefined,
      };
      const result = validateContract(GasResponseContract, withUndefined);
      // undefined is allowed for optional fields
      expect(result.valid).toBe(true);
    });

    it('should handle very large numbers', () => {
      const withLargeNumber = {
        success: true,
        current_gas: Number.MAX_SAFE_INTEGER,
      };
      const result = validateContract(GasResponseContract, withLargeNumber);
      expect(result.valid).toBe(true);
    });

    it('should handle scientific notation', () => {
      const withScientific = {
        success: true,
        current_gas: 2.5e10,
      };
      const result = validateContract(GasResponseContract, withScientific);
      expect(result.valid).toBe(true);
    });
  });
});

describe('Contract Schema Exports', () => {
  it('should export valid Zod schemas', () => {
    expect(GasResponseContract).toBeDefined();
    expect(PredictionsResponseContract).toBeDefined();
    expect(AccuracyResponseContract).toBeDefined();
    expect(HealthResponseContract).toBeDefined();
    expect(ChainsResponseContract).toBeDefined();
    expect(WebSocketGasUpdateContract).toBeDefined();
    expect(ErrorResponseContract).toBeDefined();
  });

  it('should provide type inference', () => {
    type GasResponse = z.infer<typeof GasResponseContract>;
    type PredictionsResponse = z.infer<typeof PredictionsResponseContract>;

    // Type assertions (compile-time check)
    const gas: GasResponse = mockResponses.gas.valid;
    const predictions: PredictionsResponse = mockResponses.predictions.valid;

    expect(gas.success).toBe(true);
    expect(predictions.success).toBe(true);
  });
});
