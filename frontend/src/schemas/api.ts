/**
 * Zod schemas for API response validation
 * Use these schemas to validate API responses at runtime
 */

import { z } from 'zod';

// ========================================
// Base Schemas
// ========================================

export const BiasCorrection = z.object({
  applied: z.boolean(),
  type: z.enum(['time_period', 'overall']).optional(),
  period: z.enum(['night', 'morning', 'afternoon', 'evening']).optional(),
  correction: z.number().optional(),
  original: z.number().optional(),
  corrected: z.number().optional()
});

export const Probabilities = z.object({
  wait: z.number().min(0).max(1).optional(),
  normal: z.number().min(0).max(1).optional(),
  urgent: z.number().min(0).max(1).optional(),
  elevated: z.number().min(0).max(1).optional(),
  spike: z.number().min(0).max(1).optional()
});

// ========================================
// Gas Data Schemas
// ========================================

export const CurrentGasDataSchema = z.object({
  timestamp: z.string(),
  current_gas: z.number(),
  base_fee: z.number(),
  priority_fee: z.number(),
  block_number: z.number()
});

export const GraphDataPointSchema = z.object({
  time: z.string(),
  gwei: z.number().optional(),
  predictedGwei: z.number().optional(),
  lowerBound: z.number().optional(),
  upperBound: z.number().optional(),
  confidence: z.number().min(0).max(1).optional(),
  confidenceLevel: z.enum(['high', 'medium', 'low']).optional(),
  confidenceEmoji: z.string().optional(),
  confidenceColor: z.string().optional(),
  trend_signal_4h: z.number().min(-1).max(1).optional(),
  bias_correction: BiasCorrection.optional(),
  classification: z.object({
    class: z.string().optional(),
    emoji: z.string().optional(),
    color: z.string().optional(),
    probabilities: Probabilities.optional(),
    trend_signal_4h: z.number().optional()
  }).optional(),
  probabilities: Probabilities.optional()
});

export const PredictionsResponseSchema = z.object({
  current: CurrentGasDataSchema,
  predictions: z.object({
    '1h': z.array(GraphDataPointSchema),
    '4h': z.array(GraphDataPointSchema),
    '24h': z.array(GraphDataPointSchema),
    historical: z.array(GraphDataPointSchema)
  }),
  model_info: z.record(z.object({
    name: z.string(),
    mae: z.number()
  })).optional()
});

// ========================================
// Hybrid Prediction Schema
// ========================================

export const HybridPredictionSchema = z.object({
  action: z.enum(['WAIT', 'NORMAL', 'URGENT']),
  confidence: z.number().min(0).max(1),
  trend_signal_4h: z.number().min(-1).max(1),
  probabilities: z.object({
    wait: z.number().min(0).max(1),
    normal: z.number().min(0).max(1),
    urgent: z.number().min(0).max(1)
  })
});

// ========================================
// Config & Stats Schemas
// ========================================

export const ConfigResponseSchema = z.object({
  chains: z.array(z.object({
    id: z.number(),
    name: z.string(),
    enabled: z.boolean()
  })),
  features: z.record(z.boolean()),
  version: z.string()
});

export const AccuracyResponseSchema = z.object({
  horizons: z.record(z.object({
    mae: z.number(),
    rmse: z.number().optional(),
    r2: z.number().optional(),
    directional_accuracy: z.number().optional(),
    n_samples: z.number()
  })),
  overall: z.object({
    mae: z.number(),
    rmse: z.number().optional(),
    r2: z.number().optional()
  }),
  updated_at: z.string()
});

export const GlobalStatsResponseSchema = z.object({
  total_users: z.number(),
  total_transactions: z.number(),
  total_savings_usd: z.number(),
  predictions_made: z.number(),
  average_accuracy: z.number(),
  active_chains: z.number()
});

// ========================================
// Helper Types (inferred from schemas)
// ========================================

export type CurrentGasDataValidated = z.infer<typeof CurrentGasDataSchema>;
export type GraphDataPointValidated = z.infer<typeof GraphDataPointSchema>;
export type PredictionsResponseValidated = z.infer<typeof PredictionsResponseSchema>;
export type HybridPredictionValidated = z.infer<typeof HybridPredictionSchema>;
export type ConfigResponseValidated = z.infer<typeof ConfigResponseSchema>;
export type AccuracyResponseValidated = z.infer<typeof AccuracyResponseSchema>;
export type GlobalStatsResponseValidated = z.infer<typeof GlobalStatsResponseSchema>;

// ========================================
// Validation Helpers
// ========================================

/**
 * Safely parse and validate API response
 * Returns the validated data or null if validation fails
 */
export function validateApiResponse<T>(
  schema: z.ZodSchema<T>,
  data: unknown
): T | null {
  const result = schema.safeParse(data);
  if (result.success) {
    return result.data;
  }
  console.warn('API response validation failed:', result.error.format());
  return null;
}

/**
 * Parse API response with fallback
 * Returns validated data or the fallback value if validation fails
 */
export function validateWithFallback<T>(
  schema: z.ZodSchema<T>,
  data: unknown,
  fallback: T
): T {
  const result = schema.safeParse(data);
  return result.success ? result.data : fallback;
}
