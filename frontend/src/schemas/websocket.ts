/**
 * WebSocket message validation schemas using Zod.
 *
 * Provides type-safe validation for all WebSocket messages
 * to ensure data integrity and prevent runtime errors.
 *
 * @module schemas/websocket
 */

import { z } from 'zod';

/**
 * Gas price update schema.
 */
export const GasPriceUpdateSchema = z.object({
  current_gas: z.number().nonnegative(),
  base_fee: z.number().nonnegative(),
  priority_fee: z.number().nonnegative(),
  timestamp: z.string(),
  collection_count: z.number().int().nonnegative(),
});

export type GasPriceUpdate = z.infer<typeof GasPriceUpdateSchema>;

/**
 * Prediction horizon data schema.
 */
export const PredictionHorizonSchema = z.object({
  price: z.number().nonnegative(),
  confidence: z.number().min(0).max(1),
  lower_bound: z.number().nonnegative(),
  upper_bound: z.number().nonnegative(),
});

export type PredictionHorizon = z.infer<typeof PredictionHorizonSchema>;

/**
 * Prediction update schema.
 */
export const PredictionUpdateSchema = z.object({
  current_price: z.number().nonnegative(),
  predictions: z.record(z.string(), PredictionHorizonSchema),
  timestamp: z.string(),
});

export type PredictionUpdate = z.infer<typeof PredictionUpdateSchema>;

/**
 * Mempool update schema.
 */
export const MempoolUpdateSchema = z.object({
  pending_count: z.number().int().nonnegative(),
  avg_gas_price: z.number().nonnegative(),
  is_congested: z.boolean(),
  gas_momentum: z.number(),
  count_momentum: z.number(),
  timestamp: z.string(),
});

export type MempoolUpdate = z.infer<typeof MempoolUpdateSchema>;

/**
 * Combined gas update schema.
 */
export const CombinedGasUpdateSchema = z.object({
  gas: GasPriceUpdateSchema,
  predictions: PredictionUpdateSchema.optional(),
  mempool: MempoolUpdateSchema.optional(),
  timestamp: z.string(),
});

export type CombinedGasUpdate = z.infer<typeof CombinedGasUpdateSchema>;

/**
 * Connection established message schema.
 */
export const ConnectionEstablishedSchema = z.object({
  message: z.string(),
});

export type ConnectionEstablished = z.infer<typeof ConnectionEstablishedSchema>;

/**
 * Error message schema.
 */
export const ErrorMessageSchema = z.object({
  error: z.string(),
  code: z.string().optional(),
  details: z.unknown().optional(),
});

export type ErrorMessage = z.infer<typeof ErrorMessageSchema>;

/**
 * All WebSocket message types.
 */
export type WebSocketMessage =
  | { type: 'gas_price_update'; data: GasPriceUpdate }
  | { type: 'prediction_update'; data: PredictionUpdate }
  | { type: 'mempool_update'; data: MempoolUpdate }
  | { type: 'combined_update'; data: CombinedGasUpdate }
  | { type: 'connection_established'; data: ConnectionEstablished }
  | { type: 'error'; data: ErrorMessage };

/**
 * Validation result type.
 */
export interface ValidationResult<T> {
  success: boolean;
  data?: T;
  error?: string;
}

/**
 * Validate a gas price update message.
 */
export function validateGasPriceUpdate(data: unknown): ValidationResult<GasPriceUpdate> {
  const result = GasPriceUpdateSchema.safeParse(data);
  if (result.success) {
    return { success: true, data: result.data };
  }
  return { success: false, error: formatZodError(result.error) };
}

/**
 * Validate a prediction update message.
 */
export function validatePredictionUpdate(data: unknown): ValidationResult<PredictionUpdate> {
  const result = PredictionUpdateSchema.safeParse(data);
  if (result.success) {
    return { success: true, data: result.data };
  }
  return { success: false, error: formatZodError(result.error) };
}

/**
 * Validate a mempool update message.
 */
export function validateMempoolUpdate(data: unknown): ValidationResult<MempoolUpdate> {
  const result = MempoolUpdateSchema.safeParse(data);
  if (result.success) {
    return { success: true, data: result.data };
  }
  return { success: false, error: formatZodError(result.error) };
}

/**
 * Validate a combined gas update message.
 */
export function validateCombinedUpdate(data: unknown): ValidationResult<CombinedGasUpdate> {
  const result = CombinedGasUpdateSchema.safeParse(data);
  if (result.success) {
    return { success: true, data: result.data };
  }
  return { success: false, error: formatZodError(result.error) };
}

/**
 * Validate any WebSocket message based on event type.
 */
export function validateWebSocketMessage(
  eventType: string,
  data: unknown
): ValidationResult<unknown> {
  switch (eventType) {
    case 'gas_price_update':
      return validateGasPriceUpdate(data);
    case 'prediction_update':
      return validatePredictionUpdate(data);
    case 'mempool_update':
      return validateMempoolUpdate(data);
    case 'combined_update':
      return validateCombinedUpdate(data);
    case 'connection_established':
      const connResult = ConnectionEstablishedSchema.safeParse(data);
      if (connResult.success) {
        return { success: true, data: connResult.data };
      }
      return { success: false, error: formatZodError(connResult.error) };
    default:
      return { success: true, data }; // Allow unknown event types to pass through
  }
}

/**
 * Format Zod error to a readable string.
 */
function formatZodError(error: z.ZodError): string {
  return error.issues
    .map((e: z.ZodIssue) => `${e.path.join('.')}: ${e.message}`)
    .join(', ');
}

/**
 * Create a validated message handler wrapper.
 *
 * @param schema - Zod schema for validation
 * @param handler - Handler function for valid messages
 * @param onError - Optional error handler
 * @returns Wrapped handler that validates before calling
 *
 * @example
 * ```ts
 * socket.on('gas_price_update', createValidatedHandler(
 *   GasPriceUpdateSchema,
 *   (data) => {
 *     // data is typed and validated
 *     setGasPrice(data);
 *   },
 *   (error) => console.error('Invalid message:', error)
 * ));
 * ```
 */
export function createValidatedHandler<T>(
  schema: z.ZodSchema<T>,
  handler: (data: T) => void,
  onError?: (error: string, rawData: unknown) => void
): (data: unknown) => void {
  return (data: unknown) => {
    const result = schema.safeParse(data);
    if (result.success) {
      handler(result.data);
    } else {
      const errorMessage = formatZodError(result.error);
      onError?.(errorMessage, data);
      console.warn('[WS Validation] Invalid message:', errorMessage, data);
    }
  };
}

export default {
  GasPriceUpdateSchema,
  PredictionUpdateSchema,
  MempoolUpdateSchema,
  CombinedGasUpdateSchema,
  validateGasPriceUpdate,
  validatePredictionUpdate,
  validateMempoolUpdate,
  validateCombinedUpdate,
  validateWebSocketMessage,
  createValidatedHandler,
};
