import { trackEvent } from './analytics';

export type SecurityAuditEvent =
  | 'csrf_token_missing'
  | 'api_key_rotated'
  | 'https_redirect'
  | 'session_expired'
  | 'suspicious_input';

export type SensitiveAction =
  | 'wallet_connect'
  | 'wallet_disconnect'
  | 'wallet_export'
  | 'settings_update'
  | 'account_switch'
  | 'api_token_viewed';

export interface SecurityAuditPayload extends Record<string, string | number | boolean | null | undefined> {
  detail?: string;
  route?: string;
  statusCode?: number;
}

export interface SensitiveActionPayload extends Record<string, string | number | boolean | null | undefined> {
  detail?: string;
  route?: string;
  resourceId?: string;
}

export function logSecurityEvent(event: SecurityAuditEvent, payload: SecurityAuditPayload = {}): void {
  trackEvent(`security:${event}`, payload);
}

export function logSensitiveAction(
  action: SensitiveAction,
  payload: SensitiveActionPayload = {}
): void {
  trackEvent('audit:sensitive_action', { action, ...payload });
}
