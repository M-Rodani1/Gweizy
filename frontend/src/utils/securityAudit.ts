import { trackEvent } from './analytics';

export type SecurityAuditEvent =
  | 'csrf_token_missing'
  | 'api_key_rotated'
  | 'https_redirect'
  | 'session_expired'
  | 'suspicious_input';

export interface SecurityAuditPayload {
  detail?: string;
  route?: string;
  statusCode?: number;
}

export function logSecurityEvent(event: SecurityAuditEvent, payload: SecurityAuditPayload = {}): void {
  trackEvent(`security:${event}`, payload);
}
