// Push Notification utilities for Gas Alerts

export type NotificationPermission = 'granted' | 'denied' | 'default';

export interface GasAlert {
  id: number;
  alert_type: 'below' | 'above';
  threshold_gwei: number;
  is_active: boolean;
}

// Check if browser supports notifications
export function isNotificationSupported(): boolean {
  return 'Notification' in window;
}

// Get current permission status
export function getNotificationPermission(): NotificationPermission {
  if (!isNotificationSupported()) return 'denied';
  return Notification.permission;
}

// Request notification permission
export async function requestNotificationPermission(): Promise<NotificationPermission> {
  if (!isNotificationSupported()) {
    console.warn('Notifications not supported in this browser');
    return 'denied';
  }

  try {
    const permission = await Notification.requestPermission();
    return permission;
  } catch (error) {
    console.error('Failed to request notification permission:', error);
    return 'denied';
  }
}

// Send a gas alert notification
export function sendGasAlertNotification(
  currentGas: number,
  alert: GasAlert
): void {
  if (getNotificationPermission() !== 'granted') return;

  const direction = alert.alert_type === 'below' ? 'dropped below' : 'rose above';
  const title = `Gas Price Alert`;
  const body = `Gas ${direction} ${alert.threshold_gwei.toFixed(4)} gwei! Current: ${currentGas.toFixed(4)} gwei`;
  const icon = '/logo.svg';

  try {
    const notification = new Notification(title, {
      body,
      icon,
      badge: icon,
      tag: `gas-alert-${alert.id}`, // Prevents duplicate notifications
      requireInteraction: true, // Notification stays until user interacts
      data: {
        alertId: alert.id,
        currentGas,
        threshold: alert.threshold_gwei
      }
    });

    // Auto-close after 10 seconds
    setTimeout(() => notification.close(), 10000);

    // Handle notification click
    notification.onclick = () => {
      window.focus();
      notification.close();
    };
  } catch (error) {
    console.error('Failed to send notification:', error);
  }
}

// Check if an alert should trigger based on current gas price
export function shouldTriggerAlert(
  currentGas: number,
  alert: GasAlert,
  previousGas: number | null
): boolean {
  if (!alert.is_active) return false;
  if (currentGas <= 0) return false;

  // Prevent triggering on initial load (when previousGas is null)
  if (previousGas === null) return false;

  if (alert.alert_type === 'below') {
    // Trigger when gas crosses below threshold (was above, now below)
    return previousGas >= alert.threshold_gwei && currentGas < alert.threshold_gwei;
  } else {
    // Trigger when gas crosses above threshold (was below, now above)
    return previousGas <= alert.threshold_gwei && currentGas > alert.threshold_gwei;
  }
}

// Store for tracking triggered alerts (to prevent duplicate notifications)
const triggeredAlerts = new Map<number, number>();

// Check and trigger alerts with cooldown
export function checkAndTriggerAlerts(
  currentGas: number,
  alerts: GasAlert[],
  previousGas: number | null
): void {
  const now = Date.now();
  const COOLDOWN_MS = 5 * 60 * 1000; // 5 minute cooldown between same alert

  for (const alert of alerts) {
    if (!shouldTriggerAlert(currentGas, alert, previousGas)) continue;

    const lastTriggered = triggeredAlerts.get(alert.id);
    if (lastTriggered && now - lastTriggered < COOLDOWN_MS) {
      continue; // Still in cooldown
    }

    // Trigger the notification
    sendGasAlertNotification(currentGas, alert);
    triggeredAlerts.set(alert.id, now);

    console.log(`Alert triggered: ${alert.alert_type} ${alert.threshold_gwei} gwei`);
  }
}

// Clear triggered alerts cache (useful for testing)
export function clearTriggeredAlerts(): void {
  triggeredAlerts.clear();
}
