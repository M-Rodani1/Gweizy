let sentryRef: any = null;

export const setSentryRef = (sentry: any) => {
  sentryRef = sentry;
};

type AnalyticsPayload = Record<string, string | number | boolean | null | undefined>;

export const trackEvent = (event: string, payload: AnalyticsPayload = {}) => {
  const data = { event, ...payload };

  if (sentryRef?.captureMessage) {
    sentryRef.captureMessage(event, {
      level: 'info',
      tags: payload,
      extra: payload
    });
  } else if ((window as any).Sentry?.captureMessage) {
    (window as any).Sentry.captureMessage(event, { level: 'info', tags: payload, extra: payload });
  } else {
    // eslint-disable-next-line no-console
    console.debug('[analytics]', data);
  }
};
