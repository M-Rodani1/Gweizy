/**
 * Toast Notification Provider
 * Wraps app with toast notifications
 */

import React, { ReactNode } from 'react';
import { Toaster, useToasterStore } from 'react-hot-toast';

interface ToastProviderProps {
  children: ReactNode;
}

const srOnlyStyles: React.CSSProperties = {
  position: 'absolute',
  width: '1px',
  height: '1px',
  padding: 0,
  margin: '-1px',
  overflow: 'hidden',
  clip: 'rect(0, 0, 0, 0)',
  border: 0,
  whiteSpace: 'nowrap',
};

function getToastText(message: unknown): string {
  if (typeof message === 'string' || typeof message === 'number') {
    return String(message);
  }

  if (React.isValidElement<{ children?: ReactNode }>(message)) {
    const { children } = message.props;
    if (typeof children === 'string' || typeof children === 'number') {
      return String(children);
    }
    if (Array.isArray(children)) {
      return children
        .filter((child) => typeof child === 'string' || typeof child === 'number')
        .map(String)
        .join(' ');
    }
  }

  return '';
}

const ToastAnnouncer: React.FC = () => {
  const { toasts } = useToasterStore();
  const announcement = toasts
    .filter((toast) => toast.visible)
    .map((toast) => getToastText(toast.message))
    .filter(Boolean)
    .join(' ');

  return (
    <div role="status" aria-live="polite" aria-atomic="true" style={srOnlyStyles}>
      {announcement}
    </div>
  );
};

export const ToastProvider: React.FC<ToastProviderProps> = ({ children }) => {
  return (
    <>
      {children}
      <ToastAnnouncer />
      <Toaster
        position="top-right"
        toastOptions={{
          duration: 4000,
          style: {
            background: '#1f2937',
            color: '#fff',
            border: '1px solid #374151',
          },
          success: {
            iconTheme: {
              primary: '#10b981',
              secondary: '#fff',
            },
          },
          error: {
            iconTheme: {
              primary: '#ef4444',
              secondary: '#fff',
            },
          },
        }}
      />
    </>
  );
};
