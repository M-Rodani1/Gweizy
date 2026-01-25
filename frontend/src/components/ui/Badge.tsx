import React from 'react';

type BadgeVariant = 'accent' | 'success' | 'warning' | 'danger' | 'neutral';

interface BadgeProps {
  children: React.ReactNode;
  variant?: BadgeVariant;
  icon?: React.ReactNode;
  className?: string;
}

const variantStyles: Record<BadgeVariant, string> = {
  accent: 'bg-[var(--accent-light)] text-[var(--accent)] border border-[var(--accent-border)]',
  success: 'bg-[var(--success-bg)] text-[var(--success)] border border-[var(--success-border)]',
  warning: 'bg-[var(--warning-bg)] text-[var(--warning)] border border-[var(--warning-border)]',
  danger: 'bg-[var(--danger-bg)] text-[var(--danger)] border border-[var(--danger-border)]',
  neutral: 'bg-[var(--surface-2)] text-[var(--text-secondary)] border border-[var(--border)]'
};

export const Badge: React.FC<BadgeProps> = ({
  children,
  variant = 'accent',
  icon,
  className = ''
}) => (
  <span className={`inline-flex items-center gap-1 px-2.5 py-1 rounded-full text-xs font-semibold ${variantStyles[variant]} ${className}`}>
    {icon && <span className="shrink-0">{icon}</span>}
    {children}
  </span>
);

export default Badge;
