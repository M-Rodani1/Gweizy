import React from 'react';

interface CardProps {
  title?: React.ReactNode;
  subtitle?: React.ReactNode;
  action?: React.ReactNode;
  padding?: 'sm' | 'md' | 'lg';
  children: React.ReactNode;
  role?: string;
  tabIndex?: number;
  className?: string;
}

const paddingMap = {
  sm: 'p-4',
  md: 'p-6',
  lg: 'p-8'
};

export const Card: React.FC<CardProps> = ({
  title,
  subtitle,
  action,
  padding = 'md',
  children,
  role,
  tabIndex,
  className = ''
}) => (
  <div
    className={`card bg-[var(--surface)] border border-[var(--border)] rounded-[var(--radius-lg)] shadow-card transition-all ${paddingMap[padding]} ${className}`}
    role={role}
    tabIndex={tabIndex}
  >
    {(title || action || subtitle) && (
      <header className="flex items-start justify-between gap-3 mb-4">
        <div>
          {title && <h3 className="text-base font-semibold text-[var(--text)]">{title}</h3>}
          {subtitle && <p className="text-sm text-[var(--text-secondary)] mt-1">{subtitle}</p>}
        </div>
        {action && <div className="shrink-0">{action}</div>}
      </header>
    )}
    {children}
  </div>
);

export default Card;
