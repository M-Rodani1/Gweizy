import React from 'react';

interface SectionHeaderProps {
  eyebrow?: string;
  title: string;
  description?: string;
  action?: React.ReactNode;
  align?: 'left' | 'center';
}

export const SectionHeader: React.FC<SectionHeaderProps> = ({
  eyebrow,
  title,
  description,
  action,
  align = 'left'
}) => (
  <div className={`flex ${align === 'center' ? 'flex-col items-center text-center' : 'flex-col gap-2'} w-full`}>
    <div className="flex w-full items-start justify-between gap-2">
      <div className={`${align === 'center' ? 'w-full' : ''}`}>
        {eyebrow && <span className="text-xs font-semibold uppercase tracking-[0.12em] text-[var(--text-muted)]">{eyebrow}</span>}
        <h2 className="text-2xl font-semibold text-[var(--text)] leading-tight">{title}</h2>
        {description && <p className="text-[var(--text-secondary)] mt-2 max-w-2xl">{description}</p>}
      </div>
      {action && <div className="shrink-0">{action}</div>}
    </div>
  </div>
);

export default SectionHeader;
