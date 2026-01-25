import React from 'react';

type ButtonVariant = 'primary' | 'secondary' | 'ghost' | 'outline' | 'danger' | 'success' | 'link';
type ButtonSize = 'sm' | 'md' | 'lg';

export interface ButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: ButtonVariant;
  size?: ButtonSize;
  fullWidth?: boolean;
  iconLeft?: React.ReactNode;
  iconRight?: React.ReactNode;
}

const sizeClasses: Record<ButtonSize, string> = {
  sm: 'px-3 py-2 text-sm rounded-md',
  md: 'px-4 py-2.5 text-sm rounded-lg',
  lg: 'px-5 py-3 text-base rounded-xl'
};

const variantClasses: Record<ButtonVariant, string> = {
  primary: 'bg-[var(--accent)] text-white hover:bg-[var(--accent-hover)] shadow-sm shadow-cyan-500/30',
  secondary: 'bg-[var(--surface-2)] text-[var(--text)] border border-[var(--border)] hover:border-[var(--accent-border)]',
  ghost: 'bg-transparent text-[var(--text-secondary)] hover:bg-[var(--surface-2)]',
  outline: 'border border-[var(--accent-border)] text-[var(--accent)] hover:bg-[var(--accent-light)]',
  danger: 'bg-[var(--danger)] text-white hover:bg-red-500',
  success: 'bg-[var(--success)] text-white hover:bg-emerald-500',
  link: 'bg-transparent text-[var(--accent)] hover:text-[var(--accent-hover)] underline underline-offset-4'
};

export const Button: React.FC<ButtonProps> = ({
  children,
  variant = 'primary',
  size = 'md',
  fullWidth = false,
  iconLeft,
  iconRight,
  className = '',
  ...props
}) => {
  return (
    <button
      className={`btn-reset inline-flex items-center justify-center gap-2 font-semibold transition-all duration-200 focus:outline-none focus-visible:ring-2 focus-visible:ring-[var(--accent)] focus-visible:ring-offset-2 focus-visible:ring-offset-[var(--bg)] ${sizeClasses[size]} ${variantClasses[variant]} ${fullWidth ? 'w-full' : ''} ${className}`}
      {...props}
    >
      {iconLeft && <span className="shrink-0">{iconLeft}</span>}
      <span className="whitespace-nowrap">{children}</span>
      {iconRight && <span className="shrink-0">{iconRight}</span>}
    </button>
  );
};

export default Button;
