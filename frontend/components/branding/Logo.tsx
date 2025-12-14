import React from 'react';

interface LogoProps {
  size?: 'sm' | 'md' | 'lg';
  className?: string;
}

export const Logo: React.FC<LogoProps> = ({ size = 'md', className = '' }) => {
  const sizes = {
    sm: 'w-6 h-6',
    md: 'w-8 h-8',
    lg: 'w-12 h-12'
  };

  return (
    <svg 
      className={`${sizes[size]} ${className}`} 
      viewBox="0 0 100 100" 
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
    >
      {/* Lightning bolt + Gas pump icon */}
      <defs>
        <linearGradient id="logoGradient" x1="0%" y1="0%" x2="100%" y2="100%">
          <stop offset="0%" stopColor="#06B6D4" />
          <stop offset="100%" stopColor="#10B981" />
        </linearGradient>
      </defs>
      <path 
        d="M50 10 L30 50 H45 L40 90 L70 45 H55 L50 10Z" 
        fill="url(#logoGradient)" 
      />
      {/* Gas pump base */}
      <rect x="40" y="70" width="20" height="15" rx="2" fill="url(#logoGradient)" opacity="0.8" />
    </svg>
  );
};

export default Logo;

