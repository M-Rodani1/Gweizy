import React from 'react';

const LoadingSpinner: React.FC<{ message?: string }> = ({ message = 'Loading...' }) => {
  return (
    <div
      className="flex flex-col items-center justify-center p-8"
      role="status"
      aria-live="polite"
      aria-atomic="true"
    >
      <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-cyan-400" aria-hidden="true"></div>
      <p className="mt-4 text-gray-400">{message}</p>
    </div>
  );
};

export default LoadingSpinner;
