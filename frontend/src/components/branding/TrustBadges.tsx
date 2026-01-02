import React from 'react';

interface TrustBadgeProps {
  icon: string;
  text: string;
}

export const TrustBadge: React.FC<TrustBadgeProps> = ({ icon, text }) => (
  <div className="flex items-center space-x-2 text-gray-400 text-sm">
    <span>{icon}</span>
    <span>{text}</span>
  </div>
);

export default TrustBadge;

