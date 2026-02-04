import React, { useState } from 'react';

interface CollapsibleSectionProps {
  title: string;
  icon?: React.ReactNode;
  defaultOpen?: boolean;
  defaultExpanded?: boolean; // Alias for defaultOpen
  badge?: string | number;
  children: React.ReactNode;
}

const CollapsibleSection: React.FC<CollapsibleSectionProps> = ({
  title,
  icon,
  defaultOpen = false,
  defaultExpanded,
  badge,
  children
}) => {
  const [isOpen, setIsOpen] = useState(defaultExpanded ?? defaultOpen);

  return (
    <div className="bg-gray-800/50 border border-gray-700 rounded-xl overflow-hidden">
      {/* Header - clickable */}
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="w-full px-4 py-3 flex items-center justify-between hover:bg-gray-700/30 transition-colors"
      >
        <div className="flex items-center gap-3">
          {icon && <span className="text-lg">{icon}</span>}
          <span className="font-semibold text-white">{title}</span>
          {badge !== undefined && (
            <span className="px-2 py-0.5 text-xs bg-cyan-500/20 text-cyan-400 rounded-full">
              {badge}
            </span>
          )}
        </div>
        <svg
          className={`w-5 h-5 text-gray-400 transition-transform duration-200 ${isOpen ? 'rotate-180' : ''}`}
          fill="none"
          viewBox="0 0 24 24"
          stroke="currentColor"
        >
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
        </svg>
      </button>

      {/* Content */}
      <div className={`collapsible-content ${isOpen ? 'expanded' : 'collapsed'}`}>
        <div className="border-t border-gray-700/50">
          {children}
        </div>
      </div>
    </div>
  );
};

export default CollapsibleSection;
