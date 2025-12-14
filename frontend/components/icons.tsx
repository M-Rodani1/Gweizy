import React from 'react';

export const EthereumIcon: React.FC<{ className?: string }> = ({ className }) => (
  <svg className={className} viewBox="0 0 32 32" xmlns="http://www.w3.org/2000/svg" fill="currentColor">
    <path d="M16 32C7.163 32 0 24.837 0 16S7.163 0 16 0s16 7.163 16 16-7.163 16-16 16zm-.225-18.413l-6.5 3.75V16l6.5 3.75 6.5-3.75V13.587l-6.5 3.75zm0-1.5l6.5-3.75-6.5-3.75-6.5 3.75 6.5 3.75z"/>
  </svg>
);

export const PolygonIcon: React.FC<{ className?: string }> = ({ className }) => (
  <svg className={className} viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg" fill="currentColor">
    <path d="M6.2 4.6l5.8 3.35v6.7L6.2 11.3zm11.6 0l-5.8 3.35v6.7l5.8-3.35zM12 2.5l-7.73 4.46v8.92L12 21.34l7.73-4.46v-8.92z"/>
  </svg>
);

export const ArbitrumIcon: React.FC<{ className?: string }> = ({ className }) => (
  <svg className={className} viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <path d="M12 2 L2 7 L12 12 L22 7 Z"/>
    <path d="M2 17 L12 22 L22 17"/>
    <path d="M2 12 L12 17 L22 12"/>
  </svg>
);

export const OptimismIcon: React.FC<{ className?: string }> = ({ className }) => (
  <svg className={className} viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg" fill="currentColor">
    <path d="M12,2A10,10,0,1,0,22,12,10,10,0,0,0,12,2Zm5.66,8.25-3.5,2a.5.5,0,0,1-.5.05L10,10.5V14a.5.5,0,0,1-1,0V9.5a.5.5,0,0,1,.25-.43l4-2.25a.5.5,0,0,1,.5.05l3.5,2A.5.5,0,0,1,17.66,10.25Z"/>
  </svg>
);

export const GasIcon: React.FC<{ className?: string }> = ({ className }) => (
    <svg xmlns="http://www.w3.org/2000/svg" className={className} fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
        <path strokeLinecap="round" strokeLinejoin="round" d="M12 6V4m0 2a2 2 0 100 4m0-4a2 2 0 110 4m-6 8a2 2 0 100-4m0 4a2 2 0 110-4m0 4v2m0-6V4m6 6v10m6-2a2 2 0 100-4m0 4a2 2 0 110-4m0 4v2m0-6V4" />
    </svg>
);
