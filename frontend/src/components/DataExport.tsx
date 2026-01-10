import React, { useState } from 'react';
import { Download, FileJson, FileSpreadsheet } from 'lucide-react';
import { API_CONFIG, getApiUrl } from '../config/api';

interface DataExportProps {
  className?: string;
}

const DataExport: React.FC<DataExportProps> = ({ className = '' }) => {
  const [hours, setHours] = useState(24);
  const [loading, setLoading] = useState(false);

  const handleExport = async (format: 'csv' | 'json') => {
    setLoading(true);
    try {
      const url = getApiUrl(API_CONFIG.ENDPOINTS.EXPORT, { format, hours });

      if (format === 'csv') {
        // Download CSV directly
        window.open(url, '_blank');
      } else {
        // Fetch JSON and download as file
        const response = await fetch(url);
        const data = await response.json();

        const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
        const downloadUrl = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = downloadUrl;
        a.download = `gas_data_${hours}h.json`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(downloadUrl);
      }
    } catch (error) {
      console.error('Export failed:', error);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className={`bg-gray-800/50 rounded-xl p-4 border border-gray-700/50 ${className}`}>
      <div className="flex items-center gap-2 mb-3">
        <Download className="w-4 h-4 text-cyan-400" />
        <span className="text-sm font-medium text-white">Export Data</span>
      </div>

      <div className="flex items-center gap-2 mb-3">
        <select
          value={hours}
          onChange={(e) => setHours(Number(e.target.value))}
          className="flex-1 bg-gray-700 border border-gray-600 rounded-lg px-3 py-2 text-sm text-white focus:outline-none focus:border-cyan-500"
        >
          <option value={1}>Last 1 hour</option>
          <option value={6}>Last 6 hours</option>
          <option value={24}>Last 24 hours</option>
          <option value={168}>Last 7 days</option>
          <option value={720}>Last 30 days</option>
        </select>
      </div>

      <div className="flex gap-2">
        <button
          onClick={() => handleExport('csv')}
          disabled={loading}
          className="flex-1 flex items-center justify-center gap-2 px-3 py-2 bg-green-600 hover:bg-green-700 disabled:bg-gray-600 text-white text-sm rounded-lg transition-colors"
        >
          <FileSpreadsheet className="w-4 h-4" />
          CSV
        </button>
        <button
          onClick={() => handleExport('json')}
          disabled={loading}
          className="flex-1 flex items-center justify-center gap-2 px-3 py-2 bg-blue-600 hover:bg-blue-700 disabled:bg-gray-600 text-white text-sm rounded-lg transition-colors"
        >
          <FileJson className="w-4 h-4" />
          JSON
        </button>
      </div>
    </div>
  );
};

export default DataExport;
