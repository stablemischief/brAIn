import React, { useState } from 'react';
import { motion } from 'framer-motion';
import {
  DocumentArrowDownIcon,
  DocumentTextIcon,
  TableCellsIcon,
  ChartBarIcon,
  CalendarIcon,
  Cog6ToothIcon,
  CheckCircleIcon,
  ExclamationTriangleIcon,
} from '@heroicons/react/24/outline';
import { CostAnalytics, DailyCost } from '@/types';
import { format } from 'date-fns';

interface ExportFormat {
  id: string;
  name: string;
  extension: string;
  icon: React.ComponentType<{ className?: string }>;
  description: string;
  size?: string;
}

interface ExportOptions {
  format: string;
  timeRange: 'last7days' | 'last30days' | 'last90days' | 'custom';
  includeCharts: boolean;
  includeProjections: boolean;
  includeRecommendations: boolean;
  customStartDate?: string;
  customEndDate?: string;
}

interface ExportControlsProps {
  costData: CostAnalytics;
  historicalData: DailyCost[];
  onExport: (options: ExportOptions) => Promise<{ url: string; filename: string }>;
  className?: string;
}

export const ExportControls: React.FC<ExportControlsProps> = ({
  costData,
  historicalData,
  onExport,
  className = '',
}) => {
  const [isOpen, setIsOpen] = useState(false);
  const [exporting, setExporting] = useState<string | null>(null);
  const [options, setOptions] = useState<ExportOptions>({
    format: 'pdf',
    timeRange: 'last30days',
    includeCharts: true,
    includeProjections: false,
    includeRecommendations: false,
  });

  const exportFormats: ExportFormat[] = [
    {
      id: 'pdf',
      name: 'PDF Report',
      extension: 'pdf',
      icon: DocumentTextIcon,
      description: 'Comprehensive report with charts and analysis',
      size: '~2-5 MB',
    },
    {
      id: 'csv',
      name: 'CSV Data',
      extension: 'csv',
      icon: TableCellsIcon,
      description: 'Raw data for further analysis',
      size: '~50-200 KB',
    },
    {
      id: 'excel',
      name: 'Excel Workbook',
      extension: 'xlsx',
      icon: TableCellsIcon,
      description: 'Multi-sheet workbook with formulas',
      size: '~100-500 KB',
    },
    {
      id: 'json',
      name: 'JSON Data',
      extension: 'json',
      icon: Cog6ToothIcon,
      description: 'Structured data for API integration',
      size: '~10-100 KB',
    },
  ];

  const timeRangeOptions = [
    { value: 'last7days', label: 'Last 7 days' },
    { value: 'last30days', label: 'Last 30 days' },
    { value: 'last90days', label: 'Last 90 days' },
    { value: 'custom', label: 'Custom range' },
  ];

  const handleExport = async (formatId: string) => {
    setExporting(formatId);
    try {
      const exportOptions = { ...options, format: formatId };
      const result = await onExport(exportOptions);
      
      // Create download link
      const link = document.createElement('a');
      link.href = result.url;
      link.download = result.filename;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      
      // Show success message or close modal
      setIsOpen(false);
    } catch (error) {
      console.error('Export failed:', error);
      // Handle error - could show toast notification
    } finally {
      setExporting(null);
    }
  };

  const getEstimatedSize = () => {
    const baseSize = historicalData.length * 0.1; // KB
    let multiplier = 1;
    
    if (options.includeCharts) multiplier += 0.5;
    if (options.includeProjections) multiplier += 0.3;
    if (options.includeRecommendations) multiplier += 0.2;
    
    switch (options.format) {
      case 'pdf':
        return `~${Math.round(baseSize * multiplier * 10)}KB - ${Math.round(baseSize * multiplier * 20)}KB`;
      case 'csv':
        return `~${Math.round(baseSize * multiplier * 2)}KB`;
      case 'excel':
        return `~${Math.round(baseSize * multiplier * 5)}KB`;
      case 'json':
        return `~${Math.round(baseSize * multiplier)}KB`;
      default:
        return 'Unknown';
    }
  };

  const getPreview = () => {
    const selectedFormat = exportFormats.find(f => f.id === options.format);
    const timeRangeLabel = timeRangeOptions.find(t => t.value === options.timeRange)?.label || 'Custom';
    
    const sections = [];
    if (options.includeCharts) sections.push('Charts');
    if (options.includeProjections) sections.push('Projections');
    if (options.includeRecommendations) sections.push('Recommendations');
    
    return {
      format: selectedFormat?.name || 'Unknown',
      timeRange: timeRangeLabel,
      sections: sections.length > 0 ? sections : ['Basic data only'],
      estimatedSize: getEstimatedSize(),
    };
  };

  if (!isOpen) {
    return (
      <button
        onClick={() => setIsOpen(true)}
        className={`flex items-center gap-2 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors ${className}`}
      >
        <DocumentArrowDownIcon className="h-4 w-4" />
        Export
      </button>
    );
  }

  return (
    <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50 p-4">
      <motion.div
        initial={{ opacity: 0, scale: 0.95 }}
        animate={{ opacity: 1, scale: 1 }}
        className="bg-white dark:bg-gray-800 rounded-lg shadow-xl max-w-4xl w-full max-h-[90vh] overflow-hidden"
      >
        {/* Header */}
        <div className="flex items-center justify-between p-6 border-b border-gray-200 dark:border-gray-700">
          <div className="flex items-center gap-3">
            <DocumentArrowDownIcon className="h-6 w-6 text-blue-600 dark:text-blue-400" />
            <div>
              <h2 className="text-xl font-semibold text-gray-900 dark:text-white">
                Export Cost Analytics
              </h2>
              <p className="text-sm text-gray-600 dark:text-gray-400">
                Generate reports and download data in various formats
              </p>
            </div>
          </div>
          
          <button
            onClick={() => setIsOpen(false)}
            className="text-gray-400 hover:text-gray-600 dark:hover:text-gray-300 transition-colors"
          >
            <svg className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        </div>

        <div className="flex max-h-[calc(90vh-120px)]">
          {/* Left Panel - Options */}
          <div className="flex-1 p-6 overflow-y-auto border-r border-gray-200 dark:border-gray-700">
            {/* Format Selection */}
            <div className="mb-6">
              <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
                Export Format
              </h3>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                {exportFormats.map((format) => {
                  const Icon = format.icon;
                  return (
                    <label
                      key={format.id}
                      className={`flex items-start gap-3 p-4 border rounded-lg cursor-pointer transition-colors ${
                        options.format === format.id
                          ? 'border-blue-500 bg-blue-50 dark:bg-blue-900/20'
                          : 'border-gray-200 dark:border-gray-600 hover:bg-gray-50 dark:hover:bg-gray-700'
                      }`}
                    >
                      <input
                        type="radio"
                        name="format"
                        value={format.id}
                        checked={options.format === format.id}
                        onChange={(e) => setOptions(prev => ({ ...prev, format: e.target.value }))}
                        className="mt-1"
                      />
                      <Icon className="h-5 w-5 text-gray-600 dark:text-gray-400 mt-0.5" />
                      <div className="flex-1">
                        <p className="font-medium text-gray-900 dark:text-white">
                          {format.name}
                        </p>
                        <p className="text-sm text-gray-600 dark:text-gray-400">
                          {format.description}
                        </p>
                        {format.size && (
                          <p className="text-xs text-gray-500 dark:text-gray-500 mt-1">
                            {format.size}
                          </p>
                        )}
                      </div>
                    </label>
                  );
                })}
              </div>
            </div>

            {/* Time Range */}
            <div className="mb-6">
              <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
                Time Range
              </h3>
              <div className="space-y-3">
                {timeRangeOptions.map((range) => (
                  <label key={range.value} className="flex items-center">
                    <input
                      type="radio"
                      name="timeRange"
                      value={range.value}
                      checked={options.timeRange === range.value}
                      onChange={(e) => setOptions(prev => ({ ...prev, timeRange: e.target.value as any }))}
                      className="h-4 w-4 text-blue-600 focus:ring-blue-500"
                    />
                    <span className="ml-3 text-sm font-medium text-gray-700 dark:text-gray-300">
                      {range.label}
                    </span>
                  </label>
                ))}
              </div>

              {options.timeRange === 'custom' && (
                <div className="mt-4 grid grid-cols-2 gap-4">
                  <div>
                    <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                      Start Date
                    </label>
                    <input
                      type="date"
                      value={options.customStartDate || ''}
                      onChange={(e) => setOptions(prev => ({ ...prev, customStartDate: e.target.value }))}
                      className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent dark:bg-gray-700 dark:text-white"
                    />
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                      End Date
                    </label>
                    <input
                      type="date"
                      value={options.customEndDate || ''}
                      onChange={(e) => setOptions(prev => ({ ...prev, customEndDate: e.target.value }))}
                      className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent dark:bg-gray-700 dark:text-white"
                    />
                  </div>
                </div>
              )}
            </div>

            {/* Content Options */}
            {(options.format === 'pdf' || options.format === 'excel') && (
              <div className="mb-6">
                <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
                  Include in Export
                </h3>
                <div className="space-y-3">
                  <label className="flex items-center">
                    <input
                      type="checkbox"
                      checked={options.includeCharts}
                      onChange={(e) => setOptions(prev => ({ ...prev, includeCharts: e.target.checked }))}
                      className="h-4 w-4 text-blue-600 focus:ring-blue-500 rounded"
                    />
                    <span className="ml-3 text-sm font-medium text-gray-700 dark:text-gray-300">
                      Charts and Visualizations
                    </span>
                  </label>

                  <label className="flex items-center">
                    <input
                      type="checkbox"
                      checked={options.includeProjections}
                      onChange={(e) => setOptions(prev => ({ ...prev, includeProjections: e.target.checked }))}
                      className="h-4 w-4 text-blue-600 focus:ring-blue-500 rounded"
                    />
                    <span className="ml-3 text-sm font-medium text-gray-700 dark:text-gray-300">
                      Cost Projections
                    </span>
                  </label>

                  <label className="flex items-center">
                    <input
                      type="checkbox"
                      checked={options.includeRecommendations}
                      onChange={(e) => setOptions(prev => ({ ...prev, includeRecommendations: e.target.checked }))}
                      className="h-4 w-4 text-blue-600 focus:ring-blue-500 rounded"
                    />
                    <span className="ml-3 text-sm font-medium text-gray-700 dark:text-gray-300">
                      Optimization Recommendations
                    </span>
                  </label>
                </div>
              </div>
            )}
          </div>

          {/* Right Panel - Preview */}
          <div className="w-80 p-6 bg-gray-50 dark:bg-gray-900">
            <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
              Export Preview
            </h3>
            
            <div className="space-y-4">
              <div className="p-4 bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700">
                <div className="flex items-center gap-2 mb-2">
                  <DocumentTextIcon className="h-4 w-4 text-blue-600 dark:text-blue-400" />
                  <span className="text-sm font-medium text-gray-900 dark:text-white">
                    {getPreview().format}
                  </span>
                </div>
                
                <div className="text-xs space-y-1">
                  <div className="flex justify-between">
                    <span className="text-gray-600 dark:text-gray-400">Time Range:</span>
                    <span className="text-gray-900 dark:text-white">{getPreview().timeRange}</span>
                  </div>
                  
                  <div className="flex justify-between">
                    <span className="text-gray-600 dark:text-gray-400">Est. Size:</span>
                    <span className="text-gray-900 dark:text-white">{getPreview().estimatedSize}</span>
                  </div>
                </div>
              </div>

              <div className="p-4 bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700">
                <h4 className="text-sm font-medium text-gray-900 dark:text-white mb-2">
                  Content Sections
                </h4>
                <div className="space-y-1">
                  {getPreview().sections.map((section, index) => (
                    <div key={index} className="flex items-center gap-2 text-xs">
                      <CheckCircleIcon className="h-3 w-3 text-green-600 dark:text-green-400" />
                      <span className="text-gray-700 dark:text-gray-300">{section}</span>
                    </div>
                  ))}
                </div>
              </div>

              {/* Data Summary */}
              <div className="p-4 bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700">
                <h4 className="text-sm font-medium text-gray-900 dark:text-white mb-2">
                  Data Summary
                </h4>
                <div className="text-xs space-y-1">
                  <div className="flex justify-between">
                    <span className="text-gray-600 dark:text-gray-400">Total Cost:</span>
                    <span className="text-gray-900 dark:text-white">${costData.total_cost.toFixed(2)}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-600 dark:text-gray-400">Days:</span>
                    <span className="text-gray-900 dark:text-white">{historicalData.length}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-600 dark:text-gray-400">Models:</span>
                    <span className="text-gray-900 dark:text-white">
                      {Object.keys(costData.cost_by_model).length}
                    </span>
                  </div>
                </div>
              </div>

              {/* Warning for Large Exports */}
              {(options.includeCharts || options.timeRange === 'last90days') && (
                <div className="p-3 bg-yellow-50 dark:bg-yellow-900/20 rounded-lg border border-yellow-200 dark:border-yellow-800">
                  <div className="flex items-start gap-2">
                    <ExclamationTriangleIcon className="h-4 w-4 text-yellow-600 dark:text-yellow-400 mt-0.5 flex-shrink-0" />
                    <div>
                      <p className="text-xs font-medium text-yellow-700 dark:text-yellow-300">
                        Large Export
                      </p>
                      <p className="text-xs text-yellow-600 dark:text-yellow-400">
                        This export may take longer to generate due to the amount of data or included charts.
                      </p>
                    </div>
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>

        {/* Footer */}
        <div className="flex items-center justify-between p-6 border-t border-gray-200 dark:border-gray-700">
          <button
            onClick={() => setIsOpen(false)}
            className="px-4 py-2 text-gray-700 dark:text-gray-300 bg-gray-100 dark:bg-gray-700 rounded-lg hover:bg-gray-200 dark:hover:bg-gray-600 transition-colors"
          >
            Cancel
          </button>
          
          <button
            onClick={() => handleExport(options.format)}
            disabled={exporting === options.format}
            className="flex items-center gap-2 px-6 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
          >
            {exporting === options.format ? (
              <div className="animate-spin rounded-full h-4 w-4 border-2 border-white border-t-transparent" />
            ) : (
              <DocumentArrowDownIcon className="h-4 w-4" />
            )}
            {exporting === options.format ? 'Generating...' : `Export ${getPreview().format}`}
          </button>
        </div>
      </motion.div>
    </div>
  );
};

export default ExportControls;