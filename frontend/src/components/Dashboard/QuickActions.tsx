import React, { useState } from 'react';
import { SystemHealth, ProcessingStatus } from '@/types';
import {
  FolderPlusIcon,
  PlayIcon,
  StopIcon,
  CogIcon,
  DocumentMagnifyingGlassIcon,
  ArrowUpTrayIcon,
  ChartBarIcon,
  WrenchScrewdriverIcon
} from '@heroicons/react/24/outline';

interface QuickActionsProps {
  systemHealth: SystemHealth | null;
  processingStatus: ProcessingStatus | null;
  className?: string;
}

interface ActionButtonProps {
  icon: React.ReactNode;
  label: string;
  description: string;
  onClick: () => void;
  disabled?: boolean;
  variant?: 'primary' | 'secondary' | 'danger' | 'success';
  className?: string;
}

const ActionButton: React.FC<ActionButtonProps> = ({
  icon,
  label,
  description,
  onClick,
  disabled = false,
  variant = 'secondary',
  className = ''
}) => {
  const getVariantClasses = () => {
    const baseClasses = "group relative p-4 rounded-lg border transition-all duration-200 disabled:opacity-50 disabled:cursor-not-allowed";
    
    switch (variant) {
      case 'primary':
        return `${baseClasses} bg-blue-50 border-blue-200 hover:bg-blue-100 dark:bg-blue-900/20 dark:border-blue-800 dark:hover:bg-blue-900/30 text-blue-700 dark:text-blue-300`;
      case 'success':
        return `${baseClasses} bg-green-50 border-green-200 hover:bg-green-100 dark:bg-green-900/20 dark:border-green-800 dark:hover:bg-green-900/30 text-green-700 dark:text-green-300`;
      case 'danger':
        return `${baseClasses} bg-red-50 border-red-200 hover:bg-red-100 dark:bg-red-900/20 dark:border-red-800 dark:hover:bg-red-900/30 text-red-700 dark:text-red-300`;
      default:
        return `${baseClasses} bg-white border-gray-200 hover:bg-gray-50 dark:bg-gray-700 dark:border-gray-600 dark:hover:bg-gray-600 text-gray-700 dark:text-gray-300`;
    }
  };

  return (
    <button
      onClick={onClick}
      disabled={disabled}
      className={`${getVariantClasses()} ${className}`}
    >
      <div className="flex items-start space-x-3">
        <div className="flex-shrink-0">
          {icon}
        </div>
        <div className="flex-1 text-left">
          <h4 className="font-medium text-sm mb-1">
            {label}
          </h4>
          <p className="text-xs opacity-75">
            {description}
          </p>
        </div>
      </div>
    </button>
  );
};

export const QuickActions: React.FC<QuickActionsProps> = ({ 
  systemHealth, 
  processingStatus, 
  className = '' 
}) => {
  const [showUploadModal, setShowUploadModal] = useState(false);
  
  // Determine system status
  const isSystemHealthy = systemHealth?.status === 'healthy';
  const isProcessing = processingStatus?.current_status === 'processing';
  const hasDocuments = processingStatus && processingStatus.total_documents > 0;
  
  const handleAddFolder = () => {
    // Navigate to folder addition page or open modal
    console.log('Add folder clicked');
  };

  const handleStartProcessing = () => {
    // Start processing documents
    console.log('Start processing clicked');
  };

  const handleStopProcessing = () => {
    // Stop processing
    console.log('Stop processing clicked');
  };

  const handleUploadDocuments = () => {
    setShowUploadModal(true);
  };

  const handleSearchDocuments = () => {
    // Navigate to search page
    console.log('Search documents clicked');
  };

  const handleViewAnalytics = () => {
    // Navigate to analytics page
    console.log('View analytics clicked');
  };

  const handleSystemConfiguration = () => {
    // Navigate to configuration page
    console.log('System configuration clicked');
  };

  const handleSystemMaintenance = () => {
    // Navigate to maintenance page
    console.log('System maintenance clicked');
  };

  return (
    <div className={`bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 p-6 ${className}`}>
      <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-6">
        Quick Actions
      </h3>

      <div className="space-y-3">
        {/* Primary Actions */}
        <ActionButton
          icon={<FolderPlusIcon className="h-5 w-5" />}
          label="Add Folder"
          description="Connect Google Drive folder for processing"
          onClick={handleAddFolder}
          variant="primary"
          disabled={!isSystemHealthy}
        />

        <ActionButton
          icon={<ArrowUpTrayIcon className="h-5 w-5" />}
          label="Upload Documents"
          description="Upload individual files for processing"
          onClick={handleUploadDocuments}
          variant="primary"
          disabled={!isSystemHealthy}
        />

        {/* Processing Controls */}
        {hasDocuments && !isProcessing && (
          <ActionButton
            icon={<PlayIcon className="h-5 w-5" />}
            label="Start Processing"
            description="Begin processing queued documents"
            onClick={handleStartProcessing}
            variant="success"
            disabled={!isSystemHealthy}
          />
        )}

        {isProcessing && (
          <ActionButton
            icon={<StopIcon className="h-5 w-5" />}
            label="Stop Processing"
            description="Pause document processing"
            onClick={handleStopProcessing}
            variant="danger"
          />
        )}

        {/* Search and Analytics */}
        {hasDocuments && (
          <ActionButton
            icon={<DocumentMagnifyingGlassIcon className="h-5 w-5" />}
            label="Search Documents"
            description="Search through processed documents"
            onClick={handleSearchDocuments}
          />
        )}

        <ActionButton
          icon={<ChartBarIcon className="h-5 w-5" />}
          label="View Analytics"
          description="Detailed cost and usage analytics"
          onClick={handleViewAnalytics}
        />

        {/* System Management */}
        <div className="pt-4 border-t border-gray-200 dark:border-gray-600">
          <ActionButton
            icon={<CogIcon className="h-5 w-5" />}
            label="Configuration"
            description="System settings and API keys"
            onClick={handleSystemConfiguration}
          />

          <div className="mt-3">
            <ActionButton
              icon={<WrenchScrewdriverIcon className="h-5 w-5" />}
              label="Maintenance"
              description="System health and diagnostics"
              onClick={handleSystemMaintenance}
              disabled={systemHealth?.status === 'unhealthy'}
            />
          </div>
        </div>
      </div>

      {/* System Status Banner */}
      {!isSystemHealthy && (
        <div className="mt-6 p-3 bg-yellow-50 dark:bg-yellow-900/20 border border-yellow-200 dark:border-yellow-800 rounded-lg">
          <div className="flex items-center space-x-2">
            <WrenchScrewdriverIcon className="h-5 w-5 text-yellow-500" />
            <div>
              <p className="text-sm font-medium text-yellow-800 dark:text-yellow-200">
                System Status: {systemHealth?.status || 'Unknown'}
              </p>
              <p className="text-xs text-yellow-600 dark:text-yellow-300 mt-1">
                Some actions may be unavailable until system health improves
              </p>
            </div>
          </div>
        </div>
      )}

      {/* Processing Info */}
      {processingStatus && (
        <div className="mt-6 p-3 bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-800 rounded-lg">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-blue-800 dark:text-blue-200">
                Processing Status
              </p>
              <p className="text-xs text-blue-600 dark:text-blue-300 mt-1">
                {processingStatus.processed_documents} / {processingStatus.total_documents} completed
              </p>
            </div>
            <div className="text-right">
              <p className="text-sm font-medium text-blue-800 dark:text-blue-200">
                {processingStatus.processing_rate.toFixed(1)} docs/min
              </p>
            </div>
          </div>
        </div>
      )}

      {/* Upload Modal Placeholder */}
      {showUploadModal && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
          <div className="bg-white dark:bg-gray-800 rounded-lg p-6 max-w-md w-full mx-4">
            <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
              Upload Documents
            </h3>
            <p className="text-gray-600 dark:text-gray-400 mb-6">
              Upload functionality will be implemented with file picker and drag-drop support.
            </p>
            <div className="flex justify-end space-x-3">
              <button
                onClick={() => setShowUploadModal(false)}
                className="px-4 py-2 text-gray-600 dark:text-gray-400 hover:text-gray-800 dark:hover:text-gray-200"
              >
                Cancel
              </button>
              <button
                onClick={() => setShowUploadModal(false)}
                className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
              >
                Got it
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default QuickActions;