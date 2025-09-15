import React from 'react';
import { ProcessingStatus } from '../../types';
import {
  DocumentTextIcon,
  ClockIcon,
  CheckCircleIcon,
  XCircleIcon,
  ArrowPathIcon,
  PlayIcon,
  PauseIcon,
  ExclamationCircleIcon
} from '@heroicons/react/24/outline';

interface ProcessingStatusCardProps {
  status: ProcessingStatus | null;
  connected: boolean;
  className?: string;
}

interface ProgressBarProps {
  current: number;
  total: number;
  className?: string;
}

const ProgressBar: React.FC<ProgressBarProps> = ({ current, total, className = '' }) => {
  const percentage = total > 0 ? (current / total) * 100 : 0;
  
  return (
    <div className={`w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2 ${className}`}>
      <div 
        className="bg-blue-600 dark:bg-blue-400 h-2 rounded-full transition-all duration-300 ease-out"
        style={{ width: `${Math.min(100, Math.max(0, percentage))}%` }}
      />
    </div>
  );
};

interface StatusBadgeProps {
  status: ProcessingStatus['current_status'];
  className?: string;
}

const StatusBadge: React.FC<StatusBadgeProps> = ({ status, className = '' }) => {
  const getStatusConfig = () => {
    switch (status) {
      case 'idle':
        return {
          icon: <PauseIcon className="h-4 w-4" />,
          text: 'Idle',
          colors: 'bg-gray-100 text-gray-800 dark:bg-gray-800 dark:text-gray-200'
        };
      case 'processing':
        return {
          icon: <ArrowPathIcon className="h-4 w-4 animate-spin" />,
          text: 'Processing',
          colors: 'bg-blue-100 text-blue-800 dark:bg-blue-900 dark:text-blue-200'
        };
      case 'error':
        return {
          icon: <ExclamationCircleIcon className="h-4 w-4" />,
          text: 'Error',
          colors: 'bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-200'
        };
      default:
        return {
          icon: <ClockIcon className="h-4 w-4" />,
          text: 'Unknown',
          colors: 'bg-gray-100 text-gray-800 dark:bg-gray-800 dark:text-gray-200'
        };
    }
  };

  const config = getStatusConfig();

  return (
    <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${config.colors} ${className}`}>
      {config.icon}
      <span className="ml-1">{config.text}</span>
    </span>
  );
};

interface StatCardProps {
  icon: React.ReactNode;
  label: string;
  value: number;
  color: string;
  className?: string;
}

const StatCard: React.FC<StatCardProps> = ({ icon, label, value, color, className = '' }) => (
  <div className={`bg-white dark:bg-gray-700 p-4 rounded-lg border border-gray-200 dark:border-gray-600 ${className}`}>
    <div className="flex items-center">
      <div className={`flex-shrink-0 ${color}`}>
        {icon}
      </div>
      <div className="ml-3">
        <p className="text-sm text-gray-500 dark:text-gray-400">{label}</p>
        <p className="text-2xl font-semibold text-gray-900 dark:text-white">
          {value.toLocaleString()}
        </p>
      </div>
    </div>
  </div>
);

export const ProcessingStatusCard: React.FC<ProcessingStatusCardProps> = ({ 
  status, 
  connected, 
  className = '' 
}) => {
  if (!connected) {
    return (
      <div className={`bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 p-6 ${className}`}>
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-lg font-semibold text-gray-900 dark:text-white">
            Processing Status
          </h3>
          <StatusBadge status="error" />
        </div>
        
        <div className="text-center py-8">
          <ExclamationCircleIcon className="h-12 w-12 text-gray-400 mx-auto mb-4" />
          <p className="text-gray-500 dark:text-gray-400">
            Connection lost - unable to retrieve processing status
          </p>
        </div>
      </div>
    );
  }

  if (!status) {
    return (
      <div className={`bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 p-6 ${className}`}>
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-lg font-semibold text-gray-900 dark:text-white">
            Processing Status
          </h3>
          <div className="flex items-center space-x-2">
            <ArrowPathIcon className="h-4 w-4 text-gray-400 animate-spin" />
            <span className="text-sm text-gray-500 dark:text-gray-400">Loading...</span>
          </div>
        </div>
        
        <div className="space-y-4">
          <div className="animate-pulse">
            <div className="h-4 bg-gray-200 dark:bg-gray-700 rounded w-3/4 mb-2"></div>
            <div className="h-2 bg-gray-200 dark:bg-gray-700 rounded"></div>
          </div>
        </div>
      </div>
    );
  }

  const completionPercentage = status.total_documents > 0 
    ? (status.processed_documents / status.total_documents) * 100 
    : 0;

  const estimatedCompletion = status.estimated_completion 
    ? new Date(status.estimated_completion) 
    : null;

  const isProcessing = status.current_status === 'processing';
  const hasError = status.current_status === 'error';
  const isIdle = status.current_status === 'idle';

  return (
    <div className={`bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 p-6 ${className}`}>
      {/* Header */}
      <div className="flex items-center justify-between mb-6">
        <h3 className="text-lg font-semibold text-gray-900 dark:text-white">
          Processing Status
        </h3>
        <StatusBadge status={status.current_status} />
      </div>

      {/* Main Progress Section */}
      <div className="mb-6">
        <div className="flex items-center justify-between mb-2">
          <span className="text-sm font-medium text-gray-700 dark:text-gray-300">
            Overall Progress
          </span>
          <span className="text-sm text-gray-500 dark:text-gray-400">
            {status.processed_documents} / {status.total_documents} documents
          </span>
        </div>
        
        <ProgressBar 
          current={status.processed_documents} 
          total={status.total_documents}
          className="mb-2" 
        />
        
        <div className="flex items-center justify-between text-sm text-gray-500 dark:text-gray-400">
          <span>{completionPercentage.toFixed(1)}% complete</span>
          {estimatedCompletion && (
            <span>
              ETA: {estimatedCompletion.toLocaleString()}
            </span>
          )}
        </div>
      </div>

      {/* Statistics Grid */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
        <StatCard
          icon={<DocumentTextIcon className="h-6 w-6" />}
          label="Total"
          value={status.total_documents}
          color="text-gray-600 dark:text-gray-400"
        />
        
        <StatCard
          icon={<CheckCircleIcon className="h-6 w-6" />}
          label="Processed"
          value={status.processed_documents}
          color="text-green-600 dark:text-green-400"
        />
        
        <StatCard
          icon={<ArrowPathIcon className="h-6 w-6" />}
          label="Processing"
          value={status.processing_documents}
          color="text-blue-600 dark:text-blue-400"
        />
        
        <StatCard
          icon={<XCircleIcon className="h-6 w-6" />}
          label="Failed"
          value={status.failed_documents}
          color="text-red-600 dark:text-red-400"
        />
      </div>

      {/* Processing Rate */}
      <div className="flex items-center justify-between p-4 bg-gray-50 dark:bg-gray-700/50 rounded-lg">
        <div className="flex items-center space-x-3">
          <div className={`p-2 rounded-lg ${
            isProcessing ? 'bg-blue-100 text-blue-600 dark:bg-blue-900/20 dark:text-blue-400' :
            hasError ? 'bg-red-100 text-red-600 dark:bg-red-900/20 dark:text-red-400' :
            'bg-gray-100 text-gray-600 dark:bg-gray-800 dark:text-gray-400'
          }`}>
            {isProcessing ? <PlayIcon className="h-5 w-5" /> :
             hasError ? <ExclamationCircleIcon className="h-5 w-5" /> :
             <PauseIcon className="h-5 w-5" />}
          </div>
          
          <div>
            <p className="text-sm font-medium text-gray-900 dark:text-white">
              Processing Rate
            </p>
            <p className="text-xs text-gray-500 dark:text-gray-400">
              Documents per minute
            </p>
          </div>
        </div>
        
        <div className="text-right">
          <p className="text-lg font-semibold text-gray-900 dark:text-white">
            {status.processing_rate.toFixed(1)}
          </p>
          <p className="text-xs text-gray-500 dark:text-gray-400">
            docs/min
          </p>
        </div>
      </div>

      {/* Status Messages */}
      {hasError && (
        <div className="mt-4 p-3 bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg">
          <div className="flex items-center space-x-2">
            <ExclamationCircleIcon className="h-5 w-5 text-red-500" />
            <p className="text-sm text-red-700 dark:text-red-300">
              Processing has encountered errors. Check system health for details.
            </p>
          </div>
        </div>
      )}

      {isIdle && status.total_documents === 0 && (
        <div className="mt-4 p-3 bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-800 rounded-lg">
          <div className="flex items-center space-x-2">
            <DocumentTextIcon className="h-5 w-5 text-blue-500" />
            <p className="text-sm text-blue-700 dark:text-blue-300">
              No documents to process. Add folders or upload documents to begin.
            </p>
          </div>
        </div>
      )}
    </div>
  );
};

export default ProcessingStatusCard;