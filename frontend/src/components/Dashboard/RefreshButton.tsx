import React from 'react';
import { ArrowPathIcon } from '@heroicons/react/24/outline';

interface RefreshButtonProps {
  onRefresh: () => void;
  refreshing?: boolean;
  disabled?: boolean;
  className?: string;
}

export const RefreshButton: React.FC<RefreshButtonProps> = ({
  onRefresh,
  refreshing = false,
  disabled = false,
  className = ''
}) => {
  return (
    <button
      onClick={onRefresh}
      disabled={disabled || refreshing}
      className={`
        inline-flex items-center px-3 py-2 border border-gray-300 dark:border-gray-600 
        shadow-sm text-sm font-medium rounded-md text-gray-700 dark:text-gray-200 
        bg-white dark:bg-gray-700 hover:bg-gray-50 dark:hover:bg-gray-600
        focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500
        disabled:opacity-50 disabled:cursor-not-allowed
        transition-all duration-200
        ${className}
      `}
      title={disabled ? 'Refresh unavailable' : 'Refresh data'}
    >
      <ArrowPathIcon 
        className={`h-4 w-4 mr-2 ${refreshing ? 'animate-spin' : ''}`} 
      />
      {refreshing ? 'Refreshing...' : 'Refresh'}
    </button>
  );
};

export default RefreshButton;