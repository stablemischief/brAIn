import React from 'react';
import { LoadingProps } from '../types';

export const LoadingSpinner: React.FC<LoadingProps> = ({
  size = 'md',
  color = 'text-blue-600',
  text = 'Loading...',
  className = ''
}) => {
  const getSizeClasses = () => {
    switch (size) {
      case 'sm':
        return 'h-4 w-4';
      case 'lg':
        return 'h-8 w-8';
      default:
        return 'h-6 w-6';
    }
  };

  return (
    <div className={`flex flex-col items-center justify-center p-4 ${className}`}>
      <div className={`animate-spin rounded-full border-2 border-gray-300 dark:border-gray-700 border-t-current ${getSizeClasses()} ${color} mb-2`} />
      {text && (
        <p className="text-sm text-gray-600 dark:text-gray-400">
          {text}
        </p>
      )}
    </div>
  );
};

export default LoadingSpinner;