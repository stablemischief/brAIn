import React from 'react';
import { Info, AlertCircle } from 'lucide-react';

interface EnvironmentStepProps {
  config: any;
  errors: Record<string, string>;
  onUpdate: (updates: any) => void;
  isValidating?: boolean;
}

export const EnvironmentStep: React.FC<EnvironmentStepProps> = ({
  config,
  errors,
  onUpdate,
  isValidating
}) => {
  const handleChange = (field: string, value: any) => {
    onUpdate({ ...config, [field]: value });
  };

  return (
    <div className="space-y-6">
      {/* Environment Selection */}
      <div>
        <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
          Environment Type
        </label>
        <select
          value={config.environment || 'development'}
          onChange={(e) => handleChange('environment', e.target.value)}
          className="w-full px-4 py-2 border border-gray-300 dark:border-gray-600 rounded-lg
                   bg-white dark:bg-gray-800 text-gray-900 dark:text-gray-100
                   focus:ring-2 focus:ring-blue-500 focus:border-transparent"
        >
          <option value="development">Development</option>
          <option value="staging">Staging</option>
          <option value="production">Production</option>
        </select>
        {errors.environment && (
          <p className="mt-1 text-sm text-red-600 dark:text-red-400">{errors.environment}</p>
        )}
      </div>

      {/* Application Name */}
      <div>
        <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
          Application Name
        </label>
        <input
          type="text"
          value={config.appName || ''}
          onChange={(e) => handleChange('appName', e.target.value)}
          placeholder="my-brain-app"
          className="w-full px-4 py-2 border border-gray-300 dark:border-gray-600 rounded-lg
                   bg-white dark:bg-gray-800 text-gray-900 dark:text-gray-100
                   focus:ring-2 focus:ring-blue-500 focus:border-transparent"
        />
        {errors.appName && (
          <p className="mt-1 text-sm text-red-600 dark:text-red-400">{errors.appName}</p>
        )}
      </div>

      {/* Port Configuration */}
      <div className="grid grid-cols-2 gap-4">
        <div>
          <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
            Backend Port
          </label>
          <input
            type="number"
            value={config.backendPort || 8000}
            onChange={(e) => handleChange('backendPort', parseInt(e.target.value))}
            className="w-full px-4 py-2 border border-gray-300 dark:border-gray-600 rounded-lg
                     bg-white dark:bg-gray-800 text-gray-900 dark:text-gray-100
                     focus:ring-2 focus:ring-blue-500 focus:border-transparent"
          />
          {errors.backendPort && (
            <p className="mt-1 text-sm text-red-600 dark:text-red-400">{errors.backendPort}</p>
          )}
        </div>
        <div>
          <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
            Frontend Port
          </label>
          <input
            type="number"
            value={config.frontendPort || 3000}
            onChange={(e) => handleChange('frontendPort', parseInt(e.target.value))}
            className="w-full px-4 py-2 border border-gray-300 dark:border-gray-600 rounded-lg
                     bg-white dark:bg-gray-800 text-gray-900 dark:text-gray-100
                     focus:ring-2 focus:ring-blue-500 focus:border-transparent"
          />
          {errors.frontendPort && (
            <p className="mt-1 text-sm text-red-600 dark:text-red-400">{errors.frontendPort}</p>
          )}
        </div>
      </div>

      {/* Base URL */}
      <div>
        <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
          Base URL
        </label>
        <input
          type="url"
          value={config.baseUrl || ''}
          onChange={(e) => handleChange('baseUrl', e.target.value)}
          placeholder="http://localhost:3000"
          className="w-full px-4 py-2 border border-gray-300 dark:border-gray-600 rounded-lg
                   bg-white dark:bg-gray-800 text-gray-900 dark:text-gray-100
                   focus:ring-2 focus:ring-blue-500 focus:border-transparent"
        />
        {errors.baseUrl && (
          <p className="mt-1 text-sm text-red-600 dark:text-red-400">{errors.baseUrl}</p>
        )}
      </div>

      {/* Debug Mode */}
      <div className="flex items-center space-x-3">
        <input
          type="checkbox"
          id="debugMode"
          checked={config.debugMode || false}
          onChange={(e) => handleChange('debugMode', e.target.checked)}
          className="w-4 h-4 text-blue-600 rounded border-gray-300 focus:ring-blue-500"
        />
        <label htmlFor="debugMode" className="text-sm font-medium text-gray-700 dark:text-gray-300">
          Enable Debug Mode
        </label>
      </div>

      {/* Environment Variables Preview */}
      <div className="mt-6 p-4 bg-gray-50 dark:bg-gray-900 rounded-lg">
        <div className="flex items-start space-x-2">
          <Info className="w-5 h-5 text-blue-500 mt-0.5" />
          <div className="flex-1">
            <p className="text-sm font-medium text-gray-700 dark:text-gray-300">
              Environment Variables Preview
            </p>
            <pre className="mt-2 text-xs text-gray-600 dark:text-gray-400 font-mono">
{`ENVIRONMENT=${config.environment || 'development'}
APP_NAME=${config.appName || 'my-brain-app'}
BACKEND_PORT=${config.backendPort || 8000}
FRONTEND_PORT=${config.frontendPort || 3000}
BASE_URL=${config.baseUrl || 'http://localhost:3000'}
DEBUG=${config.debugMode ? 'true' : 'false'}`}
            </pre>
          </div>
        </div>
      </div>

      {/* Info Box */}
      {config.environment === 'production' && (
        <div className="p-4 bg-yellow-50 dark:bg-yellow-900/20 rounded-lg">
          <div className="flex items-start space-x-2">
            <AlertCircle className="w-5 h-5 text-yellow-600 dark:text-yellow-400 mt-0.5" />
            <div>
              <p className="text-sm font-medium text-yellow-800 dark:text-yellow-200">
                Production Environment Selected
              </p>
              <p className="mt-1 text-sm text-yellow-700 dark:text-yellow-300">
                Make sure to configure SSL certificates and secure API keys in the following steps.
              </p>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};