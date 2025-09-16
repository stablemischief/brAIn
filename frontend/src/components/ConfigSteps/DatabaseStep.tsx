import React, { useState } from 'react';
import { Database, TestTube, Loader2, CheckCircle, XCircle } from 'lucide-react';

interface DatabaseStepProps {
  config: any;
  errors: Record<string, string>;
  onUpdate: (updates: any) => void;
  isValidating?: boolean;
}

export const DatabaseStep: React.FC<DatabaseStepProps> = ({
  config,
  errors,
  onUpdate,
  isValidating
}) => {
  const [testStatus, setTestStatus] = useState<'idle' | 'testing' | 'success' | 'error'>('idle');
  const [testMessage, setTestMessage] = useState('');

  const handleChange = (field: string, value: any) => {
    onUpdate({ ...config, database: { ...config.database, [field]: value } });
  };

  const testConnection = async () => {
    setTestStatus('testing');
    setTestMessage('Testing database connection...');

    try {
      // Simulate API call to test database connection
      const response = await fetch('/api/config/test-database', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(config.database)
      });

      if (response.ok) {
        setTestStatus('success');
        setTestMessage('Database connection successful!');
      } else {
        const error = await response.json();
        setTestStatus('error');
        setTestMessage(error.message || 'Connection failed');
      }
    } catch (error) {
      setTestStatus('error');
      setTestMessage('Failed to test connection. Check your settings.');
    }
  };

  return (
    <div className="space-y-6">
      {/* Database Type */}
      <div>
        <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
          Database Type
        </label>
        <select
          value={config.database?.type || 'postgresql'}
          onChange={(e) => handleChange('type', e.target.value)}
          className="w-full px-4 py-2 border border-gray-300 dark:border-gray-600 rounded-lg
                   bg-white dark:bg-gray-800 text-gray-900 dark:text-gray-100
                   focus:ring-2 focus:ring-blue-500 focus:border-transparent"
        >
          <option value="postgresql">PostgreSQL</option>
          <option value="mysql">MySQL</option>
          <option value="sqlite">SQLite</option>
        </select>
      </div>

      {/* Connection Details */}
      {config.database?.type !== 'sqlite' && (
        <>
          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                Host
              </label>
              <input
                type="text"
                value={config.database?.host || ''}
                onChange={(e) => handleChange('host', e.target.value)}
                placeholder="localhost"
                className="w-full px-4 py-2 border border-gray-300 dark:border-gray-600 rounded-lg
                         bg-white dark:bg-gray-800 text-gray-900 dark:text-gray-100
                         focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              />
              {errors['database.host'] && (
                <p className="mt-1 text-sm text-red-600 dark:text-red-400">{errors['database.host']}</p>
              )}
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                Port
              </label>
              <input
                type="number"
                value={config.database?.port || (config.database?.type === 'postgresql' ? 5432 : 3306)}
                onChange={(e) => handleChange('port', parseInt(e.target.value))}
                className="w-full px-4 py-2 border border-gray-300 dark:border-gray-600 rounded-lg
                         bg-white dark:bg-gray-800 text-gray-900 dark:text-gray-100
                         focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              />
              {errors['database.port'] && (
                <p className="mt-1 text-sm text-red-600 dark:text-red-400">{errors['database.port']}</p>
              )}
            </div>
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
              Database Name
            </label>
            <input
              type="text"
              value={config.database?.name || ''}
              onChange={(e) => handleChange('name', e.target.value)}
              placeholder="brain_db"
              className="w-full px-4 py-2 border border-gray-300 dark:border-gray-600 rounded-lg
                       bg-white dark:bg-gray-800 text-gray-900 dark:text-gray-100
                       focus:ring-2 focus:ring-blue-500 focus:border-transparent"
            />
            {errors['database.name'] && (
              <p className="mt-1 text-sm text-red-600 dark:text-red-400">{errors['database.name']}</p>
            )}
          </div>

          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                Username
              </label>
              <input
                type="text"
                value={config.database?.username || ''}
                onChange={(e) => handleChange('username', e.target.value)}
                placeholder="admin"
                className="w-full px-4 py-2 border border-gray-300 dark:border-gray-600 rounded-lg
                         bg-white dark:bg-gray-800 text-gray-900 dark:text-gray-100
                         focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              />
              {errors['database.username'] && (
                <p className="mt-1 text-sm text-red-600 dark:text-red-400">{errors['database.username']}</p>
              )}
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                Password
              </label>
              <input
                type="password"
                value={config.database?.password || ''}
                onChange={(e) => handleChange('password', e.target.value)}
                placeholder="••••••••"
                className="w-full px-4 py-2 border border-gray-300 dark:border-gray-600 rounded-lg
                         bg-white dark:bg-gray-800 text-gray-900 dark:text-gray-100
                         focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              />
              {errors['database.password'] && (
                <p className="mt-1 text-sm text-red-600 dark:text-red-400">{errors['database.password']}</p>
              )}
            </div>
          </div>
        </>
      )}

      {/* SQLite File Path */}
      {config.database?.type === 'sqlite' && (
        <div>
          <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
            Database File Path
          </label>
          <input
            type="text"
            value={config.database?.path || ''}
            onChange={(e) => handleChange('path', e.target.value)}
            placeholder="./data/brain.db"
            className="w-full px-4 py-2 border border-gray-300 dark:border-gray-600 rounded-lg
                     bg-white dark:bg-gray-800 text-gray-900 dark:text-gray-100
                     focus:ring-2 focus:ring-blue-500 focus:border-transparent"
          />
          {errors['database.path'] && (
            <p className="mt-1 text-sm text-red-600 dark:text-red-400">{errors['database.path']}</p>
          )}
        </div>
      )}

      {/* Advanced Options */}
      <details className="group">
        <summary className="cursor-pointer text-sm font-medium text-gray-700 dark:text-gray-300">
          Advanced Options
        </summary>
        <div className="mt-4 space-y-4">
          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                Connection Pool Size
              </label>
              <input
                type="number"
                value={config.database?.poolSize || 10}
                onChange={(e) => handleChange('poolSize', parseInt(e.target.value))}
                className="w-full px-4 py-2 border border-gray-300 dark:border-gray-600 rounded-lg
                         bg-white dark:bg-gray-800 text-gray-900 dark:text-gray-100
                         focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                Connection Timeout (ms)
              </label>
              <input
                type="number"
                value={config.database?.timeout || 30000}
                onChange={(e) => handleChange('timeout', parseInt(e.target.value))}
                className="w-full px-4 py-2 border border-gray-300 dark:border-gray-600 rounded-lg
                         bg-white dark:bg-gray-800 text-gray-900 dark:text-gray-100
                         focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              />
            </div>
          </div>

          <div className="flex items-center space-x-3">
            <input
              type="checkbox"
              id="enableSSL"
              checked={config.database?.ssl || false}
              onChange={(e) => handleChange('ssl', e.target.checked)}
              className="w-4 h-4 text-blue-600 rounded border-gray-300 focus:ring-blue-500"
            />
            <label htmlFor="enableSSL" className="text-sm font-medium text-gray-700 dark:text-gray-300">
              Enable SSL Connection
            </label>
          </div>
        </div>
      </details>

      {/* Test Connection */}
      <div className="flex items-center justify-between p-4 bg-gray-50 dark:bg-gray-900 rounded-lg">
        <div className="flex items-center space-x-3">
          <Database className="w-5 h-5 text-gray-500" />
          <div>
            <p className="text-sm font-medium text-gray-700 dark:text-gray-300">
              Test Database Connection
            </p>
            {testStatus !== 'idle' && (
              <p className={`text-xs mt-1 ${
                testStatus === 'success' ? 'text-green-600 dark:text-green-400' :
                testStatus === 'error' ? 'text-red-600 dark:text-red-400' :
                'text-gray-600 dark:text-gray-400'
              }`}>
                {testMessage}
              </p>
            )}
          </div>
        </div>
        <button
          onClick={testConnection}
          disabled={testStatus === 'testing'}
          className="flex items-center space-x-2 px-4 py-2 bg-blue-600 text-white rounded-lg
                   hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
        >
          {testStatus === 'testing' ? (
            <Loader2 className="w-4 h-4 animate-spin" />
          ) : testStatus === 'success' ? (
            <CheckCircle className="w-4 h-4" />
          ) : testStatus === 'error' ? (
            <XCircle className="w-4 h-4" />
          ) : (
            <TestTube className="w-4 h-4" />
          )}
          <span>{testStatus === 'testing' ? 'Testing...' : 'Test Connection'}</span>
        </button>
      </div>
    </div>
  );
};