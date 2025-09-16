import React, { useState } from 'react';
import { Check, X, Loader2, AlertTriangle, FileText, Download, Copy } from 'lucide-react';

interface ReviewStepProps {
  config: any;
  errors: Record<string, string>;
  onUpdate: (updates: any) => void;
  onTest: () => void;
  isValidating?: boolean;
}

interface TestResult {
  name: string;
  status: 'pending' | 'testing' | 'success' | 'error' | 'warning';
  message?: string;
}

export const ReviewStep: React.FC<ReviewStepProps> = ({
  config,
  errors,
  onUpdate,
  onTest,
  isValidating
}) => {
  const [testResults, setTestResults] = useState<TestResult[]>([
    { name: 'Environment Variables', status: 'pending' },
    { name: 'Database Connection', status: 'pending' },
    { name: 'API Keys Validation', status: 'pending' },
    { name: 'Service Dependencies', status: 'pending' },
    { name: 'File Permissions', status: 'pending' },
    { name: 'Network Connectivity', status: 'pending' }
  ]);
  const [isTestingAll, setIsTestingAll] = useState(false);
  const [showEnvVars, setShowEnvVars] = useState(false);

  const runAllTests = async () => {
    setIsTestingAll(true);

    // Simulate running tests sequentially
    for (let i = 0; i < testResults.length; i++) {
      setTestResults(prev => prev.map((result, index) =>
        index === i ? { ...result, status: 'testing' } : result
      ));

      // Simulate API call
      await new Promise(resolve => setTimeout(resolve, 1000));

      const success = Math.random() > 0.3; // 70% success rate for demo
      setTestResults(prev => prev.map((result, index) =>
        index === i
          ? {
              ...result,
              status: success ? 'success' : 'error',
              message: success ? 'Test passed' : 'Connection failed'
            }
          : result
      ));
    }

    setIsTestingAll(false);
  };

  const getStatusIcon = (status: TestResult['status']) => {
    switch (status) {
      case 'success':
        return <Check className="w-4 h-4 text-green-500" />;
      case 'error':
        return <X className="w-4 h-4 text-red-500" />;
      case 'warning':
        return <AlertTriangle className="w-4 h-4 text-yellow-500" />;
      case 'testing':
        return <Loader2 className="w-4 h-4 text-blue-500 animate-spin" />;
      default:
        return <div className="w-4 h-4 rounded-full bg-gray-300 dark:bg-gray-600" />;
    }
  };

  const generateEnvFile = () => {
    const envContent = `# brAIn Configuration
# Generated on ${new Date().toISOString()}

# Environment
ENVIRONMENT=${config.environment || 'development'}
APP_NAME=${config.appName || 'brain-app'}
BACKEND_PORT=${config.backendPort || 8000}
FRONTEND_PORT=${config.frontendPort || 3000}
BASE_URL=${config.baseUrl || 'http://localhost:3000'}
DEBUG=${config.debugMode ? 'true' : 'false'}

# Database
DB_TYPE=${config.database?.type || 'postgresql'}
DB_HOST=${config.database?.host || 'localhost'}
DB_PORT=${config.database?.port || 5432}
DB_NAME=${config.database?.name || 'brain_db'}
DB_USER=${config.database?.username || ''}
DB_PASSWORD=${config.database?.password || ''}
DB_SSL=${config.database?.ssl ? 'true' : 'false'}

# API Keys
OPENAI_API_KEY=${config.apiKeys?.openai || ''}
ANTHROPIC_API_KEY=${config.apiKeys?.anthropic || ''}
SUPABASE_URL=${config.apiKeys?.supabase_url || ''}
SUPABASE_ANON_KEY=${config.apiKeys?.supabase_key || ''}
GOOGLE_DRIVE_API_KEY=${config.apiKeys?.google_drive || ''}
LANGFUSE_API_KEY=${config.apiKeys?.langfuse || ''}

# Services
${config.services?.map((s: any) => `SERVICE_${s.id.toUpperCase()}_ENABLED=${s.enabled}`).join('\n') || ''}
`;
    return envContent;
  };

  const copyToClipboard = () => {
    navigator.clipboard.writeText(generateEnvFile());
  };

  const downloadEnvFile = () => {
    const blob = new Blob([generateEnvFile()], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = '.env';
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="space-y-6">
      {/* Configuration Summary */}
      <div className="bg-gray-50 dark:bg-gray-900 rounded-lg p-6">
        <h3 className="text-lg font-semibold text-gray-900 dark:text-gray-100 mb-4">
          Configuration Summary
        </h3>

        <div className="space-y-4">
          {/* Environment */}
          <div>
            <h4 className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
              Environment Settings
            </h4>
            <div className="grid grid-cols-2 gap-2 text-sm">
              <div className="text-gray-600 dark:text-gray-400">Environment:</div>
              <div className="font-medium text-gray-900 dark:text-gray-100">{config.environment}</div>
              <div className="text-gray-600 dark:text-gray-400">Application:</div>
              <div className="font-medium text-gray-900 dark:text-gray-100">{config.appName}</div>
              <div className="text-gray-600 dark:text-gray-400">Ports:</div>
              <div className="font-medium text-gray-900 dark:text-gray-100">
                Backend: {config.backendPort}, Frontend: {config.frontendPort}
              </div>
            </div>
          </div>

          {/* Database */}
          <div>
            <h4 className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
              Database Configuration
            </h4>
            <div className="grid grid-cols-2 gap-2 text-sm">
              <div className="text-gray-600 dark:text-gray-400">Type:</div>
              <div className="font-medium text-gray-900 dark:text-gray-100">{config.database?.type}</div>
              <div className="text-gray-600 dark:text-gray-400">Host:</div>
              <div className="font-medium text-gray-900 dark:text-gray-100">
                {config.database?.host}:{config.database?.port}
              </div>
              <div className="text-gray-600 dark:text-gray-400">Database:</div>
              <div className="font-medium text-gray-900 dark:text-gray-100">{config.database?.name}</div>
            </div>
          </div>

          {/* API Keys */}
          <div>
            <h4 className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
              API Keys Status
            </h4>
            <div className="grid grid-cols-2 gap-2 text-sm">
              <div className="text-gray-600 dark:text-gray-400">OpenAI:</div>
              <div className="font-medium">
                {config.apiKeys?.openai ? '✅ Configured' : '❌ Missing'}
              </div>
              <div className="text-gray-600 dark:text-gray-400">Supabase:</div>
              <div className="font-medium">
                {config.apiKeys?.supabase_url && config.apiKeys?.supabase_key ? '✅ Configured' : '❌ Missing'}
              </div>
              <div className="text-gray-600 dark:text-gray-400">Optional Keys:</div>
              <div className="font-medium">
                {[config.apiKeys?.anthropic, config.apiKeys?.google_drive, config.apiKeys?.langfuse]
                  .filter(Boolean).length} of 3 configured
              </div>
            </div>
          </div>

          {/* Services */}
          <div>
            <h4 className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
              Enabled Services
            </h4>
            <div className="flex flex-wrap gap-2">
              {config.services?.filter((s: any) => s.enabled).map((service: any) => (
                <span
                  key={service.id}
                  className="px-2 py-1 text-xs bg-blue-100 dark:bg-blue-900/30 text-blue-700
                           dark:text-blue-300 rounded"
                >
                  {service.name}
                </span>
              ))}
            </div>
          </div>
        </div>
      </div>

      {/* Test Configuration */}
      <div className="bg-white dark:bg-gray-800 rounded-lg p-6 border border-gray-200 dark:border-gray-700">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-lg font-semibold text-gray-900 dark:text-gray-100">
            Configuration Tests
          </h3>
          <button
            onClick={runAllTests}
            disabled={isTestingAll}
            className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700
                     disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
          >
            {isTestingAll ? 'Testing...' : 'Run All Tests'}
          </button>
        </div>

        <div className="space-y-3">
          {testResults.map((result, index) => (
            <div
              key={index}
              className="flex items-center justify-between p-3 bg-gray-50 dark:bg-gray-900 rounded-lg"
            >
              <div className="flex items-center space-x-3">
                {getStatusIcon(result.status)}
                <span className="text-sm font-medium text-gray-700 dark:text-gray-300">
                  {result.name}
                </span>
              </div>
              {result.message && (
                <span className={`text-xs ${
                  result.status === 'success' ? 'text-green-600 dark:text-green-400' :
                  result.status === 'error' ? 'text-red-600 dark:text-red-400' :
                  'text-gray-600 dark:text-gray-400'
                }`}>
                  {result.message}
                </span>
              )}
            </div>
          ))}
        </div>
      </div>

      {/* Export Configuration */}
      <div className="bg-white dark:bg-gray-800 rounded-lg p-6 border border-gray-200 dark:border-gray-700">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-lg font-semibold text-gray-900 dark:text-gray-100">
            Export Configuration
          </h3>
          <div className="flex items-center space-x-2">
            <button
              onClick={() => setShowEnvVars(!showEnvVars)}
              className="p-2 text-gray-600 hover:text-gray-800 dark:text-gray-400
                       dark:hover:text-gray-200"
            >
              <FileText className="w-4 h-4" />
            </button>
            <button
              onClick={copyToClipboard}
              className="p-2 text-gray-600 hover:text-gray-800 dark:text-gray-400
                       dark:hover:text-gray-200"
            >
              <Copy className="w-4 h-4" />
            </button>
            <button
              onClick={downloadEnvFile}
              className="p-2 text-gray-600 hover:text-gray-800 dark:text-gray-400
                       dark:hover:text-gray-200"
            >
              <Download className="w-4 h-4" />
            </button>
          </div>
        </div>

        {showEnvVars && (
          <pre className="p-4 bg-gray-900 text-gray-100 rounded-lg text-xs overflow-x-auto">
            {generateEnvFile()}
          </pre>
        )}

        <div className="mt-4 flex items-center justify-between">
          <p className="text-sm text-gray-600 dark:text-gray-400">
            Ready to complete setup?
          </p>
          <button
            onClick={onTest}
            disabled={testResults.some(r => r.status === 'error')}
            className="px-6 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700
                     disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
          >
            Complete Configuration
          </button>
        </div>
      </div>
    </div>
  );
};