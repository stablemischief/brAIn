import React, { useState } from 'react';
import { Key, Eye, EyeOff, CheckCircle, XCircle, Info } from 'lucide-react';

interface APIKeysStepProps {
  config: any;
  errors: Record<string, string>;
  onUpdate: (updates: any) => void;
  isValidating?: boolean;
}

interface APIKeyField {
  id: string;
  label: string;
  placeholder: string;
  required: boolean;
  description: string;
  validateEndpoint?: string;
}

const API_KEY_FIELDS: APIKeyField[] = [
  {
    id: 'openai',
    label: 'OpenAI API Key',
    placeholder: 'sk-...',
    required: true,
    description: 'Required for AI text generation and embeddings',
    validateEndpoint: '/api/config/validate-openai'
  },
  {
    id: 'anthropic',
    label: 'Anthropic API Key',
    placeholder: 'sk-ant-...',
    required: false,
    description: 'Optional: For Claude AI integration',
    validateEndpoint: '/api/config/validate-anthropic'
  },
  {
    id: 'supabase_url',
    label: 'Supabase URL',
    placeholder: 'https://xxxxx.supabase.co',
    required: true,
    description: 'Your Supabase project URL',
    validateEndpoint: '/api/config/validate-supabase'
  },
  {
    id: 'supabase_key',
    label: 'Supabase Anon Key',
    placeholder: 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...',
    required: true,
    description: 'Your Supabase anonymous key',
    validateEndpoint: '/api/config/validate-supabase'
  },
  {
    id: 'google_drive',
    label: 'Google Drive API Key',
    placeholder: 'AIza...',
    required: false,
    description: 'Optional: For Google Drive integration',
    validateEndpoint: '/api/config/validate-google'
  },
  {
    id: 'langfuse',
    label: 'Langfuse API Key',
    placeholder: 'sk-lf-...',
    required: false,
    description: 'Optional: For LLM observability and monitoring',
    validateEndpoint: '/api/config/validate-langfuse'
  }
];

export const APIKeysStep: React.FC<APIKeysStepProps> = ({
  config,
  errors,
  onUpdate,
  isValidating
}) => {
  const [showKeys, setShowKeys] = useState<Record<string, boolean>>({});
  const [validationStatus, setValidationStatus] = useState<Record<string, 'idle' | 'validating' | 'valid' | 'invalid'>>({});

  const handleChange = (field: string, value: string) => {
    onUpdate({
      ...config,
      apiKeys: {
        ...config.apiKeys,
        [field]: value
      }
    });
    // Reset validation status when key changes
    setValidationStatus(prev => ({ ...prev, [field]: 'idle' }));
  };

  const toggleShowKey = (field: string) => {
    setShowKeys(prev => ({ ...prev, [field]: !prev[field] }));
  };

  const validateKey = async (field: APIKeyField) => {
    if (!config.apiKeys?.[field.id] || !field.validateEndpoint) return;

    setValidationStatus(prev => ({ ...prev, [field.id]: 'validating' }));

    try {
      const response = await fetch(field.validateEndpoint, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ key: config.apiKeys[field.id] })
      });

      setValidationStatus(prev => ({
        ...prev,
        [field.id]: response.ok ? 'valid' : 'invalid'
      }));
    } catch (error) {
      setValidationStatus(prev => ({ ...prev, [field.id]: 'invalid' }));
    }
  };

  const getStatusIcon = (fieldId: string) => {
    const status = validationStatus[fieldId];
    if (status === 'valid') return <CheckCircle className="w-4 h-4 text-green-500" />;
    if (status === 'invalid') return <XCircle className="w-4 h-4 text-red-500" />;
    return null;
  };

  return (
    <div className="space-y-6">
      {/* Introduction */}
      <div className="p-4 bg-blue-50 dark:bg-blue-900/20 rounded-lg">
        <div className="flex items-start space-x-2">
          <Info className="w-5 h-5 text-blue-500 mt-0.5" />
          <div>
            <p className="text-sm font-medium text-blue-800 dark:text-blue-200">
              API Key Configuration
            </p>
            <p className="mt-1 text-sm text-blue-700 dark:text-blue-300">
              Enter your API keys below. Required keys are marked with an asterisk (*).
              Your keys are encrypted and stored securely.
            </p>
          </div>
        </div>
      </div>

      {/* API Key Fields */}
      {API_KEY_FIELDS.map((field) => (
        <div key={field.id} className="space-y-2">
          <div className="flex items-center justify-between">
            <label className="block text-sm font-medium text-gray-700 dark:text-gray-300">
              {field.label} {field.required && <span className="text-red-500">*</span>}
            </label>
            {getStatusIcon(field.id)}
          </div>
          <div className="relative">
            <input
              type={showKeys[field.id] ? 'text' : 'password'}
              value={config.apiKeys?.[field.id] || ''}
              onChange={(e) => handleChange(field.id, e.target.value)}
              placeholder={field.placeholder}
              className="w-full pr-20 px-4 py-2 border border-gray-300 dark:border-gray-600 rounded-lg
                       bg-white dark:bg-gray-800 text-gray-900 dark:text-gray-100
                       focus:ring-2 focus:ring-blue-500 focus:border-transparent"
            />
            <div className="absolute right-2 top-1/2 -translate-y-1/2 flex items-center space-x-2">
              <button
                type="button"
                onClick={() => toggleShowKey(field.id)}
                className="p-1 text-gray-500 hover:text-gray-700 dark:text-gray-400
                         dark:hover:text-gray-200"
              >
                {showKeys[field.id] ? (
                  <EyeOff className="w-4 h-4" />
                ) : (
                  <Eye className="w-4 h-4" />
                )}
              </button>
              {field.validateEndpoint && config.apiKeys?.[field.id] && (
                <button
                  type="button"
                  onClick={() => validateKey(field)}
                  disabled={validationStatus[field.id] === 'validating'}
                  className="text-xs px-2 py-1 text-blue-600 hover:text-blue-700
                           dark:text-blue-400 dark:hover:text-blue-300
                           disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  {validationStatus[field.id] === 'validating' ? 'Validating...' : 'Validate'}
                </button>
              )}
            </div>
          </div>
          <p className="text-xs text-gray-500 dark:text-gray-400">{field.description}</p>
          {errors[`apiKeys.${field.id}`] && (
            <p className="text-sm text-red-600 dark:text-red-400">
              {errors[`apiKeys.${field.id}`]}
            </p>
          )}
        </div>
      ))}

      {/* Security Note */}
      <div className="mt-6 p-4 bg-gray-50 dark:bg-gray-900 rounded-lg">
        <div className="flex items-start space-x-2">
          <Key className="w-5 h-5 text-gray-500 mt-0.5" />
          <div>
            <p className="text-sm font-medium text-gray-700 dark:text-gray-300">
              Security Note
            </p>
            <ul className="mt-2 text-xs text-gray-600 dark:text-gray-400 space-y-1">
              <li>• API keys are encrypted before storage</li>
              <li>• Never commit API keys to version control</li>
              <li>• Use environment variables in production</li>
              <li>• Rotate keys regularly for security</li>
            </ul>
          </div>
        </div>
      </div>

      {/* Validation Summary */}
      {Object.values(validationStatus).some(s => s !== 'idle') && (
        <div className="p-4 bg-gray-50 dark:bg-gray-900 rounded-lg">
          <p className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
            Validation Status
          </p>
          <div className="space-y-1">
            {API_KEY_FIELDS.map(field => {
              const status = validationStatus[field.id];
              if (status === 'idle') return null;
              return (
                <div key={field.id} className="flex items-center justify-between text-xs">
                  <span className="text-gray-600 dark:text-gray-400">{field.label}</span>
                  <span className={
                    status === 'valid' ? 'text-green-600 dark:text-green-400' :
                    status === 'invalid' ? 'text-red-600 dark:text-red-400' :
                    'text-gray-500 dark:text-gray-500'
                  }>
                    {status === 'validating' ? 'Validating...' :
                     status === 'valid' ? 'Valid' :
                     status === 'invalid' ? 'Invalid' : ''}
                  </span>
                </div>
              );
            })}
          </div>
        </div>
      )}
    </div>
  );
};