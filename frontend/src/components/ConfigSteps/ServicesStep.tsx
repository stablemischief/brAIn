import React from 'react';
import { Settings, ToggleLeft, ToggleRight, Info } from 'lucide-react';

interface Service {
  id: string;
  name: string;
  description: string;
  enabled: boolean;
  configurable: boolean;
  settings?: Record<string, any>;
}

interface ServicesStepProps {
  config: any;
  errors: Record<string, string>;
  onUpdate: (updates: any) => void;
  isValidating?: boolean;
}

const DEFAULT_SERVICES: Service[] = [
  {
    id: 'realtime',
    name: 'Real-time Updates',
    description: 'WebSocket connections for live dashboard updates',
    enabled: true,
    configurable: true,
    settings: {
      reconnectInterval: 5000,
      maxRetries: 10
    }
  },
  {
    id: 'monitoring',
    name: 'System Monitoring',
    description: 'Health checks and performance monitoring',
    enabled: true,
    configurable: true,
    settings: {
      checkInterval: 60000,
      alertThreshold: 0.8
    }
  },
  {
    id: 'cost_tracking',
    name: 'Cost Tracking',
    description: 'Track API usage and costs in real-time',
    enabled: true,
    configurable: true,
    settings: {
      budgetLimit: 100,
      alertPercentage: 80
    }
  },
  {
    id: 'knowledge_graph',
    name: 'Knowledge Graph',
    description: 'Document relationship mapping and visualization',
    enabled: true,
    configurable: true,
    settings: {
      maxNodes: 1000,
      clusteringEnabled: true
    }
  },
  {
    id: 'ai_assistant',
    name: 'AI Configuration Assistant',
    description: 'AI-powered help for configuration and troubleshooting',
    enabled: true,
    configurable: false
  },
  {
    id: 'auto_backup',
    name: 'Automatic Backups',
    description: 'Scheduled database and configuration backups',
    enabled: false,
    configurable: true,
    settings: {
      frequency: 'daily',
      retention: 7
    }
  },
  {
    id: 'rate_limiting',
    name: 'Rate Limiting',
    description: 'API rate limiting for security',
    enabled: true,
    configurable: true,
    settings: {
      requestsPerMinute: 60,
      burstLimit: 100
    }
  }
];

export const ServicesStep: React.FC<ServicesStepProps> = ({
  config,
  errors,
  onUpdate,
  isValidating
}) => {
  const services = config.services || DEFAULT_SERVICES;

  const toggleService = (serviceId: string) => {
    const updatedServices = services.map((service: Service) =>
      service.id === serviceId
        ? { ...service, enabled: !service.enabled }
        : service
    );
    onUpdate({ ...config, services: updatedServices });
  };

  const updateServiceSetting = (serviceId: string, setting: string, value: any) => {
    const updatedServices = services.map((service: Service) =>
      service.id === serviceId
        ? { ...service, settings: { ...service.settings, [setting]: value } }
        : service
    );
    onUpdate({ ...config, services: updatedServices });
  };

  const renderServiceSettings = (service: Service) => {
    if (!service.configurable || !service.enabled || !service.settings) return null;

    return (
      <div className="mt-3 pl-12 space-y-3 border-l-2 border-gray-200 dark:border-gray-700">
        {Object.entries(service.settings).map(([key, value]) => (
          <div key={key} className="pl-4">
            <label className="text-xs font-medium text-gray-600 dark:text-gray-400 capitalize">
              {key.replace(/_/g, ' ')}
            </label>
            {typeof value === 'boolean' ? (
              <button
                onClick={() => updateServiceSetting(service.id, key, !value)}
                className="ml-2"
              >
                {value ? (
                  <ToggleRight className="w-5 h-5 text-green-500" />
                ) : (
                  <ToggleLeft className="w-5 h-5 text-gray-400" />
                )}
              </button>
            ) : typeof value === 'number' ? (
              <input
                type="number"
                value={value}
                onChange={(e) => updateServiceSetting(service.id, key, parseInt(e.target.value))}
                className="ml-2 w-24 px-2 py-1 text-xs border border-gray-300 dark:border-gray-600
                         rounded bg-white dark:bg-gray-800 text-gray-900 dark:text-gray-100"
              />
            ) : (
              <input
                type="text"
                value={value}
                onChange={(e) => updateServiceSetting(service.id, key, e.target.value)}
                className="ml-2 w-32 px-2 py-1 text-xs border border-gray-300 dark:border-gray-600
                         rounded bg-white dark:bg-gray-800 text-gray-900 dark:text-gray-100"
              />
            )}
          </div>
        ))}
      </div>
    );
  };

  return (
    <div className="space-y-6">
      {/* Introduction */}
      <div className="p-4 bg-blue-50 dark:bg-blue-900/20 rounded-lg">
        <div className="flex items-start space-x-2">
          <Info className="w-5 h-5 text-blue-500 mt-0.5" />
          <div>
            <p className="text-sm font-medium text-blue-800 dark:text-blue-200">
              Service Configuration
            </p>
            <p className="mt-1 text-sm text-blue-700 dark:text-blue-300">
              Enable or disable services based on your needs. Some services have additional
              configuration options that appear when enabled.
            </p>
          </div>
        </div>
      </div>

      {/* Services List */}
      <div className="space-y-4">
        {services.map((service: Service) => (
          <div
            key={service.id}
            className={`p-4 rounded-lg border transition-colors ${
              service.enabled
                ? 'border-blue-200 dark:border-blue-800 bg-blue-50/50 dark:bg-blue-900/10'
                : 'border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-800'
            }`}
          >
            <div className="flex items-start justify-between">
              <div className="flex items-start space-x-3">
                <button
                  onClick={() => toggleService(service.id)}
                  className="mt-0.5"
                >
                  {service.enabled ? (
                    <ToggleRight className="w-6 h-6 text-blue-500" />
                  ) : (
                    <ToggleLeft className="w-6 h-6 text-gray-400" />
                  )}
                </button>
                <div className="flex-1">
                  <h3 className="text-sm font-medium text-gray-900 dark:text-gray-100">
                    {service.name}
                  </h3>
                  <p className="mt-1 text-xs text-gray-600 dark:text-gray-400">
                    {service.description}
                  </p>
                  {renderServiceSettings(service)}
                </div>
              </div>
              {service.configurable && service.enabled && (
                <Settings className="w-4 h-4 text-gray-400" />
              )}
            </div>
          </div>
        ))}
      </div>

      {/* Service Dependencies */}
      <div className="p-4 bg-yellow-50 dark:bg-yellow-900/20 rounded-lg">
        <p className="text-sm font-medium text-yellow-800 dark:text-yellow-200">
          Service Dependencies
        </p>
        <ul className="mt-2 text-xs text-yellow-700 dark:text-yellow-300 space-y-1">
          <li>• Real-time Updates requires WebSocket support</li>
          <li>• Cost Tracking requires valid OpenAI/Anthropic API keys</li>
          <li>• Knowledge Graph requires PostgreSQL with pgvector extension</li>
          <li>• Auto Backups requires write permissions to backup directory</li>
        </ul>
      </div>

      {/* Resource Usage Estimate */}
      <div className="p-4 bg-gray-50 dark:bg-gray-900 rounded-lg">
        <p className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
          Estimated Resource Usage
        </p>
        <div className="space-y-2 text-xs">
          <div className="flex justify-between">
            <span className="text-gray-600 dark:text-gray-400">Memory Usage:</span>
            <span className="font-medium text-gray-900 dark:text-gray-100">
              {services.filter((s: Service) => s.enabled).length * 50} MB
            </span>
          </div>
          <div className="flex justify-between">
            <span className="text-gray-600 dark:text-gray-400">CPU Usage:</span>
            <span className="font-medium text-gray-900 dark:text-gray-100">
              {services.filter((s: Service) => s.enabled).length * 5}%
            </span>
          </div>
          <div className="flex justify-between">
            <span className="text-gray-600 dark:text-gray-400">Network Connections:</span>
            <span className="font-medium text-gray-900 dark:text-gray-100">
              {services.filter((s: Service) => s.enabled && ['realtime', 'monitoring'].includes(s.id)).length * 10}
            </span>
          </div>
        </div>
      </div>
    </div>
  );
};