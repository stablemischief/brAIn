import React from 'react';
import { SystemHealth, ServiceStatus } from '@/types';
import { 
  CheckCircleIcon, 
  ExclamationTriangleIcon, 
  XCircleIcon,
  ClockIcon,
  WifiIcon,
  CpuChipIcon,
  ServerIcon,
  CloudIcon
} from '@heroicons/react/24/outline';

interface HealthIndicatorProps {
  health: SystemHealth | null;
  connected: boolean;
  className?: string;
}

interface ServiceCardProps {
  name: string;
  status: ServiceStatus;
  icon: React.ReactNode;
  description: string;
}

const ServiceCard: React.FC<ServiceCardProps> = ({ name, status, icon, description }) => {
  const getStatusColor = (status: ServiceStatus['status']) => {
    switch (status) {
      case 'healthy':
        return 'text-green-600 bg-green-50 border-green-200 dark:text-green-400 dark:bg-green-900/20 dark:border-green-800';
      case 'degraded':
        return 'text-yellow-600 bg-yellow-50 border-yellow-200 dark:text-yellow-400 dark:bg-yellow-900/20 dark:border-yellow-800';
      case 'unhealthy':
        return 'text-red-600 bg-red-50 border-red-200 dark:text-red-400 dark:bg-red-900/20 dark:border-red-800';
      default:
        return 'text-gray-600 bg-gray-50 border-gray-200 dark:text-gray-400 dark:bg-gray-900/20 dark:border-gray-800';
    }
  };

  const getStatusIcon = (status: ServiceStatus['status']) => {
    switch (status) {
      case 'healthy':
        return <CheckCircleIcon className="h-4 w-4 text-green-500" />;
      case 'degraded':
        return <ExclamationTriangleIcon className="h-4 w-4 text-yellow-500" />;
      case 'unhealthy':
        return <XCircleIcon className="h-4 w-4 text-red-500" />;
      default:
        return <ClockIcon className="h-4 w-4 text-gray-500" />;
    }
  };

  return (
    <div className={`p-4 rounded-lg border transition-colors duration-200 ${getStatusColor(status.status)}`}>
      <div className="flex items-center justify-between">
        <div className="flex items-center space-x-3">
          <div className="flex-shrink-0">
            {icon}
          </div>
          <div>
            <h4 className="font-medium text-sm">
              {name}
            </h4>
            <p className="text-xs opacity-75">
              {description}
            </p>
          </div>
        </div>
        
        <div className="flex items-center space-x-2">
          {getStatusIcon(status.status)}
          {status.response_time_ms && (
            <span className="text-xs opacity-75">
              {status.response_time_ms}ms
            </span>
          )}
        </div>
      </div>
      
      {status.error && (
        <div className="mt-2 text-xs opacity-75 truncate">
          Error: {status.error}
        </div>
      )}
      
      {status.last_check && (
        <div className="mt-1 text-xs opacity-60">
          Last check: {new Date(status.last_check).toLocaleTimeString()}
        </div>
      )}
    </div>
  );
};

const MetricCard: React.FC<{ 
  label: string; 
  value: string | number; 
  unit?: string;
  status: 'good' | 'warning' | 'critical';
  icon: React.ReactNode;
}> = ({ label, value, unit, status, icon }) => {
  const getStatusColor = () => {
    switch (status) {
      case 'good':
        return 'text-green-600 dark:text-green-400';
      case 'warning':
        return 'text-yellow-600 dark:text-yellow-400';
      case 'critical':
        return 'text-red-600 dark:text-red-400';
      default:
        return 'text-gray-600 dark:text-gray-400';
    }
  };

  return (
    <div className="flex items-center space-x-3 p-3 bg-white dark:bg-gray-700 rounded-lg border border-gray-200 dark:border-gray-600">
      <div className={`flex-shrink-0 ${getStatusColor()}`}>
        {icon}
      </div>
      <div className="flex-1 min-w-0">
        <p className="text-xs text-gray-500 dark:text-gray-400 truncate">
          {label}
        </p>
        <p className={`text-lg font-semibold ${getStatusColor()}`}>
          {value}{unit}
        </p>
      </div>
    </div>
  );
};

export const HealthIndicator: React.FC<HealthIndicatorProps> = ({ 
  health, 
  connected, 
  className = '' 
}) => {
  if (!connected) {
    return (
      <div className={`bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg p-6 ${className}`}>
        <div className="flex items-center space-x-3">
          <WifiIcon className="h-8 w-8 text-red-500" />
          <div>
            <h3 className="text-lg font-semibold text-red-900 dark:text-red-100">
              Connection Lost
            </h3>
            <p className="text-red-700 dark:text-red-300">
              Unable to receive real-time health updates
            </p>
          </div>
        </div>
      </div>
    );
  }

  if (!health) {
    return (
      <div className={`bg-gray-50 dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-lg p-6 ${className}`}>
        <div className="flex items-center justify-center space-x-3">
          <ClockIcon className="h-6 w-6 text-gray-400 animate-spin" />
          <p className="text-gray-600 dark:text-gray-400">
            Loading system health...
          </p>
        </div>
      </div>
    );
  }

  const getOverallStatusColor = () => {
    switch (health.status) {
      case 'healthy':
        return 'bg-green-50 dark:bg-green-900/20 border-green-200 dark:border-green-800';
      case 'degraded':
        return 'bg-yellow-50 dark:bg-yellow-900/20 border-yellow-200 dark:border-yellow-800';
      case 'unhealthy':
        return 'bg-red-50 dark:bg-red-900/20 border-red-200 dark:border-red-800';
      default:
        return 'bg-gray-50 dark:bg-gray-800 border-gray-200 dark:border-gray-700';
    }
  };

  const getMetricStatus = (value: number, thresholds: { warning: number; critical: number }) => {
    if (value >= thresholds.critical) return 'critical';
    if (value >= thresholds.warning) return 'warning';
    return 'good';
  };

  const formatUptime = (seconds: number) => {
    const days = Math.floor(seconds / 86400);
    const hours = Math.floor((seconds % 86400) / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    
    if (days > 0) return `${days}d ${hours}h`;
    if (hours > 0) return `${hours}h ${minutes}m`;
    return `${minutes}m`;
  };

  return (
    <div className={`border rounded-lg p-6 ${getOverallStatusColor()} ${className}`}>
      {/* Overall Status Header */}
      <div className="flex items-center justify-between mb-6">
        <div className="flex items-center space-x-3">
          {health.status === 'healthy' && <CheckCircleIcon className="h-8 w-8 text-green-500" />}
          {health.status === 'degraded' && <ExclamationTriangleIcon className="h-8 w-8 text-yellow-500" />}
          {health.status === 'unhealthy' && <XCircleIcon className="h-8 w-8 text-red-500" />}
          
          <div>
            <h3 className="text-lg font-semibold text-gray-900 dark:text-white">
              System Status: {health.status.toUpperCase()}
            </h3>
            <p className="text-sm text-gray-600 dark:text-gray-400">
              Last updated: {new Date(health.timestamp).toLocaleString()}
            </p>
          </div>
        </div>

        <div className="text-right">
          <p className="text-sm text-gray-500 dark:text-gray-400">Uptime</p>
          <p className="text-lg font-semibold text-gray-900 dark:text-white">
            {formatUptime(health.metrics.uptime_seconds)}
          </p>
        </div>
      </div>

      {/* System Metrics */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
        <MetricCard
          label="CPU Usage"
          value={health.metrics.cpu_usage_percent.toFixed(1)}
          unit="%"
          status={getMetricStatus(health.metrics.cpu_usage_percent, { warning: 70, critical: 90 })}
          icon={<CpuChipIcon className="h-5 w-5" />}
        />
        
        <MetricCard
          label="Memory Usage"
          value={health.metrics.memory_usage_percent.toFixed(1)}
          unit="%"
          status={getMetricStatus(health.metrics.memory_usage_percent, { warning: 80, critical: 95 })}
          icon={<ServerIcon className="h-5 w-5" />}
        />
        
        <MetricCard
          label="Disk Usage"
          value={health.metrics.disk_usage_percent.toFixed(1)}
          unit="%"
          status={getMetricStatus(health.metrics.disk_usage_percent, { warning: 80, critical: 95 })}
          icon={<ServerIcon className="h-5 w-5" />}
        />
        
        <MetricCard
          label="Connections"
          value={health.metrics.active_connections}
          unit=""
          status={getMetricStatus(health.metrics.active_connections, { warning: 100, critical: 200 })}
          icon={<WifiIcon className="h-5 w-5" />}
        />
      </div>

      {/* Service Status Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        <ServiceCard
          name="Database"
          status={health.services.database}
          icon={<ServerIcon className="h-5 w-5" />}
          description="PostgreSQL + pgvector"
        />
        
        <ServiceCard
          name="WebSocket"
          status={health.services.websocket}
          icon={<WifiIcon className="h-5 w-5" />}
          description="Real-time communication"
        />
        
        <ServiceCard
          name="OpenAI"
          status={health.services.openai}
          icon={<CloudIcon className="h-5 w-5" />}
          description="LLM processing service"
        />
        
        <ServiceCard
          name="Supabase"
          status={health.services.supabase}
          icon={<CloudIcon className="h-5 w-5" />}
          description="Backend services"
        />
        
        {health.services.langfuse && (
          <ServiceCard
            name="Langfuse"
            status={health.services.langfuse}
            icon={<CloudIcon className="h-5 w-5" />}
            description="LLM observability"
          />
        )}
      </div>
    </div>
  );
};

export default HealthIndicator;