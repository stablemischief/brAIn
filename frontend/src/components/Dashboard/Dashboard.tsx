import React, { useState, useEffect } from 'react';
import { 
  useProcessingStatus, 
  useSystemHealth, 
  useCostUpdates,
  useWebSocket 
} from '../../hooks/useWebSocket';
import { SystemHealth, ProcessingStatus, CostAnalytics } from '../../types';
import { WS_CHANNELS } from '../../utils/websocket';
import { HealthIndicator } from './HealthIndicator';
import { ProcessingStatusCard } from './ProcessingStatusCard';
import { CostAnalyticsCard } from './CostAnalyticsCard';
import { RecentActivity } from './RecentActivity';
import { QuickActions } from './QuickActions';
import { RefreshButton } from './RefreshButton';

interface DashboardProps {
  className?: string;
}

export const Dashboard: React.FC<DashboardProps> = ({ className = '' }) => {
  // Real-time data hooks
  const processingStatus = useProcessingStatus();
  const systemHealth = useSystemHealth();
  const costData = useCostUpdates();
  
  // WebSocket connection
  const { 
    connected, 
    connecting, 
    error: wsError,
    sendMessage
  } = useWebSocket({
    autoConnect: true,
    channels: [
      WS_CHANNELS.PROCESSING,
      WS_CHANNELS.SYSTEM_HEALTH,
      WS_CHANNELS.COST_MONITORING,
      WS_CHANNELS.USER_ACTIVITY
    ]
  });

  // Local state for dashboard
  const [lastUpdated, setLastUpdated] = useState<Date>(new Date());
  const [refreshing, setRefreshing] = useState(false);

  // Update timestamp when data changes
  useEffect(() => {
    if (processingStatus || systemHealth || costData) {
      setLastUpdated(new Date());
    }
  }, [processingStatus, systemHealth, costData]);

  // Manual refresh function
  const handleRefresh = async () => {
    setRefreshing(true);
    
    // Send refresh requests via WebSocket
    sendMessage({
      type: 'request_update',
      channels: [
        WS_CHANNELS.PROCESSING,
        WS_CHANNELS.SYSTEM_HEALTH,
        WS_CHANNELS.COST_MONITORING
      ]
    });

    // Reset refreshing state after a delay
    setTimeout(() => setRefreshing(false), 1000);
  };

  // Connection status indicator
  const getConnectionStatus = () => {
    if (connecting) return { status: 'connecting', message: 'Connecting...' };
    if (!connected) return { status: 'disconnected', message: 'Disconnected' };
    if (wsError) return { status: 'error', message: `Error: ${wsError}` };
    return { status: 'connected', message: 'Live Updates Active' };
  };

  const connectionStatus = getConnectionStatus();

  return (
    <div className={`min-h-screen bg-gradient-to-br from-gray-50 to-gray-100 dark:from-gray-900 dark:to-gray-800 ${className}`}>
      {/* Header */}
      <div className="px-6 py-4 bg-white dark:bg-gray-800 border-b border-gray-200 dark:border-gray-700 shadow-sm">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-bold text-gray-900 dark:text-white">
              brAIn Dashboard
            </h1>
            <p className="text-sm text-gray-600 dark:text-gray-400 mt-1">
              Intelligent Document Processing System
            </p>
          </div>
          
          <div className="flex items-center space-x-4">
            {/* Connection Status */}
            <div className="flex items-center space-x-2">
              <div className={`w-2 h-2 rounded-full ${
                connectionStatus.status === 'connected' ? 'bg-green-500 animate-pulse' :
                connectionStatus.status === 'connecting' ? 'bg-yellow-500 animate-pulse' :
                'bg-red-500'
              }`} />
              <span className="text-sm text-gray-600 dark:text-gray-400">
                {connectionStatus.message}
              </span>
            </div>

            {/* Last Updated */}
            <div className="text-sm text-gray-500 dark:text-gray-400">
              Updated: {lastUpdated.toLocaleTimeString()}
            </div>

            {/* Refresh Button */}
            <RefreshButton 
              onRefresh={handleRefresh} 
              refreshing={refreshing}
              disabled={!connected}
            />
          </div>
        </div>
      </div>

      {/* Main Content */}
      <div className="p-6">
        {/* System Health Row */}
        <div className="mb-6">
          <h2 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
            System Health
          </h2>
          <HealthIndicator 
            health={systemHealth} 
            connected={connected}
          />
        </div>

        {/* Main Dashboard Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Left Column - Processing & Activity */}
          <div className="lg:col-span-2 space-y-6">
            {/* Processing Status */}
            <ProcessingStatusCard 
              status={processingStatus}
              connected={connected}
            />

            {/* Recent Activity */}
            <RecentActivity 
              connected={connected}
            />
          </div>

          {/* Right Column - Analytics & Actions */}
          <div className="space-y-6">
            {/* Cost Analytics */}
            <CostAnalyticsCard 
              data={costData}
              connected={connected}
            />

            {/* Quick Actions */}
            <QuickActions 
              systemHealth={systemHealth}
              processingStatus={processingStatus}
            />
          </div>
        </div>
      </div>

      {/* Footer */}
      <footer className="mt-12 px-6 py-4 bg-white dark:bg-gray-800 border-t border-gray-200 dark:border-gray-700">
        <div className="flex items-center justify-between text-sm text-gray-500 dark:text-gray-400">
          <div>
            brAIn v2.0 - Intelligent Document Processing
          </div>
          <div className="flex items-center space-x-4">
            <span>
              {processingStatus?.processed_documents || 0} documents processed
            </span>
            <span>•</span>
            <span>
              ${costData?.total_cost?.toFixed(4) || '0.0000'} total cost
            </span>
          </div>
        </div>
      </footer>
    </div>
  );
};

export default Dashboard;