import React, { useState, useEffect } from 'react';
import { useWebSocketEvent } from '../../hooks/useWebSocket';
import { WS_EVENTS } from '../../utils/websocket';
import {
  DocumentTextIcon,
  CheckCircleIcon,
  XCircleIcon,
  ClockIcon,
  ExclamationTriangleIcon,
  ArrowPathIcon
} from '@heroicons/react/24/outline';

interface ActivityItem {
  id: string;
  type: 'document_processed' | 'processing_started' | 'processing_failed' | 'system_alert';
  title: string;
  description: string;
  timestamp: string;
  metadata?: Record<string, any>;
}

interface RecentActivityProps {
  connected: boolean;
  className?: string;
  maxItems?: number;
}

const ActivityIcon: React.FC<{ type: ActivityItem['type'] }> = ({ type }) => {
  const iconClass = "h-5 w-5";
  
  switch (type) {
    case 'document_processed':
      return <CheckCircleIcon className={`${iconClass} text-green-500`} />;
    case 'processing_started':
      return <ArrowPathIcon className={`${iconClass} text-blue-500`} />;
    case 'processing_failed':
      return <XCircleIcon className={`${iconClass} text-red-500`} />;
    case 'system_alert':
      return <ExclamationTriangleIcon className={`${iconClass} text-yellow-500`} />;
    default:
      return <ClockIcon className={`${iconClass} text-gray-500`} />;
  }
};

const ActivityItemComponent: React.FC<{ item: ActivityItem; isNew?: boolean }> = ({ 
  item, 
  isNew = false 
}) => {
  const timeAgo = (timestamp: string) => {
    const now = new Date();
    const time = new Date(timestamp);
    const diffMs = now.getTime() - time.getTime();
    const diffMins = Math.floor(diffMs / 60000);
    
    if (diffMins < 1) return 'Just now';
    if (diffMins < 60) return `${diffMins}m ago`;
    
    const diffHours = Math.floor(diffMins / 60);
    if (diffHours < 24) return `${diffHours}h ago`;
    
    const diffDays = Math.floor(diffHours / 24);
    return `${diffDays}d ago`;
  };

  return (
    <div className={`flex items-start space-x-3 p-3 rounded-lg transition-all duration-300 ${
      isNew ? 'bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-800' :
      'hover:bg-gray-50 dark:hover:bg-gray-700/50'
    }`}>
      <div className="flex-shrink-0 mt-0.5">
        <ActivityIcon type={item.type} />
      </div>
      
      <div className="flex-1 min-w-0">
        <div className="flex items-start justify-between">
          <div className="flex-1">
            <p className="text-sm font-medium text-gray-900 dark:text-white">
              {item.title}
            </p>
            <p className="text-sm text-gray-600 dark:text-gray-400 mt-1">
              {item.description}
            </p>
            
            {item.metadata && Object.keys(item.metadata).length > 0 && (
              <div className="mt-2 text-xs text-gray-500 dark:text-gray-500">
                {Object.entries(item.metadata).map(([key, value]) => (
                  <span key={key} className="mr-3">
                    {key}: {String(value)}
                  </span>
                ))}
              </div>
            )}
          </div>
          
          <div className="flex-shrink-0 ml-2">
            <span className="text-xs text-gray-500 dark:text-gray-400">
              {timeAgo(item.timestamp)}
            </span>
          </div>
        </div>
      </div>
    </div>
  );
};

export const RecentActivity: React.FC<RecentActivityProps> = ({ 
  connected, 
  className = '',
  maxItems = 10 
}) => {
  const [activities, setActivities] = useState<ActivityItem[]>([
    // Mock data for demo
    {
      id: '1',
      type: 'document_processed',
      title: 'Document processed successfully',
      description: 'Annual_Report_2024.pdf completed processing',
      timestamp: new Date(Date.now() - 5 * 60000).toISOString(),
      metadata: { file_size: '2.3MB', tokens: 45678 }
    },
    {
      id: '2', 
      type: 'processing_started',
      title: 'Batch processing started',
      description: '15 documents queued for processing',
      timestamp: new Date(Date.now() - 15 * 60000).toISOString(),
      metadata: { folder: 'Q4_Reports' }
    },
    {
      id: '3',
      type: 'system_alert',
      title: 'High memory usage detected',
      description: 'System memory usage at 85%',
      timestamp: new Date(Date.now() - 30 * 60000).toISOString(),
      metadata: { usage: '85%' }
    },
    {
      id: '4',
      type: 'processing_failed',
      title: 'Processing failed',
      description: 'corrupted_file.pdf failed to process',
      timestamp: new Date(Date.now() - 45 * 60000).toISOString(),
      metadata: { error: 'Invalid PDF format' }
    }
  ]);

  const [newItemIds, setNewItemIds] = useState<Set<string>>(new Set());

  // Listen for real-time activity updates
  useWebSocketEvent(WS_EVENTS.ACTIVITY_FEED, (message: any) => {
    if (message.payload && message.payload.activity) {
      const newActivity: ActivityItem = {
        id: Date.now().toString(),
        ...message.payload.activity,
        timestamp: message.timestamp
      };
      
      setActivities(prev => {
        const updated = [newActivity, ...prev].slice(0, maxItems);
        return updated;
      });
      
      // Mark as new for animation
      setNewItemIds(prev => new Set(Array.from(prev).concat(newActivity.id)));
      
      // Remove new status after animation
      setTimeout(() => {
        setNewItemIds(prev => {
          const updated = new Set(prev);
          updated.delete(newActivity.id);
          return updated;
        });
      }, 3000);
    }
  }, [maxItems]);

  if (!connected) {
    return (
      <div className={`bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 p-6 ${className}`}>
        <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
          Recent Activity
        </h3>
        <div className="text-center py-8">
          <ExclamationTriangleIcon className="h-12 w-12 text-gray-400 mx-auto mb-4" />
          <p className="text-gray-500 dark:text-gray-400">
            Connection lost - activity feed unavailable
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className={`bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 p-6 ${className}`}>
      {/* Header */}
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-semibold text-gray-900 dark:text-white">
          Recent Activity
        </h3>
        <div className="flex items-center space-x-2">
          <div className={`w-2 h-2 rounded-full ${
            connected ? 'bg-green-500 animate-pulse' : 'bg-gray-400'
          }`} />
          <span className="text-xs text-gray-500 dark:text-gray-400">
            Live Feed
          </span>
        </div>
      </div>

      {/* Activity List */}
      <div className="space-y-1 max-h-96 overflow-y-auto">
        {activities.length === 0 ? (
          <div className="text-center py-8">
            <ClockIcon className="h-8 w-8 text-gray-400 mx-auto mb-3" />
            <p className="text-sm text-gray-500 dark:text-gray-400">
              No recent activity
            </p>
            <p className="text-xs text-gray-400 dark:text-gray-500 mt-1">
              Activity will appear here as your system processes documents
            </p>
          </div>
        ) : (
          activities.map((activity) => (
            <ActivityItemComponent
              key={activity.id}
              item={activity}
              isNew={newItemIds.has(activity.id)}
            />
          ))
        )}
      </div>

      {activities.length >= maxItems && (
        <div className="mt-4 pt-4 border-t border-gray-200 dark:border-gray-600">
          <button className="w-full text-sm text-blue-600 dark:text-blue-400 hover:text-blue-800 dark:hover:text-blue-300 transition-colors">
            View all activity →
          </button>
        </div>
      )}
    </div>
  );
};

export default RecentActivity;