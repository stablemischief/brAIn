import React, { createContext, useContext, useReducer, useEffect, ReactNode } from 'react';
import {
  SystemHealth,
  ProcessingStatus,
  CostAnalytics,
  ProcessingJob,
  WebSocketMessage,
  RealtimeUpdate,
  Notification
} from '../types';
import { useWebSocket, useWebSocketEvent } from '../hooks/useWebSocket';
import { WS_EVENTS, WS_CHANNELS } from '../utils/websocket';

// State interface
interface RealtimeState {
  // Connection status
  connected: boolean;
  connecting: boolean;
  error: string | null;
  
  // System data
  systemHealth: SystemHealth | null;
  processingStatus: ProcessingStatus | null;
  costAnalytics: CostAnalytics | null;
  
  // Processing jobs
  activeJobs: ProcessingJob[];
  recentJobs: ProcessingJob[];
  
  // Notifications
  notifications: Notification[];
  unreadCount: number;
  
  // Real-time metrics
  realTimeMetrics: {
    documentsProcessedToday: number;
    currentCostToday: number;
    averageProcessingTime: number;
    queueSize: number;
    activeConnections: number;
  };
  
  // Last update timestamps
  lastUpdated: {
    systemHealth: string | null;
    processingStatus: string | null;
    costAnalytics: string | null;
  };
}

// Action types
type RealtimeAction =
  | { type: 'SET_CONNECTION_STATUS'; payload: { connected: boolean; connecting: boolean; error?: string | null } }
  | { type: 'UPDATE_SYSTEM_HEALTH'; payload: SystemHealth }
  | { type: 'UPDATE_PROCESSING_STATUS'; payload: ProcessingStatus }
  | { type: 'UPDATE_COST_ANALYTICS'; payload: CostAnalytics }
  | { type: 'UPDATE_JOB_PROGRESS'; payload: ProcessingJob }
  | { type: 'ADD_JOB'; payload: ProcessingJob }
  | { type: 'REMOVE_JOB'; payload: string }
  | { type: 'ADD_NOTIFICATION'; payload: Notification }
  | { type: 'MARK_NOTIFICATION_READ'; payload: string }
  | { type: 'CLEAR_NOTIFICATIONS' }
  | { type: 'UPDATE_REAL_TIME_METRICS'; payload: Partial<RealtimeState['realTimeMetrics']> }
  | { type: 'SET_ERROR'; payload: string };

// Initial state
const initialState: RealtimeState = {
  connected: false,
  connecting: false,
  error: null,
  systemHealth: null,
  processingStatus: null,
  costAnalytics: null,
  activeJobs: [],
  recentJobs: [],
  notifications: [],
  unreadCount: 0,
  realTimeMetrics: {
    documentsProcessedToday: 0,
    currentCostToday: 0,
    averageProcessingTime: 0,
    queueSize: 0,
    activeConnections: 0,
  },
  lastUpdated: {
    systemHealth: null,
    processingStatus: null,
    costAnalytics: null,
  },
};

// Reducer
const realtimeReducer = (state: RealtimeState, action: RealtimeAction): RealtimeState => {
  switch (action.type) {
    case 'SET_CONNECTION_STATUS':
      return {
        ...state,
        connected: action.payload.connected,
        connecting: action.payload.connecting,
        error: action.payload.error || null,
      };

    case 'UPDATE_SYSTEM_HEALTH':
      return {
        ...state,
        systemHealth: action.payload,
        lastUpdated: {
          ...state.lastUpdated,
          systemHealth: new Date().toISOString(),
        },
      };

    case 'UPDATE_PROCESSING_STATUS':
      return {
        ...state,
        processingStatus: action.payload,
        lastUpdated: {
          ...state.lastUpdated,
          processingStatus: new Date().toISOString(),
        },
        realTimeMetrics: {
          ...state.realTimeMetrics,
          queueSize: action.payload.processing_documents,
        },
      };

    case 'UPDATE_COST_ANALYTICS':
      return {
        ...state,
        costAnalytics: action.payload,
        lastUpdated: {
          ...state.lastUpdated,
          costAnalytics: new Date().toISOString(),
        },
        realTimeMetrics: {
          ...state.realTimeMetrics,
          currentCostToday: action.payload.daily_costs[action.payload.daily_costs.length - 1]?.cost || 0,
        },
      };

    case 'UPDATE_JOB_PROGRESS':
      const updatedJob = action.payload;
      const existingJobIndex = state.activeJobs.findIndex(job => job.id === updatedJob.id);
      
      let newActiveJobs = [...state.activeJobs];
      let newRecentJobs = [...state.recentJobs];
      
      if (existingJobIndex >= 0) {
        // Update existing job
        newActiveJobs[existingJobIndex] = updatedJob;
        
        // Move completed/failed jobs to recent jobs
        if (['completed', 'failed', 'cancelled'].includes(updatedJob.status)) {
          newActiveJobs.splice(existingJobIndex, 1);
          newRecentJobs.unshift(updatedJob);
          
          // Keep only last 20 recent jobs
          newRecentJobs = newRecentJobs.slice(0, 20);
        }
      }
      
      return {
        ...state,
        activeJobs: newActiveJobs,
        recentJobs: newRecentJobs,
      };

    case 'ADD_JOB':
      return {
        ...state,
        activeJobs: [action.payload, ...state.activeJobs],
      };

    case 'REMOVE_JOB':
      return {
        ...state,
        activeJobs: state.activeJobs.filter(job => job.id !== action.payload),
      };

    case 'ADD_NOTIFICATION':
      const notification = action.payload;
      return {
        ...state,
        notifications: [notification, ...state.notifications],
        unreadCount: notification.read ? state.unreadCount : state.unreadCount + 1,
      };

    case 'MARK_NOTIFICATION_READ':
      const updatedNotifications = state.notifications.map(n => 
        n.id === action.payload ? { ...n, read: true } : n
      );
      const wasUnread = state.notifications.find(n => n.id === action.payload && !n.read);
      
      return {
        ...state,
        notifications: updatedNotifications,
        unreadCount: wasUnread ? state.unreadCount - 1 : state.unreadCount,
      };

    case 'CLEAR_NOTIFICATIONS':
      return {
        ...state,
        notifications: [],
        unreadCount: 0,
      };

    case 'UPDATE_REAL_TIME_METRICS':
      return {
        ...state,
        realTimeMetrics: {
          ...state.realTimeMetrics,
          ...action.payload,
        },
      };

    case 'SET_ERROR':
      return {
        ...state,
        error: action.payload,
      };

    default:
      return state;
  }
};

// Context interface
interface RealtimeContextType extends RealtimeState {
  // Actions
  markNotificationAsRead: (id: string) => void;
  clearAllNotifications: () => void;
  refreshData: () => void;
}

// Create context
const RealtimeContext = createContext<RealtimeContextType | undefined>(undefined);

// Provider props
interface RealtimeProviderProps {
  children: ReactNode;
  userId?: string;
  sessionId?: string;
}

// Provider component
export const RealtimeProvider: React.FC<RealtimeProviderProps> = ({ 
  children, 
  userId, 
  sessionId 
}) => {
  const [state, dispatch] = useReducer(realtimeReducer, initialState);
  
  // WebSocket connection with auto-subscription to relevant channels
  const {
    connected,
    connecting,
    error: connectionError,
    sendMessage,
  } = useWebSocket({
    userId,
    sessionId,
    autoConnect: true,
    channels: [
      WS_CHANNELS.PROCESSING,
      WS_CHANNELS.SYSTEM_HEALTH,
      WS_CHANNELS.COST_MONITORING,
      WS_CHANNELS.USER_ACTIVITY,
    ],
    onConnect: () => {
      dispatch({ 
        type: 'SET_CONNECTION_STATUS', 
        payload: { connected: true, connecting: false } 
      });
      console.log('Real-time connection established');
    },
    onDisconnect: (reason) => {
      dispatch({ 
        type: 'SET_CONNECTION_STATUS', 
        payload: { connected: false, connecting: false, error: reason } 
      });
      console.log('Real-time connection lost:', reason);
    },
    onError: (error) => {
      dispatch({ type: 'SET_ERROR', payload: error.message || 'Connection error' });
    },
  });

  // Update connection status in state
  useEffect(() => {
    dispatch({
      type: 'SET_CONNECTION_STATUS',
      payload: { connected, connecting, error: connectionError },
    });
  }, [connected, connecting, connectionError]);

  // Event handlers for different message types
  useWebSocketEvent(WS_EVENTS.SYSTEM_HEALTH, (message: WebSocketMessage) => {
    dispatch({ type: 'UPDATE_SYSTEM_HEALTH', payload: message.payload });
  });

  useWebSocketEvent(WS_EVENTS.PROCESSING_STATUS, (message: WebSocketMessage) => {
    dispatch({ type: 'UPDATE_PROCESSING_STATUS', payload: message.payload });
  });

  useWebSocketEvent(WS_EVENTS.COST_UPDATE, (message: WebSocketMessage) => {
    dispatch({ type: 'UPDATE_COST_ANALYTICS', payload: message.payload });
  });

  useWebSocketEvent(WS_EVENTS.JOB_PROGRESS, (message: WebSocketMessage) => {
    dispatch({ type: 'UPDATE_JOB_PROGRESS', payload: message.payload });
  });

  useWebSocketEvent(WS_EVENTS.NEW_DOCUMENT, (message: WebSocketMessage) => {
    // Create notification for new document
    const notification: Notification = {
      id: `doc-${Date.now()}`,
      type: 'info',
      title: 'New Document Processed',
      message: `Document "${message.payload.title}" has been processed successfully`,
      timestamp: new Date().toISOString(),
      read: false,
    };
    
    dispatch({ type: 'ADD_NOTIFICATION', payload: notification });
    
    // Update metrics
    dispatch({
      type: 'UPDATE_REAL_TIME_METRICS',
      payload: {
        documentsProcessedToday: state.realTimeMetrics.documentsProcessedToday + 1,
      },
    });
  });

  // Action functions
  const markNotificationAsRead = (id: string) => {
    dispatch({ type: 'MARK_NOTIFICATION_READ', payload: id });
  };

  const clearAllNotifications = () => {
    dispatch({ type: 'CLEAR_NOTIFICATIONS' });
  };

  const refreshData = () => {
    if (connected) {
      sendMessage({ type: 'refresh_all_data' });
    }
  };

  const contextValue: RealtimeContextType = {
    ...state,
    markNotificationAsRead,
    clearAllNotifications,
    refreshData,
  };

  return (
    <RealtimeContext.Provider value={contextValue}>
      {children}
    </RealtimeContext.Provider>
  );
};

// Hook to use the context
export const useRealtime = (): RealtimeContextType => {
  const context = useContext(RealtimeContext);
  if (!context) {
    throw new Error('useRealtime must be used within a RealtimeProvider');
  }
  return context;
};

// Selector hooks for specific data
export const useSystemHealth = () => {
  const { systemHealth, lastUpdated } = useRealtime();
  return { systemHealth, lastUpdated: lastUpdated.systemHealth };
};

export const useProcessingStatus = () => {
  const { processingStatus, lastUpdated } = useRealtime();
  return { processingStatus, lastUpdated: lastUpdated.processingStatus };
};

export const useCostAnalytics = () => {
  const { costAnalytics, lastUpdated } = useRealtime();
  return { costAnalytics, lastUpdated: lastUpdated.costAnalytics };
};

export const useActiveJobs = () => {
  const { activeJobs, recentJobs } = useRealtime();
  return { activeJobs, recentJobs };
};

export const useNotifications = () => {
  const { 
    notifications, 
    unreadCount, 
    markNotificationAsRead, 
    clearAllNotifications 
  } = useRealtime();
  
  return { 
    notifications, 
    unreadCount, 
    markNotificationAsRead, 
    clearAllNotifications 
  };
};

export const useRealTimeMetrics = () => {
  const { realTimeMetrics } = useRealtime();
  return realTimeMetrics;
};