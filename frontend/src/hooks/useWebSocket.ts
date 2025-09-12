import { useEffect, useRef, useState, useCallback } from 'react';
import { WebSocketMessage, WebSocketState, UseWebSocketResult } from '@/types';
import { getWebSocketClient, WS_EVENTS, WS_CHANNELS } from '@/utils/websocket';

interface UseWebSocketOptions {
  userId?: string;
  sessionId?: string;
  autoConnect?: boolean;
  channels?: string[];
  onMessage?: (message: WebSocketMessage) => void;
  onConnect?: () => void;
  onDisconnect?: (reason?: string) => void;
  onError?: (error: any) => void;
}

/**
 * React hook for WebSocket connection management
 */
export const useWebSocket = (options: UseWebSocketOptions = {}): UseWebSocketResult => {
  const {
    userId,
    sessionId,
    autoConnect = true,
    channels = [],
    onMessage,
    onConnect,
    onDisconnect,
    onError,
  } = options;

  const [state, setState] = useState<WebSocketState>({
    connected: false,
    connecting: false,
    error: null,
    lastMessage: null,
    subscriptions: [],
  });

  const wsClient = useRef(getWebSocketClient());
  const reconnectTimeoutRef = useRef<NodeJS.Timeout>();

  // Connect to WebSocket
  const connect = useCallback(async () => {
    if (state.connecting || state.connected) return;

    setState(prev => ({ ...prev, connecting: true, error: null }));

    try {
      await wsClient.current.connect(userId, sessionId, channels);
    } catch (error) {
      console.error('WebSocket connection failed:', error);
      setState(prev => ({ 
        ...prev, 
        connecting: false, 
        error: 'Connection failed' 
      }));
    }
  }, [userId, sessionId, channels, state.connecting, state.connected]);

  // Disconnect from WebSocket
  const disconnect = useCallback(() => {
    wsClient.current.disconnect();
    setState(prev => ({ 
      ...prev, 
      connected: false, 
      connecting: false,
      subscriptions: []
    }));
  }, []);

  // Send message
  const sendMessage = useCallback((message: any) => {
    if (state.connected) {
      wsClient.current.sendMessage({
        ...message,
        timestamp: new Date().toISOString(),
      });
    } else {
      console.warn('Cannot send message: WebSocket not connected');
    }
  }, [state.connected]);

  // Subscribe to channel
  const subscribe = useCallback((channel: string) => {
    wsClient.current.subscribe(channel);
    setState(prev => ({
      ...prev,
      subscriptions: [...new Set([...prev.subscriptions, channel])]
    }));
  }, []);

  // Unsubscribe from channel
  const unsubscribe = useCallback((channel: string) => {
    wsClient.current.unsubscribe(channel);
    setState(prev => ({
      ...prev,
      subscriptions: prev.subscriptions.filter(sub => sub !== channel)
    }));
  }, []);

  // Set up event listeners
  useEffect(() => {
    const client = wsClient.current;

    const handleConnected = () => {
      setState(prev => ({ 
        ...prev, 
        connected: true, 
        connecting: false, 
        error: null 
      }));
      onConnect?.();
    };

    const handleDisconnected = (data: any) => {
      setState(prev => ({ 
        ...prev, 
        connected: false, 
        connecting: false,
        subscriptions: []
      }));
      onDisconnect?.(`${data.code}: ${data.reason}`);
    };

    const handleError = (error: any) => {
      setState(prev => ({ 
        ...prev, 
        error: error.message || 'WebSocket error',
        connecting: false 
      }));
      onError?.(error);
    };

    const handleMessage = (message: WebSocketMessage) => {
      setState(prev => ({ 
        ...prev, 
        lastMessage: message 
      }));
      onMessage?.(message);
    };

    // Add event listeners
    client.on(WS_EVENTS.CONNECTED, handleConnected);
    client.on(WS_EVENTS.DISCONNECTED, handleDisconnected);
    client.on(WS_EVENTS.ERROR, handleError);
    client.on(WS_EVENTS.MESSAGE, handleMessage);

    // Cleanup function
    return () => {
      client.off(WS_EVENTS.CONNECTED, handleConnected);
      client.off(WS_EVENTS.DISCONNECTED, handleDisconnected);
      client.off(WS_EVENTS.ERROR, handleError);
      client.off(WS_EVENTS.MESSAGE, handleMessage);
    };
  }, [onConnect, onDisconnect, onError, onMessage]);

  // Auto-connect on mount
  useEffect(() => {
    if (autoConnect && !state.connected && !state.connecting) {
      connect();
    }

    // Cleanup on unmount
    return () => {
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current);
      }
    };
  }, [autoConnect, connect, state.connected, state.connecting]);

  // Subscribe to initial channels
  useEffect(() => {
    if (state.connected && channels.length > 0) {
      channels.forEach(channel => {
        if (!state.subscriptions.includes(channel)) {
          subscribe(channel);
        }
      });
    }
  }, [state.connected, channels, state.subscriptions, subscribe]);

  return {
    connected: state.connected,
    connecting: state.connecting,
    error: state.error,
    lastMessage: state.lastMessage,
    sendMessage,
    subscribe,
    unsubscribe,
    connect,
    disconnect,
  };
};

/**
 * Hook for subscribing to specific WebSocket events
 */
export const useWebSocketEvent = (
  eventType: string,
  callback: (data: any) => void,
  deps: any[] = []
) => {
  const wsClient = useRef(getWebSocketClient());

  useEffect(() => {
    const client = wsClient.current;
    client.on(eventType, callback);

    return () => {
      client.off(eventType, callback);
    };
  }, [eventType, ...deps]);
};

/**
 * Hook for processing status updates
 */
export const useProcessingStatus = () => {
  const [processingStatus, setProcessingStatus] = useState(null);

  useWebSocketEvent(WS_EVENTS.PROCESSING_STATUS, (message: WebSocketMessage) => {
    setProcessingStatus(message.payload);
  });

  return processingStatus;
};

/**
 * Hook for system health updates
 */
export const useSystemHealth = () => {
  const [systemHealth, setSystemHealth] = useState(null);

  useWebSocketEvent(WS_EVENTS.SYSTEM_HEALTH, (message: WebSocketMessage) => {
    setSystemHealth(message.payload);
  });

  return systemHealth;
};

/**
 * Hook for cost updates
 */
export const useCostUpdates = () => {
  const [costData, setCostData] = useState(null);

  useWebSocketEvent(WS_EVENTS.COST_UPDATE, (message: WebSocketMessage) => {
    setCostData(message.payload);
  });

  return costData;
};

/**
 * Hook for job progress updates
 */
export const useJobProgress = (jobId?: string) => {
  const [jobProgress, setJobProgress] = useState(null);

  useWebSocketEvent(WS_EVENTS.JOB_PROGRESS, (message: WebSocketMessage) => {
    if (!jobId || message.payload?.job_id === jobId) {
      setJobProgress(message.payload);
    }
  }, [jobId]);

  return jobProgress;
};

/**
 * Hook for managing WebSocket channels
 */
export const useWebSocketChannels = (initialChannels: string[] = []) => {
  const [channels, setChannels] = useState<string[]>(initialChannels);
  const { subscribe, unsubscribe } = useWebSocket({ channels });

  const addChannel = useCallback((channel: string) => {
    if (!channels.includes(channel)) {
      setChannels(prev => [...prev, channel]);
      subscribe(channel);
    }
  }, [channels, subscribe]);

  const removeChannel = useCallback((channel: string) => {
    setChannels(prev => prev.filter(ch => ch !== channel));
    unsubscribe(channel);
  }, [unsubscribe]);

  const clearChannels = useCallback(() => {
    channels.forEach(channel => unsubscribe(channel));
    setChannels([]);
  }, [channels, unsubscribe]);

  return {
    channels,
    addChannel,
    removeChannel,
    clearChannels,
  };
};