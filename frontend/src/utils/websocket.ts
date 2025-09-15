import { WebSocketMessage, RealtimeUpdate } from '../types';

export class WebSocketClient {
  private ws: WebSocket | null = null;
  private url: string;
  private reconnectAttempts = 0;
  private maxReconnectAttempts = 10;
  private reconnectDelay = 1000;
  private maxReconnectDelay = 30000;
  private listeners = new Map<string, Set<Function>>();
  private subscriptions = new Set<string>();
  private heartbeatInterval: NodeJS.Timeout | null = null;
  private reconnectTimeout: NodeJS.Timeout | null = null;
  private connectionId: string | null = null;

  constructor(baseUrl: string = 'ws://localhost:8000') {
    this.url = `${baseUrl}/ws/realtime`;
  }

  /**
   * Connect to the WebSocket server
   */
  connect(userId?: string, sessionId?: string, channels?: string[]): Promise<void> {
    return new Promise((resolve, reject) => {
      try {
        // Build connection URL with query parameters
        const urlParams = new URLSearchParams();
        if (userId) urlParams.append('user_id', userId);
        if (sessionId) urlParams.append('session_id', sessionId);
        if (channels?.length) urlParams.append('channels', channels.join(','));

        const wsUrl = `${this.url}?${urlParams.toString()}`;
        
        this.ws = new WebSocket(wsUrl);

        this.ws.onopen = () => {
          console.log('WebSocket connected');
          this.reconnectAttempts = 0;
          this.startHeartbeat();
          this.resubscribeToChannels();
          this.emit('connected', { connected: true });
          resolve();
        };

        this.ws.onmessage = (event) => {
          try {
            const message: WebSocketMessage = JSON.parse(event.data);
            this.handleMessage(message);
          } catch (error) {
            console.error('Failed to parse WebSocket message:', error);
          }
        };

        this.ws.onclose = (event) => {
          console.log('WebSocket disconnected:', event.code, event.reason);
          this.stopHeartbeat();
          this.emit('disconnected', { connected: false, code: event.code, reason: event.reason });
          
          // Attempt reconnection if not a clean close
          if (event.code !== 1000 && this.reconnectAttempts < this.maxReconnectAttempts) {
            this.scheduleReconnect();
          }
        };

        this.ws.onerror = (error) => {
          console.error('WebSocket error:', error);
          this.emit('error', { error: 'WebSocket connection error' });
          reject(error);
        };

      } catch (error) {
        console.error('Failed to create WebSocket connection:', error);
        reject(error);
      }
    });
  }

  /**
   * Disconnect from the WebSocket server
   */
  disconnect(): void {
    if (this.reconnectTimeout) {
      clearTimeout(this.reconnectTimeout);
      this.reconnectTimeout = null;
    }
    
    this.stopHeartbeat();
    
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.ws.close(1000, 'Client disconnect');
    }
    
    this.ws = null;
    this.connectionId = null;
    this.subscriptions.clear();
  }

  /**
   * Send a message to the server
   */
  sendMessage(message: any): void {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      try {
        this.ws.send(JSON.stringify(message));
      } catch (error) {
        console.error('Failed to send WebSocket message:', error);
        this.emit('error', { error: 'Failed to send message' });
      }
    } else {
      console.warn('WebSocket is not connected. Message not sent:', message);
    }
  }

  /**
   * Subscribe to a channel
   */
  subscribe(channel: string): void {
    this.subscriptions.add(channel);
    
    if (this.isConnected()) {
      this.sendMessage({
        type: 'subscribe',
        channel: channel,
        timestamp: new Date().toISOString()
      });
    }
  }

  /**
   * Unsubscribe from a channel
   */
  unsubscribe(channel: string): void {
    this.subscriptions.delete(channel);
    
    if (this.isConnected()) {
      this.sendMessage({
        type: 'unsubscribe',
        channel: channel,
        timestamp: new Date().toISOString()
      });
    }
  }

  /**
   * Add event listener
   */
  on(event: string, callback: Function): void {
    if (!this.listeners.has(event)) {
      this.listeners.set(event, new Set());
    }
    this.listeners.get(event)!.add(callback);
  }

  /**
   * Remove event listener
   */
  off(event: string, callback: Function): void {
    const eventListeners = this.listeners.get(event);
    if (eventListeners) {
      eventListeners.delete(callback);
      if (eventListeners.size === 0) {
        this.listeners.delete(event);
      }
    }
  }

  /**
   * Check if WebSocket is connected
   */
  isConnected(): boolean {
    return this.ws?.readyState === WebSocket.OPEN;
  }

  /**
   * Get connection state
   */
  getConnectionState(): 'connecting' | 'open' | 'closing' | 'closed' {
    if (!this.ws) return 'closed';
    
    switch (this.ws.readyState) {
      case WebSocket.CONNECTING:
        return 'connecting';
      case WebSocket.OPEN:
        return 'open';
      case WebSocket.CLOSING:
        return 'closing';
      case WebSocket.CLOSED:
      default:
        return 'closed';
    }
  }

  /**
   * Get current subscriptions
   */
  getSubscriptions(): string[] {
    return Array.from(this.subscriptions);
  }

  private handleMessage(message: WebSocketMessage): void {
    // Handle connection confirmation
    if (message.type === 'connection_confirmed') {
      this.connectionId = message.payload?.connection_id;
      console.log('Connection confirmed:', this.connectionId);
    }

    // Handle subscription confirmation
    if (message.type === 'subscription_confirmed') {
      console.log('Subscribed to channel:', message.payload?.channel);
    }

    // Handle errors
    if (message.type === 'error') {
      console.error('WebSocket server error:', message.payload);
      this.emit('error', message.payload);
      return;
    }

    // Emit the message to listeners
    this.emit('message', message);
    
    // Emit channel-specific events
    if (message.channel) {
      this.emit(`channel:${message.channel}`, message);
    }

    // Emit type-specific events
    this.emit(`type:${message.type}`, message);
  }

  private emit(event: string, data: any): void {
    const eventListeners = this.listeners.get(event);
    if (eventListeners) {
      eventListeners.forEach(callback => {
        try {
          callback(data);
        } catch (error) {
          console.error(`Error in WebSocket event listener for ${event}:`, error);
        }
      });
    }
  }

  private scheduleReconnect(): void {
    if (this.reconnectTimeout) return;

    const delay = Math.min(
      this.reconnectDelay * Math.pow(2, this.reconnectAttempts),
      this.maxReconnectDelay
    );

    console.log(`Attempting to reconnect in ${delay}ms... (attempt ${this.reconnectAttempts + 1}/${this.maxReconnectAttempts})`);

    this.reconnectTimeout = setTimeout(() => {
      this.reconnectTimeout = null;
      this.reconnectAttempts++;
      this.connect();
    }, delay);
  }

  private resubscribeToChannels(): void {
    // Re-subscribe to all channels after reconnection
    this.subscriptions.forEach(channel => {
      this.sendMessage({
        type: 'subscribe',
        channel: channel,
        timestamp: new Date().toISOString()
      });
    });
  }

  private startHeartbeat(): void {
    this.stopHeartbeat();
    
    this.heartbeatInterval = setInterval(() => {
      if (this.isConnected()) {
        this.sendMessage({
          type: 'ping',
          timestamp: new Date().toISOString()
        });
      }
    }, 30000); // Send ping every 30 seconds
  }

  private stopHeartbeat(): void {
    if (this.heartbeatInterval) {
      clearInterval(this.heartbeatInterval);
      this.heartbeatInterval = null;
    }
  }
}

// Singleton WebSocket client instance
let wsClientInstance: WebSocketClient | null = null;

/**
 * Get the singleton WebSocket client instance
 */
export const getWebSocketClient = (): WebSocketClient => {
  if (!wsClientInstance) {
    const baseUrl = process.env.NODE_ENV === 'production' 
      ? 'wss://your-domain.com'  // Replace with your production WebSocket URL
      : 'ws://localhost:8000';
    
    wsClientInstance = new WebSocketClient(baseUrl);
  }
  
  return wsClientInstance;
};

/**
 * WebSocket event types for type safety
 */
export const WS_EVENTS = {
  CONNECTED: 'connected',
  DISCONNECTED: 'disconnected',
  ERROR: 'error',
  MESSAGE: 'message',
  PROCESSING_STATUS: 'type:processing_status',
  SYSTEM_HEALTH: 'type:system_health',
  COST_UPDATE: 'type:cost_update',
  NEW_DOCUMENT: 'type:new_document',
  JOB_PROGRESS: 'type:job_progress',
  ACTIVITY_FEED: 'type:activity_feed',
} as const;

/**
 * Channel names for subscriptions
 */
export const WS_CHANNELS = {
  PROCESSING: 'processing',
  SYSTEM_HEALTH: 'system_health',
  COST_MONITORING: 'cost_monitoring',
  USER_ACTIVITY: 'user_activity',
  KNOWLEDGE_GRAPH: 'knowledge_graph',
} as const;