// Core API Types
export interface ApiResponse<T = any> {
  success: boolean;
  data?: T;
  error?: {
    code: string;
    message: string;
    details?: any;
  };
}

// User and Authentication Types
export interface User {
  id: string;
  email: string;
  role: string;
  created_at: string;
}

export interface AuthState {
  user: User | null;
  isAuthenticated: boolean;
  isLoading: boolean;
  error: string | null;
}

// Dashboard and System Health Types
export interface SystemHealth {
  status: 'healthy' | 'degraded' | 'unhealthy';
  services: {
    database: ServiceStatus;
    websocket: ServiceStatus;
    openai: ServiceStatus;
    supabase: ServiceStatus;
    langfuse?: ServiceStatus;
  };
  metrics: SystemMetrics;
  timestamp: string;
}

export interface ServiceStatus {
  status: 'healthy' | 'degraded' | 'unhealthy';
  response_time_ms?: number;
  error?: string;
  last_check?: string;
}

export interface SystemMetrics {
  cpu_usage_percent: number;
  memory_usage_percent: number;
  memory_total_gb: number;
  memory_available_gb: number;
  disk_usage_percent: number;
  active_connections: number;
  uptime_seconds: number;
}

// Processing Types
export interface ProcessingStatus {
  total_documents: number;
  processed_documents: number;
  processing_documents: number;
  failed_documents: number;
  processing_rate: number;
  estimated_completion: string | null;
  current_status: 'idle' | 'processing' | 'error';
}

export interface ProcessingJob {
  id: string;
  job_type: string;
  status: 'queued' | 'processing' | 'completed' | 'failed' | 'cancelled';
  total_items: number;
  processed_items: number;
  failed_items: number;
  progress_percentage: number;
  created_at: string;
  updated_at: string;
  completed_at?: string;
  error_message?: string;
  options: Record<string, any>;
}

// Folder and Document Types
export interface Folder {
  id: string;
  name: string;
  google_drive_id: string;
  total_documents: number;
  processed_documents: number;
  status: 'pending' | 'processing' | 'completed' | 'error';
  created_at: string;
  updated_at: string;
}

export interface Document {
  id: string;
  title: string;
  file_path: string;
  file_type: string;
  file_size: number;
  processing_status: 'pending' | 'processing' | 'completed' | 'failed';
  created_at: string;
  updated_at: string;
  metadata?: Record<string, any>;
}

// Cost and Analytics Types
export interface CostAnalytics {
  total_cost: number;
  daily_costs: DailyCost[];
  cost_by_model: Record<string, number>;
  cost_by_operation: Record<string, number>;
  token_usage: {
    total_tokens: number;
    input_tokens: number;
    output_tokens: number;
  };
  projected_monthly_cost: number;
  timeframe: string;
}

export interface DailyCost {
  date: string;
  cost: number;
  tokens: number;
  requests: number;
}

export interface UsageAnalytics {
  total_documents_processed: number;
  total_search_queries: number;
  average_documents_per_day: number;
  average_search_queries_per_day: number;
  most_active_hours: Record<string, number>;
  most_processed_file_types: Record<string, number>;
  search_patterns: {
    avg_query_length: number;
    most_common_terms: string[];
    successful_search_rate: number;
  };
}

export interface BudgetInfo {
  monthly_limit: number;
  current_month_spend: number;
  remaining_budget: number;
  budget_utilization_percentage: number;
  projected_month_end_spend: number;
  alerts: {
    approaching_limit: boolean;
    over_budget: boolean;
    high_daily_spend: boolean;
  };
}

// Search Types
export interface SearchResult {
  document_id: string;
  title: string;
  content_snippet: string;
  relevance_score: number;
  file_type: string;
  file_path: string;
  metadata: Record<string, any>;
  created_at: string;
  highlighted_text: string[];
}

export interface SearchResponse {
  results: SearchResult[];
  total_results: number;
  query: string;
  search_time_ms: number;
  suggestions: string[];
  facets: Record<string, Record<string, number>>;
}

// Knowledge Graph Types
export interface KnowledgeNode {
  id: string;
  label: string;
  type: string;
  properties: Record<string, any>;
  x?: number;
  y?: number;
}

export interface KnowledgeEdge {
  id: string;
  source: string;
  target: string;
  relationship: string;
  weight: number;
  properties: Record<string, any>;
}

export interface KnowledgeGraph {
  nodes: KnowledgeNode[];
  edges: KnowledgeEdge[];
  total_nodes: number;
  total_edges: number;
}

// WebSocket Types
export interface WebSocketMessage {
  type: string;
  payload: any;
  timestamp: string;
  channel?: string;
}

export interface WebSocketState {
  connected: boolean;
  connecting: boolean;
  error: string | null;
  lastMessage: WebSocketMessage | null;
  subscriptions: string[];
}

// Real-time Update Types
export interface RealtimeUpdate {
  type: 'processing_status' | 'system_health' | 'cost_update' | 'new_document' | 'error';
  data: any;
  timestamp: string;
}

// Dashboard Layout Types
export interface DashboardPanel {
  id: string;
  title: string;
  component: string;
  position: { x: number; y: number };
  size: { width: number; height: number };
  visible: boolean;
  config?: Record<string, any>;
}

export interface DashboardLayout {
  id: string;
  name: string;
  panels: DashboardPanel[];
  is_default: boolean;
}

// Configuration Types
export interface ConfigurationStep {
  step_number: number;
  title: string;
  description: string;
  fields: ConfigField[];
  is_completed: boolean;
  validation_status: 'valid' | 'invalid' | 'pending';
  next_step?: number;
}

export interface ConfigField {
  name: string;
  label: string;
  type: 'text' | 'password' | 'boolean' | 'select' | 'textarea' | 'multiselect';
  required: boolean;
  placeholder?: string;
  help_text?: string;
  options?: string[];
  default?: any;
  validation_pattern?: string;
}

export interface ConfigurationTemplate {
  id: string;
  name: string;
  description: string;
  category: string;
  configuration: Record<string, any>;
  required_secrets: string[];
  estimated_monthly_cost: number;
  features: string[];
}

// Error Types
export interface AppError {
  code: string;
  message: string;
  details?: any;
  timestamp: string;
  user_message?: string;
}

// Chart and Visualization Types
export interface ChartDataPoint {
  x: string | number;
  y: number;
  label?: string;
  color?: string;
  metadata?: Record<string, any>;
}

export interface ChartSeries {
  name: string;
  data: ChartDataPoint[];
  color?: string;
  type?: 'line' | 'bar' | 'area' | 'pie';
}

// Notification Types
export interface Notification {
  id: string;
  type: 'info' | 'success' | 'warning' | 'error';
  title: string;
  message: string;
  timestamp: string;
  read: boolean;
  actions?: NotificationAction[];
}

export interface NotificationAction {
  label: string;
  action: string;
  style?: 'primary' | 'secondary' | 'danger';
}

// Theme Types
export interface Theme {
  mode: 'light' | 'dark';
  primaryColor: string;
  accentColor: string;
  customization?: Record<string, any>;
}

// Component Props Types
export interface BaseComponentProps {
  className?: string;
  children?: React.ReactNode;
}

export interface LoadingProps extends BaseComponentProps {
  size?: 'sm' | 'md' | 'lg';
  color?: string;
  text?: string;
}

export interface StatusBadgeProps extends BaseComponentProps {
  status: 'online' | 'offline' | 'processing' | 'warning' | 'error';
  label?: string;
  pulse?: boolean;
}

// Utility Types
export type DeepPartial<T> = {
  [P in keyof T]?: T[P] extends object ? DeepPartial<T[P]> : T[P];
};

export type Optional<T, K extends keyof T> = Omit<T, K> & Partial<Pick<T, K>>;

export type RequireAtLeastOne<T, Keys extends keyof T = keyof T> = 
  Pick<T, Exclude<keyof T, Keys>> & 
  {[K in Keys]-?: Required<Pick<T, K>> & Partial<Pick<T, Exclude<Keys, K>>>}[Keys];

// Hook Return Types
export interface UseApiResult<T> {
  data: T | null;
  loading: boolean;
  error: string | null;
  refetch: () => Promise<void>;
}

export interface UseWebSocketResult {
  connected: boolean;
  connecting: boolean;
  error: string | null;
  sendMessage: (message: any) => void;
  subscribe: (channel: string) => void;
  unsubscribe: (channel: string) => void;
  lastMessage: WebSocketMessage | null;
  connect: () => Promise<void>;
  disconnect: () => void;
}

// API Endpoint Types
export type ApiEndpoint = 
  | '/api/health'
  | '/api/auth/login'
  | '/api/auth/logout'
  | '/api/folders'
  | '/api/processing/status'
  | '/api/search'
  | '/api/analytics/costs'
  | '/api/analytics/usage'
  | '/api/config/wizard/steps';

export type HttpMethod = 'GET' | 'POST' | 'PUT' | 'DELETE' | 'PATCH';