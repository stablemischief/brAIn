-- brAIn v2.0 Monitoring and Analytics Schema
-- Migration 003: LLM usage tracking, system health, and analytics
-- Created: 2025-09-11
-- Purpose: Monitor AI operations, costs, performance, and system health

-- Create monitoring specific types
CREATE TYPE service_status AS ENUM (
    'healthy',
    'warning', 
    'critical',
    'down',
    'maintenance'
);

CREATE TYPE operation_type AS ENUM (
    'embedding',
    'completion',
    'extraction',
    'analysis',
    'classification',
    'summarization',
    'validation'
);

-- LLM usage tracking for cost analysis and monitoring
CREATE TABLE IF NOT EXISTS llm_usage (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    
    -- Operation details
    operation_type operation_type NOT NULL,
    model_name TEXT NOT NULL, -- 'gpt-4-turbo-preview', 'text-embedding-3-small', etc.
    provider TEXT NOT NULL DEFAULT 'openai', -- 'openai', 'anthropic', 'local'
    
    -- Request/Response tracking
    input_text TEXT,
    output_text TEXT,
    input_tokens INTEGER NOT NULL DEFAULT 0,
    output_tokens INTEGER NOT NULL DEFAULT 0,
    total_tokens INTEGER GENERATED ALWAYS AS (input_tokens + output_tokens) STORED,
    
    -- Cost calculation
    input_cost_per_token DECIMAL(12,8) NOT NULL DEFAULT 0.0, -- Cost per input token
    output_cost_per_token DECIMAL(12,8) NOT NULL DEFAULT 0.0, -- Cost per output token
    total_cost DECIMAL(8,4) GENERATED ALWAYS AS (
        (input_tokens * input_cost_per_token) + (output_tokens * output_cost_per_token)
    ) STORED,
    
    -- Performance metrics
    latency_ms INTEGER, -- Response time in milliseconds
    started_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP WITH TIME ZONE,
    
    -- Context and metadata
    document_id UUID REFERENCES documents(id) ON DELETE SET NULL,
    folder_id UUID REFERENCES folders(id) ON DELETE SET NULL,
    session_id UUID, -- Link to user session if applicable
    request_id TEXT, -- Provider's request ID for tracking
    
    -- Error handling
    error_message TEXT,
    error_code TEXT,
    retry_count INTEGER DEFAULT 0,
    
    -- Langfuse integration
    trace_id TEXT, -- Langfuse trace identifier
    span_id TEXT, -- Langfuse span identifier
    
    -- Quality metrics
    response_quality_score DECIMAL(3,2), -- 0.0-1.0 quality assessment
    
    -- Audit fields
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    
    -- Constraints
    CHECK (input_tokens >= 0),
    CHECK (output_tokens >= 0),
    CHECK (total_cost >= 0),
    CHECK (latency_ms >= 0),
    CHECK (retry_count >= 0)
);

-- System health monitoring for all services
CREATE TABLE IF NOT EXISTS system_health (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    
    -- Service identification
    service_name TEXT NOT NULL, -- 'backend', 'postgres', 'redis', 'websocket', etc.
    service_version TEXT,
    host_name TEXT DEFAULT 'localhost',
    
    -- Health status
    status service_status NOT NULL,
    status_message TEXT,
    
    -- Performance metrics
    response_time_ms INTEGER, -- Service response time
    cpu_percent DECIMAL(5,2), -- CPU usage percentage
    memory_mb INTEGER, -- Memory usage in MB
    disk_usage_percent DECIMAL(5,2), -- Disk usage percentage
    
    -- Network metrics
    active_connections INTEGER DEFAULT 0,
    requests_per_minute INTEGER DEFAULT 0,
    error_rate_percent DECIMAL(5,2) DEFAULT 0.0,
    
    -- Database specific metrics (for postgres service)
    db_connections_active INTEGER,
    db_connections_max INTEGER,
    db_query_avg_time_ms DECIMAL(8,3),
    db_cache_hit_ratio DECIMAL(5,4), -- PostgreSQL buffer cache hit ratio
    
    -- Custom metrics (JSON for flexibility)
    custom_metrics JSONB DEFAULT '{}',
    
    -- Health check details
    check_duration_ms INTEGER,
    last_error TEXT,
    consecutive_failures INTEGER DEFAULT 0,
    
    -- Timestamps
    measured_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    
    -- Constraints
    CHECK (cpu_percent >= 0 AND cpu_percent <= 100),
    CHECK (disk_usage_percent >= 0 AND disk_usage_percent <= 100),
    CHECK (error_rate_percent >= 0 AND error_rate_percent <= 100),
    CHECK (consecutive_failures >= 0)
);

-- Processing analytics for performance monitoring and optimization
CREATE TABLE IF NOT EXISTS processing_analytics (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    
    -- Time aggregation
    date DATE NOT NULL,
    hour INTEGER NOT NULL, -- 0-23 for hourly aggregation
    
    -- Folder/source context
    folder_id UUID REFERENCES folders(id) ON DELETE CASCADE,
    
    -- Processing statistics
    files_processed INTEGER DEFAULT 0,
    files_failed INTEGER DEFAULT 0,
    files_skipped INTEGER DEFAULT 0,
    total_files INTEGER GENERATED ALWAYS AS (files_processed + files_failed + files_skipped) STORED,
    
    -- Performance metrics
    avg_processing_time_ms DECIMAL(8,2) DEFAULT 0.0,
    min_processing_time_ms INTEGER DEFAULT 0,
    max_processing_time_ms INTEGER DEFAULT 0,
    total_processing_time_ms BIGINT DEFAULT 0,
    
    -- Size and content metrics
    total_file_size_bytes BIGINT DEFAULT 0,
    total_text_extracted_chars BIGINT DEFAULT 0,
    avg_file_size_mb DECIMAL(8,3) DEFAULT 0.0,
    
    -- Cost analytics
    total_cost DECIMAL(8,4) DEFAULT 0.0,
    total_tokens INTEGER DEFAULT 0,
    avg_cost_per_document DECIMAL(8,4) DEFAULT 0.0,
    
    -- Quality metrics
    avg_extraction_quality DECIMAL(3,2) DEFAULT 0.0,
    avg_confidence_score DECIMAL(3,2) DEFAULT 0.0,
    
    -- Error analysis
    error_types JSONB DEFAULT '{}', -- Count of different error types
    retry_count INTEGER DEFAULT 0,
    
    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    
    -- Constraints
    UNIQUE(user_id, date, hour, folder_id),
    CHECK (hour >= 0 AND hour <= 23),
    CHECK (files_processed >= 0),
    CHECK (files_failed >= 0),
    CHECK (files_skipped >= 0),
    CHECK (total_cost >= 0),
    CHECK (avg_extraction_quality >= 0.0 AND avg_extraction_quality <= 1.0)
);

-- Daily cost summaries for budget tracking
CREATE TABLE IF NOT EXISTS daily_cost_summary (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    
    -- Date and context
    date DATE NOT NULL,
    
    -- Cost breakdown
    embedding_cost DECIMAL(8,4) DEFAULT 0.0,
    completion_cost DECIMAL(8,4) DEFAULT 0.0,
    other_operations_cost DECIMAL(8,4) DEFAULT 0.0,
    total_cost DECIMAL(8,4) GENERATED ALWAYS AS (
        embedding_cost + completion_cost + other_operations_cost
    ) STORED,
    
    -- Token usage
    total_input_tokens INTEGER DEFAULT 0,
    total_output_tokens INTEGER DEFAULT 0,
    total_tokens INTEGER GENERATED ALWAYS AS (total_input_tokens + total_output_tokens) STORED,
    
    -- Operation counts
    total_operations INTEGER DEFAULT 0,
    successful_operations INTEGER DEFAULT 0,
    failed_operations INTEGER DEFAULT 0,
    success_rate DECIMAL(5,2) GENERATED ALWAYS AS (
        CASE 
            WHEN total_operations > 0 THEN (successful_operations::decimal / total_operations * 100)
            ELSE 0
        END
    ) STORED,
    
    -- Budget tracking
    budget_limit DECIMAL(8,4), -- Daily budget limit
    budget_remaining DECIMAL(8,4) GENERATED ALWAYS AS (
        CASE 
            WHEN budget_limit IS NOT NULL THEN GREATEST(budget_limit - total_cost, 0)
            ELSE NULL
        END
    ) STORED,
    budget_exceeded BOOLEAN GENERATED ALWAYS AS (
        budget_limit IS NOT NULL AND total_cost > budget_limit
    ) STORED,
    
    -- Performance metrics
    avg_latency_ms DECIMAL(8,2) DEFAULT 0.0,
    documents_processed INTEGER DEFAULT 0,
    avg_cost_per_document DECIMAL(8,4) DEFAULT 0.0,
    
    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    
    -- Constraints
    UNIQUE(user_id, date),
    CHECK (total_cost >= 0),
    CHECK (total_input_tokens >= 0),
    CHECK (total_output_tokens >= 0),
    CHECK (successful_operations >= 0),
    CHECK (failed_operations >= 0),
    CHECK (total_operations >= successful_operations + failed_operations)
);

-- Alert rules for proactive monitoring
CREATE TABLE IF NOT EXISTS alert_rules (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    
    -- Rule identification
    rule_name TEXT NOT NULL,
    rule_type TEXT NOT NULL, -- 'cost_threshold', 'error_rate', 'performance', 'health'
    description TEXT,
    
    -- Alert conditions
    metric_name TEXT NOT NULL, -- 'daily_cost', 'error_rate', 'response_time', etc.
    operator TEXT NOT NULL, -- 'greater_than', 'less_than', 'equals', 'not_equals'
    threshold_value DECIMAL(10,4) NOT NULL,
    evaluation_window_minutes INTEGER DEFAULT 60,
    
    -- Alert behavior
    enabled BOOLEAN DEFAULT TRUE,
    severity TEXT DEFAULT 'warning', -- 'info', 'warning', 'critical'
    notification_channels TEXT[] DEFAULT ARRAY['email'], -- 'email', 'slack', 'webhook'
    
    -- Rate limiting
    cooldown_minutes INTEGER DEFAULT 60, -- Minimum time between alerts
    max_alerts_per_day INTEGER DEFAULT 10,
    
    -- Status tracking
    last_triggered_at TIMESTAMP WITH TIME ZONE,
    alerts_sent_today INTEGER DEFAULT 0,
    total_alerts_sent INTEGER DEFAULT 0,
    
    -- Rule metadata
    created_by UUID REFERENCES users(id),
    metadata JSONB DEFAULT '{}',
    
    -- Audit fields
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    
    -- Constraints
    UNIQUE(user_id, rule_name),
    CHECK (evaluation_window_minutes > 0),
    CHECK (cooldown_minutes >= 0),
    CHECK (max_alerts_per_day > 0)
);

-- Alert history for tracking notifications
CREATE TABLE IF NOT EXISTS alert_history (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    rule_id UUID NOT NULL REFERENCES alert_rules(id) ON DELETE CASCADE,
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    
    -- Alert details
    alert_message TEXT NOT NULL,
    severity TEXT NOT NULL,
    metric_value DECIMAL(10,4),
    threshold_value DECIMAL(10,4),
    
    -- Notification details
    channels_notified TEXT[] DEFAULT ARRAY[]::TEXT[],
    notification_status JSONB DEFAULT '{}', -- Status per channel
    
    -- Context
    context_data JSONB DEFAULT '{}',
    
    -- Resolution
    acknowledged BOOLEAN DEFAULT FALSE,
    acknowledged_by UUID REFERENCES users(id),
    acknowledged_at TIMESTAMP WITH TIME ZONE,
    resolution_notes TEXT,
    
    -- Timestamps
    triggered_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Functions for cost calculation and aggregation
CREATE OR REPLACE FUNCTION update_daily_cost_summary()
RETURNS TRIGGER AS $$
DECLARE
    target_date DATE;
    target_user_id UUID;
BEGIN
    -- Determine the target date and user from the new/old record
    IF TG_OP = 'INSERT' OR TG_OP = 'UPDATE' THEN
        target_date := DATE(NEW.created_at);
        target_user_id := NEW.user_id;
    ELSE
        target_date := DATE(OLD.created_at);
        target_user_id := OLD.user_id;
    END IF;
    
    -- Recalculate the daily summary
    INSERT INTO daily_cost_summary (
        user_id, 
        date, 
        embedding_cost, 
        completion_cost, 
        other_operations_cost,
        total_input_tokens,
        total_output_tokens,
        total_operations,
        successful_operations,
        failed_operations,
        avg_latency_ms,
        documents_processed,
        avg_cost_per_document
    )
    SELECT 
        user_id,
        target_date,
        COALESCE(SUM(CASE WHEN operation_type = 'embedding' THEN total_cost END), 0),
        COALESCE(SUM(CASE WHEN operation_type = 'completion' THEN total_cost END), 0),
        COALESCE(SUM(CASE WHEN operation_type NOT IN ('embedding', 'completion') THEN total_cost END), 0),
        COALESCE(SUM(input_tokens), 0),
        COALESCE(SUM(output_tokens), 0),
        COUNT(*),
        COUNT(*) FILTER (WHERE error_message IS NULL),
        COUNT(*) FILTER (WHERE error_message IS NOT NULL),
        COALESCE(AVG(latency_ms), 0),
        COUNT(DISTINCT document_id) FILTER (WHERE document_id IS NOT NULL),
        CASE 
            WHEN COUNT(DISTINCT document_id) FILTER (WHERE document_id IS NOT NULL) > 0 
            THEN COALESCE(SUM(total_cost), 0) / COUNT(DISTINCT document_id) FILTER (WHERE document_id IS NOT NULL)
            ELSE 0 
        END
    FROM llm_usage 
    WHERE user_id = target_user_id 
    AND DATE(created_at) = target_date
    GROUP BY user_id
    ON CONFLICT (user_id, date) DO UPDATE SET
        embedding_cost = EXCLUDED.embedding_cost,
        completion_cost = EXCLUDED.completion_cost,
        other_operations_cost = EXCLUDED.other_operations_cost,
        total_input_tokens = EXCLUDED.total_input_tokens,
        total_output_tokens = EXCLUDED.total_output_tokens,
        total_operations = EXCLUDED.total_operations,
        successful_operations = EXCLUDED.successful_operations,
        failed_operations = EXCLUDED.failed_operations,
        avg_latency_ms = EXCLUDED.avg_latency_ms,
        documents_processed = EXCLUDED.documents_processed,
        avg_cost_per_document = EXCLUDED.avg_cost_per_document,
        updated_at = CURRENT_TIMESTAMP;
    
    RETURN COALESCE(NEW, OLD);
END;
$$ LANGUAGE plpgsql;

-- Trigger to update daily cost summary
CREATE TRIGGER trigger_update_daily_cost_summary
    AFTER INSERT OR UPDATE OR DELETE ON llm_usage
    FOR EACH ROW EXECUTE FUNCTION update_daily_cost_summary();

-- Apply standard update triggers
CREATE TRIGGER update_processing_analytics_updated_at 
    BEFORE UPDATE ON processing_analytics 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_daily_cost_summary_updated_at 
    BEFORE UPDATE ON daily_cost_summary 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_alert_rules_updated_at 
    BEFORE UPDATE ON alert_rules 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Indexes for monitoring performance
CREATE INDEX IF NOT EXISTS idx_llm_usage_user_date ON llm_usage(user_id, DATE(created_at));
CREATE INDEX IF NOT EXISTS idx_llm_usage_operation_type ON llm_usage(operation_type);
CREATE INDEX IF NOT EXISTS idx_llm_usage_model_name ON llm_usage(model_name);
CREATE INDEX IF NOT EXISTS idx_llm_usage_document_id ON llm_usage(document_id);
CREATE INDEX IF NOT EXISTS idx_llm_usage_trace_id ON llm_usage(trace_id);
CREATE INDEX IF NOT EXISTS idx_llm_usage_created_at ON llm_usage(created_at);
CREATE INDEX IF NOT EXISTS idx_llm_usage_total_cost ON llm_usage(total_cost DESC);

CREATE INDEX IF NOT EXISTS idx_system_health_service ON system_health(service_name);
CREATE INDEX IF NOT EXISTS idx_system_health_status ON system_health(status);
CREATE INDEX IF NOT EXISTS idx_system_health_measured_at ON system_health(measured_at DESC);

CREATE INDEX IF NOT EXISTS idx_processing_analytics_user_date ON processing_analytics(user_id, date);
CREATE INDEX IF NOT EXISTS idx_processing_analytics_folder_date ON processing_analytics(folder_id, date);
CREATE INDEX IF NOT EXISTS idx_processing_analytics_date_hour ON processing_analytics(date, hour);

CREATE INDEX IF NOT EXISTS idx_daily_cost_summary_user_date ON daily_cost_summary(user_id, date);
CREATE INDEX IF NOT EXISTS idx_daily_cost_summary_date ON daily_cost_summary(date);
CREATE INDEX IF NOT EXISTS idx_daily_cost_summary_total_cost ON daily_cost_summary(total_cost DESC);
CREATE INDEX IF NOT EXISTS idx_daily_cost_summary_budget_exceeded ON daily_cost_summary(budget_exceeded) WHERE budget_exceeded = TRUE;

CREATE INDEX IF NOT EXISTS idx_alert_rules_user_enabled ON alert_rules(user_id, enabled) WHERE enabled = TRUE;
CREATE INDEX IF NOT EXISTS idx_alert_rules_rule_type ON alert_rules(rule_type);

CREATE INDEX IF NOT EXISTS idx_alert_history_rule_id ON alert_history(rule_id);
CREATE INDEX IF NOT EXISTS idx_alert_history_user_triggered ON alert_history(user_id, triggered_at DESC);
CREATE INDEX IF NOT EXISTS idx_alert_history_acknowledged ON alert_history(acknowledged) WHERE acknowledged = FALSE;

-- Composite indexes for complex queries
CREATE INDEX IF NOT EXISTS idx_llm_usage_user_operation_date ON llm_usage(user_id, operation_type, DATE(created_at));
CREATE INDEX IF NOT EXISTS idx_system_health_service_status_time ON system_health(service_name, status, measured_at DESC);

-- Comments for documentation
COMMENT ON TABLE llm_usage IS 'Detailed tracking of all LLM operations with cost and performance metrics';
COMMENT ON TABLE system_health IS 'Real-time monitoring of all system services and infrastructure';
COMMENT ON TABLE processing_analytics IS 'Hourly aggregated analytics for document processing performance';
COMMENT ON TABLE daily_cost_summary IS 'Daily cost summaries with budget tracking and alerts';
COMMENT ON TABLE alert_rules IS 'User-configurable alert rules for proactive monitoring';
COMMENT ON TABLE alert_history IS 'Historical record of all triggered alerts and notifications';

COMMENT ON COLUMN llm_usage.trace_id IS 'Langfuse trace ID for distributed tracing and debugging';
COMMENT ON COLUMN llm_usage.total_cost IS 'Calculated total cost in USD for this operation';
COMMENT ON COLUMN system_health.db_cache_hit_ratio IS 'PostgreSQL buffer cache hit ratio (higher is better)';
COMMENT ON COLUMN daily_cost_summary.budget_exceeded IS 'Automatically calculated based on daily budget limit';