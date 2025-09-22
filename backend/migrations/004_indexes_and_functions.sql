-- brAIn v2.0 Advanced Indexes and Functions
-- Migration 004: Vector indexes, materialized views, and advanced functions
-- Created: 2025-09-11
-- Purpose: Optimize performance for vector similarity, search, and analytics

-- ========================================
-- VECTOR SIMILARITY INDEXES
-- ========================================

-- HNSW index for document embeddings (most important for semantic search)
CREATE INDEX IF NOT EXISTS idx_documents_embedding_hnsw 
ON documents USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);

-- Alternative indexes for different distance metrics
CREATE INDEX IF NOT EXISTS idx_documents_embedding_l2 
ON documents USING hnsw (embedding vector_l2_ops)
WITH (m = 16, ef_construction = 64);

-- HNSW index for knowledge node embeddings (smaller dimension)
CREATE INDEX IF NOT EXISTS idx_knowledge_nodes_embedding_hnsw 
ON knowledge_nodes USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);

-- HNSW index for document cluster centroids
CREATE INDEX IF NOT EXISTS idx_document_clusters_centroid_hnsw 
ON document_clusters USING hnsw (centroid_embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);

-- ========================================
-- MATERIALIZED VIEWS FOR PERFORMANCE
-- ========================================

-- Document search view with precomputed metrics
CREATE MATERIALIZED VIEW IF NOT EXISTS document_search_view AS
SELECT 
    d.id,
    d.user_id,
    d.folder_id,
    d.file_name,
    d.processed_text,
    d.embedding,
    d.document_type,
    d.language_code,
    d.extraction_quality,
    d.processing_cost,
    d.token_count,
    d.created_at,
    d.updated_at,
    f.folder_name,
    f.folder_path,
    -- Precomputed text search vector
    to_tsvector('english', COALESCE(d.processed_text, d.raw_text, d.file_name, '')) as search_vector,
    -- Document statistics
    COALESCE(d.text_length, 0) as text_length,
    -- Cost metrics
    d.processing_cost,
    -- Quality metrics
    COALESCE(d.extraction_quality, 0.0) as quality_score
FROM documents d
LEFT JOIN folders f ON d.folder_id = f.id
WHERE d.deleted_at IS NULL 
AND d.processing_status = 'completed'
AND d.embedding IS NOT NULL;

-- Create indexes on the materialized view
CREATE INDEX IF NOT EXISTS idx_document_search_view_user_id ON document_search_view(user_id);
CREATE INDEX IF NOT EXISTS idx_document_search_view_folder_id ON document_search_view(folder_id);
CREATE INDEX IF NOT EXISTS idx_document_search_view_search_vector ON document_search_view USING GIN(search_vector);
CREATE INDEX IF NOT EXISTS idx_document_search_view_embedding ON document_search_view USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64);
CREATE INDEX IF NOT EXISTS idx_document_search_view_quality ON document_search_view(quality_score DESC);
CREATE INDEX IF NOT EXISTS idx_document_search_view_created_at ON document_search_view(created_at DESC);

-- Knowledge graph analytics view
CREATE MATERIALIZED VIEW IF NOT EXISTS knowledge_graph_analytics_view AS
SELECT 
    kn.user_id,
    kn.node_type,
    COUNT(*) as node_count,
    AVG(kn.confidence_score) as avg_confidence,
    SUM(kn.mention_count) as total_mentions,
    COUNT(DISTINCT kn.source_document_id) as document_count,
    AVG(kn.relationship_count) as avg_relationships,
    COUNT(*) FILTER (WHERE kn.confidence_score >= 0.8) as high_confidence_nodes,
    COUNT(*) FILTER (WHERE kn.validated = true) as validated_nodes
FROM knowledge_nodes kn
GROUP BY kn.user_id, kn.node_type;

CREATE INDEX IF NOT EXISTS idx_knowledge_graph_analytics_view_user_type ON knowledge_graph_analytics_view(user_id, node_type);

-- Cost analytics view by user and time period
CREATE MATERIALIZED VIEW IF NOT EXISTS cost_analytics_view AS
SELECT 
    user_id,
    DATE_TRUNC('day', created_at) as day,
    DATE_TRUNC('week', created_at) as week,
    DATE_TRUNC('month', created_at) as month,
    operation_type,
    model_name,
    COUNT(*) as operation_count,
    SUM(total_cost) as total_cost,
    SUM(input_tokens) as total_input_tokens,
    SUM(output_tokens) as total_output_tokens,
    AVG(latency_ms) as avg_latency_ms,
    COUNT(*) FILTER (WHERE error_message IS NOT NULL) as error_count,
    (COUNT(*) FILTER (WHERE error_message IS NULL)::decimal / COUNT(*) * 100) as success_rate
FROM llm_usage
GROUP BY user_id, DATE_TRUNC('day', created_at), DATE_TRUNC('week', created_at), 
         DATE_TRUNC('month', created_at), operation_type, model_name;

CREATE INDEX IF NOT EXISTS idx_cost_analytics_view_user_day ON cost_analytics_view(user_id, day);
CREATE INDEX IF NOT EXISTS idx_cost_analytics_view_operation_day ON cost_analytics_view(operation_type, day);

-- ========================================
-- SEARCH AND SIMILARITY FUNCTIONS
-- ========================================

-- Hybrid search function combining vector and text search
CREATE OR REPLACE FUNCTION hybrid_search(
    p_user_id UUID,
    p_query TEXT,
    p_embedding VECTOR(1536),
    p_limit INTEGER DEFAULT 10,
    p_vector_weight DECIMAL DEFAULT 0.6,
    p_text_weight DECIMAL DEFAULT 0.4,
    p_min_quality DECIMAL DEFAULT 0.0
)
RETURNS TABLE (
    document_id UUID,
    file_name TEXT,
    folder_name TEXT,
    similarity_score DECIMAL,
    text_rank DECIMAL,
    combined_score DECIMAL,
    extraction_quality DECIMAL,
    snippet TEXT
) AS $$
BEGIN
    RETURN QUERY
    WITH vector_results AS (
        SELECT 
            dsv.id as document_id,
            dsv.file_name,
            dsv.folder_name,
            dsv.extraction_quality,
            (1 - (dsv.embedding <=> p_embedding)) as vector_similarity,
            ts_rank_cd(dsv.search_vector, plainto_tsquery('english', p_query)) as text_rank,
            ts_headline(
                'english',
                COALESCE(dsv.processed_text, ''),
                plainto_tsquery('english', p_query),
                'MaxWords=25, MinWords=10'
            ) as snippet
        FROM document_search_view dsv
        WHERE dsv.user_id = p_user_id
        AND dsv.quality_score >= p_min_quality
        AND (
            dsv.embedding <=> p_embedding < 0.8  -- Similarity threshold
            OR dsv.search_vector @@ plainto_tsquery('english', p_query)
        )
    )
    SELECT 
        vr.document_id,
        vr.file_name,
        vr.folder_name,
        vr.vector_similarity,
        vr.text_rank,
        (vr.vector_similarity * p_vector_weight + vr.text_rank * p_text_weight) as combined_score,
        vr.extraction_quality,
        vr.snippet
    FROM vector_results vr
    ORDER BY combined_score DESC
    LIMIT p_limit;
END;
$$ LANGUAGE plpgsql;

-- Semantic similarity search for documents
CREATE OR REPLACE FUNCTION semantic_search(
    p_user_id UUID,
    p_embedding VECTOR(1536),
    p_limit INTEGER DEFAULT 10,
    p_similarity_threshold DECIMAL DEFAULT 0.7
)
RETURNS TABLE (
    document_id UUID,
    file_name TEXT,
    similarity_score DECIMAL,
    extraction_quality DECIMAL,
    processing_cost DECIMAL
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        d.id,
        d.file_name,
        (1 - (d.embedding <=> p_embedding)) as similarity,
        COALESCE(d.extraction_quality, 0.0),
        COALESCE(d.processing_cost, 0.0)
    FROM documents d
    WHERE d.user_id = p_user_id
    AND d.embedding IS NOT NULL
    AND d.deleted_at IS NULL
    AND d.processing_status = 'completed'
    AND (1 - (d.embedding <=> p_embedding)) >= p_similarity_threshold
    ORDER BY d.embedding <=> p_embedding
    LIMIT p_limit;
END;
$$ LANGUAGE plpgsql;

-- Find similar knowledge nodes
CREATE OR REPLACE FUNCTION find_similar_nodes(
    p_user_id UUID,
    p_node_id UUID,
    p_limit INTEGER DEFAULT 5
)
RETURNS TABLE (
    node_id UUID,
    node_type node_type,
    node_value TEXT,
    similarity_score DECIMAL,
    confidence_score DECIMAL
) AS $$
DECLARE
    source_embedding VECTOR(384);
BEGIN
    -- Get the embedding of the source node
    SELECT embedding INTO source_embedding
    FROM knowledge_nodes 
    WHERE id = p_node_id;
    
    IF source_embedding IS NULL THEN
        RETURN;
    END IF;
    
    RETURN QUERY
    SELECT 
        kn.id,
        kn.node_type,
        kn.node_value,
        (1 - (kn.embedding <=> source_embedding)) as similarity,
        kn.confidence_score
    FROM knowledge_nodes kn
    WHERE kn.user_id = p_user_id
    AND kn.id != p_node_id
    AND kn.embedding IS NOT NULL
    ORDER BY kn.embedding <=> source_embedding
    LIMIT p_limit;
END;
$$ LANGUAGE plpgsql;

-- ========================================
-- ANALYTICS AND AGGREGATION FUNCTIONS
-- ========================================

-- Calculate knowledge graph metrics
CREATE OR REPLACE FUNCTION calculate_graph_metrics(p_user_id UUID)
RETURNS JSONB AS $$
DECLARE
    total_nodes INTEGER;
    total_edges INTEGER;
    graph_density DECIMAL;
    avg_degree DECIMAL;
    result JSONB;
BEGIN
    -- Get basic counts
    SELECT COUNT(*) INTO total_nodes
    FROM knowledge_nodes WHERE user_id = p_user_id;
    
    SELECT COUNT(*) INTO total_edges
    FROM knowledge_edges WHERE user_id = p_user_id;
    
    -- Calculate graph density
    IF total_nodes > 1 THEN
        graph_density := total_edges::decimal / (total_nodes * (total_nodes - 1));
    ELSE
        graph_density := 0;
    END IF;
    
    -- Calculate average degree
    IF total_nodes > 0 THEN
        avg_degree := (total_edges * 2)::decimal / total_nodes;
    ELSE
        avg_degree := 0;
    END IF;
    
    -- Build result JSON
    result := jsonb_build_object(
        'total_nodes', total_nodes,
        'total_edges', total_edges,
        'graph_density', graph_density,
        'avg_degree', avg_degree,
        'calculated_at', CURRENT_TIMESTAMP
    );
    
    RETURN result;
END;
$$ LANGUAGE plpgsql;

-- Get user cost summary for date range
CREATE OR REPLACE FUNCTION get_cost_summary(
    p_user_id UUID,
    p_start_date DATE,
    p_end_date DATE
)
RETURNS TABLE (
    total_cost DECIMAL,
    total_operations INTEGER,
    avg_cost_per_operation DECIMAL,
    cost_by_operation JSONB,
    cost_by_model JSONB,
    daily_breakdown JSONB
) AS $$
BEGIN
    RETURN QUERY
    WITH cost_data AS (
        SELECT 
            operation_type,
            model_name,
            DATE(created_at) as day,
            COUNT(*) as ops,
            SUM(llm_usage.total_cost) as cost
        FROM llm_usage
        WHERE user_id = p_user_id
        AND DATE(created_at) BETWEEN p_start_date AND p_end_date
        GROUP BY operation_type, model_name, DATE(created_at)
    )
    SELECT 
        COALESCE(SUM(cd.cost), 0) as total_cost,
        COALESCE(SUM(cd.ops), 0)::INTEGER as total_operations,
        CASE 
            WHEN SUM(cd.ops) > 0 THEN SUM(cd.cost) / SUM(cd.ops)
            ELSE 0
        END as avg_cost_per_operation,
        COALESCE(
            jsonb_object_agg(
                cd.operation_type, 
                cd.cost
            ) FILTER (WHERE cd.operation_type IS NOT NULL),
            '{}'::jsonb
        ) as cost_by_operation,
        COALESCE(
            jsonb_object_agg(
                cd.model_name,
                cd.cost
            ) FILTER (WHERE cd.model_name IS NOT NULL),
            '{}'::jsonb
        ) as cost_by_model,
        COALESCE(
            jsonb_object_agg(
                cd.day::text,
                cd.cost
            ) FILTER (WHERE cd.day IS NOT NULL),
            '{}'::jsonb
        ) as daily_breakdown
    FROM cost_data cd;
END;
$$ LANGUAGE plpgsql;

-- ========================================
-- REAL-TIME NOTIFICATION FUNCTIONS
-- ========================================

-- Function to notify real-time subscriptions
CREATE OR REPLACE FUNCTION notify_document_change()
RETURNS TRIGGER AS $$
BEGIN
    -- Notify Supabase real-time subscribers
    PERFORM pg_notify(
        'document_changes',
        json_build_object(
            'operation', TG_OP,
            'table', TG_TABLE_NAME,
            'user_id', COALESCE(NEW.user_id, OLD.user_id),
            'document_id', COALESCE(NEW.id, OLD.id),
            'timestamp', CURRENT_TIMESTAMP
        )::text
    );
    
    RETURN COALESCE(NEW, OLD);
END;
$$ LANGUAGE plpgsql;

CREATE OR REPLACE FUNCTION notify_processing_status_change()
RETURNS TRIGGER AS $$
BEGIN
    -- Only notify on status changes
    IF TG_OP = 'UPDATE' AND OLD.processing_status = NEW.processing_status THEN
        RETURN NEW;
    END IF;
    
    PERFORM pg_notify(
        'processing_status',
        json_build_object(
            'user_id', NEW.user_id,
            'document_id', NEW.id,
            'old_status', OLD.processing_status,
            'new_status', NEW.processing_status,
            'timestamp', CURRENT_TIMESTAMP
        )::text
    );
    
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE OR REPLACE FUNCTION notify_cost_update()
RETURNS TRIGGER AS $$
BEGIN
    PERFORM pg_notify(
        'cost_updates',
        json_build_object(
            'user_id', NEW.user_id,
            'operation_type', NEW.operation_type,
            'cost', NEW.total_cost,
            'timestamp', CURRENT_TIMESTAMP
        )::text
    );
    
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Apply real-time notification triggers
CREATE TRIGGER trigger_notify_document_change
    AFTER INSERT OR UPDATE OR DELETE ON documents
    FOR EACH ROW EXECUTE FUNCTION notify_document_change();

CREATE TRIGGER trigger_notify_processing_status
    AFTER UPDATE ON documents
    FOR EACH ROW EXECUTE FUNCTION notify_processing_status_change();

CREATE TRIGGER trigger_notify_cost_update
    AFTER INSERT ON llm_usage
    FOR EACH ROW EXECUTE FUNCTION notify_cost_update();

-- ========================================
-- MAINTENANCE FUNCTIONS
-- ========================================

-- Refresh all materialized views
CREATE OR REPLACE FUNCTION refresh_all_materialized_views()
RETURNS TEXT AS $$
DECLARE
    start_time TIMESTAMP;
    end_time TIMESTAMP;
BEGIN
    start_time := CURRENT_TIMESTAMP;
    
    REFRESH MATERIALIZED VIEW CONCURRENTLY document_search_view;
    REFRESH MATERIALIZED VIEW CONCURRENTLY knowledge_graph_analytics_view;
    REFRESH MATERIALIZED VIEW CONCURRENTLY cost_analytics_view;
    
    end_time := CURRENT_TIMESTAMP;
    
    RETURN 'Materialized views refreshed in ' || 
           EXTRACT(EPOCH FROM (end_time - start_time)) || ' seconds';
END;
$$ LANGUAGE plpgsql;

-- Clean up old monitoring data
CREATE OR REPLACE FUNCTION cleanup_old_monitoring_data(days_to_keep INTEGER DEFAULT 90)
RETURNS TEXT AS $$
DECLARE
    cutoff_date TIMESTAMP;
    deleted_count INTEGER := 0;
BEGIN
    cutoff_date := CURRENT_TIMESTAMP - INTERVAL '1 day' * days_to_keep;
    
    -- Clean up old system health records (keep daily aggregates)
    DELETE FROM system_health 
    WHERE measured_at < cutoff_date
    AND EXTRACT(HOUR FROM measured_at) != 0; -- Keep midnight records as daily snapshots
    
    GET DIAGNOSTICS deleted_count = ROW_COUNT;
    
    RETURN 'Cleaned up ' || deleted_count || ' old monitoring records';
END;
$$ LANGUAGE plpgsql;

-- Analyze and optimize database performance
CREATE OR REPLACE FUNCTION analyze_database_performance()
RETURNS TABLE (
    table_name TEXT,
    table_size TEXT,
    index_usage DECIMAL,
    seq_scan_ratio DECIMAL,
    recommendation TEXT
) AS $$
BEGIN
    RETURN QUERY
    WITH table_stats AS (
        SELECT 
            schemaname,
            tablename,
            pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) as size,
            seq_scan::decimal,
            seq_tup_read::decimal,
            idx_scan::decimal,
            idx_tup_fetch::decimal,
            n_tup_ins + n_tup_upd + n_tup_del as total_writes
        FROM pg_stat_user_tables
        WHERE schemaname = 'public'
    )
    SELECT 
        ts.tablename::TEXT,
        ts.size::TEXT,
        CASE 
            WHEN (ts.seq_scan + ts.idx_scan) > 0 
            THEN ts.idx_scan / (ts.seq_scan + ts.idx_scan) * 100
            ELSE 0 
        END as index_usage,
        CASE 
            WHEN (ts.seq_tup_read + ts.idx_tup_fetch) > 0 
            THEN ts.seq_tup_read / (ts.seq_tup_read + ts.idx_tup_fetch) * 100
            ELSE 0 
        END as seq_scan_ratio,
        CASE 
            WHEN ts.seq_tup_read / GREATEST(ts.seq_scan, 1) > 10000 
            THEN 'Consider adding indexes - high sequential scan cost'
            WHEN ts.idx_scan = 0 AND ts.total_writes > 1000 
            THEN 'Table has writes but no index usage'
            WHEN ts.seq_scan > ts.idx_scan * 2 
            THEN 'Sequential scans dominate - review query patterns'
            ELSE 'Performance looks good'
        END as recommendation
    FROM table_stats ts
    ORDER BY pg_total_relation_size('public.'||ts.tablename) DESC;
END;
$$ LANGUAGE plpgsql;

-- Comments for documentation
COMMENT ON FUNCTION hybrid_search IS 'Combines vector similarity and full-text search with configurable weights';
COMMENT ON FUNCTION semantic_search IS 'Pure vector similarity search for documents';
COMMENT ON FUNCTION find_similar_nodes IS 'Finds knowledge nodes similar to a given node using embeddings';
COMMENT ON FUNCTION calculate_graph_metrics IS 'Calculates comprehensive knowledge graph statistics';
COMMENT ON FUNCTION get_cost_summary IS 'Provides detailed cost analysis for a user and date range';
COMMENT ON FUNCTION refresh_all_materialized_views IS 'Refreshes all materialized views for updated search performance';
COMMENT ON FUNCTION cleanup_old_monitoring_data IS 'Removes old monitoring data while preserving daily snapshots';
COMMENT ON FUNCTION analyze_database_performance IS 'Analyzes table and index usage patterns for optimization recommendations';

-- Set up automatic materialized view refresh (run every hour)
-- This would typically be handled by a cron job or scheduled task
-- INSERT INTO processing_queue (task_type, task_data, scheduled_for)
-- VALUES ('refresh_materialized_views', '{}', CURRENT_TIMESTAMP + INTERVAL '1 hour');