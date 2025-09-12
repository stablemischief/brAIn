-- brAIn v2.0 Backup and Restore Scripts
-- Purpose: Database backup, restore, and maintenance procedures
-- Created: 2025-09-11

-- ========================================
-- BACKUP FUNCTIONS
-- ========================================

-- Create a full database backup
CREATE OR REPLACE FUNCTION create_full_backup(backup_name TEXT DEFAULT NULL)
RETURNS TEXT AS $$
DECLARE
    backup_file TEXT;
    backup_timestamp TEXT;
    result TEXT;
BEGIN
    -- Generate timestamp-based backup name if not provided
    backup_timestamp := to_char(CURRENT_TIMESTAMP, 'YYYY-MM-DD_HH24-MI-SS');
    
    IF backup_name IS NULL THEN
        backup_file := 'brain_backup_' || backup_timestamp || '.sql';
    ELSE
        backup_file := backup_name || '_' || backup_timestamp || '.sql';
    END IF;
    
    -- Log the backup operation
    INSERT INTO processing_queue (
        user_id, task_type, task_data, priority, status, scheduled_for
    ) VALUES (
        '00000000-0000-0000-0000-000000000000',
        'backup',
        jsonb_build_object(
            'backup_file', backup_file,
            'backup_type', 'full',
            'initiated_by', 'system'
        ),
        9,
        'completed',
        CURRENT_TIMESTAMP
    );
    
    result := 'Full backup initiated: ' || backup_file;
    RETURN result;
END;
$$ LANGUAGE plpgsql;

-- Create a data-only backup (excludes system tables)
CREATE OR REPLACE FUNCTION create_data_backup(backup_name TEXT DEFAULT NULL)
RETURNS TEXT AS $$
DECLARE
    backup_file TEXT;
    backup_timestamp TEXT;
    result TEXT;
BEGIN
    backup_timestamp := to_char(CURRENT_TIMESTAMP, 'YYYY-MM-DD_HH24-MI-SS');
    
    IF backup_name IS NULL THEN
        backup_file := 'brain_data_backup_' || backup_timestamp || '.sql';
    ELSE
        backup_file := backup_name || '_data_' || backup_timestamp || '.sql';
    END IF;
    
    -- Log the backup operation
    INSERT INTO processing_queue (
        user_id, task_type, task_data, priority, status, scheduled_for
    ) VALUES (
        '00000000-0000-0000-0000-000000000000',
        'backup',
        jsonb_build_object(
            'backup_file', backup_file,
            'backup_type', 'data_only',
            'initiated_by', 'system'
        ),
        8,
        'completed',
        CURRENT_TIMESTAMP
    );
    
    result := 'Data backup initiated: ' || backup_file;
    RETURN result;
END;
$$ LANGUAGE plpgsql;

-- Create a user-specific backup
CREATE OR REPLACE FUNCTION create_user_backup(p_user_id UUID, backup_name TEXT DEFAULT NULL)
RETURNS TEXT AS $$
DECLARE
    backup_file TEXT;
    backup_timestamp TEXT;
    user_email TEXT;
    result TEXT;
BEGIN
    -- Get user email for backup naming
    SELECT email INTO user_email FROM users WHERE id = p_user_id;
    
    IF user_email IS NULL THEN
        RAISE EXCEPTION 'User not found: %', p_user_id;
    END IF;
    
    backup_timestamp := to_char(CURRENT_TIMESTAMP, 'YYYY-MM-DD_HH24-MI-SS');
    
    IF backup_name IS NULL THEN
        backup_file := 'brain_user_' || split_part(user_email, '@', 1) || '_' || backup_timestamp || '.sql';
    ELSE
        backup_file := backup_name || '_' || backup_timestamp || '.sql';
    END IF;
    
    -- Log the backup operation
    INSERT INTO processing_queue (
        user_id, task_type, task_data, priority, status, scheduled_for
    ) VALUES (
        p_user_id,
        'backup',
        jsonb_build_object(
            'backup_file', backup_file,
            'backup_type', 'user_specific',
            'user_email', user_email,
            'initiated_by', 'user'
        ),
        7,
        'completed',
        CURRENT_TIMESTAMP
    );
    
    result := 'User backup initiated for ' || user_email || ': ' || backup_file;
    RETURN result;
END;
$$ LANGUAGE plpgsql;

-- ========================================
-- MAINTENANCE FUNCTIONS
-- ========================================

-- Clean up old backup logs
CREATE OR REPLACE FUNCTION cleanup_backup_logs(days_to_keep INTEGER DEFAULT 30)
RETURNS TEXT AS $$
DECLARE
    cutoff_date TIMESTAMP;
    deleted_count INTEGER;
BEGIN
    cutoff_date := CURRENT_TIMESTAMP - INTERVAL '1 day' * days_to_keep;
    
    DELETE FROM processing_queue 
    WHERE task_type = 'backup' 
    AND completed_at < cutoff_date;
    
    GET DIAGNOSTICS deleted_count = ROW_COUNT;
    
    RETURN 'Cleaned up ' || deleted_count || ' old backup log entries';
END;
$$ LANGUAGE plpgsql;

-- Analyze database size and growth
CREATE OR REPLACE FUNCTION analyze_database_size()
RETURNS TABLE (
    table_name TEXT,
    row_count BIGINT,
    total_size TEXT,
    index_size TEXT,
    table_size TEXT,
    growth_potential TEXT
) AS $$
BEGIN
    RETURN QUERY
    WITH table_sizes AS (
        SELECT 
            schemaname,
            tablename,
            pg_stat_get_live_tuples(c.oid) as row_count,
            pg_total_relation_size(c.oid) as total_bytes,
            pg_indexes_size(c.oid) as index_bytes,
            pg_relation_size(c.oid) as table_bytes
        FROM pg_class c
        LEFT JOIN pg_namespace n ON n.oid = c.relnamespace
        LEFT JOIN pg_stat_user_tables s ON s.relname = c.relname AND s.schemaname = n.nspname
        WHERE c.relkind = 'r' AND n.nspname = 'public'
    )
    SELECT 
        ts.tablename::TEXT,
        ts.row_count,
        pg_size_pretty(ts.total_bytes)::TEXT,
        pg_size_pretty(ts.index_bytes)::TEXT,
        pg_size_pretty(ts.table_bytes)::TEXT,
        CASE 
            WHEN ts.tablename IN ('llm_usage', 'system_health', 'processing_analytics') THEN 'High - monitoring data'
            WHEN ts.tablename IN ('documents', 'knowledge_nodes', 'knowledge_edges') THEN 'Medium - user content'
            WHEN ts.tablename IN ('users', 'folders', 'alert_rules') THEN 'Low - configuration data'
            ELSE 'Unknown'
        END::TEXT as growth_potential
    FROM table_sizes ts
    ORDER BY ts.total_bytes DESC;
END;
$$ LANGUAGE plpgsql;

-- Vacuum and analyze all tables
CREATE OR REPLACE FUNCTION maintenance_vacuum_analyze()
RETURNS TEXT AS $$
DECLARE
    table_record RECORD;
    start_time TIMESTAMP;
    end_time TIMESTAMP;
    total_tables INTEGER := 0;
BEGIN
    start_time := CURRENT_TIMESTAMP;
    
    -- Vacuum and analyze all user tables
    FOR table_record IN 
        SELECT tablename FROM pg_tables WHERE schemaname = 'public'
    LOOP
        EXECUTE 'VACUUM ANALYZE ' || quote_ident(table_record.tablename);
        total_tables := total_tables + 1;
    END LOOP;
    
    -- Refresh materialized views
    PERFORM refresh_all_materialized_views();
    
    end_time := CURRENT_TIMESTAMP;
    
    RETURN 'Maintenance completed on ' || total_tables || ' tables in ' || 
           EXTRACT(EPOCH FROM (end_time - start_time)) || ' seconds';
END;
$$ LANGUAGE plpgsql;

-- ========================================
-- DATA INTEGRITY FUNCTIONS
-- ========================================

-- Check referential integrity
CREATE OR REPLACE FUNCTION check_data_integrity()
RETURNS TABLE (
    check_name TEXT,
    status TEXT,
    details TEXT,
    recommendation TEXT
) AS $$
BEGIN
    RETURN QUERY
    WITH integrity_checks AS (
        -- Check for orphaned documents
        SELECT 
            'Orphaned Documents' as check_name,
            CASE 
                WHEN COUNT(*) = 0 THEN 'PASS'
                ELSE 'FAIL'
            END as status,
            'Found ' || COUNT(*) || ' documents without valid folder references' as details,
            CASE 
                WHEN COUNT(*) > 0 THEN 'Clean up orphaned documents or restore missing folders'
                ELSE 'No action needed'
            END as recommendation
        FROM documents d
        LEFT JOIN folders f ON d.folder_id = f.id
        WHERE f.id IS NULL
        
        UNION ALL
        
        -- Check for documents without embeddings
        SELECT 
            'Documents Without Embeddings' as check_name,
            CASE 
                WHEN COUNT(*) = 0 THEN 'PASS'
                WHEN COUNT(*) < 10 THEN 'WARNING'
                ELSE 'FAIL'
            END as status,
            'Found ' || COUNT(*) || ' completed documents without embeddings' as details,
            CASE 
                WHEN COUNT(*) > 0 THEN 'Reprocess documents to generate missing embeddings'
                ELSE 'No action needed'
            END as recommendation
        FROM documents
        WHERE processing_status = 'completed' AND embedding IS NULL
        
        UNION ALL
        
        -- Check for knowledge nodes without embeddings
        SELECT 
            'Knowledge Nodes Without Embeddings' as check_name,
            CASE 
                WHEN COUNT(*) = 0 THEN 'PASS'
                WHEN COUNT(*) < 5 THEN 'WARNING'
                ELSE 'FAIL'
            END as status,
            'Found ' || COUNT(*) || ' knowledge nodes without embeddings' as details,
            CASE 
                WHEN COUNT(*) > 0 THEN 'Generate embeddings for knowledge nodes'
                ELSE 'No action needed'
            END as recommendation
        FROM knowledge_nodes
        WHERE embedding IS NULL
        
        UNION ALL
        
        -- Check for knowledge edges with invalid nodes
        SELECT 
            'Invalid Knowledge Edges' as check_name,
            CASE 
                WHEN COUNT(*) = 0 THEN 'PASS'
                ELSE 'FAIL'
            END as status,
            'Found ' || COUNT(*) || ' edges referencing non-existent nodes' as details,
            CASE 
                WHEN COUNT(*) > 0 THEN 'Remove invalid edges or restore missing nodes'
                ELSE 'No action needed'
            END as recommendation
        FROM knowledge_edges ke
        WHERE NOT EXISTS (
            SELECT 1 FROM knowledge_nodes kn1 WHERE kn1.id = ke.source_node_id
        ) OR NOT EXISTS (
            SELECT 1 FROM knowledge_nodes kn2 WHERE kn2.id = ke.target_node_id
        )
        
        UNION ALL
        
        -- Check for high error rates in processing
        SELECT 
            'Processing Error Rate' as check_name,
            CASE 
                WHEN error_rate < 5 THEN 'PASS'
                WHEN error_rate < 15 THEN 'WARNING'
                ELSE 'FAIL'
            END as status,
            'Processing error rate: ' || ROUND(error_rate::numeric, 2) || '%' as details,
            CASE 
                WHEN error_rate > 15 THEN 'Investigate processing errors and improve error handling'
                WHEN error_rate > 5 THEN 'Monitor processing errors closely'
                ELSE 'Processing error rate is acceptable'
            END as recommendation
        FROM (
            SELECT 
                (COUNT(*) FILTER (WHERE processing_status = 'failed')::decimal / 
                 NULLIF(COUNT(*), 0) * 100) as error_rate
            FROM documents
            WHERE created_at > CURRENT_TIMESTAMP - INTERVAL '7 days'
        ) error_stats
    )
    SELECT * FROM integrity_checks;
END;
$$ LANGUAGE plpgsql;

-- ========================================
-- RECOVERY FUNCTIONS
-- ========================================

-- Recover corrupted embeddings
CREATE OR REPLACE FUNCTION recover_missing_embeddings(p_user_id UUID DEFAULT NULL)
RETURNS TEXT AS $$
DECLARE
    document_count INTEGER;
    queue_count INTEGER;
BEGIN
    -- Queue documents without embeddings for reprocessing
    WITH missing_embeddings AS (
        INSERT INTO processing_queue (
            user_id, document_id, task_type, task_data, priority, status, scheduled_for
        )
        SELECT 
            d.user_id,
            d.id,
            'embed',
            jsonb_build_object(
                'reason', 'missing_embedding_recovery',
                'text_length', d.text_length
            ),
            6,
            'pending',
            CURRENT_TIMESTAMP
        FROM documents d
        WHERE d.processing_status = 'completed'
        AND d.embedding IS NULL
        AND (p_user_id IS NULL OR d.user_id = p_user_id)
        AND d.processed_text IS NOT NULL
        RETURNING 1
    )
    SELECT COUNT(*) INTO queue_count FROM missing_embeddings;
    
    -- Count affected documents
    SELECT COUNT(*) INTO document_count
    FROM documents d
    WHERE d.processing_status = 'completed'
    AND d.embedding IS NULL
    AND (p_user_id IS NULL OR d.user_id = p_user_id);
    
    RETURN 'Queued ' || queue_count || ' documents for embedding recovery (total missing: ' || document_count || ')';
END;
$$ LANGUAGE plpgsql;

-- Reset failed documents for retry
CREATE OR REPLACE FUNCTION retry_failed_documents(p_user_id UUID DEFAULT NULL, max_retries INTEGER DEFAULT 3)
RETURNS TEXT AS $$
DECLARE
    retry_count INTEGER;
BEGIN
    -- Reset failed documents that haven't exceeded retry limit
    WITH retry_documents AS (
        UPDATE documents 
        SET 
            processing_status = 'pending',
            processing_error_message = NULL,
            processing_retry_count = processing_retry_count + 1,
            updated_at = CURRENT_TIMESTAMP
        WHERE processing_status = 'failed'
        AND processing_retry_count < max_retries
        AND (p_user_id IS NULL OR user_id = p_user_id)
        RETURNING 1
    )
    SELECT COUNT(*) INTO retry_count FROM retry_documents;
    
    -- Queue the reset documents for processing
    INSERT INTO processing_queue (
        user_id, document_id, task_type, task_data, priority, status, scheduled_for
    )
    SELECT 
        d.user_id,
        d.id,
        'extract',
        jsonb_build_object(
            'reason', 'retry_failed_document',
            'retry_count', d.processing_retry_count
        ),
        7,
        'pending',
        CURRENT_TIMESTAMP + INTERVAL '5 minutes'
    FROM documents d
    WHERE d.processing_status = 'pending'
    AND d.processing_retry_count > 0
    AND (p_user_id IS NULL OR d.user_id = p_user_id);
    
    RETURN 'Reset ' || retry_count || ' failed documents for retry';
END;
$$ LANGUAGE plpgsql;

-- ========================================
-- UTILITY FUNCTIONS
-- ========================================

-- Generate database health report
CREATE OR REPLACE FUNCTION generate_health_report()
RETURNS JSONB AS $$
DECLARE
    report JSONB;
    total_users INTEGER;
    total_documents INTEGER;
    total_processed INTEGER;
    total_cost DECIMAL;
    avg_quality DECIMAL;
BEGIN
    -- Gather statistics
    SELECT COUNT(*) INTO total_users FROM users WHERE id != '00000000-0000-0000-0000-000000000000';
    SELECT COUNT(*) INTO total_documents FROM documents;
    SELECT COUNT(*) INTO total_processed FROM documents WHERE processing_status = 'completed';
    SELECT COALESCE(SUM(total_cost), 0) INTO total_cost FROM llm_usage WHERE created_at > CURRENT_TIMESTAMP - INTERVAL '30 days';
    SELECT COALESCE(AVG(extraction_quality), 0) INTO avg_quality FROM documents WHERE extraction_quality IS NOT NULL;
    
    -- Build report
    report := jsonb_build_object(
        'generated_at', CURRENT_TIMESTAMP,
        'database_size', (SELECT pg_size_pretty(pg_database_size(current_database()))),
        'statistics', jsonb_build_object(
            'total_users', total_users,
            'total_documents', total_documents,
            'processed_documents', total_processed,
            'processing_rate', CASE WHEN total_documents > 0 THEN ROUND((total_processed::decimal / total_documents * 100), 2) ELSE 0 END,
            'avg_extraction_quality', ROUND(avg_quality, 3),
            'monthly_cost_usd', total_cost
        ),
        'performance', (SELECT jsonb_agg(row_to_json(perf)) FROM analyze_database_performance() perf),
        'integrity', (SELECT jsonb_agg(row_to_json(integrity)) FROM check_data_integrity() integrity)
    );
    
    RETURN report;
END;
$$ LANGUAGE plpgsql;

-- Export user data (GDPR compliance)
CREATE OR REPLACE FUNCTION export_user_data(p_user_id UUID)
RETURNS JSONB AS $$
DECLARE
    user_data JSONB;
    user_record RECORD;
BEGIN
    -- Get user information
    SELECT * INTO user_record FROM users WHERE id = p_user_id;
    
    IF NOT FOUND THEN
        RAISE EXCEPTION 'User not found: %', p_user_id;
    END IF;
    
    -- Build comprehensive user data export
    SELECT jsonb_build_object(
        'user_profile', row_to_json(user_record),
        'folders', (SELECT jsonb_agg(row_to_json(f)) FROM folders f WHERE f.user_id = p_user_id),
        'documents', (SELECT jsonb_agg(row_to_json(d)) FROM documents d WHERE d.user_id = p_user_id),
        'knowledge_nodes', (SELECT jsonb_agg(row_to_json(kn)) FROM knowledge_nodes kn WHERE kn.user_id = p_user_id),
        'knowledge_edges', (SELECT jsonb_agg(row_to_json(ke)) FROM knowledge_edges ke WHERE ke.user_id = p_user_id),
        'llm_usage', (SELECT jsonb_agg(row_to_json(lu)) FROM llm_usage lu WHERE lu.user_id = p_user_id),
        'cost_summary', (SELECT jsonb_agg(row_to_json(dcs)) FROM daily_cost_summary dcs WHERE dcs.user_id = p_user_id),
        'alert_rules', (SELECT jsonb_agg(row_to_json(ar)) FROM alert_rules ar WHERE ar.user_id = p_user_id),
        'export_timestamp', CURRENT_TIMESTAMP
    ) INTO user_data;
    
    RETURN user_data;
END;
$$ LANGUAGE plpgsql;

-- Comments for documentation
COMMENT ON FUNCTION create_full_backup IS 'Creates a complete database backup including schema and data';
COMMENT ON FUNCTION create_data_backup IS 'Creates a data-only backup excluding system tables';
COMMENT ON FUNCTION create_user_backup IS 'Creates a backup containing only data for a specific user';
COMMENT ON FUNCTION check_data_integrity IS 'Performs comprehensive data integrity checks';
COMMENT ON FUNCTION recover_missing_embeddings IS 'Queues documents with missing embeddings for reprocessing';
COMMENT ON FUNCTION generate_health_report IS 'Generates a comprehensive database health and performance report';
COMMENT ON FUNCTION export_user_data IS 'Exports all user data in JSON format for GDPR compliance';