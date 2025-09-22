"""
SQL script generator for database setup and configuration.

This module generates SQL scripts for database initialization,
schema creation, and required extensions based on configuration.
"""

from typing import List, Dict, Any, Optional
from datetime import datetime

from config.models import DatabaseConfig, SystemConfig


class SQLScriptGenerator:
    """Generate SQL scripts for database setup."""

    def __init__(self, config: SystemConfig):
        """
        Initialize SQL script generator.

        Args:
            config: System configuration
        """
        self.config = config
        self.db_config = config.database

    def generate_complete_setup(self) -> str:
        """
        Generate complete database setup script.

        Returns:
            Complete SQL setup script
        """
        scripts = [
            self._generate_header(),
            self._generate_extensions(),
            self._generate_schema(),
            self._generate_tables(),
            self._generate_indexes(),
            self._generate_functions(),
            self._generate_triggers(),
            self._generate_initial_data(),
            self._generate_permissions()
        ]

        return "\n\n".join(filter(None, scripts))

    def _generate_header(self) -> str:
        """Generate SQL script header."""
        return f"""-- brAIn Enhanced RAG Pipeline Database Setup
-- Generated: {datetime.utcnow().isoformat()}
-- Database: {self.db_config.database}
-- Schema: {self.db_config.schema}
-- Environment: {self.config.environment}

-- This script sets up the complete database structure for brAIn
-- WARNING: This will create new tables. Ensure you have backups if updating existing database.

\\echo 'Starting database setup...'"""

    def _generate_extensions(self) -> str:
        """Generate extension installation commands."""
        return """-- Install required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgvector";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";  -- For text search
CREATE EXTENSION IF NOT EXISTS "btree_gin"; -- For composite indexes

\\echo 'Extensions installed successfully'"""

    def _generate_schema(self) -> str:
        """Generate schema creation commands."""
        schema = self.db_config.schema
        if schema == "public":
            return "-- Using public schema"

        return f"""-- Create schema if not exists
CREATE SCHEMA IF NOT EXISTS {schema};

-- Set search path
SET search_path TO {schema}, public;

\\echo 'Schema {schema} configured'"""

    def _generate_tables(self) -> str:
        """Generate table creation commands."""
        return """-- Main documents table with enhanced fields
CREATE TABLE IF NOT EXISTS documents (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    google_id VARCHAR(255) UNIQUE,
    name VARCHAR(500) NOT NULL,
    content TEXT,
    folder_id UUID,

    -- Enhanced fields for AI processing
    embedding vector(1536),
    metadata JSONB DEFAULT '{}'::jsonb,
    processing_status VARCHAR(50) DEFAULT 'pending',
    processing_started_at TIMESTAMP,
    processing_completed_at TIMESTAMP,
    processing_error TEXT,

    -- Quality and cost tracking
    content_hash VARCHAR(64),
    language_code VARCHAR(10),
    extraction_method VARCHAR(50),
    extraction_quality FLOAT,
    processing_cost DECIMAL(10, 6),
    token_count INTEGER,

    -- Timestamps
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    indexed_at TIMESTAMP,
    last_accessed_at TIMESTAMP
);

-- Folders table
CREATE TABLE IF NOT EXISTS folders (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    google_id VARCHAR(255) UNIQUE,
    name VARCHAR(500) NOT NULL,
    parent_id UUID REFERENCES folders(id) ON DELETE CASCADE,
    metadata JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Knowledge graph nodes table
CREATE TABLE IF NOT EXISTS knowledge_nodes (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    document_id UUID REFERENCES documents(id) ON DELETE CASCADE,
    node_type VARCHAR(100) NOT NULL,
    node_value TEXT NOT NULL,
    confidence FLOAT DEFAULT 1.0,
    metadata JSONB DEFAULT '{}'::jsonb,
    embedding vector(1536),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Knowledge graph edges table
CREATE TABLE IF NOT EXISTS knowledge_edges (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    source_node_id UUID REFERENCES knowledge_nodes(id) ON DELETE CASCADE,
    target_node_id UUID REFERENCES knowledge_nodes(id) ON DELETE CASCADE,
    relationship_type VARCHAR(100) NOT NULL,
    weight FLOAT DEFAULT 1.0,
    metadata JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(source_node_id, target_node_id, relationship_type)
);

-- LLM usage tracking table
CREATE TABLE IF NOT EXISTS llm_usage (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    document_id UUID REFERENCES documents(id) ON DELETE SET NULL,
    operation_type VARCHAR(100) NOT NULL,
    model VARCHAR(100) NOT NULL,
    prompt_tokens INTEGER,
    completion_tokens INTEGER,
    total_tokens INTEGER,
    cost DECIMAL(10, 6),
    latency_ms INTEGER,
    metadata JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Processing analytics table
CREATE TABLE IF NOT EXISTS processing_analytics (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    time_bucket TIMESTAMP NOT NULL,
    bucket_type VARCHAR(20) NOT NULL, -- 'hour', 'day', 'week'
    documents_processed INTEGER DEFAULT 0,
    total_tokens INTEGER DEFAULT 0,
    total_cost DECIMAL(10, 4) DEFAULT 0,
    avg_processing_time_ms INTEGER,
    error_count INTEGER DEFAULT 0,
    metadata JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(time_bucket, bucket_type)
);

-- System health monitoring table
CREATE TABLE IF NOT EXISTS system_health (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    service_name VARCHAR(100) NOT NULL,
    status VARCHAR(50) NOT NULL,
    health_score FLOAT,
    metrics JSONB DEFAULT '{}'::jsonb,
    error_details TEXT,
    checked_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    response_time_ms INTEGER
);

-- Search history table
CREATE TABLE IF NOT EXISTS search_history (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id VARCHAR(255),
    query TEXT NOT NULL,
    query_embedding vector(1536),
    results_count INTEGER,
    result_ids UUID[],
    search_type VARCHAR(50), -- 'semantic', 'keyword', 'hybrid'
    latency_ms INTEGER,
    metadata JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Configuration versions table
CREATE TABLE IF NOT EXISTS configuration_versions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    version VARCHAR(50) NOT NULL,
    config JSONB NOT NULL,
    checksum VARCHAR(64),
    environment VARCHAR(50),
    deployed_by VARCHAR(255),
    deployed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    rollback_from UUID REFERENCES configuration_versions(id),
    is_active BOOLEAN DEFAULT FALSE
);

\\echo 'Tables created successfully'"""

    def _generate_indexes(self) -> str:
        """Generate index creation commands."""
        return """-- Create indexes for performance optimization

-- Documents table indexes
CREATE INDEX IF NOT EXISTS idx_documents_folder_id ON documents(folder_id);
CREATE INDEX IF NOT EXISTS idx_documents_processing_status ON documents(processing_status);
CREATE INDEX IF NOT EXISTS idx_documents_created_at ON documents(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_documents_metadata ON documents USING GIN(metadata);
CREATE INDEX IF NOT EXISTS idx_documents_content_search ON documents USING GIN(to_tsvector('english', content));

-- Vector similarity search index (HNSW)
CREATE INDEX IF NOT EXISTS idx_documents_embedding ON documents
USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);

-- Knowledge nodes indexes
CREATE INDEX IF NOT EXISTS idx_knowledge_nodes_document_id ON knowledge_nodes(document_id);
CREATE INDEX IF NOT EXISTS idx_knowledge_nodes_type ON knowledge_nodes(node_type);
CREATE INDEX IF NOT EXISTS idx_knowledge_nodes_embedding ON knowledge_nodes
USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);

-- Knowledge edges indexes
CREATE INDEX IF NOT EXISTS idx_knowledge_edges_source ON knowledge_edges(source_node_id);
CREATE INDEX IF NOT EXISTS idx_knowledge_edges_target ON knowledge_edges(target_node_id);
CREATE INDEX IF NOT EXISTS idx_knowledge_edges_type ON knowledge_edges(relationship_type);

-- LLM usage indexes
CREATE INDEX IF NOT EXISTS idx_llm_usage_created_at ON llm_usage(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_llm_usage_model ON llm_usage(model);
CREATE INDEX IF NOT EXISTS idx_llm_usage_operation ON llm_usage(operation_type);

-- Analytics indexes
CREATE INDEX IF NOT EXISTS idx_analytics_bucket ON processing_analytics(time_bucket, bucket_type);

-- System health indexes
CREATE INDEX IF NOT EXISTS idx_health_service ON system_health(service_name, checked_at DESC);
CREATE INDEX IF NOT EXISTS idx_health_status ON system_health(status);

-- Search history indexes
CREATE INDEX IF NOT EXISTS idx_search_history_user ON search_history(user_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_search_history_embedding ON search_history
USING hnsw (query_embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);

\\echo 'Indexes created successfully'"""

    def _generate_functions(self) -> str:
        """Generate stored functions."""
        return """-- Utility functions

-- Update timestamp trigger function
CREATE OR REPLACE FUNCTION update_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Similarity search function
CREATE OR REPLACE FUNCTION search_similar_documents(
    query_embedding vector(1536),
    match_count int DEFAULT 10,
    match_threshold float DEFAULT 0.7
)
RETURNS TABLE (
    id UUID,
    name VARCHAR,
    content TEXT,
    similarity float
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        d.id,
        d.name,
        d.content,
        1 - (d.embedding <=> query_embedding) as similarity
    FROM documents d
    WHERE d.embedding IS NOT NULL
    AND 1 - (d.embedding <=> query_embedding) > match_threshold
    ORDER BY d.embedding <=> query_embedding
    LIMIT match_count;
END;
$$ LANGUAGE plpgsql;

-- Calculate processing cost function
CREATE OR REPLACE FUNCTION calculate_processing_cost(
    model VARCHAR,
    prompt_tokens INT,
    completion_tokens INT
)
RETURNS DECIMAL AS $$
DECLARE
    cost_per_1k DECIMAL;
    total_cost DECIMAL;
BEGIN
    -- Define costs per model (in dollars per 1000 tokens)
    CASE model
        WHEN 'gpt-4o' THEN cost_per_1k := 0.005;
        WHEN 'gpt-4o-mini' THEN cost_per_1k := 0.00015;
        WHEN 'claude-3-5-sonnet-20241022' THEN cost_per_1k := 0.003;
        WHEN 'text-embedding-3-small' THEN cost_per_1k := 0.00002;
        ELSE cost_per_1k := 0.001; -- Default cost
    END CASE;

    total_cost := ((prompt_tokens + completion_tokens) / 1000.0) * cost_per_1k;
    RETURN total_cost;
END;
$$ LANGUAGE plpgsql;

\\echo 'Functions created successfully'"""

    def _generate_triggers(self) -> str:
        """Generate trigger creation commands."""
        return """-- Create triggers

-- Update timestamp triggers
CREATE TRIGGER update_documents_updated_at
    BEFORE UPDATE ON documents
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at();

CREATE TRIGGER update_folders_updated_at
    BEFORE UPDATE ON folders
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at();

CREATE TRIGGER update_knowledge_nodes_updated_at
    BEFORE UPDATE ON knowledge_nodes
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at();

CREATE TRIGGER update_knowledge_edges_updated_at
    BEFORE UPDATE ON knowledge_edges
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at();

-- Real-time notification triggers for Supabase
CREATE OR REPLACE FUNCTION notify_document_change()
RETURNS TRIGGER AS $$
BEGIN
    PERFORM pg_notify(
        'document_changes',
        json_build_object(
            'operation', TG_OP,
            'document_id', COALESCE(NEW.id, OLD.id),
            'status', COALESCE(NEW.processing_status, OLD.processing_status)
        )::text
    );
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER document_change_notification
    AFTER INSERT OR UPDATE OR DELETE ON documents
    FOR EACH ROW
    EXECUTE FUNCTION notify_document_change();

-- Cost tracking trigger
CREATE OR REPLACE FUNCTION track_llm_cost()
RETURNS TRIGGER AS $$
BEGIN
    NEW.cost := calculate_processing_cost(
        NEW.model,
        NEW.prompt_tokens,
        NEW.completion_tokens
    );
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER calculate_llm_cost
    BEFORE INSERT ON llm_usage
    FOR EACH ROW
    EXECUTE FUNCTION track_llm_cost();

\\echo 'Triggers created successfully'"""

    def _generate_initial_data(self) -> str:
        """Generate initial data insertion commands."""
        return """-- Insert initial configuration
INSERT INTO configuration_versions (
    version,
    environment,
    config,
    is_active,
    deployed_by
) VALUES (
    '1.0.0',
    '{environment}',
    '{{"generated_by": "sql_generator", "timestamp": "{timestamp}"}}'::jsonb,
    true,
    'system'
) ON CONFLICT DO NOTHING;

-- Create root folder
INSERT INTO folders (
    name,
    google_id,
    metadata
) VALUES (
    'Root',
    'root',
    '{{"is_root": true}}'::jsonb
) ON CONFLICT (google_id) DO NOTHING;

\\echo 'Initial data inserted successfully'""".format(
            environment=self.config.environment,
            timestamp=datetime.utcnow().isoformat()
        )

    def _generate_permissions(self) -> str:
        """Generate permission grant commands."""
        username = self.db_config.username
        schema = self.db_config.schema

        return f"""-- Grant permissions to application user
GRANT USAGE ON SCHEMA {schema} TO {username};
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA {schema} TO {username};
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA {schema} TO {username};
GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA {schema} TO {username};

-- Grant permissions on future objects
ALTER DEFAULT PRIVILEGES IN SCHEMA {schema}
    GRANT ALL PRIVILEGES ON TABLES TO {username};
ALTER DEFAULT PRIVILEGES IN SCHEMA {schema}
    GRANT ALL PRIVILEGES ON SEQUENCES TO {username};
ALTER DEFAULT PRIVILEGES IN SCHEMA {schema}
    GRANT EXECUTE ON FUNCTIONS TO {username};

\\echo 'Permissions granted successfully'
\\echo 'Database setup completed!'"""

    def generate_rollback_script(self) -> str:
        """
        Generate rollback script to undo database setup.

        Returns:
            Rollback SQL script
        """
        schema = self.db_config.schema

        return f"""-- Rollback script for brAIn database setup
-- WARNING: This will DROP all tables and data!
-- Generated: {datetime.utcnow().isoformat()}

\\echo 'Starting rollback...'

-- Drop triggers first
DROP TRIGGER IF EXISTS update_documents_updated_at ON documents;
DROP TRIGGER IF EXISTS update_folders_updated_at ON folders;
DROP TRIGGER IF EXISTS update_knowledge_nodes_updated_at ON knowledge_nodes;
DROP TRIGGER IF EXISTS update_knowledge_edges_updated_at ON knowledge_edges;
DROP TRIGGER IF EXISTS document_change_notification ON documents;
DROP TRIGGER IF EXISTS calculate_llm_cost ON llm_usage;

-- Drop functions
DROP FUNCTION IF EXISTS update_updated_at();
DROP FUNCTION IF EXISTS search_similar_documents(vector, int, float);
DROP FUNCTION IF EXISTS calculate_processing_cost(varchar, int, int);
DROP FUNCTION IF EXISTS notify_document_change();
DROP FUNCTION IF EXISTS track_llm_cost();

-- Drop tables in dependency order
DROP TABLE IF EXISTS search_history CASCADE;
DROP TABLE IF EXISTS system_health CASCADE;
DROP TABLE IF EXISTS processing_analytics CASCADE;
DROP TABLE IF EXISTS llm_usage CASCADE;
DROP TABLE IF EXISTS knowledge_edges CASCADE;
DROP TABLE IF EXISTS knowledge_nodes CASCADE;
DROP TABLE IF EXISTS documents CASCADE;
DROP TABLE IF EXISTS folders CASCADE;
DROP TABLE IF EXISTS configuration_versions CASCADE;

-- Drop schema if not public
{"DROP SCHEMA IF EXISTS " + schema + " CASCADE;" if schema != "public" else "-- Using public schema"}

\\echo 'Rollback completed!'"""

    def generate_migration_script(self, from_version: str, to_version: str) -> str:
        """
        Generate migration script between versions.

        Args:
            from_version: Source version
            to_version: Target version

        Returns:
            Migration SQL script
        """
        # This would contain version-specific migrations
        # For now, return a template
        return f"""-- Migration script from {from_version} to {to_version}
-- Generated: {datetime.utcnow().isoformat()}

\\echo 'Starting migration from {from_version} to {to_version}...'

-- Add migration commands here based on version differences

\\echo 'Migration completed!'"""