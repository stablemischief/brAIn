-- brAIn v2.0 Enhanced Database Schema
-- Migration 001: Base schema with AI-first enhancements
-- Created: 2025-09-11
-- Purpose: Enhanced document storage with vector embeddings, cost tracking, and quality metrics

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "vector";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";
CREATE EXTENSION IF NOT EXISTS "btree_gin";

-- Create custom types for better type safety and performance
CREATE TYPE processing_status AS ENUM (
    'pending',
    'processing', 
    'completed',
    'failed',
    'skipped'
);

CREATE TYPE document_type AS ENUM (
    'pdf',
    'docx',
    'xlsx', 
    'pptx',
    'txt',
    'md',
    'html',
    'csv',
    'json',
    'xml',
    'epub',
    'rtf',
    'odt',
    'other'
);

CREATE TYPE language_code AS ENUM (
    'en',
    'es',
    'fr',
    'de',
    'it',
    'pt',
    'ru',
    'zh',
    'ja',
    'ko',
    'ar',
    'hi',
    'unknown'
);

-- Users table with enhanced authentication support
CREATE TABLE IF NOT EXISTS users (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    email TEXT UNIQUE NOT NULL,
    display_name TEXT,
    avatar_url TEXT,
    role TEXT DEFAULT 'user',
    
    -- Authentication fields
    auth_provider TEXT DEFAULT 'supabase',
    external_id TEXT,
    email_verified BOOLEAN DEFAULT FALSE,
    
    -- Preferences and settings
    preferences JSONB DEFAULT '{}',
    settings JSONB DEFAULT '{}',
    
    -- Cost tracking per user
    monthly_budget_limit DECIMAL(10,2) DEFAULT 100.00,
    current_month_spend DECIMAL(10,2) DEFAULT 0.00,
    
    -- Audit fields
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    last_seen_at TIMESTAMP WITH TIME ZONE,
    
    -- Soft delete support
    deleted_at TIMESTAMP WITH TIME ZONE
);

-- Google Drive folders table with enhanced sync capabilities
CREATE TABLE IF NOT EXISTS folders (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    
    -- Google Drive fields
    google_folder_id TEXT NOT NULL,
    folder_name TEXT NOT NULL,
    folder_path TEXT,
    parent_folder_id TEXT,
    
    -- Sync configuration
    auto_sync_enabled BOOLEAN DEFAULT TRUE,
    sync_frequency_minutes INTEGER DEFAULT 60,
    include_subfolders BOOLEAN DEFAULT TRUE,
    file_type_filters TEXT[] DEFAULT ARRAY[]::TEXT[],
    max_file_size_mb INTEGER DEFAULT 50,
    
    -- Status and monitoring
    last_sync_at TIMESTAMP WITH TIME ZONE,
    last_successful_sync_at TIMESTAMP WITH TIME ZONE,
    sync_status processing_status DEFAULT 'pending',
    sync_error_message TEXT,
    
    -- Statistics
    total_files INTEGER DEFAULT 0,
    processed_files INTEGER DEFAULT 0,
    failed_files INTEGER DEFAULT 0,
    total_size_bytes BIGINT DEFAULT 0,
    
    -- Audit fields
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    
    -- Constraints
    UNIQUE(user_id, google_folder_id)
);

-- Enhanced documents table with AI-first features
CREATE TABLE IF NOT EXISTS documents (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    folder_id UUID NOT NULL REFERENCES folders(id) ON DELETE CASCADE,
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    
    -- Google Drive metadata
    google_file_id TEXT UNIQUE NOT NULL,
    file_name TEXT NOT NULL,
    file_path TEXT,
    mime_type TEXT,
    file_size_bytes BIGINT,
    google_modified_at TIMESTAMP WITH TIME ZONE,
    
    -- Enhanced AI-focused fields
    document_type document_type,
    content_hash TEXT UNIQUE, -- SHA-256 hash for duplicate detection
    language_code language_code DEFAULT 'unknown',
    
    -- Text extraction and processing
    raw_text TEXT,
    processed_text TEXT,
    text_length INTEGER,
    
    -- Quality assessment
    extraction_quality DECIMAL(3,2) DEFAULT 0.0, -- 0.0-1.0 quality score
    extraction_confidence DECIMAL(3,2) DEFAULT 0.0, -- 0.0-1.0 confidence score
    processing_notes JSONB DEFAULT '{}',
    
    -- Vector embeddings (OpenAI text-embedding-3-small: 1536 dimensions)
    embedding VECTOR(1536),
    embedding_model TEXT DEFAULT 'text-embedding-3-small',
    embedding_created_at TIMESTAMP WITH TIME ZONE,
    
    -- Cost tracking
    processing_cost DECIMAL(8,4) DEFAULT 0.0000, -- Track processing costs
    token_count INTEGER DEFAULT 0, -- Input tokens for embedding
    
    -- Processing status and errors
    processing_status processing_status DEFAULT 'pending',
    processing_started_at TIMESTAMP WITH TIME ZONE,
    processing_completed_at TIMESTAMP WITH TIME ZONE,
    processing_error_message TEXT,
    processing_retry_count INTEGER DEFAULT 0,
    
    -- Metadata and tags
    metadata JSONB DEFAULT '{}',
    tags TEXT[] DEFAULT ARRAY[]::TEXT[],
    
    -- Audit fields
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    
    -- Soft delete support
    deleted_at TIMESTAMP WITH TIME ZONE
);

-- User sessions for activity tracking
CREATE TABLE IF NOT EXISTS user_sessions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    
    session_token TEXT UNIQUE NOT NULL,
    ip_address INET,
    user_agent TEXT,
    
    -- Session data
    session_data JSONB DEFAULT '{}',
    
    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    last_accessed_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    expires_at TIMESTAMP WITH TIME ZONE DEFAULT (CURRENT_TIMESTAMP + INTERVAL '7 days'),
    
    -- Activity tracking
    page_views INTEGER DEFAULT 0,
    api_calls INTEGER DEFAULT 0,
    last_activity JSONB DEFAULT '{}'
);

-- Processing queue for background tasks
CREATE TABLE IF NOT EXISTS processing_queue (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    document_id UUID REFERENCES documents(id) ON DELETE CASCADE,
    folder_id UUID REFERENCES folders(id) ON DELETE CASCADE,
    
    -- Task details
    task_type TEXT NOT NULL, -- 'extract', 'embed', 'sync', 'analyze'
    task_data JSONB DEFAULT '{}',
    priority INTEGER DEFAULT 5, -- 1-10, higher = more urgent
    
    -- Processing details
    status processing_status DEFAULT 'pending',
    started_at TIMESTAMP WITH TIME ZONE,
    completed_at TIMESTAMP WITH TIME ZONE,
    error_message TEXT,
    retry_count INTEGER DEFAULT 0,
    max_retries INTEGER DEFAULT 3,
    
    -- Scheduling
    scheduled_for TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    worker_id TEXT,
    
    -- Audit fields
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Update timestamps automatically
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Apply update triggers to all tables with updated_at
CREATE TRIGGER update_users_updated_at BEFORE UPDATE ON users FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_folders_updated_at BEFORE UPDATE ON folders FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_documents_updated_at BEFORE UPDATE ON documents FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_user_sessions_updated_at BEFORE UPDATE ON user_sessions FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_processing_queue_updated_at BEFORE UPDATE ON processing_queue FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Basic indexes for core performance
CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);
CREATE INDEX IF NOT EXISTS idx_users_external_id ON users(external_id);
CREATE INDEX IF NOT EXISTS idx_folders_user_id ON folders(user_id);
CREATE INDEX IF NOT EXISTS idx_folders_google_folder_id ON folders(google_folder_id);
CREATE INDEX IF NOT EXISTS idx_folders_sync_status ON folders(sync_status);
CREATE INDEX IF NOT EXISTS idx_documents_folder_id ON documents(folder_id);
CREATE INDEX IF NOT EXISTS idx_documents_user_id ON documents(user_id);
CREATE INDEX IF NOT EXISTS idx_documents_google_file_id ON documents(google_file_id);
CREATE INDEX IF NOT EXISTS idx_documents_content_hash ON documents(content_hash);
CREATE INDEX IF NOT EXISTS idx_documents_processing_status ON documents(processing_status);
CREATE INDEX IF NOT EXISTS idx_documents_created_at ON documents(created_at);
CREATE INDEX IF NOT EXISTS idx_processing_queue_status ON processing_queue(status);
CREATE INDEX IF NOT EXISTS idx_processing_queue_scheduled_for ON processing_queue(scheduled_for);
CREATE INDEX IF NOT EXISTS idx_processing_queue_priority ON processing_queue(priority DESC);

-- Composite indexes for common query patterns
CREATE INDEX IF NOT EXISTS idx_documents_user_status ON documents(user_id, processing_status);
CREATE INDEX IF NOT EXISTS idx_documents_folder_status ON documents(folder_id, processing_status);
CREATE INDEX IF NOT EXISTS idx_folders_user_sync ON folders(user_id, sync_status);

-- GIN indexes for JSONB and array fields
CREATE INDEX IF NOT EXISTS idx_documents_metadata_gin ON documents USING GIN(metadata);
CREATE INDEX IF NOT EXISTS idx_documents_tags_gin ON documents USING GIN(tags);
CREATE INDEX IF NOT EXISTS idx_users_preferences_gin ON users USING GIN(preferences);
CREATE INDEX IF NOT EXISTS idx_folders_file_filters_gin ON folders USING GIN(file_type_filters);

-- Text search indexes
CREATE INDEX IF NOT EXISTS idx_documents_text_search ON documents USING GIN(to_tsvector('english', COALESCE(processed_text, raw_text, '')));
CREATE INDEX IF NOT EXISTS idx_documents_file_name_trgm ON documents USING GIN(file_name gin_trgm_ops);

-- Add comments for documentation
COMMENT ON TABLE users IS 'User accounts with authentication and preference management';
COMMENT ON TABLE folders IS 'Google Drive folders with sync configuration and monitoring';
COMMENT ON TABLE documents IS 'Enhanced document storage with AI processing, embeddings, and cost tracking';
COMMENT ON TABLE user_sessions IS 'User session tracking for security and analytics';
COMMENT ON TABLE processing_queue IS 'Background task queue for document processing';

COMMENT ON COLUMN documents.content_hash IS 'SHA-256 hash for duplicate detection and content verification';
COMMENT ON COLUMN documents.embedding IS '1536-dimensional vector embedding for semantic search';
COMMENT ON COLUMN documents.extraction_quality IS 'AI-assessed quality score (0.0-1.0) of text extraction';
COMMENT ON COLUMN documents.processing_cost IS 'Cumulative cost in USD for all AI operations on this document';
COMMENT ON COLUMN documents.language_code IS 'Detected language code for content localization';

-- Initial data for system
INSERT INTO users (id, email, display_name, role) 
VALUES ('00000000-0000-0000-0000-000000000000', 'system@brain.ai', 'System', 'admin')
ON CONFLICT (email) DO NOTHING;