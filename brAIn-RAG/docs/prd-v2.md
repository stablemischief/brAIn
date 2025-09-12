# brAIn - Product Requirements Document (PRD)

## Version 2.0 - Enhanced with AI-First Architecture

### Version History
- v1.0: Initial MVP specification (15-day timeline)
- v2.0: Enhanced with Archon knowledge base insights (Claude Code development)

---

## 1. Product Overview

### 1.1 Product Name
**brAIn** - Intelligent RAG Pipeline Management System

### 1.2 Product Description
brAIn is a containerized web application that transforms an existing CLI-based RAG Pipeline tool into an enterprise-ready, team-accessible platform with real-time monitoring, AI-powered validation, and comprehensive observability.

### 1.3 Key Enhancements (v2.0)
- **Real-time Architecture**: Supabase real-time subscriptions for live updates
- **AI-Powered Validation**: Pydantic AI for configuration and data validation
- **LLM Observability**: Langfuse integration for token tracking and cost management
- **Knowledge Graphs**: Document relationship tracking with Mem0/Zep patterns
- **Advanced Vector Search**: Optimized pgvector with HNSW indexing

### 1.4 Target Users
- **Primary:** James (Product Owner)
- **Secondary:** Mitch (Partner)
- **Tertiary:** Selected colleague for testing
- **Team Size:** 2-3 concurrent users (expandable architecture)

### 1.5 Deployment Environment
- **Platform:** Docker container on VPS
- **Access:** Web-based with real-time capabilities
- **Database:** Supabase with PGVector (enhanced indexing)
- **APIs:** OpenAI, Google Drive, Supabase Real-time
- **Monitoring:** Langfuse, custom dashboards

---

## 2. Enhanced User Stories & Acceptance Criteria

### Story 1: Google Drive Folder Management (Enhanced)
**As a team member, I want intelligent folder management with real-time sync status**

**Original Acceptance Criteria:** ✅ Maintained

**New Enhancements:**
- ✅ Real-time sync status via Supabase subscriptions
- ✅ Duplicate document detection using vector similarity
- ✅ Smart folder discovery with AI-suggested related folders
- ✅ Knowledge graph visualization of folder relationships

**Implementation with Pydantic:**
```python
from pydantic import BaseModel, Field, validator
from typing import Optional, List
import re

class GoogleDriveFolderInput(BaseModel):
    """Validated folder input with AI enhancement"""
    folder_id: str = Field(..., regex="^[a-zA-Z0-9_-]+$")
    folder_name: Optional[str] = None
    auto_discover_subfolders: bool = False
    similarity_threshold: float = Field(default=0.95, ge=0.0, le=1.0)
    
    @validator('folder_id')
    def validate_folder_id(cls, v):
        """Ensure valid Google Drive folder ID format"""
        if not re.match(r'^[a-zA-Z0-9_-]{19,}$', v):
            raise ValueError('Invalid Google Drive folder ID format')
        return v

class FolderSyncStatus(BaseModel):
    """Real-time sync status model"""
    folder_id: str
    status: Literal["syncing", "idle", "error", "discovering"]
    files_processed: int = 0
    files_total: int = 0
    current_file: Optional[str] = None
    errors: List[str] = Field(default_factory=list)
```

### Story 2: Processing Status & Logs (Enhanced)
**As a team member, I want comprehensive observability with cost tracking**

**Original Acceptance Criteria:** ✅ Maintained

**New Enhancements:**
- ✅ Token usage tracking per file/folder
- ✅ Cost projection and alerts
- ✅ Processing performance metrics with Langfuse
- ✅ Anomaly detection for unusual patterns
- ✅ Export detailed analytics reports

**LLM Monitoring Schema:**
```python
from datetime import datetime
from decimal import Decimal

class ProcessingMetrics(BaseModel):
    """Enhanced metrics with LLM tracking"""
    file_id: str
    file_name: str
    file_size_bytes: int
    processing_time_ms: int
    
    # Token tracking
    tokens_used: int
    embedding_model: str
    embedding_cost: Decimal
    
    # Quality metrics
    chunk_count: int
    avg_chunk_size: int
    extraction_confidence: float = Field(ge=0.0, le=1.0)
    
    # Langfuse trace
    trace_id: Optional[str] = None
    span_id: Optional[str] = None

class CostAlert(BaseModel):
    """Cost monitoring and alerts"""
    alert_type: Literal["daily_limit", "projection", "spike"]
    current_cost: Decimal
    threshold: Decimal
    projected_daily: Optional[Decimal] = None
    recommendation: str
```

### Story 3: Configuration Without CLI (Enhanced)
**As a team admin, I want AI-assisted configuration with validation**

**Original Acceptance Criteria:** ✅ Maintained

**New Enhancements:**
- ✅ AI-powered configuration assistant
- ✅ Environment validation before deployment
- ✅ Configuration templates for common scenarios
- ✅ Automated security scanning of credentials
- ✅ Migration assistant for existing installations

**AI-Assisted Configuration:**
```python
from pydantic_ai import Agent
from pydantic import HttpUrl, SecretStr

class InstallationConfigV2(BaseModel):
    """Enhanced configuration with AI validation"""
    # Core settings (validated)
    supabase_url: HttpUrl
    supabase_service_key: SecretStr
    supabase_anon_key: SecretStr
    
    # OpenAI settings with model selection
    openai_api_key: SecretStr
    embedding_model: Literal[
        "text-embedding-3-small",
        "text-embedding-3-large",
        "text-embedding-ada-002"
    ] = "text-embedding-3-small"
    
    # Google Drive with enhanced options
    google_service_account_json: str
    google_workspace_domain: Optional[str] = None
    
    # Monitoring configuration
    enable_langfuse: bool = True
    langfuse_public_key: Optional[str] = None
    langfuse_secret_key: Optional[SecretStr] = None
    
    # Advanced settings
    enable_real_time: bool = True
    enable_cost_tracking: bool = True
    daily_token_limit: int = Field(default=1000000, gt=0)
    alert_email: EmailStr
    
    # Performance tuning
    chunk_size: int = Field(default=400, ge=100, le=1000)
    batch_size: int = Field(default=10, ge=1, le=100)
    max_workers: int = Field(default=4, ge=1, le=16)

# AI Configuration Assistant
config_assistant = Agent(
    'claude-3-5-sonnet',
    system_prompt="""You are a configuration assistant for brAIn.
    Help users set up their environment correctly and securely.
    Validate all inputs and suggest optimal settings based on their use case."""
)
```

### Story 4: Search Functionality Testing (Enhanced)
**As a team member, I want semantic search with knowledge graph context**

**Original Acceptance Criteria:** ✅ Maintained

**New Enhancements:**
- ✅ Semantic search with context awareness
- ✅ Related document suggestions
- ✅ Search history and saved queries
- ✅ Knowledge graph traversal
- ✅ Multi-modal search (text + metadata)

**Advanced Search Implementation:**
```python
class SemanticSearchRequest(BaseModel):
    """Enhanced search with multiple strategies"""
    query: str
    search_strategy: Literal["semantic", "keyword", "hybrid", "graph"]
    include_context: bool = True
    similarity_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    max_results: int = Field(default=10, ge=1, le=100)
    
    # Filters
    file_types: Optional[List[str]] = None
    date_range: Optional[DateRange] = None
    folders: Optional[List[str]] = None
    
    # Knowledge graph options
    traverse_depth: int = Field(default=1, ge=0, le=3)
    include_related: bool = True

class SearchResult(BaseModel):
    """Enhanced search result with context"""
    document_id: str
    content: str
    similarity_score: float
    
    # Metadata
    file_name: str
    file_type: str
    folder_path: str
    
    # Context
    related_documents: List[str] = Field(default_factory=list)
    knowledge_path: Optional[List[str]] = None
    
    # Highlights
    matched_chunks: List[str]
    context_before: Optional[str]
    context_after: Optional[str]
```

### Story 5: Dashboard Status Monitoring (Enhanced)
**As a team member, I want predictive monitoring with intelligent alerts**

**Original Acceptance Criteria:** ✅ Maintained

**New Enhancements:**
- ✅ Predictive failure detection
- ✅ API quota forecasting
- ✅ Performance trend analysis
- ✅ Intelligent alert grouping
- ✅ Self-healing capabilities

**Intelligent Monitoring System:**
```python
class HealthStatus(BaseModel):
    """Enhanced health monitoring with predictions"""
    service: str
    status: Literal["up", "degraded", "down", "predicted_failure"]
    latency_ms: Optional[int]
    success_rate: float = Field(ge=0.0, le=1.0)
    
    # Predictions
    predicted_failure_time: Optional[datetime] = None
    failure_probability: Optional[float] = None
    recommended_action: Optional[str] = None
    
    # Resource usage
    cpu_percent: Optional[float]
    memory_mb: Optional[int]
    disk_usage_percent: Optional[float]

class QuotaMonitor(BaseModel):
    """API quota tracking and forecasting"""
    service: str
    current_usage: int
    daily_limit: int
    reset_time: datetime
    
    # Forecasting
    projected_usage: int
    will_exceed: bool
    recommended_rate_limit: Optional[float]
    
    # Historical
    yesterday_usage: int
    weekly_average: float
    trend: Literal["increasing", "stable", "decreasing"]
```

### Story 6: File Type Handling (Enhanced)
**As a team member, I want intelligent file processing with format detection**

**Original Acceptance Criteria:** ✅ Maintained

**New Enhancements:**
- ✅ AI-powered format detection for unknown types
- ✅ Automatic OCR for scanned documents
- ✅ Language detection and translation options
- ✅ Custom extraction rules per file type
- ✅ Preview generation for all formats

**Intelligent File Processing:**
```python
class FileProcessor(BaseModel):
    """Enhanced file processing with AI capabilities"""
    file_id: str
    file_path: str
    detected_type: str
    confidence_score: float
    
    # Processing options
    enable_ocr: bool = False
    target_language: Optional[str] = None
    extract_tables: bool = True
    extract_images: bool = False
    
    # Custom rules
    custom_extractors: Optional[Dict[str, Any]] = None
    preprocessing_steps: List[str] = Field(default_factory=list)
    
    # Quality checks
    min_text_length: int = Field(default=50, ge=0)
    max_error_rate: float = Field(default=0.1, ge=0.0, le=1.0)

class ProcessingResult(BaseModel):
    """Processing result with quality metrics"""
    success: bool
    extracted_text: Optional[str]
    metadata: Dict[str, Any]
    
    # Quality metrics
    extraction_quality: float = Field(ge=0.0, le=1.0)
    language_detected: Optional[str]
    contains_tables: bool = False
    contains_images: bool = False
    
    # Warnings
    warnings: List[str] = Field(default_factory=list)
    suggestions: List[str] = Field(default_factory=list)
```

### Story 7: Folder Association Cleanup (Enhanced)
**As a team member, I want intelligent cleanup with relationship preservation**

**Original Acceptance Criteria:** ✅ Maintained

**New Enhancements:**
- ✅ Relationship preservation options
- ✅ Orphaned document detection
- ✅ Cleanup impact analysis
- ✅ Undo capability with versioning
- ✅ Archival option instead of deletion

**Smart Cleanup System:**
```python
class CleanupOperation(BaseModel):
    """Enhanced cleanup with intelligence"""
    folder_id: str
    operation_type: Literal["delete", "archive", "move"]
    
    # Impact analysis
    documents_affected: int
    relationships_affected: int
    storage_recovered_mb: float
    
    # Options
    preserve_relationships: bool = False
    create_backup: bool = True
    cascade_to_subfolders: bool = False
    
    # Archive settings
    archive_location: Optional[str] = None
    retention_days: Optional[int] = Field(default=30, ge=1)

class CleanupImpactAnalysis(BaseModel):
    """Pre-cleanup impact analysis"""
    total_documents: int
    unique_documents: int
    shared_documents: int
    
    # Relationships
    outgoing_references: List[str]
    incoming_references: List[str]
    orphaned_after_cleanup: List[str]
    
    # Recommendations
    suggested_action: str
    risk_level: Literal["low", "medium", "high"]
    warnings: List[str]
```

---

## 3. Enhanced Technical Architecture

### 3.1 System Architecture (v2.0)

```
┌─────────────────────────────────────────────────────────┐
│                   Docker Container                       │
│  ┌─────────────────────────────────────────────────────┐│
│  │         FastAPI Backend (Port 8000)                 ││
│  │  ┌─────────────────────────────────────────────────┐││
│  │  │   Core API Endpoints                           │││
│  │  │   - /api/auth/* (Supabase Auth + Magic Links)  │││
│  │  │   - /api/folders/* (CRUD + Real-time)          │││
│  │  │   - /api/status/* (Health + Predictions)       │││
│  │  │   - /api/search/* (Semantic + Graph)           │││
│  │  │   - /api/analytics/* (Langfuse + Custom)       │││
│  │  │   - /ws/realtime (Supabase Subscriptions)      │││
│  │  └─────────────────────────────────────────────────┘││
│  │  ┌─────────────────────────────────────────────────┐││
│  │  │   Enhanced RAG Pipeline                        │││
│  │  │   - Pydantic Validation Layer                  │││
│  │  │   - Intelligent Text Extraction                │││
│  │  │   - Multi-Strategy Embedding                   │││
│  │  │   - Knowledge Graph Builder                    │││
│  │  │   - Cost-Aware Processing                      │││
│  │  └─────────────────────────────────────────────────┘││
│  │  ┌─────────────────────────────────────────────────┐││
│  │  │   Monitoring & Observability                   │││
│  │  │   - Langfuse LLM Tracking                      │││
│  │  │   - Custom Metrics Collector                   │││
│  │  │   - Predictive Analytics Engine               │││
│  │  └─────────────────────────────────────────────────┘││
│  └─────────────────────────────────────────────────────┘│
│  ┌─────────────────────────────────────────────────────┐│
│  │       React Frontend (Port 3000)                    ││
│  │   - Real-time Dashboard (WebSocket)                 ││
│  │   - Cost Analytics Views                            ││
│  │   - Knowledge Graph Visualizer                      ││
│  │   - AI Configuration Assistant                      ││
│  └─────────────────────────────────────────────────────┘│
│  ┌─────────────────────────────────────────────────────┐│
│  │          Background Services                        ││
│  │   - Intelligent Polling Scheduler                   ││
│  │   - Predictive Health Monitor                       ││
│  │   - Cost Optimization Engine                        ││
│  │   - Knowledge Graph Indexer                         ││
│  └─────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────┘
                           │
        ┌──────────────────┼──────────────────┐
        ▼                  ▼                  ▼
   Supabase DB        OpenAI API       Google Drive API
   (+ Real-time)      (+ Langfuse)      (+ Workspace)
```

### 3.2 Database Schema (Enhanced)

```sql
-- Enhanced schema with v2.0 features

-- Enable new extensions
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_stat_statements;
CREATE EXTENSION IF NOT EXISTS pg_trgm; -- For text search
CREATE EXTENSION IF NOT EXISTS btree_gin; -- For composite indexes

-- Enhanced documents table with better indexing
CREATE TABLE IF NOT EXISTS documents (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    content TEXT NOT NULL,
    content_hash TEXT GENERATED ALWAYS AS (md5(content)) STORED,
    metadata JSONB NOT NULL,
    embedding vector(1536),
    
    -- New fields for v2.0
    language_code VARCHAR(10),
    extraction_quality FLOAT CHECK (extraction_quality >= 0 AND extraction_quality <= 1),
    processing_cost DECIMAL(10, 6),
    token_count INTEGER,
    
    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    last_accessed_at TIMESTAMP WITH TIME ZONE,
    
    -- Duplicate detection
    UNIQUE(content_hash, metadata->>'folder_id')
);

-- Enhanced indexing strategy
CREATE INDEX idx_documents_embedding_hnsw 
    ON documents USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

CREATE INDEX idx_documents_metadata_gin 
    ON documents USING gin(metadata);

CREATE INDEX idx_documents_language 
    ON documents(language_code) 
    WHERE language_code IS NOT NULL;

CREATE INDEX idx_documents_folder_created 
    ON documents((metadata->>'folder_id'), created_at DESC);

-- Knowledge graph tables
CREATE TABLE IF NOT EXISTS knowledge_nodes (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    document_id UUID REFERENCES documents(id) ON DELETE CASCADE,
    node_type VARCHAR(50) NOT NULL,
    node_value TEXT NOT NULL,
    confidence FLOAT DEFAULT 1.0,
    metadata JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS knowledge_edges (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    source_node_id UUID REFERENCES knowledge_nodes(id) ON DELETE CASCADE,
    target_node_id UUID REFERENCES knowledge_nodes(id) ON DELETE CASCADE,
    edge_type VARCHAR(50) NOT NULL,
    weight FLOAT DEFAULT 1.0,
    metadata JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(source_node_id, target_node_id, edge_type)
);

-- LLM usage tracking
CREATE TABLE IF NOT EXISTS llm_usage (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    document_id UUID REFERENCES documents(id) ON DELETE SET NULL,
    operation_type VARCHAR(50) NOT NULL,
    model_name VARCHAR(100) NOT NULL,
    input_tokens INTEGER NOT NULL,
    output_tokens INTEGER NOT NULL,
    cost DECIMAL(10, 6) NOT NULL,
    latency_ms INTEGER,
    trace_id VARCHAR(255),
    metadata JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Processing analytics
CREATE TABLE IF NOT EXISTS processing_analytics (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    date DATE NOT NULL,
    hour INTEGER CHECK (hour >= 0 AND hour < 24),
    folder_id TEXT,
    
    -- Metrics
    files_processed INTEGER DEFAULT 0,
    files_succeeded INTEGER DEFAULT 0,
    files_failed INTEGER DEFAULT 0,
    
    -- Performance
    avg_processing_time_ms FLOAT,
    p95_processing_time_ms FLOAT,
    
    -- Cost
    total_tokens INTEGER DEFAULT 0,
    total_cost DECIMAL(10, 6) DEFAULT 0,
    
    -- Unique constraint for aggregation
    UNIQUE(date, hour, folder_id)
);

-- Real-time monitoring table
CREATE TABLE IF NOT EXISTS system_health (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    service_name VARCHAR(50) NOT NULL,
    status VARCHAR(20) NOT NULL,
    latency_ms INTEGER,
    success_rate FLOAT,
    cpu_percent FLOAT,
    memory_mb INTEGER,
    error_count INTEGER DEFAULT 0,
    last_error TEXT,
    metadata JSONB,
    checked_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Configuration versioning
CREATE TABLE IF NOT EXISTS config_versions (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    version INTEGER NOT NULL,
    config JSONB NOT NULL,
    changed_by VARCHAR(255),
    change_summary TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(version)
);

-- Create materialized view for search performance
CREATE MATERIALIZED VIEW IF NOT EXISTS document_search_index AS
SELECT 
    d.id,
    d.content,
    d.metadata,
    d.language_code,
    to_tsvector('english', d.content) as content_tsv,
    dm.title,
    dm.url
FROM documents d
LEFT JOIN document_metadata dm ON d.id::text = dm.id;

CREATE INDEX idx_search_content_tsv 
    ON document_search_index USING gin(content_tsv);

-- Functions for real-time subscriptions
CREATE OR REPLACE FUNCTION notify_processing_status()
RETURNS TRIGGER AS $$
BEGIN
    PERFORM pg_notify(
        'processing_status',
        json_build_object(
            'operation', TG_OP,
            'table', TG_TABLE_NAME,
            'data', row_to_json(NEW)
        )::text
    );
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Triggers for real-time updates
CREATE TRIGGER processing_logs_notify
AFTER INSERT OR UPDATE ON processing_logs
FOR EACH ROW EXECUTE FUNCTION notify_processing_status();

CREATE TRIGGER system_health_notify
AFTER INSERT OR UPDATE ON system_health
FOR EACH ROW EXECUTE FUNCTION notify_processing_status();

-- Smart search function with multiple strategies
CREATE OR REPLACE FUNCTION smart_search(
    query_text TEXT,
    query_embedding vector(1536) DEFAULT NULL,
    search_strategy TEXT DEFAULT 'hybrid',
    match_count INT DEFAULT 10,
    similarity_threshold FLOAT DEFAULT 0.7
)
RETURNS TABLE (
    id UUID,
    content TEXT,
    metadata JSONB,
    similarity_score FLOAT,
    rank INTEGER
)
LANGUAGE plpgsql
AS $$
DECLARE
    semantic_weight FLOAT := 0.7;
    keyword_weight FLOAT := 0.3;
BEGIN
    IF search_strategy = 'semantic' AND query_embedding IS NOT NULL THEN
        -- Pure semantic search
        RETURN QUERY
        SELECT 
            d.id,
            d.content,
            d.metadata,
            1 - (d.embedding <=> query_embedding) AS similarity_score,
            ROW_NUMBER() OVER (ORDER BY d.embedding <=> query_embedding) AS rank
        FROM documents d
        WHERE 1 - (d.embedding <=> query_embedding) >= similarity_threshold
        ORDER BY d.embedding <=> query_embedding
        LIMIT match_count;
        
    ELSIF search_strategy = 'keyword' THEN
        -- Pure keyword search
        RETURN QUERY
        SELECT 
            d.id,
            d.content,
            d.metadata,
            ts_rank(to_tsvector('english', d.content), plainto_tsquery('english', query_text)) AS similarity_score,
            ROW_NUMBER() OVER (ORDER BY ts_rank(to_tsvector('english', d.content), plainto_tsquery('english', query_text)) DESC) AS rank
        FROM documents d
        WHERE to_tsvector('english', d.content) @@ plainto_tsquery('english', query_text)
        ORDER BY similarity_score DESC
        LIMIT match_count;
        
    ELSE
        -- Hybrid search (default)
        RETURN QUERY
        WITH semantic_results AS (
            SELECT 
                d.id,
                1 - (d.embedding <=> query_embedding) AS semantic_score
            FROM documents d
            WHERE query_embedding IS NOT NULL
            ORDER BY d.embedding <=> query_embedding
            LIMIT match_count * 2
        ),
        keyword_results AS (
            SELECT 
                d.id,
                ts_rank(to_tsvector('english', d.content), plainto_tsquery('english', query_text)) AS keyword_score
            FROM documents d
            WHERE to_tsvector('english', d.content) @@ plainto_tsquery('english', query_text)
            ORDER BY keyword_score DESC
            LIMIT match_count * 2
        )
        SELECT 
            d.id,
            d.content,
            d.metadata,
            COALESCE(sr.semantic_score * semantic_weight, 0) + 
            COALESCE(kr.keyword_score * keyword_weight, 0) AS similarity_score,
            ROW_NUMBER() OVER (ORDER BY 
                COALESCE(sr.semantic_score * semantic_weight, 0) + 
                COALESCE(kr.keyword_score * keyword_weight, 0) DESC
            ) AS rank
        FROM documents d
        LEFT JOIN semantic_results sr ON d.id = sr.id
        LEFT JOIN keyword_results kr ON d.id = kr.id
        WHERE sr.id IS NOT NULL OR kr.id IS NOT NULL
        ORDER BY similarity_score DESC
        LIMIT match_count;
    END IF;
END;
$$;
```

### 3.3 Real-time Subscription Implementation

```python
# realtime/subscriptions.py
from typing import AsyncGenerator, Dict, Any
import asyncio
from supabase import create_client, Client
from fastapi import WebSocket

class RealtimeManager:
    """Manage real-time subscriptions and broadcasts"""
    
    def __init__(self, supabase: Client):
        self.supabase = supabase
        self.active_connections: List[WebSocket] = []
        self.channels = {}
    
    async def connect(self, websocket: WebSocket):
        """Accept WebSocket connection"""
        await websocket.accept()
        self.active_connections.append(websocket)
        
        # Subscribe to Supabase channels
        await self.setup_subscriptions(websocket)
    
    async def setup_subscriptions(self, websocket: WebSocket):
        """Setup Supabase real-time subscriptions"""
        
        # Processing status channel
        processing_channel = self.supabase.channel('processing-status')
        processing_channel.on_postgres_changes(
            event='*',
            schema='public',
            table='processing_logs',
            callback=lambda payload: asyncio.create_task(
                self.broadcast_update(websocket, 'processing', payload)
            )
        ).subscribe()
        
        # System health channel
        health_channel = self.supabase.channel('system-health')
        health_channel.on_postgres_changes(
            event='*',
            schema='public',
            table='system_health',
            callback=lambda payload: asyncio.create_task(
                self.broadcast_update(websocket, 'health', payload)
            )
        ).subscribe()
        
        # Cost alerts channel
        cost_channel = self.supabase.channel('cost-alerts')
        cost_channel.on_postgres_changes(
            event='INSERT',
            schema='public',
            table='llm_usage',
            callback=lambda payload: asyncio.create_task(
                self.check_cost_threshold(websocket, payload)
            )
        ).subscribe()
        
        self.channels[websocket] = [processing_channel, health_channel, cost_channel]
    
    async def broadcast_update(self, websocket: WebSocket, channel: str, payload: Dict[str, Any]):
        """Broadcast updates to connected clients"""
        message = {
            'type': 'realtime_update',
            'channel': channel,
            'data': payload['new'] if 'new' in payload else payload,
            'timestamp': datetime.now().isoformat()
        }
        
        try:
            await websocket.send_json(message)
        except Exception as e:
            logger.error(f"Failed to send update: {e}")
            await self.disconnect(websocket)
    
    async def check_cost_threshold(self, websocket: WebSocket, payload: Dict[str, Any]):
        """Check if cost threshold exceeded and send alert"""
        if 'new' in payload:
            cost_data = payload['new']
            
            # Get daily total
            daily_total = await self.get_daily_cost_total()
            
            if daily_total > float(os.getenv('DAILY_COST_LIMIT', '10.0')):
                alert = {
                    'type': 'cost_alert',
                    'severity': 'high',
                    'message': f'Daily cost limit exceeded: ${daily_total:.2f}',
                    'data': {
                        'daily_total': daily_total,
                        'limit': float(os.getenv('DAILY_COST_LIMIT', '10.0')),
                        'last_operation': cost_data
                    }
                }
                await websocket.send_json(alert)
    
    async def disconnect(self, websocket: WebSocket):
        """Handle WebSocket disconnection"""
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        
        # Unsubscribe from channels
        if websocket in self.channels:
            for channel in self.channels[websocket]:
                channel.unsubscribe()
            del self.channels[websocket]
```

### 3.4 Langfuse Integration

```python
# monitoring/langfuse_integration.py
from langfuse import Langfuse
from langfuse.decorators import observe, langfuse_context
from typing import Dict, Any, Optional
import functools

# Initialize Langfuse client
langfuse = Langfuse(
    public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
    secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
    host=os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")
)

class LLMObservability:
    """Enhanced LLM observability with Langfuse"""
    
    @observe(name="document-processing")
    async def track_document_processing(
        self, 
        document: Dict[str, Any],
        processing_result: ProcessingResult
    ) -> None:
        """Track document processing with detailed metrics"""
        
        # Create trace
        trace = langfuse.trace(
            name="document-processing",
            input={
                "file_name": document['file_name'],
                "file_type": document['file_type'],
                "file_size": document['file_size']
            },
            metadata={
                "folder_id": document['folder_id'],
                "user_id": document.get('user_id')
            }
        )
        
        # Track extraction
        extraction_span = trace.span(
            name="text-extraction",
            input={"file_type": document['file_type']},
            output={
                "text_length": len(processing_result.extracted_text),
                "quality": processing_result.extraction_quality
            }
        )
        
        # Track chunking
        chunking_span = trace.span(
            name="text-chunking",
            input={"text_length": len(processing_result.extracted_text)},
            output={"chunk_count": len(processing_result.chunks)}
        )
        
        # Track embedding generation
        for i, chunk in enumerate(processing_result.chunks):
            generation = trace.generation(
                name=f"embedding-{i}",
                model=self.embedding_model,
                input=chunk[:100],  # First 100 chars for reference
                output={"dimensions": 1536},
                usage={
                    "input_tokens": self.count_tokens(chunk),
                    "total_tokens": self.count_tokens(chunk),
                    "unit": "TOKENS"
                },
                metadata={
                    "chunk_index": i,
                    "chunk_size": len(chunk)
                }
            )
        
        # Calculate total cost
        total_tokens = sum(self.count_tokens(chunk) for chunk in processing_result.chunks)
        total_cost = self.calculate_cost(total_tokens, self.embedding_model)
        
        # Update trace with final metrics
        trace.update(
            output={
                "success": processing_result.success,
                "chunks_processed": len(processing_result.chunks),
                "total_tokens": total_tokens,
                "total_cost": total_cost
            }
        )
        
        # Store in database for analytics
        await self.store_usage_metrics({
            "document_id": document['id'],
            "operation_type": "embedding",
            "model_name": self.embedding_model,
            "input_tokens": total_tokens,
            "output_tokens": 0,
            "cost": total_cost,
            "trace_id": trace.id
        })
    
    @observe(name="semantic-search")
    async def track_search(
        self,
        query: str,
        results: List[SearchResult]
    ) -> None:
        """Track semantic search operations"""
        
        langfuse_context.update_current_trace(
            name="semantic-search",
            input={"query": query},
            output={
                "result_count": len(results),
                "avg_similarity": sum(r.similarity_score for r in results) / len(results) if results else 0
            },
            metadata={
                "search_strategy": "hybrid",
                "has_results": len(results) > 0
            }
        )
    
    def calculate_cost(self, tokens: int, model: str) -> float:
        """Calculate cost based on token usage"""
        pricing = {
            "text-embedding-3-small": 0.00002,  # per 1k tokens
            "text-embedding-3-large": 0.00013,
            "text-embedding-ada-002": 0.00010
        }
        return (tokens / 1000) * pricing.get(model, 0.00002)
```

---

## 4. Implementation Approach (Claude Code Optimized)

### 4.1 Development Phases (Activity-Based)

#### Phase 1: Foundation & Infrastructure
**Activities:**
- Set up Docker environment with multi-stage builds
- Initialize FastAPI with Pydantic validation
- Configure Supabase with enhanced schema
- Implement authentication with magic links
- Set up Langfuse monitoring
- Create base Pydantic models

**Key Deliverables:**
- Working Docker container
- Database migrations
- Authentication flow
- Base API structure
- Monitoring foundation

#### Phase 2: Core Pipeline Enhancement
**Activities:**
- Integrate existing RAG pipeline
- Add Pydantic validation layer
- Implement real-time subscriptions
- Create knowledge graph builder
- Add duplicate detection
- Implement cost tracking

**Key Deliverables:**
- Enhanced processing pipeline
- Real-time status updates
- Knowledge graph structure
- Cost management system

#### Phase 3: Intelligent Features
**Activities:**
- Build AI configuration assistant
- Implement predictive monitoring
- Create semantic search with context
- Add anomaly detection
- Build recommendation engine

**Key Deliverables:**
- AI-powered features
- Advanced search capabilities
- Predictive analytics
- Smart recommendations

#### Phase 4: Dashboard & Visualization
**Activities:**
- Create responsive React UI
- Implement real-time WebSocket updates
- Build cost analytics dashboard
- Create knowledge graph visualizer
- Add interactive configuration wizard

**Key Deliverables:**
- Complete web interface
- Real-time dashboard
- Analytics views
- Configuration UI

#### Phase 5: Testing & Optimization
**Activities:**
- Comprehensive testing suite
- Performance optimization
- Security audit
- Documentation completion
- Deployment preparation

**Key Deliverables:**
- Test coverage >80%
- Performance benchmarks
- Security report
- Complete documentation
- Production-ready system

---

## 5. Success Metrics (Enhanced)

### Primary KPIs
- **Processing Efficiency:** 10-30 documents/minute (improved from 6-20)
- **Cost Optimization:** <$0.01 per document average
- **System Reliability:** 99.9% uptime
- **User Satisfaction:** Zero CLI commands required
- **Search Accuracy:** >90% relevance score

### Advanced Metrics
- **Duplicate Detection Rate:** >95% accuracy
- **Prediction Accuracy:** >80% for failure forecasting
- **Real-time Latency:** <100ms for status updates
- **Knowledge Graph Coverage:** >70% of documents connected
- **Cost Savings:** 30% reduction through optimization

---

## 6. Risk Mitigation (Enhanced)

### Technical Risks

1. **Real-time Scalability**
   - Mitigation: Implement connection pooling
   - Fallback: Graceful degradation to polling

2. **Cost Overruns**
   - Mitigation: Aggressive rate limiting
   - Fallback: Automatic processing pause

3. **Knowledge Graph Complexity**
   - Mitigation: Incremental relationship building
   - Fallback: Simple tagging system

4. **AI Configuration Errors**
   - Mitigation: Multi-layer validation
   - Fallback: Manual configuration option

---

## 7. Future Enhancements (Post-MVP)

### Near-term (1-3 months)
- Multi-tenant architecture
- Advanced OCR capabilities
- Custom embedding models
- Workflow automation
- API marketplace integration

### Mid-term (3-6 months)
- Horizontal scaling with Kubernetes
- Multi-language support
- Custom ML models
- Enterprise SSO
- Advanced compliance features

### Long-term (6+ months)
- AI-powered content generation
- Automated knowledge synthesis
- Cross-platform plugins
- White-label options
- SaaS offering

---

## Appendix A: Technology Stack

### Core Technologies
- **Backend:** Python 3.11+, FastAPI, Pydantic
- **Frontend:** React 18, TypeScript, Tailwind CSS
- **Database:** Supabase (PostgreSQL + PGVector)
- **Container:** Docker, Docker Compose
- **Monitoring:** Langfuse, Custom Dashboards

### AI/ML Stack
- **Embeddings:** OpenAI API (text-embedding-3-*)
- **Validation:** Pydantic AI
- **Monitoring:** Langfuse
- **Knowledge:** Graph algorithms

### Infrastructure
- **Deployment:** VPS with Docker
- **CI/CD:** GitHub Actions
- **Monitoring:** Prometheus + Grafana (optional)
- **Logging:** Structured JSON logs

---

## Appendix B: API Documentation

### Core Endpoints

```yaml
openapi: 3.0.0
info:
  title: brAIn API
  version: 2.0.0
  description: Intelligent RAG Pipeline Management System

paths:
  /api/folders:
    post:
      summary: Add folder with validation
      requestBody:
        content:
          application/json:
            schema:
              $ref: '#/components/schemas/GoogleDriveFolderInput'
      responses:
        200:
          description: Folder added successfully
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/FolderSyncStatus'
  
  /api/search:
    post:
      summary: Semantic search with context
      requestBody:
        content:
          application/json:
            schema:
              $ref: '#/components/schemas/SemanticSearchRequest'
      responses:
        200:
          description: Search results with context
          content:
            application/json:
              schema:
                type: array
                items:
                  $ref: '#/components/schemas/SearchResult'
  
  /ws/realtime:
    get:
      summary: WebSocket for real-time updates
      responses:
        101:
          description: Switching Protocols
```

---

*End of Product Requirements Document v2.0*
*Development Approach: Claude Code with Activity-Based Structure*
*Architecture: AI-First with Real-time Capabilities*