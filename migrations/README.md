# brAIn v2.0 Database Schema

## Overview

This directory contains the complete database schema for brAIn v2.0, an AI-first RAG pipeline management system. The schema is designed for PostgreSQL with pgvector extension for vector similarity search.

## Schema Features

- **AI-First Design**: Optimized for vector embeddings, cost tracking, and quality metrics
- **Knowledge Graph Support**: Entities, relationships, and document clustering
- **Real-time Monitoring**: System health, LLM usage, and performance analytics
- **Cost Management**: Detailed cost tracking with budget alerts
- **Performance Optimized**: HNSW indexes, materialized views, and query optimization

## Migration Files

### 001_enhanced_schema.sql
- Core tables: users, folders, documents, processing_queue
- Enhanced document fields: content_hash, extraction_quality, processing_cost
- Vector embedding storage with 1536-dimensional support
- Processing status tracking and retry logic
- Basic indexes and triggers

### 002_knowledge_graph.sql  
- Knowledge nodes: entities, concepts, people, organizations
- Knowledge edges: relationships with confidence scoring
- Document clustering and similarity analysis
- Graph metrics and statistics tracking
- Entity extraction and relationship detection

### 003_monitoring_tables.sql
- LLM usage tracking with cost calculation
- System health monitoring for all services  
- Processing analytics with hourly aggregation
- Daily cost summaries with budget management
- Alert rules and notification history

### 004_indexes_and_functions.sql
- HNSW vector indexes for similarity search
- Materialized views for search performance
- Hybrid search functions (vector + text)
- Real-time notification triggers
- Maintenance and cleanup functions

## Setup Instructions

### Prerequisites
```sql
-- Required PostgreSQL extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "vector";      -- pgvector for embeddings
CREATE EXTENSION IF NOT EXISTS "pg_trgm";     -- Trigram search
CREATE EXTENSION IF NOT EXISTS "btree_gin";   -- GIN indexes
```

### Development Setup
```bash
# 1. Start PostgreSQL with Docker
docker-compose up postgres

# 2. Run migrations in order
psql -h localhost -U brain_user -d brain_db -f migrations/001_enhanced_schema.sql
psql -h localhost -U brain_user -d brain_db -f migrations/002_knowledge_graph.sql  
psql -h localhost -U brain_user -d brain_db -f migrations/003_monitoring_tables.sql
psql -h localhost -U brain_user -d brain_db -f migrations/004_indexes_and_functions.sql

# 3. Load seed data (optional)
psql -h localhost -U brain_user -d brain_db -f seeds/development_data.sql
```

### Production Setup
```bash
# 1. Create production database
createdb brain_production

# 2. Run migrations
for migration in migrations/*.sql; do
    psql -h your-db-host -U brain_user -d brain_production -f "$migration"
done

# 3. Verify installation
psql -h your-db-host -U brain_user -d brain_production -c "SELECT generate_health_report();"
```

## Key Tables

### Core Data
- **users**: User accounts with authentication and preferences
- **folders**: Google Drive folders with sync configuration
- **documents**: Enhanced document storage with AI processing metadata
- **processing_queue**: Background task queue for document processing

### Knowledge Graph
- **knowledge_nodes**: Extracted entities and concepts with embeddings
- **knowledge_edges**: Relationships between entities with confidence scores
- **document_clusters**: Groups of similar documents
- **knowledge_graph_metrics**: Graph statistics and health metrics

### Monitoring
- **llm_usage**: Detailed tracking of all LLM operations and costs
- **system_health**: Real-time monitoring of all services
- **processing_analytics**: Hourly performance aggregations
- **daily_cost_summary**: Budget tracking and cost analysis

## Vector Search

### Embedding Storage
- Documents: 1536-dimensional vectors (OpenAI text-embedding-3-small)
- Knowledge nodes: 384-dimensional vectors for entities
- Clusters: Centroid embeddings for similarity grouping

### Search Functions
```sql
-- Hybrid search (vector + text)
SELECT * FROM hybrid_search(
    'user-uuid',
    'search query',
    embedding_vector,
    10,  -- limit
    0.6, -- vector weight
    0.4  -- text weight
);

-- Pure semantic search
SELECT * FROM semantic_search(
    'user-uuid', 
    embedding_vector,
    10,  -- limit
    0.7  -- similarity threshold
);
```

## Performance Features

### Indexes
- HNSW indexes for vector similarity (cosine and L2 distance)
- GIN indexes for JSONB metadata and array fields
- Composite indexes for common query patterns
- Text search indexes with trigram support

### Materialized Views
- **document_search_view**: Optimized document search with precomputed vectors
- **knowledge_graph_analytics_view**: Graph metrics by user and type
- **cost_analytics_view**: Cost aggregations by time period and operation

### Maintenance
```sql
-- Refresh materialized views
SELECT refresh_all_materialized_views();

-- Database maintenance
SELECT maintenance_vacuum_analyze();

-- Health check
SELECT * FROM check_data_integrity();
```

## Cost Tracking

### Automatic Cost Calculation
- Input/output token counting
- Per-model pricing configuration
- Real-time cost accumulation
- Daily budget tracking with alerts

### Budget Management
```sql
-- Set user budget limits
UPDATE users SET monthly_budget_limit = 500.00 WHERE email = 'user@example.com';

-- Check current spending
SELECT * FROM daily_cost_summary WHERE user_id = 'user-uuid' AND date = CURRENT_DATE;

-- Cost analysis
SELECT * FROM get_cost_summary('user-uuid', '2025-09-01', '2025-09-30');
```

## Monitoring & Alerts

### System Health
- Service status monitoring
- Performance metrics collection
- Error rate tracking
- Resource usage monitoring

### Custom Alerts
```sql
-- Create cost alert
INSERT INTO alert_rules (user_id, rule_name, rule_type, metric_name, operator, threshold_value)
VALUES ('user-uuid', 'Daily Budget Alert', 'cost_threshold', 'daily_cost', 'greater_than', 25.00);

-- Create performance alert
INSERT INTO alert_rules (user_id, rule_name, rule_type, metric_name, operator, threshold_value)
VALUES ('user-uuid', 'High Error Rate', 'error_rate', 'error_rate', 'greater_than', 5.0);
```

## Backup & Recovery

### Backup Functions
```sql
-- Full database backup
SELECT create_full_backup('production_backup');

-- User-specific backup
SELECT create_user_backup('user-uuid', 'user_export');

-- Data integrity check
SELECT * FROM check_data_integrity();
```

### Recovery Operations
```sql
-- Recover missing embeddings
SELECT recover_missing_embeddings('user-uuid');

-- Retry failed documents
SELECT retry_failed_documents('user-uuid', 3);
```

## Real-time Features

### Notification Channels
- `document_changes`: Document processing updates
- `processing_status`: Status change notifications  
- `cost_updates`: Real-time cost tracking

### Supabase Integration
The schema includes triggers for real-time subscriptions compatible with Supabase's real-time engine.

## Development Tools

### Seed Data
Run `seeds/development_data.sql` to populate the database with sample data for development and testing.

### Health Reports
```sql
-- Generate comprehensive health report
SELECT generate_health_report();

-- Database performance analysis
SELECT * FROM analyze_database_performance();

-- Database size analysis
SELECT * FROM analyze_database_size();
```

## Security Considerations

- User data isolation through row-level security patterns
- Soft deletes with `deleted_at` timestamps
- Audit trails with `created_at`/`updated_at` timestamps
- Input validation through CHECK constraints
- Foreign key constraints for referential integrity

## Optimization Tips

1. **Vector Search**: Use appropriate HNSW parameters (m=16, ef_construction=64)
2. **Materialized Views**: Refresh hourly or after bulk operations
3. **Cost Tracking**: Archive old LLM usage data after analysis
4. **Monitoring**: Clean up old system health data regularly
5. **Knowledge Graph**: Validate and prune low-confidence relationships

## Troubleshooting

### Common Issues
- **Slow vector search**: Check HNSW index creation and statistics
- **High storage usage**: Run cleanup functions and archive old data
- **Processing failures**: Check processing_queue and error messages
- **Cost discrepancies**: Verify token counting and pricing models

### Performance Tuning
```sql
-- Check index usage
SELECT * FROM analyze_database_performance();

-- Update table statistics
SELECT maintenance_vacuum_analyze();

-- Check vector index effectiveness
EXPLAIN (ANALYZE, BUFFERS) 
SELECT * FROM documents ORDER BY embedding <=> '[query_vector]' LIMIT 10;
```

For detailed API documentation and integration examples, see the backend documentation.