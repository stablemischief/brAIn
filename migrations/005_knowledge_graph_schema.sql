-- Knowledge Graph Database Schema for brAIn v2.0
-- Creates tables for nodes, edges, and relationships with optimized indexes

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "vector";
CREATE EXTENSION IF NOT EXISTS "btree_gin";

-- Entity types enum
CREATE TYPE entity_type AS ENUM (
    'person',
    'organization', 
    'location',
    'concept',
    'technology',
    'product',
    'event',
    'topic',
    'keyword',
    'date',
    'number',
    'custom'
);

-- Relationship types enum
CREATE TYPE relationship_type AS ENUM (
    'mentions',
    'related_to',
    'contains',
    'part_of',
    'created_by',
    'associated_with',
    'located_in',
    'occurred_on',
    'depends_on',
    'influences',
    'similar_to',
    'derived_from',
    'custom'
);

-- Graph nodes table - represents entities
CREATE TABLE knowledge_graph_nodes (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(500) NOT NULL,
    entity_type entity_type NOT NULL DEFAULT 'concept',
    description TEXT,
    properties JSONB DEFAULT '{}',
    embedding vector(1536), -- OpenAI embeddings dimension
    confidence FLOAT DEFAULT 1.0 CHECK (confidence >= 0 AND confidence <= 1),
    source_documents UUID[] DEFAULT '{}', -- References to documents
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    created_by UUID, -- User who created this entity
    metadata JSONB DEFAULT '{}'
);

-- Graph edges table - represents relationships
CREATE TABLE knowledge_graph_edges (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    source_node_id UUID NOT NULL REFERENCES knowledge_graph_nodes(id) ON DELETE CASCADE,
    target_node_id UUID NOT NULL REFERENCES knowledge_graph_nodes(id) ON DELETE CASCADE,
    relationship_type relationship_type NOT NULL DEFAULT 'related_to',
    relationship_name VARCHAR(200),
    description TEXT,
    strength FLOAT DEFAULT 0.5 CHECK (strength >= 0 AND strength <= 1),
    properties JSONB DEFAULT '{}',
    evidence TEXT[], -- Supporting evidence for this relationship
    source_documents UUID[] DEFAULT '{}', -- Documents where this relationship was found
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    created_by UUID,
    metadata JSONB DEFAULT '{}'
);

-- Document-entity relationships for tracking which documents mention which entities
CREATE TABLE document_entities (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    document_id UUID NOT NULL, -- References to processed documents
    entity_id UUID NOT NULL REFERENCES knowledge_graph_nodes(id) ON DELETE CASCADE,
    mention_count INTEGER DEFAULT 1,
    importance_score FLOAT DEFAULT 0.5,
    context_snippets TEXT[],
    positions INTEGER[], -- Character positions where entity appears
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Entity clusters for grouping similar entities
CREATE TABLE entity_clusters (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(300) NOT NULL,
    description TEXT,
    entity_ids UUID[] NOT NULL,
    cluster_type VARCHAR(100) DEFAULT 'similarity',
    centroid_embedding vector(1536),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    metadata JSONB DEFAULT '{}'
);

-- Graph analytics and metrics
CREATE TABLE graph_metrics (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    metric_name VARCHAR(100) NOT NULL,
    metric_value FLOAT NOT NULL,
    node_id UUID REFERENCES knowledge_graph_nodes(id) ON DELETE CASCADE,
    edge_id UUID REFERENCES knowledge_graph_edges(id) ON DELETE CASCADE,
    calculated_at TIMESTAMPTZ DEFAULT NOW(),
    metadata JSONB DEFAULT '{}'
);

-- Performance-optimized indexes
-- Node indexes
CREATE INDEX idx_nodes_name ON knowledge_graph_nodes USING GIN (name gin_trgm_ops);
CREATE INDEX idx_nodes_entity_type ON knowledge_graph_nodes (entity_type);
CREATE INDEX idx_nodes_embedding ON knowledge_graph_nodes USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
CREATE INDEX idx_nodes_created_at ON knowledge_graph_nodes (created_at DESC);
CREATE INDEX idx_nodes_properties ON knowledge_graph_nodes USING GIN (properties);
CREATE INDEX idx_nodes_source_docs ON knowledge_graph_nodes USING GIN (source_documents);
CREATE INDEX idx_nodes_confidence ON knowledge_graph_nodes (confidence DESC);

-- Edge indexes
CREATE INDEX idx_edges_source_target ON knowledge_graph_edges (source_node_id, target_node_id);
CREATE INDEX idx_edges_source ON knowledge_graph_edges (source_node_id);
CREATE INDEX idx_edges_target ON knowledge_graph_edges (target_node_id);
CREATE INDEX idx_edges_relationship_type ON knowledge_graph_edges (relationship_type);
CREATE INDEX idx_edges_strength ON knowledge_graph_edges (strength DESC);
CREATE INDEX idx_edges_created_at ON knowledge_graph_edges (created_at DESC);
CREATE INDEX idx_edges_properties ON knowledge_graph_edges USING GIN (properties);
CREATE INDEX idx_edges_source_docs ON knowledge_graph_edges USING GIN (source_documents);

-- Document-entity relationship indexes
CREATE INDEX idx_doc_entities_document ON document_entities (document_id);
CREATE INDEX idx_doc_entities_entity ON document_entities (entity_id);
CREATE INDEX idx_doc_entities_importance ON document_entities (importance_score DESC);
CREATE UNIQUE INDEX idx_doc_entities_unique ON document_entities (document_id, entity_id);

-- Cluster indexes
CREATE INDEX idx_clusters_entities ON entity_clusters USING GIN (entity_ids);
CREATE INDEX idx_clusters_embedding ON entity_clusters USING ivfflat (centroid_embedding vector_cosine_ops) WITH (lists = 50);
CREATE INDEX idx_clusters_type ON entity_clusters (cluster_type);

-- Metrics indexes
CREATE INDEX idx_metrics_name ON graph_metrics (metric_name);
CREATE INDEX idx_metrics_node ON graph_metrics (node_id);
CREATE INDEX idx_metrics_edge ON graph_metrics (edge_id);
CREATE INDEX idx_metrics_calculated_at ON graph_metrics (calculated_at DESC);

-- Ensure referential integrity
ALTER TABLE knowledge_graph_edges ADD CONSTRAINT chk_no_self_loops 
CHECK (source_node_id != target_node_id);

-- Function to update the updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE 'plpgsql';

-- Triggers for automatic timestamp updates
CREATE TRIGGER update_nodes_updated_at 
    BEFORE UPDATE ON knowledge_graph_nodes 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_edges_updated_at 
    BEFORE UPDATE ON knowledge_graph_edges 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_doc_entities_updated_at 
    BEFORE UPDATE ON document_entities 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Function to calculate graph metrics
CREATE OR REPLACE FUNCTION calculate_node_degree(node_uuid UUID)
RETURNS INTEGER AS $$
BEGIN
    RETURN (
        SELECT COUNT(*) 
        FROM knowledge_graph_edges 
        WHERE source_node_id = node_uuid OR target_node_id = node_uuid
    );
END;
$$ LANGUAGE 'plpgsql';

-- Function to find similar nodes using vector similarity
CREATE OR REPLACE FUNCTION find_similar_nodes(
    input_embedding vector(1536),
    similarity_threshold FLOAT DEFAULT 0.8,
    max_results INTEGER DEFAULT 10
)
RETURNS TABLE(
    node_id UUID,
    name VARCHAR(500),
    entity_type entity_type,
    similarity_score FLOAT
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        n.id,
        n.name,
        n.entity_type,
        1 - (n.embedding <=> input_embedding) as similarity_score
    FROM knowledge_graph_nodes n
    WHERE n.embedding IS NOT NULL
        AND 1 - (n.embedding <=> input_embedding) >= similarity_threshold
    ORDER BY n.embedding <=> input_embedding
    LIMIT max_results;
END;
$$ LANGUAGE 'plpgsql';

-- Function to get node neighbors
CREATE OR REPLACE FUNCTION get_node_neighbors(
    node_uuid UUID,
    max_depth INTEGER DEFAULT 1,
    relationship_filter relationship_type DEFAULT NULL
)
RETURNS TABLE(
    neighbor_id UUID,
    neighbor_name VARCHAR(500),
    relationship_type relationship_type,
    relationship_name VARCHAR(200),
    strength FLOAT,
    distance INTEGER
) AS $$
BEGIN
    RETURN QUERY
    WITH RECURSIVE graph_traversal AS (
        -- Base case: direct neighbors
        SELECT 
            CASE 
                WHEN e.source_node_id = node_uuid THEN e.target_node_id
                ELSE e.source_node_id
            END as neighbor_id,
            e.relationship_type,
            e.relationship_name,
            e.strength,
            1 as distance
        FROM knowledge_graph_edges e
        WHERE (e.source_node_id = node_uuid OR e.target_node_id = node_uuid)
            AND (relationship_filter IS NULL OR e.relationship_type = relationship_filter)
        
        UNION ALL
        
        -- Recursive case: neighbors of neighbors
        SELECT 
            CASE 
                WHEN e.source_node_id = gt.neighbor_id THEN e.target_node_id
                ELSE e.source_node_id
            END as neighbor_id,
            e.relationship_type,
            e.relationship_name,
            e.strength,
            gt.distance + 1
        FROM knowledge_graph_edges e
        JOIN graph_traversal gt ON (e.source_node_id = gt.neighbor_id OR e.target_node_id = gt.neighbor_id)
        WHERE gt.distance < max_depth
            AND (relationship_filter IS NULL OR e.relationship_type = relationship_filter)
    )
    SELECT DISTINCT
        gt.neighbor_id,
        n.name as neighbor_name,
        gt.relationship_type,
        gt.relationship_name,
        gt.strength,
        gt.distance
    FROM graph_traversal gt
    JOIN knowledge_graph_nodes n ON n.id = gt.neighbor_id
    WHERE gt.neighbor_id != node_uuid
    ORDER BY gt.distance, gt.strength DESC;
END;
$$ LANGUAGE 'plpgsql';

-- View for graph statistics
CREATE VIEW graph_statistics AS
SELECT 
    (SELECT COUNT(*) FROM knowledge_graph_nodes) as total_nodes,
    (SELECT COUNT(*) FROM knowledge_graph_edges) as total_edges,
    (SELECT COUNT(DISTINCT entity_type) FROM knowledge_graph_nodes) as entity_types,
    (SELECT COUNT(DISTINCT relationship_type) FROM knowledge_graph_edges) as relationship_types,
    (SELECT AVG(calculate_node_degree(id)) FROM knowledge_graph_nodes) as avg_node_degree,
    (SELECT MAX(calculate_node_degree(id)) FROM knowledge_graph_nodes) as max_node_degree,
    NOW() as calculated_at;

-- Create sample data for testing (optional)
-- INSERT INTO knowledge_graph_nodes (name, entity_type, description, confidence) VALUES
-- ('Artificial Intelligence', 'concept', 'The field of computer science focused on creating intelligent machines', 0.95),
-- ('Machine Learning', 'concept', 'A subset of AI that enables machines to learn from data', 0.92),
-- ('Neural Networks', 'technology', 'Computing systems inspired by biological neural networks', 0.88);

-- INSERT INTO knowledge_graph_edges (source_node_id, target_node_id, relationship_type, relationship_name, strength) VALUES
-- ((SELECT id FROM knowledge_graph_nodes WHERE name = 'Machine Learning'), 
--  (SELECT id FROM knowledge_graph_nodes WHERE name = 'Artificial Intelligence'), 
--  'part_of', 'is_subset_of', 0.9),
-- ((SELECT id FROM knowledge_graph_nodes WHERE name = 'Neural Networks'), 
--  (SELECT id FROM knowledge_graph_nodes WHERE name = 'Machine Learning'), 
--  'related_to', 'implements', 0.85);

-- Commit the transaction
COMMIT;