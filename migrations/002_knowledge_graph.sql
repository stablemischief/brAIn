-- brAIn v2.0 Knowledge Graph Schema
-- Migration 002: Knowledge graph for document relationships and entity extraction
-- Created: 2025-09-11
-- Purpose: Store and query document relationships, entities, and semantic connections

-- Create knowledge graph specific types
CREATE TYPE node_type AS ENUM (
    'document',
    'entity', 
    'concept',
    'person',
    'organization',
    'location',
    'date',
    'topic',
    'keyword',
    'category'
);

CREATE TYPE edge_type AS ENUM (
    'references',
    'contains',
    'similar_to',
    'authored_by',
    'mentions',
    'categorized_as',
    'related_to',
    'part_of',
    'derived_from',
    'cites',
    'temporal_relation',
    'causal_relation'
);

CREATE TYPE confidence_level AS ENUM (
    'very_low',   -- 0.0-0.2
    'low',        -- 0.2-0.4
    'medium',     -- 0.4-0.6
    'high',       -- 0.6-0.8
    'very_high'   -- 0.8-1.0
);

-- Knowledge nodes: Entities, concepts, and documents in the graph
CREATE TABLE IF NOT EXISTS knowledge_nodes (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    
    -- Node identification
    node_type node_type NOT NULL,
    node_value TEXT NOT NULL, -- The actual content/name of the node
    normalized_value TEXT NOT NULL, -- Normalized version for matching
    
    -- Source information
    source_document_id UUID REFERENCES documents(id) ON DELETE CASCADE,
    extraction_method TEXT DEFAULT 'ai_extraction', -- 'ai_extraction', 'manual', 'imported'
    
    -- AI confidence and quality
    confidence_score DECIMAL(3,2) NOT NULL DEFAULT 0.0, -- 0.0-1.0
    confidence_level confidence_level GENERATED ALWAYS AS (
        CASE 
            WHEN confidence_score >= 0.8 THEN 'very_high'
            WHEN confidence_score >= 0.6 THEN 'high'
            WHEN confidence_score >= 0.4 THEN 'medium'
            WHEN confidence_score >= 0.2 THEN 'low'
            ELSE 'very_low'
        END
    ) STORED,
    
    -- Context and metadata
    context TEXT, -- Surrounding text where entity was found
    position_in_text INTEGER, -- Character position in source document
    metadata JSONB DEFAULT '{}', -- Additional extracted attributes
    
    -- Embedding for semantic similarity (smaller dimension for entities)
    embedding VECTOR(384), -- Using smaller model for entities
    embedding_model TEXT DEFAULT 'text-embedding-3-small',
    
    -- Statistics and usage
    mention_count INTEGER DEFAULT 1, -- How many times this entity appears
    document_count INTEGER DEFAULT 1, -- In how many documents
    relationship_count INTEGER DEFAULT 0, -- Number of edges connected
    
    -- Quality and validation
    validated BOOLEAN DEFAULT FALSE,
    validated_by UUID REFERENCES users(id),
    validated_at TIMESTAMP WITH TIME ZONE,
    
    -- Audit fields
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    
    -- Constraints
    UNIQUE(user_id, node_type, normalized_value)
);

-- Knowledge edges: Relationships between nodes
CREATE TABLE IF NOT EXISTS knowledge_edges (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    
    -- Edge definition
    source_node_id UUID NOT NULL REFERENCES knowledge_nodes(id) ON DELETE CASCADE,
    target_node_id UUID NOT NULL REFERENCES knowledge_nodes(id) ON DELETE CASCADE,
    edge_type edge_type NOT NULL,
    
    -- Relationship strength and confidence
    weight DECIMAL(3,2) NOT NULL DEFAULT 0.5, -- 0.0-1.0 relationship strength
    confidence_score DECIMAL(3,2) NOT NULL DEFAULT 0.0, -- 0.0-1.0
    confidence_level confidence_level GENERATED ALWAYS AS (
        CASE 
            WHEN confidence_score >= 0.8 THEN 'very_high'
            WHEN confidence_score >= 0.6 THEN 'high'
            WHEN confidence_score >= 0.4 THEN 'medium'
            WHEN confidence_score >= 0.2 THEN 'low'
            ELSE 'very_low'
        END
    ) STORED,
    
    -- Source information
    source_document_id UUID REFERENCES documents(id) ON DELETE CASCADE,
    extraction_method TEXT DEFAULT 'ai_extraction',
    evidence_text TEXT, -- Text that supports this relationship
    
    -- Context and metadata
    context JSONB DEFAULT '{}', -- Additional relationship context
    metadata JSONB DEFAULT '{}', -- Extracted attributes of the relationship
    
    -- Quality and validation
    validated BOOLEAN DEFAULT FALSE,
    validated_by UUID REFERENCES users(id),
    validated_at TIMESTAMP WITH TIME ZONE,
    
    -- Usage statistics
    access_count INTEGER DEFAULT 0,
    last_accessed_at TIMESTAMP WITH TIME ZONE,
    
    -- Audit fields
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    
    -- Constraints
    UNIQUE(user_id, source_node_id, target_node_id, edge_type),
    CHECK (source_node_id != target_node_id), -- No self-loops
    CHECK (weight >= 0.0 AND weight <= 1.0),
    CHECK (confidence_score >= 0.0 AND confidence_score <= 1.0)
);

-- Document clusters: Groups of similar documents
CREATE TABLE IF NOT EXISTS document_clusters (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    
    -- Cluster information
    cluster_name TEXT NOT NULL,
    description TEXT,
    cluster_type TEXT DEFAULT 'similarity', -- 'similarity', 'topic', 'manual'
    
    -- Algorithm details
    algorithm_used TEXT DEFAULT 'kmeans',
    algorithm_parameters JSONB DEFAULT '{}',
    similarity_threshold DECIMAL(3,2) DEFAULT 0.7,
    
    -- Statistics
    document_count INTEGER DEFAULT 0,
    avg_similarity DECIMAL(3,2) DEFAULT 0.0,
    cluster_cohesion DECIMAL(3,2) DEFAULT 0.0,
    
    -- Cluster embedding (centroid)
    centroid_embedding VECTOR(1536),
    
    -- Quality metrics
    silhouette_score DECIMAL(3,2), -- Clustering quality metric
    inertia DECIMAL(10,4), -- Within-cluster sum of squares
    
    -- Audit fields
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    
    -- Constraints
    UNIQUE(user_id, cluster_name)
);

-- Document cluster membership: Which documents belong to which clusters
CREATE TABLE IF NOT EXISTS document_cluster_membership (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    
    document_id UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    cluster_id UUID NOT NULL REFERENCES document_clusters(id) ON DELETE CASCADE,
    
    -- Membership details
    similarity_to_centroid DECIMAL(3,2) NOT NULL,
    membership_confidence DECIMAL(3,2) NOT NULL DEFAULT 1.0,
    is_cluster_representative BOOLEAN DEFAULT FALSE, -- Is this a representative document?
    
    -- Audit fields
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    
    -- Constraints
    UNIQUE(document_id, cluster_id),
    CHECK (similarity_to_centroid >= 0.0 AND similarity_to_centroid <= 1.0),
    CHECK (membership_confidence >= 0.0 AND membership_confidence <= 1.0)
);

-- Knowledge graph statistics and metrics
CREATE TABLE IF NOT EXISTS knowledge_graph_metrics (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    
    -- Snapshot timestamp
    measured_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    
    -- Node statistics
    total_nodes INTEGER NOT NULL DEFAULT 0,
    nodes_by_type JSONB DEFAULT '{}', -- Count of each node type
    avg_node_confidence DECIMAL(3,2) DEFAULT 0.0,
    
    -- Edge statistics  
    total_edges INTEGER NOT NULL DEFAULT 0,
    edges_by_type JSONB DEFAULT '{}', -- Count of each edge type
    avg_edge_confidence DECIMAL(3,2) DEFAULT 0.0,
    avg_edge_weight DECIMAL(3,2) DEFAULT 0.0,
    
    -- Graph structure metrics
    graph_density DECIMAL(6,4) DEFAULT 0.0, -- edges / (nodes * (nodes-1))
    avg_degree DECIMAL(4,2) DEFAULT 0.0, -- Average connections per node
    max_degree INTEGER DEFAULT 0, -- Most connected node
    connected_components INTEGER DEFAULT 0, -- Number of disconnected subgraphs
    
    -- Quality metrics
    clustering_coefficient DECIMAL(4,3) DEFAULT 0.0, -- Local clustering measure
    avg_path_length DECIMAL(4,2) DEFAULT 0.0, -- Average shortest path
    
    -- Performance metrics
    calculation_time_ms INTEGER DEFAULT 0,
    
    UNIQUE(user_id, measured_at)
);

-- Triggers for knowledge graph maintenance
CREATE OR REPLACE FUNCTION update_node_statistics()
RETURNS TRIGGER AS $$
BEGIN
    -- Update mention and document counts for nodes
    IF TG_OP = 'INSERT' OR TG_OP = 'UPDATE' THEN
        UPDATE knowledge_nodes 
        SET mention_count = (
            SELECT COUNT(*) FROM knowledge_nodes kn2 
            WHERE kn2.user_id = NEW.user_id 
            AND kn2.normalized_value = NEW.normalized_value 
            AND kn2.node_type = NEW.node_type
        ),
        document_count = (
            SELECT COUNT(DISTINCT source_document_id) FROM knowledge_nodes kn2 
            WHERE kn2.user_id = NEW.user_id 
            AND kn2.normalized_value = NEW.normalized_value 
            AND kn2.node_type = NEW.node_type
            AND source_document_id IS NOT NULL
        )
        WHERE id = NEW.id;
        
        RETURN NEW;
    END IF;
    
    RETURN NULL;
END;
$$ LANGUAGE plpgsql;

CREATE OR REPLACE FUNCTION update_edge_statistics()
RETURNS TRIGGER AS $$
BEGIN
    -- Update relationship counts for connected nodes
    IF TG_OP = 'INSERT' THEN
        UPDATE knowledge_nodes 
        SET relationship_count = relationship_count + 1 
        WHERE id IN (NEW.source_node_id, NEW.target_node_id);
        
        RETURN NEW;
    ELSIF TG_OP = 'DELETE' THEN
        UPDATE knowledge_nodes 
        SET relationship_count = relationship_count - 1 
        WHERE id IN (OLD.source_node_id, OLD.target_node_id);
        
        RETURN OLD;
    END IF;
    
    RETURN NULL;
END;
$$ LANGUAGE plpgsql;

CREATE OR REPLACE FUNCTION update_cluster_statistics()
RETURNS TRIGGER AS $$
BEGIN
    -- Update document count in clusters
    IF TG_OP = 'INSERT' THEN
        UPDATE document_clusters 
        SET document_count = document_count + 1 
        WHERE id = NEW.cluster_id;
        RETURN NEW;
    ELSIF TG_OP = 'DELETE' THEN
        UPDATE document_clusters 
        SET document_count = document_count - 1 
        WHERE id = OLD.cluster_id;
        RETURN OLD;
    END IF;
    
    RETURN NULL;
END;
$$ LANGUAGE plpgsql;

-- Apply triggers
CREATE TRIGGER update_knowledge_nodes_updated_at 
    BEFORE UPDATE ON knowledge_nodes 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_knowledge_edges_updated_at 
    BEFORE UPDATE ON knowledge_edges 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_document_clusters_updated_at 
    BEFORE UPDATE ON document_clusters 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER trigger_update_node_statistics 
    AFTER INSERT OR UPDATE ON knowledge_nodes 
    FOR EACH ROW EXECUTE FUNCTION update_node_statistics();

CREATE TRIGGER trigger_update_edge_statistics 
    AFTER INSERT OR DELETE ON knowledge_edges 
    FOR EACH ROW EXECUTE FUNCTION update_edge_statistics();

CREATE TRIGGER trigger_update_cluster_statistics 
    AFTER INSERT OR DELETE ON document_cluster_membership 
    FOR EACH ROW EXECUTE FUNCTION update_cluster_statistics();

-- Indexes for knowledge graph performance
CREATE INDEX IF NOT EXISTS idx_knowledge_nodes_user_type ON knowledge_nodes(user_id, node_type);
CREATE INDEX IF NOT EXISTS idx_knowledge_nodes_normalized_value ON knowledge_nodes(normalized_value);
CREATE INDEX IF NOT EXISTS idx_knowledge_nodes_confidence ON knowledge_nodes(confidence_score DESC);
CREATE INDEX IF NOT EXISTS idx_knowledge_nodes_source_doc ON knowledge_nodes(source_document_id);
CREATE INDEX IF NOT EXISTS idx_knowledge_nodes_mention_count ON knowledge_nodes(mention_count DESC);

CREATE INDEX IF NOT EXISTS idx_knowledge_edges_user_type ON knowledge_edges(user_id, edge_type);
CREATE INDEX IF NOT EXISTS idx_knowledge_edges_source_node ON knowledge_edges(source_node_id);
CREATE INDEX IF NOT EXISTS idx_knowledge_edges_target_node ON knowledge_edges(target_node_id);
CREATE INDEX IF NOT EXISTS idx_knowledge_edges_weight ON knowledge_edges(weight DESC);
CREATE INDEX IF NOT EXISTS idx_knowledge_edges_confidence ON knowledge_edges(confidence_score DESC);
CREATE INDEX IF NOT EXISTS idx_knowledge_edges_source_doc ON knowledge_edges(source_document_id);

CREATE INDEX IF NOT EXISTS idx_document_clusters_user_id ON document_clusters(user_id);
CREATE INDEX IF NOT EXISTS idx_document_clusters_type ON document_clusters(cluster_type);
CREATE INDEX IF NOT EXISTS idx_document_cluster_membership_doc ON document_cluster_membership(document_id);
CREATE INDEX IF NOT EXISTS idx_document_cluster_membership_cluster ON document_cluster_membership(cluster_id);

-- Composite indexes for common graph queries
CREATE INDEX IF NOT EXISTS idx_knowledge_nodes_user_type_confidence ON knowledge_nodes(user_id, node_type, confidence_score DESC);
CREATE INDEX IF NOT EXISTS idx_knowledge_edges_user_source_type ON knowledge_edges(user_id, source_node_id, edge_type);
CREATE INDEX IF NOT EXISTS idx_knowledge_edges_user_target_type ON knowledge_edges(user_id, target_node_id, edge_type);

-- GIN indexes for metadata searches
CREATE INDEX IF NOT EXISTS idx_knowledge_nodes_metadata_gin ON knowledge_nodes USING GIN(metadata);
CREATE INDEX IF NOT EXISTS idx_knowledge_edges_metadata_gin ON knowledge_edges USING GIN(metadata);
CREATE INDEX IF NOT EXISTS idx_knowledge_edges_context_gin ON knowledge_edges USING GIN(context);

-- Text search indexes
CREATE INDEX IF NOT EXISTS idx_knowledge_nodes_value_trgm ON knowledge_nodes USING GIN(node_value gin_trgm_ops);
CREATE INDEX IF NOT EXISTS idx_knowledge_nodes_context_trgm ON knowledge_nodes USING GIN(context gin_trgm_ops);

-- Comments for documentation
COMMENT ON TABLE knowledge_nodes IS 'Entities and concepts extracted from documents with AI confidence scoring';
COMMENT ON TABLE knowledge_edges IS 'Relationships between knowledge nodes with weighted connections';
COMMENT ON TABLE document_clusters IS 'Groups of semantically similar documents';
COMMENT ON TABLE document_cluster_membership IS 'Mapping between documents and their cluster assignments';
COMMENT ON TABLE knowledge_graph_metrics IS 'Periodic snapshots of knowledge graph statistics and health';

COMMENT ON COLUMN knowledge_nodes.normalized_value IS 'Cleaned and standardized version of node_value for better matching';
COMMENT ON COLUMN knowledge_nodes.embedding IS '384-dimensional embedding for entity similarity matching';
COMMENT ON COLUMN knowledge_edges.weight IS 'Strength of relationship between nodes (0.0-1.0)';
COMMENT ON COLUMN knowledge_edges.evidence_text IS 'Text evidence that supports the existence of this relationship';