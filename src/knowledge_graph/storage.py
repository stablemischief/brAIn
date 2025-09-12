"""
Knowledge graph storage operations for PostgreSQL database.
Handles CRUD operations for nodes, edges, and graph data.
"""

import logging
import json
from typing import Dict, List, Set, Optional, Tuple, Union, Any
from datetime import datetime, timezone
from uuid import UUID, uuid4
import asyncio

import asyncpg
from pydantic import BaseModel, Field
import numpy as np

from .entities import ExtractedEntity, EntityType
from .relationships import DetectedRelationship, RelationshipType

logger = logging.getLogger(__name__)


class GraphNode(BaseModel):
    """Graph node model"""
    id: UUID = Field(default_factory=uuid4)
    name: str
    entity_type: EntityType
    description: Optional[str] = None
    properties: Dict[str, Any] = Field(default_factory=dict)
    embedding: Optional[List[float]] = None
    confidence: float = 1.0
    source_documents: List[UUID] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    created_by: Optional[UUID] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class GraphEdge(BaseModel):
    """Graph edge model"""
    id: UUID = Field(default_factory=uuid4)
    source_node_id: UUID
    target_node_id: UUID
    relationship_type: RelationshipType
    relationship_name: Optional[str] = None
    description: Optional[str] = None
    strength: float = 0.5
    properties: Dict[str, Any] = Field(default_factory=dict)
    evidence: List[str] = Field(default_factory=list)
    source_documents: List[UUID] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    created_by: Optional[UUID] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class DocumentEntity(BaseModel):
    """Document-entity relationship model"""
    id: UUID = Field(default_factory=uuid4)
    document_id: UUID
    entity_id: UUID
    mention_count: int = 1
    importance_score: float = 0.5
    context_snippets: List[str] = Field(default_factory=list)
    positions: List[int] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class GraphStorageConfig(BaseModel):
    """Configuration for graph storage"""
    database_url: str
    max_connections: int = 10
    min_connections: int = 1
    connection_timeout: float = 60.0
    command_timeout: float = 60.0
    
    # Performance settings
    batch_size: int = 100
    enable_connection_pooling: bool = True
    
    # Vector similarity settings
    similarity_threshold: float = 0.8
    max_similar_nodes: int = 10


class KnowledgeGraphStorage:
    """Advanced knowledge graph storage system"""
    
    def __init__(self, config: GraphStorageConfig):
        self.config = config
        self.pool: Optional[asyncpg.Pool] = None
        
    async def initialize(self) -> bool:
        """Initialize database connection pool"""
        try:
            if self.config.enable_connection_pooling:
                self.pool = await asyncpg.create_pool(
                    dsn=self.config.database_url,
                    min_size=self.config.min_connections,
                    max_size=self.config.max_connections,
                    timeout=self.config.connection_timeout,
                    command_timeout=self.config.command_timeout
                )
            else:
                # Create a single connection wrapped in a pool-like interface
                conn = await asyncpg.connect(self.config.database_url)
                self.pool = _SingleConnectionPool(conn)
            
            logger.info("Knowledge graph storage initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize graph storage: {e}")
            return False
    
    async def close(self):
        """Close database connections"""
        if self.pool:
            await self.pool.close()
            logger.info("Knowledge graph storage closed")
    
    # Node operations
    
    async def create_node(
        self, 
        entity: ExtractedEntity, 
        document_id: Optional[UUID] = None,
        embedding: Optional[List[float]] = None,
        created_by: Optional[UUID] = None
    ) -> GraphNode:
        """Create a new graph node from an extracted entity"""
        
        node = GraphNode(
            name=entity.name,
            entity_type=entity.entity_type,
            description=entity.description,
            properties={
                'aliases': entity.aliases,
                'mention_count': len(entity.mentions),
                'extraction_methods': entity.properties.get('extraction_methods', []),
                'type_distribution': entity.properties.get('type_distribution', {})
            },
            embedding=embedding,
            confidence=entity.confidence,
            source_documents=[document_id] if document_id else [],
            created_by=created_by,
            metadata={
                'entity_properties': entity.properties
            }
        )
        
        async with self.pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO knowledge_graph_nodes (
                    id, name, entity_type, description, properties, 
                    embedding, confidence, source_documents, created_by, metadata
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
            """, 
                node.id,
                node.name,
                node.entity_type.value,
                node.description,
                json.dumps(node.properties),
                node.embedding,
                node.confidence,
                [str(doc_id) for doc_id in node.source_documents],
                node.created_by,
                json.dumps(node.metadata)
            )
        
        logger.debug(f"Created graph node: {node.name} ({node.id})")
        return node
    
    async def get_node_by_id(self, node_id: UUID) -> Optional[GraphNode]:
        """Get a node by its ID"""
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow("""
                SELECT * FROM knowledge_graph_nodes WHERE id = $1
            """, node_id)
            
            if row:
                return self._row_to_node(row)
            return None
    
    async def get_node_by_name(
        self, 
        name: str, 
        entity_type: Optional[EntityType] = None
    ) -> Optional[GraphNode]:
        """Get a node by its name and optionally entity type"""
        query = "SELECT * FROM knowledge_graph_nodes WHERE LOWER(name) = LOWER($1)"
        params = [name]
        
        if entity_type:
            query += " AND entity_type = $2"
            params.append(entity_type.value)
        
        query += " ORDER BY confidence DESC LIMIT 1"
        
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(query, *params)
            
            if row:
                return self._row_to_node(row)
            return None
    
    async def search_nodes(
        self, 
        query: str,
        entity_types: Optional[List[EntityType]] = None,
        limit: int = 50,
        min_confidence: float = 0.0
    ) -> List[GraphNode]:
        """Search for nodes using text similarity"""
        
        sql_query = """
            SELECT *, similarity(name, $1) as sim_score
            FROM knowledge_graph_nodes 
            WHERE similarity(name, $1) > 0.1
        """
        params = [query]
        
        if min_confidence > 0:
            sql_query += " AND confidence >= $2"
            params.append(min_confidence)
        
        if entity_types:
            type_values = [t.value for t in entity_types]
            sql_query += f" AND entity_type = ANY(${len(params) + 1})"
            params.append(type_values)
        
        sql_query += " ORDER BY sim_score DESC, confidence DESC LIMIT $" + str(len(params) + 1)
        params.append(limit)
        
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(sql_query, *params)
            return [self._row_to_node(row) for row in rows]
    
    async def find_similar_nodes(
        self, 
        embedding: List[float],
        similarity_threshold: Optional[float] = None,
        max_results: Optional[int] = None
    ) -> List[Tuple[GraphNode, float]]:
        """Find nodes similar to given embedding"""
        
        threshold = similarity_threshold or self.config.similarity_threshold
        max_res = max_results or self.config.max_similar_nodes
        
        async with self.pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT *, 1 - (embedding <=> $1) as similarity_score
                FROM knowledge_graph_nodes 
                WHERE embedding IS NOT NULL 
                    AND 1 - (embedding <=> $1) >= $2
                ORDER BY embedding <=> $1
                LIMIT $3
            """, embedding, threshold, max_res)
            
            results = []
            for row in rows:
                node = self._row_to_node(row)
                similarity = row['similarity_score']
                results.append((node, similarity))
            
            return results
    
    async def update_node(self, node_id: UUID, updates: Dict[str, Any]) -> bool:
        """Update a node with new data"""
        if not updates:
            return True
        
        # Build dynamic update query
        set_clauses = []
        params = []
        param_count = 1
        
        for key, value in updates.items():
            if key in ['name', 'entity_type', 'description', 'confidence']:
                set_clauses.append(f"{key} = ${param_count}")
                params.append(value)
                param_count += 1
            elif key in ['properties', 'metadata']:
                set_clauses.append(f"{key} = ${param_count}")
                params.append(json.dumps(value))
                param_count += 1
            elif key == 'embedding':
                set_clauses.append(f"embedding = ${param_count}")
                params.append(value)
                param_count += 1
            elif key == 'source_documents':
                set_clauses.append(f"source_documents = ${param_count}")
                params.append([str(doc_id) for doc_id in value])
                param_count += 1
        
        if not set_clauses:
            return True
        
        set_clauses.append(f"updated_at = ${param_count}")
        params.append(datetime.now(timezone.utc))
        param_count += 1
        
        params.append(node_id)
        
        query = f"""
            UPDATE knowledge_graph_nodes 
            SET {', '.join(set_clauses)}
            WHERE id = ${param_count}
        """
        
        async with self.pool.acquire() as conn:
            result = await conn.execute(query, *params)
            return result.split()[-1] == '1'  # Check if one row was updated
    
    async def delete_node(self, node_id: UUID) -> bool:
        """Delete a node and its relationships"""
        async with self.pool.acquire() as conn:
            async with conn.transaction():
                # Delete edges first (foreign key constraint)
                await conn.execute("""
                    DELETE FROM knowledge_graph_edges 
                    WHERE source_node_id = $1 OR target_node_id = $1
                """, node_id)
                
                # Delete document relationships
                await conn.execute("""
                    DELETE FROM document_entities WHERE entity_id = $1
                """, node_id)
                
                # Delete the node
                result = await conn.execute("""
                    DELETE FROM knowledge_graph_nodes WHERE id = $1
                """, node_id)
                
                return result.split()[-1] == '1'
    
    # Edge operations
    
    async def create_edge(
        self, 
        relationship: DetectedRelationship,
        source_node_id: UUID,
        target_node_id: UUID,
        document_id: Optional[UUID] = None,
        created_by: Optional[UUID] = None
    ) -> GraphEdge:
        """Create a new graph edge from a detected relationship"""
        
        evidence_texts = [e.text_snippet for e in relationship.evidence[:5]]  # Limit evidence
        
        edge = GraphEdge(
            source_node_id=source_node_id,
            target_node_id=target_node_id,
            relationship_type=relationship.relationship_type,
            relationship_name=relationship.relationship_name,
            description=relationship.description,
            strength=relationship.strength,
            properties=relationship.properties,
            evidence=evidence_texts,
            source_documents=[document_id] if document_id else [],
            created_by=created_by,
            metadata={
                'bidirectional': relationship.bidirectional,
                'detection_confidence': relationship.confidence,
                'evidence_count': len(relationship.evidence)
            }
        )
        
        async with self.pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO knowledge_graph_edges (
                    id, source_node_id, target_node_id, relationship_type, 
                    relationship_name, description, strength, properties, 
                    evidence, source_documents, created_by, metadata
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
            """,
                edge.id,
                edge.source_node_id,
                edge.target_node_id,
                edge.relationship_type.value,
                edge.relationship_name,
                edge.description,
                edge.strength,
                json.dumps(edge.properties),
                edge.evidence,
                [str(doc_id) for doc_id in edge.source_documents],
                edge.created_by,
                json.dumps(edge.metadata)
            )
        
        logger.debug(f"Created graph edge: {source_node_id} -> {target_node_id} ({edge.relationship_type.value})")
        return edge
    
    async def get_edge_by_id(self, edge_id: UUID) -> Optional[GraphEdge]:
        """Get an edge by its ID"""
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow("""
                SELECT * FROM knowledge_graph_edges WHERE id = $1
            """, edge_id)
            
            if row:
                return self._row_to_edge(row)
            return None
    
    async def get_edges_by_nodes(
        self, 
        source_node_id: Optional[UUID] = None,
        target_node_id: Optional[UUID] = None,
        relationship_types: Optional[List[RelationshipType]] = None,
        min_strength: float = 0.0
    ) -> List[GraphEdge]:
        """Get edges between nodes"""
        
        conditions = []
        params = []
        param_count = 1
        
        if source_node_id:
            conditions.append(f"source_node_id = ${param_count}")
            params.append(source_node_id)
            param_count += 1
        
        if target_node_id:
            conditions.append(f"target_node_id = ${param_count}")
            params.append(target_node_id)
            param_count += 1
        
        if relationship_types:
            type_values = [t.value for t in relationship_types]
            conditions.append(f"relationship_type = ANY(${param_count})")
            params.append(type_values)
            param_count += 1
        
        if min_strength > 0:
            conditions.append(f"strength >= ${param_count}")
            params.append(min_strength)
            param_count += 1
        
        query = "SELECT * FROM knowledge_graph_edges"
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        query += " ORDER BY strength DESC"
        
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(query, *params)
            return [self._row_to_edge(row) for row in rows]
    
    async def get_node_neighbors(
        self, 
        node_id: UUID,
        max_depth: int = 1,
        relationship_filter: Optional[RelationshipType] = None,
        min_strength: float = 0.0
    ) -> List[Dict[str, Any]]:
        """Get neighboring nodes using database function"""
        
        async with self.pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT * FROM get_node_neighbors($1, $2, $3)
                WHERE strength >= $4
            """, 
                node_id, 
                max_depth, 
                relationship_filter.value if relationship_filter else None,
                min_strength
            )
            
            neighbors = []
            for row in rows:
                neighbors.append({
                    'neighbor_id': row['neighbor_id'],
                    'neighbor_name': row['neighbor_name'],
                    'relationship_type': row['relationship_type'],
                    'relationship_name': row['relationship_name'],
                    'strength': row['strength'],
                    'distance': row['distance']
                })
            
            return neighbors
    
    async def update_edge(self, edge_id: UUID, updates: Dict[str, Any]) -> bool:
        """Update an edge with new data"""
        if not updates:
            return True
        
        # Build dynamic update query
        set_clauses = []
        params = []
        param_count = 1
        
        for key, value in updates.items():
            if key in ['relationship_type', 'relationship_name', 'description', 'strength']:
                set_clauses.append(f"{key} = ${param_count}")
                params.append(value)
                param_count += 1
            elif key in ['properties', 'metadata']:
                set_clauses.append(f"{key} = ${param_count}")
                params.append(json.dumps(value))
                param_count += 1
            elif key in ['evidence', 'source_documents']:
                set_clauses.append(f"{key} = ${param_count}")
                params.append(value)
                param_count += 1
        
        if not set_clauses:
            return True
        
        set_clauses.append(f"updated_at = ${param_count}")
        params.append(datetime.now(timezone.utc))
        param_count += 1
        
        params.append(edge_id)
        
        query = f"""
            UPDATE knowledge_graph_edges 
            SET {', '.join(set_clauses)}
            WHERE id = ${param_count}
        """
        
        async with self.pool.acquire() as conn:
            result = await conn.execute(query, *params)
            return result.split()[-1] == '1'
    
    async def delete_edge(self, edge_id: UUID) -> bool:
        """Delete an edge"""
        async with self.pool.acquire() as conn:
            result = await conn.execute("""
                DELETE FROM knowledge_graph_edges WHERE id = $1
            """, edge_id)
            
            return result.split()[-1] == '1'
    
    # Document-entity relationships
    
    async def create_document_entity_relationship(
        self, 
        document_id: UUID,
        entity_id: UUID,
        mentions: List[Any],  # EntityMention objects
        importance_score: float = 0.5
    ) -> DocumentEntity:
        """Create document-entity relationship"""
        
        # Extract context snippets and positions from mentions
        context_snippets = [m.context for m in mentions[:10]]  # Limit snippets
        positions = [m.start_pos for m in mentions]
        
        doc_entity = DocumentEntity(
            document_id=document_id,
            entity_id=entity_id,
            mention_count=len(mentions),
            importance_score=importance_score,
            context_snippets=context_snippets,
            positions=positions
        )
        
        async with self.pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO document_entities (
                    id, document_id, entity_id, mention_count, 
                    importance_score, context_snippets, positions
                ) VALUES ($1, $2, $3, $4, $5, $6, $7)
                ON CONFLICT (document_id, entity_id) 
                DO UPDATE SET 
                    mention_count = EXCLUDED.mention_count,
                    importance_score = EXCLUDED.importance_score,
                    context_snippets = EXCLUDED.context_snippets,
                    positions = EXCLUDED.positions,
                    updated_at = NOW()
            """,
                doc_entity.id,
                doc_entity.document_id,
                doc_entity.entity_id,
                doc_entity.mention_count,
                doc_entity.importance_score,
                doc_entity.context_snippets,
                doc_entity.positions
            )
        
        return doc_entity
    
    async def get_document_entities(
        self, 
        document_id: UUID,
        min_importance: float = 0.0
    ) -> List[Tuple[GraphNode, DocumentEntity]]:
        """Get all entities mentioned in a document"""
        
        async with self.pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT n.*, de.*
                FROM knowledge_graph_nodes n
                JOIN document_entities de ON n.id = de.entity_id
                WHERE de.document_id = $1 AND de.importance_score >= $2
                ORDER BY de.importance_score DESC
            """, document_id, min_importance)
            
            results = []
            for row in rows:
                # Split row data between node and document_entity
                node_data = {k: v for k, v in row.items() if not k.startswith('de_')}
                de_data = {k[3:]: v for k, v in row.items() if k.startswith('de_')}
                
                node = self._row_to_node(node_data)
                doc_entity = self._row_to_document_entity(de_data)
                results.append((node, doc_entity))
            
            return results
    
    # Batch operations
    
    async def batch_create_nodes(
        self, 
        entities: List[ExtractedEntity],
        document_id: Optional[UUID] = None,
        created_by: Optional[UUID] = None,
        embeddings: Optional[List[List[float]]] = None
    ) -> List[GraphNode]:
        """Create multiple nodes in batch"""
        
        if not entities:
            return []
        
        nodes = []
        values = []
        
        for i, entity in enumerate(entities):
            embedding = embeddings[i] if embeddings and i < len(embeddings) else None
            
            node = GraphNode(
                name=entity.name,
                entity_type=entity.entity_type,
                description=entity.description,
                properties={
                    'aliases': entity.aliases,
                    'mention_count': len(entity.mentions),
                    'extraction_methods': entity.properties.get('extraction_methods', []),
                    'type_distribution': entity.properties.get('type_distribution', {})
                },
                embedding=embedding,
                confidence=entity.confidence,
                source_documents=[document_id] if document_id else [],
                created_by=created_by,
                metadata={'entity_properties': entity.properties}
            )
            
            nodes.append(node)
            values.extend([
                node.id,
                node.name,
                node.entity_type.value,
                node.description,
                json.dumps(node.properties),
                node.embedding,
                node.confidence,
                [str(doc_id) for doc_id in node.source_documents],
                node.created_by,
                json.dumps(node.metadata)
            ])
        
        # Build batch insert query
        placeholders = []
        for i in range(len(entities)):
            start_idx = i * 10 + 1
            placeholder = f"(${start_idx}, ${start_idx+1}, ${start_idx+2}, ${start_idx+3}, ${start_idx+4}, ${start_idx+5}, ${start_idx+6}, ${start_idx+7}, ${start_idx+8}, ${start_idx+9})"
            placeholders.append(placeholder)
        
        query = f"""
            INSERT INTO knowledge_graph_nodes (
                id, name, entity_type, description, properties, 
                embedding, confidence, source_documents, created_by, metadata
            ) VALUES {', '.join(placeholders)}
        """
        
        async with self.pool.acquire() as conn:
            await conn.execute(query, *values)
        
        logger.info(f"Batch created {len(nodes)} graph nodes")
        return nodes
    
    async def batch_create_edges(
        self, 
        relationships: List[Tuple[DetectedRelationship, UUID, UUID]],
        document_id: Optional[UUID] = None,
        created_by: Optional[UUID] = None
    ) -> List[GraphEdge]:
        """Create multiple edges in batch"""
        
        if not relationships:
            return []
        
        edges = []
        values = []
        
        for relationship, source_node_id, target_node_id in relationships:
            evidence_texts = [e.text_snippet for e in relationship.evidence[:5]]
            
            edge = GraphEdge(
                source_node_id=source_node_id,
                target_node_id=target_node_id,
                relationship_type=relationship.relationship_type,
                relationship_name=relationship.relationship_name,
                description=relationship.description,
                strength=relationship.strength,
                properties=relationship.properties,
                evidence=evidence_texts,
                source_documents=[document_id] if document_id else [],
                created_by=created_by,
                metadata={
                    'bidirectional': relationship.bidirectional,
                    'detection_confidence': relationship.confidence,
                    'evidence_count': len(relationship.evidence)
                }
            )
            
            edges.append(edge)
            values.extend([
                edge.id,
                edge.source_node_id,
                edge.target_node_id,
                edge.relationship_type.value,
                edge.relationship_name,
                edge.description,
                edge.strength,
                json.dumps(edge.properties),
                edge.evidence,
                [str(doc_id) for doc_id in edge.source_documents],
                edge.created_by,
                json.dumps(edge.metadata)
            ])
        
        # Build batch insert query
        placeholders = []
        for i in range(len(relationships)):
            start_idx = i * 12 + 1
            placeholder = f"(${start_idx}, ${start_idx+1}, ${start_idx+2}, ${start_idx+3}, ${start_idx+4}, ${start_idx+5}, ${start_idx+6}, ${start_idx+7}, ${start_idx+8}, ${start_idx+9}, ${start_idx+10}, ${start_idx+11})"
            placeholders.append(placeholder)
        
        query = f"""
            INSERT INTO knowledge_graph_edges (
                id, source_node_id, target_node_id, relationship_type, 
                relationship_name, description, strength, properties, 
                evidence, source_documents, created_by, metadata
            ) VALUES {', '.join(placeholders)}
        """
        
        async with self.pool.acquire() as conn:
            await conn.execute(query, *values)
        
        logger.info(f"Batch created {len(edges)} graph edges")
        return edges
    
    # Analytics and metrics
    
    async def get_graph_statistics(self) -> Dict[str, Any]:
        """Get comprehensive graph statistics"""
        async with self.pool.acquire() as conn:
            stats_row = await conn.fetchrow("SELECT * FROM graph_statistics")
            
            if stats_row:
                return dict(stats_row)
            return {}
    
    async def calculate_node_centrality_metrics(self, node_id: UUID) -> Dict[str, float]:
        """Calculate centrality metrics for a node"""
        async with self.pool.acquire() as conn:
            # Degree centrality
            degree = await conn.fetchval("""
                SELECT calculate_node_degree($1)
            """, node_id)
            
            # Additional metrics would require more complex graph algorithms
            # For now, return basic degree centrality
            return {
                'degree_centrality': float(degree),
                'betweenness_centrality': 0.0,  # Placeholder
                'closeness_centrality': 0.0,    # Placeholder
                'eigenvector_centrality': 0.0   # Placeholder
            }
    
    # Helper methods
    
    def _row_to_node(self, row) -> GraphNode:
        """Convert database row to GraphNode"""
        return GraphNode(
            id=row['id'],
            name=row['name'],
            entity_type=EntityType(row['entity_type']),
            description=row['description'],
            properties=json.loads(row['properties']) if row['properties'] else {},
            embedding=list(row['embedding']) if row['embedding'] else None,
            confidence=float(row['confidence']),
            source_documents=[UUID(doc_id) for doc_id in row['source_documents']],
            created_at=row['created_at'],
            updated_at=row['updated_at'],
            created_by=row['created_by'],
            metadata=json.loads(row['metadata']) if row['metadata'] else {}
        )
    
    def _row_to_edge(self, row) -> GraphEdge:
        """Convert database row to GraphEdge"""
        return GraphEdge(
            id=row['id'],
            source_node_id=row['source_node_id'],
            target_node_id=row['target_node_id'],
            relationship_type=RelationshipType(row['relationship_type']),
            relationship_name=row['relationship_name'],
            description=row['description'],
            strength=float(row['strength']),
            properties=json.loads(row['properties']) if row['properties'] else {},
            evidence=list(row['evidence']) if row['evidence'] else [],
            source_documents=[UUID(doc_id) for doc_id in row['source_documents']],
            created_at=row['created_at'],
            updated_at=row['updated_at'],
            created_by=row['created_by'],
            metadata=json.loads(row['metadata']) if row['metadata'] else {}
        )
    
    def _row_to_document_entity(self, row) -> DocumentEntity:
        """Convert database row to DocumentEntity"""
        return DocumentEntity(
            id=row['id'],
            document_id=row['document_id'],
            entity_id=row['entity_id'],
            mention_count=int(row['mention_count']),
            importance_score=float(row['importance_score']),
            context_snippets=list(row['context_snippets']) if row['context_snippets'] else [],
            positions=list(row['positions']) if row['positions'] else [],
            created_at=row['created_at'],
            updated_at=row['updated_at']
        )


class _SingleConnectionPool:
    """Wrapper to make a single connection behave like a pool"""
    
    def __init__(self, connection):
        self._connection = connection
    
    def acquire(self):
        return self._connection
    
    async def close(self):
        await self._connection.close()


# Global storage instance
_storage: Optional[KnowledgeGraphStorage] = None


def get_graph_storage() -> Optional[KnowledgeGraphStorage]:
    """Get global graph storage instance"""
    return _storage


def initialize_graph_storage(config: GraphStorageConfig) -> KnowledgeGraphStorage:
    """Initialize global graph storage"""
    global _storage
    _storage = KnowledgeGraphStorage(config)
    return _storage