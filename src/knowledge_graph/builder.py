"""
Main knowledge graph builder that orchestrates entity extraction, 
relationship detection, and graph construction with real-time updates.
"""

import logging
import asyncio
from typing import Dict, List, Set, Optional, Tuple, Union, Any
from datetime import datetime, timezone
from uuid import UUID, uuid4

from pydantic import BaseModel, Field
import numpy as np

from .entities import EntityExtractor, ExtractedEntity, EntityExtractionConfig, get_entity_extractor
from .relationships import RelationshipDetector, DetectedRelationship, RelationshipDetectionConfig, get_relationship_detector
from .storage import KnowledgeGraphStorage, GraphStorageConfig, GraphNode, GraphEdge, get_graph_storage
from .queries import KnowledgeGraphQuery, get_graph_query_engine, initialize_graph_query_engine

# Import real-time components if available
try:
    from realtime.supabase_client import SupabaseRealtimeClient
    from realtime.message_broadcaster import MessageBroadcaster, MessageScope, MessagePriority
    REALTIME_AVAILABLE = True
except ImportError:
    REALTIME_AVAILABLE = False
    logger.warning("Real-time components not available, knowledge graph updates will be synchronous only")

logger = logging.getLogger(__name__)


class GraphBuildConfig(BaseModel):
    """Configuration for knowledge graph builder"""
    
    # Component configurations
    entity_extraction: EntityExtractionConfig = Field(default_factory=EntityExtractionConfig)
    relationship_detection: RelationshipDetectionConfig = Field(default_factory=RelationshipDetectionConfig)
    storage: GraphStorageConfig
    
    # Processing settings
    batch_size: int = 50
    max_concurrent_tasks: int = 5
    enable_realtime_updates: bool = True
    
    # Graph construction settings
    merge_similar_entities: bool = True
    entity_similarity_threshold: float = 0.85
    min_relationship_confidence: float = 0.3
    
    # Performance settings
    enable_caching: bool = True
    cache_expiry_hours: int = 24
    enable_incremental_updates: bool = True


class GraphBuildResult(BaseModel):
    """Result from graph building operation"""
    success: bool
    document_id: Optional[UUID] = None
    entities_created: int = 0
    entities_updated: int = 0
    relationships_created: int = 0
    relationships_updated: int = 0
    processing_time_seconds: float = 0.0
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    graph_statistics: Dict[str, Any] = Field(default_factory=dict)


class KnowledgeGraphBuilder:
    """
    Main knowledge graph builder that coordinates all components
    to build and maintain a comprehensive knowledge graph.
    """
    
    def __init__(self, config: GraphBuildConfig):
        self.config = config
        
        # Initialize components
        self.entity_extractor = EntityExtractor(config.entity_extraction)
        self.relationship_detector = RelationshipDetector(config.relationship_detection) 
        self.storage: Optional[KnowledgeGraphStorage] = None
        self.query_engine: Optional[KnowledgeGraphQuery] = None
        
        # Real-time components
        self.supabase_client: Optional[SupabaseRealtimeClient] = None
        self.message_broadcaster: Optional[MessageBroadcaster] = None
        
        # Processing state
        self._processing_lock = asyncio.Lock()
        self._entity_cache: Dict[str, UUID] = {}
        self._similarity_cache: Dict[str, List[Tuple[UUID, float]]] = {}
        
    async def initialize(self) -> bool:
        """Initialize the knowledge graph builder"""
        try:
            # Initialize storage
            self.storage = KnowledgeGraphStorage(self.config.storage)
            if not await self.storage.initialize():
                return False
            
            # Initialize query engine
            self.query_engine = initialize_graph_query_engine(self.storage)
            
            # Initialize real-time components if enabled
            if self.config.enable_realtime_updates and REALTIME_AVAILABLE:
                # These would be initialized by the main application
                # self.supabase_client = get_supabase_client()
                # self.message_broadcaster = get_message_broadcaster()
                pass
            
            logger.info("Knowledge graph builder initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize knowledge graph builder: {e}")
            return False
    
    async def close(self):
        """Clean up resources"""
        if self.storage:
            await self.storage.close()
    
    # Main processing methods
    
    async def process_document(
        self,
        text: str,
        document_id: UUID,
        metadata: Optional[Dict[str, Any]] = None,
        created_by: Optional[UUID] = None
    ) -> GraphBuildResult:
        """
        Process a document to extract entities and relationships,
        then update the knowledge graph.
        """
        start_time = datetime.now()
        result = GraphBuildResult(success=False, document_id=document_id)
        
        try:
            async with self._processing_lock:
                logger.info(f"Processing document {document_id} for knowledge graph construction")
                
                # Step 1: Extract entities
                entities = await self.entity_extractor.extract_entities(text, str(document_id))
                if not entities:
                    result.warnings.append("No entities extracted from document")
                    result.success = True
                    return result
                
                logger.info(f"Extracted {len(entities)} entities from document")
                
                # Step 2: Merge similar entities if enabled
                if self.config.merge_similar_entities:
                    entities = await self._merge_similar_entities(entities)
                    logger.info(f"After merging: {len(entities)} unique entities")
                
                # Step 3: Create or update entities in graph
                entity_nodes = await self._process_entities(entities, document_id, created_by)
                result.entities_created = len([n for n in entity_nodes if n])
                
                # Step 4: Detect relationships
                relationships = await self.relationship_detector.detect_relationships(
                    entities, text, str(document_id)
                )
                
                # Filter relationships by confidence
                relationships = [r for r in relationships 
                               if r.confidence >= self.config.min_relationship_confidence]
                
                logger.info(f"Detected {len(relationships)} relationships")
                
                # Step 5: Create relationships in graph
                relationship_edges = await self._process_relationships(
                    relationships, entity_nodes, document_id, created_by
                )
                result.relationships_created = len([e for e in relationship_edges if e])
                
                # Step 6: Update document-entity relationships
                await self._update_document_entity_relationships(
                    document_id, entities, entity_nodes
                )
                
                # Step 7: Send real-time updates
                if self.config.enable_realtime_updates:
                    await self._send_graph_updates(document_id, entity_nodes, relationship_edges)
                
                # Calculate processing time
                result.processing_time_seconds = (datetime.now() - start_time).total_seconds()
                result.success = True
                
                # Get updated graph statistics
                if self.storage:
                    result.graph_statistics = await self.storage.get_graph_statistics()
                
                logger.info(f"Successfully processed document {document_id} in {result.processing_time_seconds:.2f}s")
                
        except Exception as e:
            logger.error(f"Error processing document {document_id}: {e}")
            result.errors.append(str(e))
            result.processing_time_seconds = (datetime.now() - start_time).total_seconds()
        
        return result
    
    async def process_text_batch(
        self,
        texts: List[Tuple[str, UUID]],  # (text, document_id) pairs
        metadata: Optional[Dict[str, Any]] = None,
        created_by: Optional[UUID] = None
    ) -> List[GraphBuildResult]:
        """Process multiple documents in batch for better performance"""
        
        if not texts:
            return []
        
        logger.info(f"Processing batch of {len(texts)} documents")
        
        # Process in batches to avoid overwhelming the system
        results = []
        batch_size = self.config.batch_size
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            
            # Process batch concurrently with limited concurrency
            semaphore = asyncio.Semaphore(self.config.max_concurrent_tasks)
            
            async def process_single(text_doc_pair):
                async with semaphore:
                    text, doc_id = text_doc_pair
                    return await self.process_document(text, doc_id, metadata, created_by)
            
            batch_results = await asyncio.gather(
                *[process_single(td) for td in batch],
                return_exceptions=True
            )
            
            # Handle any exceptions
            for i, result in enumerate(batch_results):
                if isinstance(result, Exception):
                    error_result = GraphBuildResult(
                        success=False,
                        document_id=batch[i][1],
                        errors=[str(result)]
                    )
                    results.append(error_result)
                else:
                    results.append(result)
        
        # Log batch summary
        successful = sum(1 for r in results if r.success)
        total_entities = sum(r.entities_created for r in results)
        total_relationships = sum(r.relationships_created for r in results)
        
        logger.info(f"Batch processing complete: {successful}/{len(results)} successful, "
                   f"{total_entities} entities, {total_relationships} relationships created")
        
        return results
    
    # Entity processing methods
    
    async def _merge_similar_entities(self, entities: List[ExtractedEntity]) -> List[ExtractedEntity]:
        """Merge entities that are likely duplicates"""
        if len(entities) <= 1:
            return entities
        
        merged = []
        used_indices = set()
        
        for i, entity1 in enumerate(entities):
            if i in used_indices:
                continue
                
            # Find similar entities
            similar_indices = [i]
            
            for j, entity2 in enumerate(entities[i+1:], start=i+1):
                if j in used_indices:
                    continue
                
                similarity = self._calculate_entity_similarity(entity1, entity2)
                if similarity >= self.config.entity_similarity_threshold:
                    similar_indices.append(j)
            
            # Merge similar entities
            if len(similar_indices) == 1:
                merged.append(entity1)
            else:
                merged_entity = self._merge_entities([entities[idx] for idx in similar_indices])
                merged.append(merged_entity)
            
            used_indices.update(similar_indices)
        
        return merged
    
    def _calculate_entity_similarity(self, entity1: ExtractedEntity, entity2: ExtractedEntity) -> float:
        """Calculate similarity score between two entities"""
        
        # Exact name match
        if entity1.name.lower() == entity2.name.lower():
            return 1.0
        
        # Check aliases
        all_names1 = {entity1.name.lower()} | {alias.lower() for alias in entity1.aliases}
        all_names2 = {entity2.name.lower()} | {alias.lower() for alias in entity2.aliases}
        
        if all_names1 & all_names2:
            return 0.9
        
        # Type mismatch penalty
        if entity1.entity_type != entity2.entity_type:
            return 0.0
        
        # String similarity (simplified Levenshtein distance)
        name1, name2 = entity1.name.lower(), entity2.name.lower()
        max_len = max(len(name1), len(name2))
        if max_len == 0:
            return 1.0
        
        # Simple character overlap similarity
        chars1, chars2 = set(name1), set(name2)
        overlap = len(chars1 & chars2)
        union = len(chars1 | chars2)
        
        char_similarity = overlap / union if union > 0 else 0.0
        
        # Length similarity
        len_similarity = 1.0 - abs(len(name1) - len(name2)) / max_len
        
        # Combined similarity
        return (char_similarity + len_similarity) / 2.0
    
    def _merge_entities(self, entities: List[ExtractedEntity]) -> ExtractedEntity:
        """Merge multiple entities into one"""
        if len(entities) == 1:
            return entities[0]
        
        # Use entity with highest confidence as base
        base_entity = max(entities, key=lambda e: e.confidence)
        
        # Combine mentions
        all_mentions = []
        for entity in entities:
            all_mentions.extend(entity.mentions)
        
        # Combine aliases
        all_aliases = set()
        for entity in entities:
            all_aliases.add(entity.name)
            all_aliases.update(entity.aliases)
        all_aliases.discard(base_entity.name)  # Remove base name from aliases
        
        # Combine properties
        combined_properties = base_entity.properties.copy()
        for entity in entities:
            combined_properties.update(entity.properties)
        
        # Calculate combined confidence
        combined_confidence = sum(e.confidence for e in entities) / len(entities)
        
        return ExtractedEntity(
            name=base_entity.name,
            entity_type=base_entity.entity_type,
            mentions=all_mentions,
            confidence=combined_confidence,
            description=base_entity.description,
            properties=combined_properties,
            aliases=list(all_aliases)
        )
    
    async def _process_entities(
        self,
        entities: List[ExtractedEntity],
        document_id: UUID,
        created_by: Optional[UUID]
    ) -> List[Optional[GraphNode]]:
        """Process entities and create/update graph nodes"""
        
        if not self.storage:
            return []
        
        nodes = []
        
        for entity in entities:
            try:
                # Check if entity already exists
                existing_node = await self.storage.get_node_by_name(
                    entity.name, entity.entity_type
                )
                
                if existing_node:
                    # Update existing node
                    updates = {
                        'confidence': max(existing_node.confidence, entity.confidence),
                        'source_documents': list(set(existing_node.source_documents + [document_id]))
                    }
                    
                    # Merge properties
                    merged_properties = existing_node.properties.copy()
                    merged_properties.update(entity.properties)
                    updates['properties'] = merged_properties
                    
                    await self.storage.update_node(existing_node.id, updates)
                    nodes.append(existing_node)
                    
                    # Cache the entity
                    self._entity_cache[entity.name.lower()] = existing_node.id
                    
                else:
                    # Create new node
                    new_node = await self.storage.create_node(
                        entity=entity,
                        document_id=document_id,
                        created_by=created_by
                    )
                    nodes.append(new_node)
                    
                    # Cache the entity
                    self._entity_cache[entity.name.lower()] = new_node.id
            
            except Exception as e:
                logger.error(f"Error processing entity {entity.name}: {e}")
                nodes.append(None)
        
        return nodes
    
    # Relationship processing methods
    
    async def _process_relationships(
        self,
        relationships: List[DetectedRelationship],
        entity_nodes: List[Optional[GraphNode]],
        document_id: UUID,
        created_by: Optional[UUID]
    ) -> List[Optional[GraphEdge]]:
        """Process relationships and create graph edges"""
        
        if not self.storage:
            return []
        
        # Create entity name to node mapping
        node_map = {}
        for node in entity_nodes:
            if node:
                node_map[node.name.lower()] = node
        
        edges = []
        
        for relationship in relationships:
            try:
                # Find source and target nodes
                source_node = node_map.get(relationship.source_entity.lower())
                target_node = node_map.get(relationship.target_entity.lower())
                
                if not source_node or not target_node:
                    logger.warning(f"Missing nodes for relationship: {relationship.source_entity} -> {relationship.target_entity}")
                    edges.append(None)
                    continue
                
                # Check if relationship already exists
                existing_edges = await self.storage.get_edges_by_nodes(
                    source_node_id=source_node.id,
                    target_node_id=target_node.id,
                    relationship_types=[relationship.relationship_type]
                )
                
                if existing_edges:
                    # Update existing edge
                    edge = existing_edges[0]
                    updates = {
                        'strength': max(edge.strength, relationship.strength),
                        'source_documents': list(set(edge.source_documents + [document_id]))
                    }
                    
                    # Merge evidence
                    new_evidence = [e.text_snippet for e in relationship.evidence[:5]]
                    merged_evidence = list(set(edge.evidence + new_evidence))
                    updates['evidence'] = merged_evidence
                    
                    await self.storage.update_edge(edge.id, updates)
                    edges.append(edge)
                    
                else:
                    # Create new edge
                    new_edge = await self.storage.create_edge(
                        relationship=relationship,
                        source_node_id=source_node.id,
                        target_node_id=target_node.id,
                        document_id=document_id,
                        created_by=created_by
                    )
                    edges.append(new_edge)
            
            except Exception as e:
                logger.error(f"Error processing relationship {relationship.source_entity} -> {relationship.target_entity}: {e}")
                edges.append(None)
        
        return edges
    
    async def _update_document_entity_relationships(
        self,
        document_id: UUID,
        entities: List[ExtractedEntity],
        entity_nodes: List[Optional[GraphNode]]
    ):
        """Update document-entity relationship records"""
        
        if not self.storage:
            return
        
        for entity, node in zip(entities, entity_nodes):
            if not node:
                continue
            
            try:
                # Calculate importance score based on mention count and confidence
                importance_score = min(len(entity.mentions) * entity.confidence / 10.0, 1.0)
                
                await self.storage.create_document_entity_relationship(
                    document_id=document_id,
                    entity_id=node.id,
                    mentions=entity.mentions,
                    importance_score=importance_score
                )
            
            except Exception as e:
                logger.error(f"Error updating document-entity relationship: {e}")
    
    # Real-time update methods
    
    async def _send_graph_updates(
        self,
        document_id: UUID,
        entity_nodes: List[Optional[GraphNode]],
        relationship_edges: List[Optional[GraphEdge]]
    ):
        """Send real-time updates about graph changes"""
        
        if not self.config.enable_realtime_updates or not REALTIME_AVAILABLE:
            return
        
        if not self.message_broadcaster:
            return
        
        try:
            # Send entity updates
            new_entities = [node for node in entity_nodes if node]
            if new_entities:
                await self.message_broadcaster.broadcast_to_channel_subscribers(
                    channel="knowledge_graph_updates",
                    message_type="entities_added",
                    payload={
                        "document_id": str(document_id),
                        "entities": [
                            {
                                "id": str(node.id),
                                "name": node.name,
                                "entity_type": node.entity_type.value,
                                "confidence": node.confidence
                            }
                            for node in new_entities
                        ],
                        "count": len(new_entities)
                    },
                    priority=MessagePriority.NORMAL
                )
            
            # Send relationship updates
            new_relationships = [edge for edge in relationship_edges if edge]
            if new_relationships:
                await self.message_broadcaster.broadcast_to_channel_subscribers(
                    channel="knowledge_graph_updates", 
                    message_type="relationships_added",
                    payload={
                        "document_id": str(document_id),
                        "relationships": [
                            {
                                "id": str(edge.id),
                                "source_node_id": str(edge.source_node_id),
                                "target_node_id": str(edge.target_node_id),
                                "relationship_type": edge.relationship_type.value,
                                "strength": edge.strength
                            }
                            for edge in new_relationships
                        ],
                        "count": len(new_relationships)
                    },
                    priority=MessagePriority.NORMAL
                )
            
            # Send graph statistics update
            if self.storage:
                stats = await self.storage.get_graph_statistics()
                await self.message_broadcaster.broadcast_to_channel_subscribers(
                    channel="knowledge_graph_updates",
                    message_type="graph_statistics_updated",
                    payload={
                        "document_id": str(document_id),
                        "statistics": stats
                    },
                    priority=MessagePriority.LOW
                )
        
        except Exception as e:
            logger.error(f"Error sending graph updates: {e}")
    
    # Utility methods
    
    async def rebuild_graph_from_documents(
        self,
        document_texts: List[Tuple[str, UUID]],
        clear_existing: bool = False,
        created_by: Optional[UUID] = None
    ) -> List[GraphBuildResult]:
        """Rebuild the entire graph from a collection of documents"""
        
        logger.info(f"Rebuilding knowledge graph from {len(document_texts)} documents")
        
        try:
            if clear_existing:
                logger.warning("Clearing existing graph data")
                # Implementation would clear existing graph data
                # This is a destructive operation and should be used carefully
                pass
            
            # Process all documents
            results = await self.process_text_batch(document_texts, created_by=created_by)
            
            # Log summary
            successful = sum(1 for r in results if r.success)
            total_entities = sum(r.entities_created for r in results)
            total_relationships = sum(r.relationships_created for r in results)
            total_time = sum(r.processing_time_seconds for r in results)
            
            logger.info(f"Graph rebuild complete: {successful}/{len(results)} documents processed, "
                       f"{total_entities} entities, {total_relationships} relationships created "
                       f"in {total_time:.2f} seconds")
            
            return results
        
        except Exception as e:
            logger.error(f"Error rebuilding graph: {e}")
            return []
    
    async def get_graph_summary(self) -> Dict[str, Any]:
        """Get a comprehensive summary of the current graph state"""
        
        if not self.storage:
            return {"error": "Storage not initialized"}
        
        try:
            stats = await self.storage.get_graph_statistics()
            
            # Add additional analysis
            summary = {
                "basic_statistics": stats,
                "last_updated": datetime.now(timezone.utc).isoformat(),
                "cache_status": {
                    "entity_cache_size": len(self._entity_cache),
                    "similarity_cache_size": len(self._similarity_cache)
                }
            }
            
            return summary
        
        except Exception as e:
            logger.error(f"Error getting graph summary: {e}")
            return {"error": str(e)}
    
    def clear_caches(self):
        """Clear internal caches"""
        self._entity_cache.clear()
        self._similarity_cache.clear()
        logger.info("Knowledge graph caches cleared")


# Global builder instance
_builder: Optional[KnowledgeGraphBuilder] = None


def get_graph_builder() -> Optional[KnowledgeGraphBuilder]:
    """Get global graph builder instance"""
    return _builder


def initialize_graph_builder(config: GraphBuildConfig) -> KnowledgeGraphBuilder:
    """Initialize global graph builder"""
    global _builder
    _builder = KnowledgeGraphBuilder(config)
    return _builder