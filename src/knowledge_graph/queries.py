"""
Knowledge graph query functions for advanced graph traversal and analysis.
Provides high-level query interfaces for the knowledge graph.
"""

import logging
from typing import Dict, List, Set, Optional, Tuple, Union, Any
from dataclasses import dataclass
from enum import Enum
import asyncio

from pydantic import BaseModel, Field
import numpy as np

from .entities import EntityType
from .relationships import RelationshipType
from .storage import KnowledgeGraphStorage, GraphNode, GraphEdge

logger = logging.getLogger(__name__)


class QueryType(str, Enum):
    """Types of graph queries"""
    ENTITY_SEARCH = "entity_search"
    RELATIONSHIP_SEARCH = "relationship_search"
    PATH_FINDING = "path_finding"
    SIMILARITY_SEARCH = "similarity_search"
    SUBGRAPH_EXTRACTION = "subgraph_extraction"
    CENTRALITY_ANALYSIS = "centrality_analysis"
    CLUSTERING = "clustering"


class TraversalDirection(str, Enum):
    """Graph traversal directions"""
    OUTGOING = "outgoing"  # Follow edges from source to target
    INCOMING = "incoming"  # Follow edges from target to source
    BOTH = "both"         # Follow edges in both directions


@dataclass
class QueryFilter:
    """Filters for graph queries"""
    entity_types: Optional[List[EntityType]] = None
    relationship_types: Optional[List[RelationshipType]] = None
    min_confidence: float = 0.0
    min_strength: float = 0.0
    node_properties: Optional[Dict[str, Any]] = None
    edge_properties: Optional[Dict[str, Any]] = None
    date_range: Optional[Tuple[str, str]] = None


class GraphPath(BaseModel):
    """Represents a path through the graph"""
    nodes: List[GraphNode] = Field(description="Nodes in the path")
    edges: List[GraphEdge] = Field(description="Edges connecting the nodes")
    total_strength: float = Field(description="Combined strength of all edges")
    path_length: int = Field(description="Number of hops in the path")
    confidence: float = Field(description="Overall path confidence")


class SubGraph(BaseModel):
    """Represents a subgraph extraction result"""
    central_node: GraphNode = Field(description="Central node of the subgraph")
    nodes: List[GraphNode] = Field(description="All nodes in the subgraph")
    edges: List[GraphEdge] = Field(description="All edges in the subgraph")
    depth: int = Field(description="Maximum depth from central node")
    statistics: Dict[str, Any] = Field(default_factory=dict, description="Subgraph statistics")


class SimilarityResult(BaseModel):
    """Result from similarity search"""
    query_node: Optional[GraphNode] = Field(None, description="Original query node")
    similar_nodes: List[Tuple[GraphNode, float]] = Field(description="Similar nodes with scores")
    query_embedding: Optional[List[float]] = Field(None, description="Query embedding used")
    search_parameters: Dict[str, Any] = Field(default_factory=dict, description="Search parameters")


class CentralityMetrics(BaseModel):
    """Centrality metrics for nodes"""
    node_id: str
    degree_centrality: float
    betweenness_centrality: float = 0.0
    closeness_centrality: float = 0.0
    eigenvector_centrality: float = 0.0
    pagerank: float = 0.0


class KnowledgeGraphQuery:
    """Advanced query interface for knowledge graph"""
    
    def __init__(self, storage: KnowledgeGraphStorage):
        self.storage = storage
        
    # Entity queries
    
    async def search_entities(
        self,
        query: str,
        filters: Optional[QueryFilter] = None,
        limit: int = 50,
        include_similarity_score: bool = False
    ) -> List[Union[GraphNode, Tuple[GraphNode, float]]]:
        """
        Search for entities by name or description.
        
        Args:
            query: Search query string
            filters: Optional filters to apply
            limit: Maximum number of results
            include_similarity_score: Whether to include similarity scores
            
        Returns:
            List of matching nodes, optionally with similarity scores
        """
        try:
            filters = filters or QueryFilter()
            
            # Search nodes using storage layer
            nodes = await self.storage.search_nodes(
                query=query,
                entity_types=filters.entity_types,
                limit=limit,
                min_confidence=filters.min_confidence
            )
            
            # Apply additional filters if specified
            if filters.node_properties:
                nodes = [node for node in nodes if self._node_matches_properties(node, filters.node_properties)]
            
            if include_similarity_score:
                # For now, return with dummy similarity scores
                # In practice, would use more sophisticated similarity calculation
                return [(node, 0.8) for node in nodes]
            
            return nodes
        
        except Exception as e:
            logger.error(f"Error in entity search: {e}")
            return []
    
    async def get_entity_by_name(
        self,
        name: str,
        entity_type: Optional[EntityType] = None,
        exact_match: bool = False
    ) -> Optional[GraphNode]:
        """Get a specific entity by name"""
        try:
            if exact_match:
                return await self.storage.get_node_by_name(name, entity_type)
            else:
                # Use search for fuzzy matching
                results = await self.search_entities(
                    query=name,
                    filters=QueryFilter(entity_types=[entity_type] if entity_type else None),
                    limit=1
                )
                return results[0] if results else None
        
        except Exception as e:
            logger.error(f"Error getting entity by name: {e}")
            return None
    
    async def get_entities_by_type(
        self,
        entity_type: EntityType,
        limit: int = 100,
        min_confidence: float = 0.0
    ) -> List[GraphNode]:
        """Get all entities of a specific type"""
        try:
            # Use search with type filter
            return await self.search_entities(
                query="",
                filters=QueryFilter(
                    entity_types=[entity_type],
                    min_confidence=min_confidence
                ),
                limit=limit
            )
        
        except Exception as e:
            logger.error(f"Error getting entities by type: {e}")
            return []
    
    # Relationship queries
    
    async def find_relationships(
        self,
        source_entity: Optional[str] = None,
        target_entity: Optional[str] = None,
        relationship_types: Optional[List[RelationshipType]] = None,
        min_strength: float = 0.0,
        limit: int = 100
    ) -> List[Tuple[GraphNode, GraphEdge, GraphNode]]:
        """
        Find relationships between entities.
        
        Returns:
            List of (source_node, edge, target_node) tuples
        """
        try:
            # Get node IDs if names provided
            source_node_id = None
            target_node_id = None
            
            if source_entity:
                source_node = await self.get_entity_by_name(source_entity)
                source_node_id = source_node.id if source_node else None
            
            if target_entity:
                target_node = await self.get_entity_by_name(target_entity)
                target_node_id = target_node.id if target_node else None
            
            # Get edges
            edges = await self.storage.get_edges_by_nodes(
                source_node_id=source_node_id,
                target_node_id=target_node_id,
                relationship_types=relationship_types,
                min_strength=min_strength
            )
            
            # Fetch related nodes for each edge
            results = []
            for edge in edges[:limit]:
                source_node = await self.storage.get_node_by_id(edge.source_node_id)
                target_node = await self.storage.get_node_by_id(edge.target_node_id)
                
                if source_node and target_node:
                    results.append((source_node, edge, target_node))
            
            return results
        
        except Exception as e:
            logger.error(f"Error finding relationships: {e}")
            return []
    
    async def get_entity_relationships(
        self,
        entity_name: str,
        direction: TraversalDirection = TraversalDirection.BOTH,
        relationship_types: Optional[List[RelationshipType]] = None,
        max_relationships: int = 50
    ) -> Dict[str, List[Tuple[GraphNode, GraphEdge]]]:
        """Get all relationships for an entity"""
        try:
            entity = await self.get_entity_by_name(entity_name)
            if not entity:
                return {}
            
            relationships = {
                'outgoing': [],
                'incoming': []
            }
            
            if direction in [TraversalDirection.OUTGOING, TraversalDirection.BOTH]:
                # Get outgoing relationships
                edges = await self.storage.get_edges_by_nodes(
                    source_node_id=entity.id,
                    relationship_types=relationship_types
                )
                
                for edge in edges[:max_relationships // 2]:
                    target_node = await self.storage.get_node_by_id(edge.target_node_id)
                    if target_node:
                        relationships['outgoing'].append((target_node, edge))
            
            if direction in [TraversalDirection.INCOMING, TraversalDirection.BOTH]:
                # Get incoming relationships
                edges = await self.storage.get_edges_by_nodes(
                    target_node_id=entity.id,
                    relationship_types=relationship_types
                )
                
                for edge in edges[:max_relationships // 2]:
                    source_node = await self.storage.get_node_by_id(edge.source_node_id)
                    if source_node:
                        relationships['incoming'].append((source_node, edge))
            
            return relationships
        
        except Exception as e:
            logger.error(f"Error getting entity relationships: {e}")
            return {}
    
    # Path finding queries
    
    async def find_shortest_path(
        self,
        source_entity: str,
        target_entity: str,
        max_depth: int = 5,
        relationship_filter: Optional[List[RelationshipType]] = None
    ) -> Optional[GraphPath]:
        """Find shortest path between two entities"""
        try:
            source_node = await self.get_entity_by_name(source_entity)
            target_node = await self.get_entity_by_name(target_entity)
            
            if not source_node or not target_node:
                return None
            
            # Use BFS to find shortest path
            path = await self._bfs_shortest_path(
                source_node.id,
                target_node.id,
                max_depth,
                relationship_filter
            )
            
            return path
        
        except Exception as e:
            logger.error(f"Error finding shortest path: {e}")
            return None
    
    async def find_all_paths(
        self,
        source_entity: str,
        target_entity: str,
        max_depth: int = 3,
        max_paths: int = 10,
        relationship_filter: Optional[List[RelationshipType]] = None
    ) -> List[GraphPath]:
        """Find all paths between two entities within depth limit"""
        try:
            source_node = await self.get_entity_by_name(source_entity)
            target_node = await self.get_entity_by_name(target_entity)
            
            if not source_node or not target_node:
                return []
            
            # Use DFS to find all paths
            paths = await self._dfs_all_paths(
                source_node.id,
                target_node.id,
                max_depth,
                max_paths,
                relationship_filter
            )
            
            return paths
        
        except Exception as e:
            logger.error(f"Error finding all paths: {e}")
            return []
    
    # Similarity queries
    
    async def find_similar_entities(
        self,
        entity_name: str,
        similarity_threshold: float = 0.7,
        max_results: int = 10,
        method: str = "embedding"
    ) -> SimilarityResult:
        """Find entities similar to the given entity"""
        try:
            query_node = await self.get_entity_by_name(entity_name)
            if not query_node or not query_node.embedding:
                return SimilarityResult(
                    query_node=query_node,
                    similar_nodes=[],
                    search_parameters={"error": "No embedding available"}
                )
            
            # Use storage layer for similarity search
            similar_nodes = await self.storage.find_similar_nodes(
                embedding=query_node.embedding,
                similarity_threshold=similarity_threshold,
                max_results=max_results
            )
            
            # Remove the query node from results
            similar_nodes = [(node, score) for node, score in similar_nodes 
                           if node.id != query_node.id]
            
            return SimilarityResult(
                query_node=query_node,
                similar_nodes=similar_nodes,
                query_embedding=query_node.embedding,
                search_parameters={
                    "method": method,
                    "threshold": similarity_threshold,
                    "max_results": max_results
                }
            )
        
        except Exception as e:
            logger.error(f"Error finding similar entities: {e}")
            return SimilarityResult(
                similar_nodes=[],
                search_parameters={"error": str(e)}
            )
    
    async def find_similar_by_embedding(
        self,
        embedding: List[float],
        similarity_threshold: float = 0.7,
        max_results: int = 10
    ) -> SimilarityResult:
        """Find entities similar to a given embedding"""
        try:
            similar_nodes = await self.storage.find_similar_nodes(
                embedding=embedding,
                similarity_threshold=similarity_threshold,
                max_results=max_results
            )
            
            return SimilarityResult(
                similar_nodes=similar_nodes,
                query_embedding=embedding,
                search_parameters={
                    "method": "embedding",
                    "threshold": similarity_threshold,
                    "max_results": max_results
                }
            )
        
        except Exception as e:
            logger.error(f"Error finding similar by embedding: {e}")
            return SimilarityResult(
                similar_nodes=[],
                search_parameters={"error": str(e)}
            )
    
    # Subgraph extraction
    
    async def extract_subgraph(
        self,
        central_entity: str,
        max_depth: int = 2,
        max_nodes: int = 50,
        relationship_filter: Optional[List[RelationshipType]] = None,
        direction: TraversalDirection = TraversalDirection.BOTH
    ) -> Optional[SubGraph]:
        """Extract a subgraph centered on an entity"""
        try:
            central_node = await self.get_entity_by_name(central_entity)
            if not central_node:
                return None
            
            # Use storage layer neighbor function for efficient traversal
            visited_nodes = {central_node.id: central_node}
            edges = []
            
            # Get neighbors at each depth level
            for depth in range(1, max_depth + 1):
                if len(visited_nodes) >= max_nodes:
                    break
                
                # Get all current node IDs
                current_nodes = list(visited_nodes.keys())
                
                for node_id in current_nodes:
                    # Get neighbors for this node
                    neighbors = await self.storage.get_node_neighbors(
                        node_id=node_id,
                        max_depth=1,  # One hop at a time
                        relationship_filter=relationship_filter.value if relationship_filter else None
                    )
                    
                    for neighbor_info in neighbors:
                        neighbor_id = neighbor_info['neighbor_id']
                        
                        # Add new nodes
                        if neighbor_id not in visited_nodes and len(visited_nodes) < max_nodes:
                            neighbor_node = await self.storage.get_node_by_id(neighbor_id)
                            if neighbor_node:
                                visited_nodes[neighbor_id] = neighbor_node
                        
                        # Find the edge between nodes
                        edge_candidates = await self.storage.get_edges_by_nodes(
                            source_node_id=node_id,
                            target_node_id=neighbor_id
                        )
                        if not edge_candidates:
                            edge_candidates = await self.storage.get_edges_by_nodes(
                                source_node_id=neighbor_id,
                                target_node_id=node_id
                            )
                        
                        if edge_candidates:
                            edges.append(edge_candidates[0])
            
            # Calculate statistics
            node_list = list(visited_nodes.values())
            statistics = {
                'total_nodes': len(node_list),
                'total_edges': len(edges),
                'max_depth': max_depth,
                'entity_types': list(set(node.entity_type.value for node in node_list)),
                'relationship_types': list(set(edge.relationship_type.value for edge in edges)),
                'avg_confidence': sum(node.confidence for node in node_list) / len(node_list) if node_list else 0,
                'avg_strength': sum(edge.strength for edge in edges) / len(edges) if edges else 0
            }
            
            return SubGraph(
                central_node=central_node,
                nodes=node_list,
                edges=edges,
                depth=max_depth,
                statistics=statistics
            )
        
        except Exception as e:
            logger.error(f"Error extracting subgraph: {e}")
            return None
    
    # Analytics queries
    
    async def calculate_centrality_metrics(
        self,
        entity_name: Optional[str] = None,
        limit: int = 100
    ) -> List[CentralityMetrics]:
        """Calculate centrality metrics for nodes"""
        try:
            if entity_name:
                entity = await self.get_entity_by_name(entity_name)
                if entity:
                    metrics = await self.storage.calculate_node_centrality_metrics(entity.id)
                    return [CentralityMetrics(
                        node_id=str(entity.id),
                        **metrics
                    )]
                return []
            else:
                # Calculate for top nodes by degree
                # This is a simplified implementation
                stats = await self.storage.get_graph_statistics()
                # Would implement full centrality calculations here
                return []
        
        except Exception as e:
            logger.error(f"Error calculating centrality metrics: {e}")
            return []
    
    async def get_graph_statistics(self) -> Dict[str, Any]:
        """Get overall graph statistics"""
        try:
            return await self.storage.get_graph_statistics()
        except Exception as e:
            logger.error(f"Error getting graph statistics: {e}")
            return {}
    
    async def analyze_entity_connections(
        self,
        entity_name: str
    ) -> Dict[str, Any]:
        """Analyze the connection patterns of an entity"""
        try:
            entity = await self.get_entity_by_name(entity_name)
            if not entity:
                return {}
            
            # Get all relationships
            relationships = await self.get_entity_relationships(entity_name)
            
            # Analyze patterns
            analysis = {
                'entity_name': entity_name,
                'entity_type': entity.entity_type.value,
                'total_connections': len(relationships['outgoing']) + len(relationships['incoming']),
                'outgoing_connections': len(relationships['outgoing']),
                'incoming_connections': len(relationships['incoming']),
                'relationship_types': {},
                'connected_entity_types': {},
                'avg_relationship_strength': 0.0,
                'strongest_connections': [],
                'weakest_connections': []
            }
            
            all_edges = []
            all_edges.extend([edge for _, edge in relationships['outgoing']])
            all_edges.extend([edge for _, edge in relationships['incoming']])
            
            # Analyze relationship types
            for edge in all_edges:
                rel_type = edge.relationship_type.value
                analysis['relationship_types'][rel_type] = analysis['relationship_types'].get(rel_type, 0) + 1
            
            # Analyze connected entity types
            all_connected_nodes = []
            all_connected_nodes.extend([node for node, _ in relationships['outgoing']])
            all_connected_nodes.extend([node for node, _ in relationships['incoming']])
            
            for node in all_connected_nodes:
                entity_type = node.entity_type.value
                analysis['connected_entity_types'][entity_type] = analysis['connected_entity_types'].get(entity_type, 0) + 1
            
            # Calculate average strength
            if all_edges:
                analysis['avg_relationship_strength'] = sum(edge.strength for edge in all_edges) / len(all_edges)
                
                # Find strongest and weakest connections
                sorted_edges = sorted(all_edges, key=lambda e: e.strength, reverse=True)
                analysis['strongest_connections'] = [
                    {
                        'relationship_type': edge.relationship_type.value,
                        'strength': edge.strength,
                        'description': edge.description
                    }
                    for edge in sorted_edges[:5]
                ]
                
                analysis['weakest_connections'] = [
                    {
                        'relationship_type': edge.relationship_type.value,
                        'strength': edge.strength,
                        'description': edge.description
                    }
                    for edge in sorted_edges[-5:]
                ]
            
            return analysis
        
        except Exception as e:
            logger.error(f"Error analyzing entity connections: {e}")
            return {}
    
    # Helper methods
    
    async def _bfs_shortest_path(
        self,
        source_id: Any,
        target_id: Any,
        max_depth: int,
        relationship_filter: Optional[List[RelationshipType]]
    ) -> Optional[GraphPath]:
        """BFS implementation for shortest path finding"""
        
        if source_id == target_id:
            source_node = await self.storage.get_node_by_id(source_id)
            if source_node:
                return GraphPath(
                    nodes=[source_node],
                    edges=[],
                    total_strength=1.0,
                    path_length=0,
                    confidence=source_node.confidence
                )
        
        # BFS queue: (current_node_id, path_nodes, path_edges, depth)
        queue = [(source_id, [source_id], [], 0)]
        visited = {source_id}
        
        while queue:
            current_id, path_nodes, path_edges, depth = queue.pop(0)
            
            if depth >= max_depth:
                continue
            
            # Get neighbors
            neighbors = await self.storage.get_node_neighbors(
                node_id=current_id,
                max_depth=1,
                relationship_filter=relationship_filter[0] if relationship_filter else None
            )
            
            for neighbor_info in neighbors:
                neighbor_id = neighbor_info['neighbor_id']
                
                if neighbor_id == target_id:
                    # Found target, reconstruct path
                    final_path_nodes = path_nodes + [neighbor_id]
                    
                    # Get all nodes and edges for the path
                    nodes = []
                    edges = []
                    total_strength = 0.0
                    total_confidence = 0.0
                    
                    for node_id in final_path_nodes:
                        node = await self.storage.get_node_by_id(node_id)
                        if node:
                            nodes.append(node)
                            total_confidence += node.confidence
                    
                    # Get edges between consecutive nodes
                    for i in range(len(final_path_nodes) - 1):
                        edge_candidates = await self.storage.get_edges_by_nodes(
                            source_node_id=final_path_nodes[i],
                            target_node_id=final_path_nodes[i + 1]
                        )
                        if not edge_candidates:
                            edge_candidates = await self.storage.get_edges_by_nodes(
                                source_node_id=final_path_nodes[i + 1],
                                target_node_id=final_path_nodes[i]
                            )
                        
                        if edge_candidates:
                            edges.append(edge_candidates[0])
                            total_strength += edge_candidates[0].strength
                    
                    avg_confidence = total_confidence / len(nodes) if nodes else 0.0
                    
                    return GraphPath(
                        nodes=nodes,
                        edges=edges,
                        total_strength=total_strength,
                        path_length=len(edges),
                        confidence=avg_confidence
                    )
                
                elif neighbor_id not in visited:
                    visited.add(neighbor_id)
                    new_path_nodes = path_nodes + [neighbor_id]
                    queue.append((neighbor_id, new_path_nodes, path_edges, depth + 1))
        
        return None
    
    async def _dfs_all_paths(
        self,
        source_id: Any,
        target_id: Any,
        max_depth: int,
        max_paths: int,
        relationship_filter: Optional[List[RelationshipType]]
    ) -> List[GraphPath]:
        """DFS implementation for finding all paths"""
        
        paths = []
        
        async def dfs_recursive(current_id, target_id, path_nodes, visited, depth):
            if len(paths) >= max_paths or depth > max_depth:
                return
            
            if current_id == target_id and len(path_nodes) > 1:
                # Found a path, convert to GraphPath
                nodes = []
                edges = []
                total_strength = 0.0
                total_confidence = 0.0
                
                # Get nodes
                for node_id in path_nodes:
                    node = await self.storage.get_node_by_id(node_id)
                    if node:
                        nodes.append(node)
                        total_confidence += node.confidence
                
                # Get edges
                for i in range(len(path_nodes) - 1):
                    edge_candidates = await self.storage.get_edges_by_nodes(
                        source_node_id=path_nodes[i],
                        target_node_id=path_nodes[i + 1]
                    )
                    if not edge_candidates:
                        edge_candidates = await self.storage.get_edges_by_nodes(
                            source_node_id=path_nodes[i + 1],
                            target_node_id=path_nodes[i]
                        )
                    
                    if edge_candidates:
                        edges.append(edge_candidates[0])
                        total_strength += edge_candidates[0].strength
                
                if nodes and edges:
                    paths.append(GraphPath(
                        nodes=nodes,
                        edges=edges,
                        total_strength=total_strength,
                        path_length=len(edges),
                        confidence=total_confidence / len(nodes)
                    ))
                return
            
            # Continue DFS
            neighbors = await self.storage.get_node_neighbors(
                node_id=current_id,
                max_depth=1,
                relationship_filter=relationship_filter[0] if relationship_filter else None
            )
            
            for neighbor_info in neighbors:
                neighbor_id = neighbor_info['neighbor_id']
                
                if neighbor_id not in visited:
                    new_visited = visited.copy()
                    new_visited.add(neighbor_id)
                    new_path = path_nodes + [neighbor_id]
                    
                    await dfs_recursive(neighbor_id, target_id, new_path, new_visited, depth + 1)
        
        await dfs_recursive(source_id, target_id, [source_id], {source_id}, 0)
        return paths
    
    def _node_matches_properties(self, node: GraphNode, property_filters: Dict[str, Any]) -> bool:
        """Check if node matches property filters"""
        for key, expected_value in property_filters.items():
            if key in node.properties:
                if node.properties[key] != expected_value:
                    return False
            elif key in node.metadata:
                if node.metadata[key] != expected_value:
                    return False
            else:
                return False
        return True


# Global query instance
_query_engine: Optional[KnowledgeGraphQuery] = None


def get_graph_query_engine() -> Optional[KnowledgeGraphQuery]:
    """Get global graph query engine"""
    return _query_engine


def initialize_graph_query_engine(storage: KnowledgeGraphStorage) -> KnowledgeGraphQuery:
    """Initialize global graph query engine"""
    global _query_engine
    _query_engine = KnowledgeGraphQuery(storage)
    return _query_engine