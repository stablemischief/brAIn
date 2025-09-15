import { useState, useEffect, useCallback, useMemo } from 'react';
import { GraphData, GraphFilters, GraphNode, GraphEdge } from '../types/graph';

// Mock data generator for testing
const generateMockData = (nodeCount: number = 100): GraphData => {
  const nodes: GraphNode[] = [];
  const edges: GraphEdge[] = [];

  const types: GraphNode['type'][] = ['document', 'entity', 'concept', 'tag', 'folder'];

  // Generate nodes
  for (let i = 0; i < nodeCount; i++) {
    nodes.push({
      id: `node-${i}`,
      label: `Node ${i}`,
      type: types[Math.floor(Math.random() * types.length)],
      group: `group-${Math.floor(i / 10)}`,
      metadata: {
        confidence: Math.random(),
        extractedAt: new Date(Date.now() - Math.random() * 30 * 24 * 60 * 60 * 1000).toISOString(),
        source: `source-${Math.floor(Math.random() * 5)}`,
        description: `This is a description for node ${i}`,
      },
    });
  }

  // Generate edges (roughly 2x nodes for good connectivity)
  const edgeCount = nodeCount * 2;
  const edgeTypes: GraphEdge['type'][] = ['references', 'contains', 'related_to', 'extracted_from', 'similar_to'];

  for (let i = 0; i < edgeCount; i++) {
    const source = Math.floor(Math.random() * nodeCount);
    const target = Math.floor(Math.random() * nodeCount);

    if (source !== target) {
      edges.push({
        id: `edge-${i}`,
        source: `node-${source}`,
        target: `node-${target}`,
        type: edgeTypes[Math.floor(Math.random() * edgeTypes.length)],
        weight: Math.random() * 5 + 1,
        metadata: {
          confidence: Math.random(),
          createdAt: new Date().toISOString(),
        },
      });
    }
  }

  return { nodes, edges };
};

interface UseGraphDataReturn {
  data: GraphData | null;
  loading: boolean;
  error: string | null;
  refetch: () => void;
}

export const useGraphData = (filters: GraphFilters = {}): UseGraphDataReturn => {
  const [data, setData] = useState<GraphData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetchGraphData = useCallback(async () => {
    setLoading(true);
    setError(null);

    try {
      // In production, this would be an API call
      const apiUrl = '/api/knowledge-graph';

      // For now, use mock data
      if (process.env.NODE_ENV === 'development') {
        // Simulate network delay
        await new Promise(resolve => setTimeout(resolve, 500));

        // Generate mock data with performance considerations
        const nodeCount = filters.searchQuery ? 50 : 200; // Reduce nodes when searching
        const mockData = generateMockData(nodeCount);

        // Apply filters
        let filteredData = { ...mockData };

        if (filters.searchQuery) {
          const query = filters.searchQuery.toLowerCase();
          filteredData.nodes = mockData.nodes.filter(node =>
            node.label.toLowerCase().includes(query) ||
            node.type.includes(query) ||
            node.metadata?.description?.toLowerCase().includes(query)
          );

          // Filter edges to only include those connecting filtered nodes
          const nodeIds = new Set(filteredData.nodes.map(n => n.id));
          filteredData.edges = mockData.edges.filter(edge =>
            nodeIds.has(typeof edge.source === 'string' ? edge.source : edge.source.id) &&
            nodeIds.has(typeof edge.target === 'string' ? edge.target : edge.target.id)
          );
        }

        if (filters.nodeTypes && filters.nodeTypes.length > 0) {
          filteredData.nodes = filteredData.nodes.filter(node =>
            filters.nodeTypes!.includes(node.type)
          );
        }

        if (filters.minConfidence !== undefined) {
          filteredData.nodes = filteredData.nodes.filter(node =>
            (node.metadata?.confidence || 0) >= filters.minConfidence!
          );
        }

        setData(filteredData);
      } else {
        // Production API call
        const params = new URLSearchParams();
        if (filters.searchQuery) params.append('q', filters.searchQuery);
        if (filters.nodeTypes) params.append('types', filters.nodeTypes.join(','));
        if (filters.minConfidence) params.append('min_confidence', filters.minConfidence.toString());

        const response = await fetch(`${apiUrl}?${params}`);
        if (!response.ok) throw new Error('Failed to fetch graph data');

        const graphData = await response.json();
        setData(graphData);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An error occurred');
      setData(null);
    } finally {
      setLoading(false);
    }
  }, [filters]);

  useEffect(() => {
    fetchGraphData();
  }, [fetchGraphData]);

  // Performance optimization: memoize filtered data
  const optimizedData = useMemo(() => {
    if (!data) return null;

    // For large graphs, implement level-of-detail (LOD) optimization
    const MAX_RENDER_NODES = 500;
    const MAX_RENDER_EDGES = 1000;

    if (data.nodes.length > MAX_RENDER_NODES) {
      // Sort by importance (confidence) and take top nodes
      const sortedNodes = [...data.nodes].sort((a, b) =>
        (b.metadata?.confidence || 0) - (a.metadata?.confidence || 0)
      );

      const visibleNodes = sortedNodes.slice(0, MAX_RENDER_NODES);
      const visibleNodeIds = new Set(visibleNodes.map(n => n.id));

      const visibleEdges = data.edges
        .filter(edge =>
          visibleNodeIds.has(typeof edge.source === 'string' ? edge.source : edge.source.id) &&
          visibleNodeIds.has(typeof edge.target === 'string' ? edge.target : edge.target.id)
        )
        .slice(0, MAX_RENDER_EDGES);

      return {
        nodes: visibleNodes,
        edges: visibleEdges,
      };
    }

    return data;
  }, [data]);

  return {
    data: optimizedData,
    loading,
    error,
    refetch: fetchGraphData,
  };
};

// Hook for graph statistics
export const useGraphStats = (data: GraphData | null) => {
  return useMemo(() => {
    if (!data) return null;

    const nodesByType: Record<string, number> = {};
    const edgesByType: Record<string, number> = {};

    data.nodes.forEach(node => {
      nodesByType[node.type] = (nodesByType[node.type] || 0) + 1;
    });

    data.edges.forEach(edge => {
      edgesByType[edge.type] = (edgesByType[edge.type] || 0) + 1;
    });

    // Calculate average degree
    const degrees = new Map<string, number>();
    data.edges.forEach(edge => {
      const sourceId = typeof edge.source === 'string' ? edge.source : edge.source.id;
      const targetId = typeof edge.target === 'string' ? edge.target : edge.target.id;
      degrees.set(sourceId, (degrees.get(sourceId) || 0) + 1);
      degrees.set(targetId, (degrees.get(targetId) || 0) + 1);
    });

    const avgDegree = Array.from(degrees.values()).reduce((a, b) => a + b, 0) / degrees.size || 0;

    // Calculate density (actual edges / possible edges)
    const possibleEdges = data.nodes.length * (data.nodes.length - 1) / 2;
    const density = possibleEdges > 0 ? data.edges.length / possibleEdges : 0;

    return {
      totalNodes: data.nodes.length,
      totalEdges: data.edges.length,
      nodesByType,
      edgesByType,
      avgDegree,
      density,
      clusters: Object.keys(nodesByType).length,
    };
  }, [data]);
};