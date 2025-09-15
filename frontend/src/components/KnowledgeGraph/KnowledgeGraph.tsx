import React, { useRef, useEffect, useState, useCallback } from 'react';
import * as d3 from 'd3';
import {
  GraphData,
  GraphNode,
  GraphEdge,
  GraphConfig,
  GraphSelection,
  GraphFilters,
  DEFAULT_GRAPH_CONFIG,
  DEFAULT_COLOR_SCHEME,
} from '../../types/graph';
import GraphControls from './GraphControls';
import GraphSearch from './GraphSearch';
import GraphLegend from './GraphLegend';
import NodeInfoPanel from './NodeInfoPanel';
import { useGraphData } from '../../hooks/useGraphData';
import LoadingSpinner from '../LoadingSpinner';

interface KnowledgeGraphProps {
  className?: string;
  initialFilters?: GraphFilters;
  config?: Partial<GraphConfig>;
  onNodeSelect?: (node: GraphNode) => void;
  onEdgeSelect?: (edge: GraphEdge) => void;
}

const KnowledgeGraph: React.FC<KnowledgeGraphProps> = ({
  className = '',
  initialFilters = {},
  config: userConfig = {},
  onNodeSelect,
  onEdgeSelect,
}) => {
  const svgRef = useRef<SVGSVGElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const simulationRef = useRef<d3.Simulation<GraphNode, GraphEdge> | null>(null);

  const [dimensions, setDimensions] = useState({ width: 800, height: 600 });
  const [selection, setSelection] = useState<GraphSelection>({
    selectedNodes: new Set(),
    selectedEdges: new Set(),
  });
  const [filters, setFilters] = useState<GraphFilters>(initialFilters);
  const [transform, setTransform] = useState<d3.ZoomTransform>(d3.zoomIdentity);

  const config = { ...DEFAULT_GRAPH_CONFIG, ...userConfig };
  const { data, loading, error, refetch } = useGraphData(filters);

  // Update dimensions on container resize
  useEffect(() => {
    const updateDimensions = () => {
      if (containerRef.current) {
        const { width, height } = containerRef.current.getBoundingClientRect();
        setDimensions({ width, height });
      }
    };

    updateDimensions();
    const resizeObserver = new ResizeObserver(updateDimensions);
    if (containerRef.current) {
      resizeObserver.observe(containerRef.current);
    }

    return () => resizeObserver.disconnect();
  }, []);

  // Initialize and update D3 force simulation
  useEffect(() => {
    if (!svgRef.current || !data || loading) return;

    const svg = d3.select(svgRef.current);
    svg.selectAll('*').remove();

    // Create container groups
    const g = svg.append('g').attr('class', 'graph-container');

    // Add zoom behavior
    const zoom = d3.zoom<SVGSVGElement, unknown>()
      .scaleExtent([config.zoomMin || 0.1, config.zoomMax || 10])
      .on('zoom', (event) => {
        g.attr('transform', event.transform);
        setTransform(event.transform);
      });

    svg.call(zoom);

    // Create arrow markers for directed edges
    const defs = svg.append('defs');
    defs.append('marker')
      .attr('id', 'arrow')
      .attr('viewBox', '0 -5 10 10')
      .attr('refX', 20)
      .attr('refY', 0)
      .attr('markerWidth', 8)
      .attr('markerHeight', 8)
      .attr('orient', 'auto')
      .append('path')
      .attr('d', 'M0,-5L10,0L0,5')
      .attr('fill', DEFAULT_COLOR_SCHEME.edge);

    // Create force simulation
    const simulation = d3.forceSimulation<GraphNode>(data.nodes)
      .force('link', d3.forceLink<GraphNode, GraphEdge>(data.edges)
        .id(d => d.id)
        .distance(config.linkDistance || 100)
        .strength(config.linkStrength || 0.5))
      .force('charge', d3.forceManyBody()
        .strength(config.chargeStrength || -300))
      .force('center', d3.forceCenter(dimensions.width / 2, dimensions.height / 2)
        .strength(config.centerForce || 0.05))
      .force('collision', d3.forceCollide()
        .radius(config.collisionRadius || 30));

    simulationRef.current = simulation;

    // Create edge elements
    const edges = g.append('g')
      .attr('class', 'edges')
      .selectAll('line')
      .data(data.edges)
      .join('line')
      .attr('stroke', DEFAULT_COLOR_SCHEME.edge)
      .attr('stroke-width', d => Math.sqrt(d.weight || 1))
      .attr('stroke-opacity', config.edgeOpacity || 0.6)
      .attr('marker-end', 'url(#arrow)')
      .on('click', (event, d) => {
        event.stopPropagation();
        handleEdgeClick(d);
      })
      .on('mouseenter', (event, d) => {
        setSelection(prev => ({ ...prev, hoveredEdge: d.id }));
      })
      .on('mouseleave', () => {
        setSelection(prev => ({ ...prev, hoveredEdge: undefined }));
      });

    // Create node elements
    const nodes = g.append('g')
      .attr('class', 'nodes')
      .selectAll('circle')
      .data(data.nodes)
      .join('circle')
      .attr('r', d => calculateNodeRadius(d))
      .attr('fill', d => getNodeColor(d))
      .attr('stroke', '#fff')
      .attr('stroke-width', 2)
      .style('cursor', 'pointer')
      .on('click', (event, d) => {
        event.stopPropagation();
        handleNodeClick(d);
      })
      .on('mouseenter', (event, d) => {
        setSelection(prev => ({ ...prev, hoveredNode: d.id }));
        showTooltip(event, d);
      })
      .on('mouseleave', () => {
        setSelection(prev => ({ ...prev, hoveredNode: undefined }));
        hideTooltip();
      })
      .call(d3.drag<SVGCircleElement, GraphNode>()
        .on('start', dragStarted)
        .on('drag', dragged)
        .on('end', dragEnded) as any);

    // Create labels if enabled
    if (config.showLabels) {
      const labels = g.append('g')
        .attr('class', 'labels')
        .selectAll('text')
        .data(data.nodes)
        .join('text')
        .text(d => d.label)
        .attr('font-size', 12)
        .attr('fill', DEFAULT_COLOR_SCHEME.text)
        .attr('text-anchor', 'middle')
        .attr('dy', -15)
        .style('pointer-events', 'none')
        .style('user-select', 'none');

      // Update label positions on simulation tick
      simulation.on('tick.labels', () => {
        labels
          .attr('x', d => d.x || 0)
          .attr('y', d => d.y || 0);
      });
    }

    // Update positions on simulation tick
    simulation.on('tick', () => {
      edges
        .attr('x1', d => (d.source as GraphNode).x || 0)
        .attr('y1', d => (d.source as GraphNode).y || 0)
        .attr('x2', d => (d.target as GraphNode).x || 0)
        .attr('y2', d => (d.target as GraphNode).y || 0);

      nodes
        .attr('cx', d => d.x || 0)
        .attr('cy', d => d.y || 0);
    });

    // Drag functions
    function dragStarted(event: d3.D3DragEvent<SVGCircleElement, GraphNode, GraphNode>, d: GraphNode) {
      if (!event.active) simulation.alphaTarget(0.3).restart();
      d.fx = d.x;
      d.fy = d.y;
    }

    function dragged(event: d3.D3DragEvent<SVGCircleElement, GraphNode, GraphNode>, d: GraphNode) {
      d.fx = event.x;
      d.fy = event.y;
    }

    function dragEnded(event: d3.D3DragEvent<SVGCircleElement, GraphNode, GraphNode>, d: GraphNode) {
      if (!event.active) simulation.alphaTarget(0);
      d.fx = null;
      d.fy = null;
    }

    return () => {
      simulation.stop();
    };
  }, [data, dimensions, config, loading]);

  const calculateNodeRadius = (node: GraphNode): number => {
    const baseRadius = config.nodeRadius || 8;
    const scaleFactor = node.metadata?.confidence || 1;
    return Math.max(
      config.nodeMinRadius || 4,
      Math.min(config.nodeMaxRadius || 20, baseRadius * scaleFactor)
    );
  };

  const getNodeColor = (node: GraphNode): string => {
    if (selection.selectedNodes.has(node.id)) {
      return DEFAULT_COLOR_SCHEME.selectedNode;
    }
    if (selection.hoveredNode === node.id) {
      return DEFAULT_COLOR_SCHEME.hoveredNode;
    }
    return DEFAULT_COLOR_SCHEME[node.type] || DEFAULT_COLOR_SCHEME.entity;
  };

  const handleNodeClick = useCallback((node: GraphNode) => {
    setSelection(prev => {
      const newSelection = new Set(prev.selectedNodes);
      if (newSelection.has(node.id)) {
        newSelection.delete(node.id);
      } else {
        newSelection.add(node.id);
      }
      return { ...prev, selectedNodes: newSelection };
    });
    onNodeSelect?.(node);
  }, [onNodeSelect]);

  const handleEdgeClick = useCallback((edge: GraphEdge) => {
    setSelection(prev => {
      const newSelection = new Set(prev.selectedEdges);
      if (newSelection.has(edge.id)) {
        newSelection.delete(edge.id);
      } else {
        newSelection.add(edge.id);
      }
      return { ...prev, selectedEdges: newSelection };
    });
    onEdgeSelect?.(edge);
  }, [onEdgeSelect]);

  const showTooltip = (event: MouseEvent, node: GraphNode) => {
    // Tooltip implementation would go here
    console.log('Show tooltip for', node);
  };

  const hideTooltip = () => {
    // Hide tooltip implementation
  };

  const handleSearch = (query: string) => {
    setFilters(prev => ({ ...prev, searchQuery: query }));
  };

  const handleFilterChange = (newFilters: GraphFilters) => {
    setFilters(newFilters);
  };

  const handleZoomIn = () => {
    if (!svgRef.current) return;
    const svg = d3.select(svgRef.current);
    svg.transition().duration(300).call(
      d3.zoom<SVGSVGElement, unknown>().scaleBy as any,
      1.3
    );
  };

  const handleZoomOut = () => {
    if (!svgRef.current) return;
    const svg = d3.select(svgRef.current);
    svg.transition().duration(300).call(
      d3.zoom<SVGSVGElement, unknown>().scaleBy as any,
      0.7
    );
  };

  const handleZoomReset = () => {
    if (!svgRef.current) return;
    const svg = d3.select(svgRef.current);
    svg.transition().duration(300).call(
      d3.zoom<SVGSVGElement, unknown>().transform as any,
      d3.zoomIdentity
    );
  };

  const handleRefresh = () => {
    refetch();
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center h-full">
        <LoadingSpinner />
      </div>
    );
  }

  if (error) {
    return (
      <div className="flex items-center justify-center h-full">
        <div className="text-red-500">Error loading graph data: {error}</div>
      </div>
    );
  }

  const selectedNode = data?.nodes.find(n => selection.selectedNodes.has(n.id));

  return (
    <div ref={containerRef} className={`relative w-full h-full bg-gray-900 ${className}`}>
      <svg
        ref={svgRef}
        width={dimensions.width}
        height={dimensions.height}
        className="w-full h-full"
      />

      <div className="absolute top-4 left-4 space-y-4">
        <GraphSearch onSearch={handleSearch} />
        <GraphControls
          onZoomIn={handleZoomIn}
          onZoomOut={handleZoomOut}
          onZoomReset={handleZoomReset}
          onRefresh={handleRefresh}
          currentZoom={transform.k}
        />
      </div>

      {config.showLegend && (
        <div className="absolute top-4 right-4">
          <GraphLegend />
        </div>
      )}

      {selectedNode && (
        <div className="absolute bottom-4 right-4 w-80">
          <NodeInfoPanel
            node={selectedNode}
            onClose={() => setSelection(prev => ({
              ...prev,
              selectedNodes: new Set(),
            }))}
          />
        </div>
      )}
    </div>
  );
};

export default KnowledgeGraph;