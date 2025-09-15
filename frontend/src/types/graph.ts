/**
 * Graph visualization types for Knowledge Graph component
 */

export interface GraphNode {
  id: string;
  label: string;
  type: 'document' | 'entity' | 'concept' | 'tag' | 'folder';
  group?: string;
  metadata?: {
    documentId?: string;
    confidence?: number;
    extractedAt?: string;
    source?: string;
    description?: string;
    [key: string]: any;
  };
  // D3 simulation properties
  x?: number;
  y?: number;
  vx?: number;
  vy?: number;
  fx?: number | null;
  fy?: number | null;
}

export interface GraphEdge {
  id: string;
  source: string | GraphNode;
  target: string | GraphNode;
  type: 'references' | 'contains' | 'related_to' | 'extracted_from' | 'similar_to';
  weight?: number;
  label?: string;
  metadata?: {
    confidence?: number;
    createdAt?: string;
    context?: string;
    [key: string]: any;
  };
}

export interface GraphData {
  nodes: GraphNode[];
  edges: GraphEdge[];
}

export interface GraphDimensions {
  width: number;
  height: number;
}

export interface GraphConfig {
  // Force simulation parameters
  chargeStrength?: number;
  linkDistance?: number;
  linkStrength?: number;
  centerForce?: number;
  collisionRadius?: number;

  // Visual parameters
  nodeRadius?: number;
  nodeMinRadius?: number;
  nodeMaxRadius?: number;
  edgeWidth?: number;
  edgeOpacity?: number;

  // Interaction parameters
  zoomMin?: number;
  zoomMax?: number;
  animationDuration?: number;

  // Performance parameters
  simulationAlpha?: number;
  simulationAlphaDecay?: number;
  simulationVelocityDecay?: number;

  // Display options
  showLabels?: boolean;
  showEdgeLabels?: boolean;
  showLegend?: boolean;
  showMinimap?: boolean;
}

export interface GraphFilters {
  nodeTypes?: string[];
  edgeTypes?: string[];
  searchQuery?: string;
  minConfidence?: number;
  dateRange?: {
    start?: Date;
    end?: Date;
  };
  clusters?: string[];
}

export interface GraphSelection {
  selectedNodes: Set<string>;
  hoveredNode?: string;
  selectedEdges: Set<string>;
  hoveredEdge?: string;
}

export interface GraphStats {
  totalNodes: number;
  totalEdges: number;
  nodesByType: Record<string, number>;
  edgesByType: Record<string, number>;
  avgDegree: number;
  density: number;
  clusters: number;
}

export interface GraphColorScheme {
  document: string;
  entity: string;
  concept: string;
  tag: string;
  folder: string;
  edge: string;
  selectedNode: string;
  selectedEdge: string;
  hoveredNode: string;
  hoveredEdge: string;
  background: string;
  text: string;
}

export const DEFAULT_GRAPH_CONFIG: GraphConfig = {
  // Force simulation
  chargeStrength: -300,
  linkDistance: 100,
  linkStrength: 0.5,
  centerForce: 0.05,
  collisionRadius: 30,

  // Visual
  nodeRadius: 8,
  nodeMinRadius: 4,
  nodeMaxRadius: 20,
  edgeWidth: 1,
  edgeOpacity: 0.6,

  // Interaction
  zoomMin: 0.1,
  zoomMax: 10,
  animationDuration: 300,

  // Performance
  simulationAlpha: 1,
  simulationAlphaDecay: 0.01,
  simulationVelocityDecay: 0.3,

  // Display
  showLabels: true,
  showEdgeLabels: false,
  showLegend: true,
  showMinimap: false,
};

export const DEFAULT_COLOR_SCHEME: GraphColorScheme = {
  document: '#3B82F6', // Blue
  entity: '#10B981',   // Emerald
  concept: '#8B5CF6',  // Violet
  tag: '#F59E0B',      // Amber
  folder: '#6B7280',   // Gray
  edge: '#9CA3AF',     // Light gray
  selectedNode: '#DC2626', // Red
  selectedEdge: '#DC2626',  // Red
  hoveredNode: '#059669',   // Green
  hoveredEdge: '#059669',   // Green
  background: '#111827',    // Dark background
  text: '#F3F4F6',         // Light text
};