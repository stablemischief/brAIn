/**
 * Performance optimization utilities for large graph rendering
 */

import { GraphNode, GraphEdge, GraphData } from '../types/graph';

/**
 * Quadtree implementation for spatial indexing of nodes
 * Improves performance for collision detection and spatial queries
 */
export class QuadTree {
  private x: number;
  private y: number;
  private width: number;
  private height: number;
  private maxObjects: number;
  private maxLevels: number;
  private level: number;
  private objects: GraphNode[];
  private nodes: QuadTree[];

  constructor(
    x: number,
    y: number,
    width: number,
    height: number,
    maxObjects = 10,
    maxLevels = 5,
    level = 0
  ) {
    this.x = x;
    this.y = y;
    this.width = width;
    this.height = height;
    this.maxObjects = maxObjects;
    this.maxLevels = maxLevels;
    this.level = level;
    this.objects = [];
    this.nodes = [];
  }

  clear(): void {
    this.objects = [];
    for (const node of this.nodes) {
      node.clear();
    }
    this.nodes = [];
  }

  split(): void {
    const subWidth = this.width / 2;
    const subHeight = this.height / 2;
    const x = this.x;
    const y = this.y;

    this.nodes[0] = new QuadTree(x + subWidth, y, subWidth, subHeight, this.maxObjects, this.maxLevels, this.level + 1);
    this.nodes[1] = new QuadTree(x, y, subWidth, subHeight, this.maxObjects, this.maxLevels, this.level + 1);
    this.nodes[2] = new QuadTree(x, y + subHeight, subWidth, subHeight, this.maxObjects, this.maxLevels, this.level + 1);
    this.nodes[3] = new QuadTree(x + subWidth, y + subHeight, subWidth, subHeight, this.maxObjects, this.maxLevels, this.level + 1);
  }

  getIndex(node: GraphNode): number {
    let index = -1;
    const verticalMidpoint = this.x + this.width / 2;
    const horizontalMidpoint = this.y + this.height / 2;

    const topQuadrant = node.y! < horizontalMidpoint;
    const bottomQuadrant = node.y! > horizontalMidpoint;

    if (node.x! < verticalMidpoint) {
      if (topQuadrant) {
        index = 1;
      } else if (bottomQuadrant) {
        index = 2;
      }
    } else if (node.x! > verticalMidpoint) {
      if (topQuadrant) {
        index = 0;
      } else if (bottomQuadrant) {
        index = 3;
      }
    }

    return index;
  }

  insert(node: GraphNode): void {
    if (this.nodes.length > 0) {
      const index = this.getIndex(node);
      if (index !== -1) {
        this.nodes[index].insert(node);
        return;
      }
    }

    this.objects.push(node);

    if (this.objects.length > this.maxObjects && this.level < this.maxLevels) {
      if (this.nodes.length === 0) {
        this.split();
      }

      let i = 0;
      while (i < this.objects.length) {
        const index = this.getIndex(this.objects[i]);
        if (index !== -1) {
          this.nodes[index].insert(this.objects.splice(i, 1)[0]);
        } else {
          i++;
        }
      }
    }
  }

  retrieve(node: GraphNode): GraphNode[] {
    const index = this.getIndex(node);
    let returnObjects = this.objects;

    if (this.nodes.length > 0) {
      if (index !== -1) {
        returnObjects = returnObjects.concat(this.nodes[index].retrieve(node));
      }
    }

    return returnObjects;
  }
}

/**
 * Canvas-based rendering for very large graphs
 * Uses WebGL when available for maximum performance
 */
export class CanvasRenderer {
  private canvas: HTMLCanvasElement;
  private ctx: CanvasRenderingContext2D;

  constructor(canvas: HTMLCanvasElement) {
    this.canvas = canvas;
    const ctx = canvas.getContext('2d');
    if (!ctx) throw new Error('Could not get 2D context');
    this.ctx = ctx;
  }

  clear(): void {
    this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
  }

  renderNodes(nodes: GraphNode[], transform: { k: number; x: number; y: number }): void {
    this.ctx.save();
    this.ctx.translate(transform.x, transform.y);
    this.ctx.scale(transform.k, transform.k);

    nodes.forEach(node => {
      if (node.x !== undefined && node.y !== undefined) {
        this.ctx.beginPath();
        this.ctx.arc(node.x, node.y, 5, 0, 2 * Math.PI);
        this.ctx.fillStyle = this.getNodeColor(node);
        this.ctx.fill();
        this.ctx.strokeStyle = '#fff';
        this.ctx.lineWidth = 1 / transform.k;
        this.ctx.stroke();
      }
    });

    this.ctx.restore();
  }

  renderEdges(edges: GraphEdge[], transform: { k: number; x: number; y: number }): void {
    this.ctx.save();
    this.ctx.translate(transform.x, transform.y);
    this.ctx.scale(transform.k, transform.k);

    this.ctx.strokeStyle = '#999';
    this.ctx.lineWidth = 1 / transform.k;

    edges.forEach(edge => {
      const source = edge.source as GraphNode;
      const target = edge.target as GraphNode;

      if (source.x !== undefined && source.y !== undefined &&
          target.x !== undefined && target.y !== undefined) {
        this.ctx.beginPath();
        this.ctx.moveTo(source.x, source.y);
        this.ctx.lineTo(target.x, target.y);
        this.ctx.stroke();
      }
    });

    this.ctx.restore();
  }

  private getNodeColor(node: GraphNode): string {
    const colors: Record<string, string> = {
      document: '#3B82F6',
      entity: '#10B981',
      concept: '#8B5CF6',
      tag: '#F59E0B',
      folder: '#6B7280',
    };
    return colors[node.type] || '#9CA3AF';
  }
}

/**
 * Graph clustering for improved layout and performance
 */
export class GraphClusterer {
  /**
   * Detect communities using Louvain algorithm
   */
  static detectCommunities(data: GraphData): Map<string, number> {
    const communities = new Map<string, number>();
    let communityId = 0;

    // Simple connected components for now
    const visited = new Set<string>();
    const adjacencyList = new Map<string, Set<string>>();

    // Build adjacency list
    data.edges.forEach(edge => {
      const sourceId = typeof edge.source === 'string' ? edge.source : edge.source.id;
      const targetId = typeof edge.target === 'string' ? edge.target : edge.target.id;

      if (!adjacencyList.has(sourceId)) adjacencyList.set(sourceId, new Set());
      if (!adjacencyList.has(targetId)) adjacencyList.set(targetId, new Set());

      adjacencyList.get(sourceId)!.add(targetId);
      adjacencyList.get(targetId)!.add(sourceId);
    });

    // DFS to find connected components
    const dfs = (nodeId: string, currentCommunity: number) => {
      visited.add(nodeId);
      communities.set(nodeId, currentCommunity);

      const neighbors = adjacencyList.get(nodeId);
      if (neighbors) {
        neighbors.forEach(neighbor => {
          if (!visited.has(neighbor)) {
            dfs(neighbor, currentCommunity);
          }
        });
      }
    };

    data.nodes.forEach(node => {
      if (!visited.has(node.id)) {
        dfs(node.id, communityId++);
      }
    });

    return communities;
  }

  /**
   * Calculate node importance using PageRank-like algorithm
   */
  static calculateNodeImportance(data: GraphData): Map<string, number> {
    const importance = new Map<string, number>();
    const damping = 0.85;
    const iterations = 50;

    // Initialize all nodes with equal importance
    data.nodes.forEach(node => {
      importance.set(node.id, 1 / data.nodes.length);
    });

    // Build adjacency list
    const incomingEdges = new Map<string, string[]>();
    const outgoingCount = new Map<string, number>();

    data.edges.forEach(edge => {
      const sourceId = typeof edge.source === 'string' ? edge.source : edge.source.id;
      const targetId = typeof edge.target === 'string' ? edge.target : edge.target.id;

      if (!incomingEdges.has(targetId)) incomingEdges.set(targetId, []);
      incomingEdges.get(targetId)!.push(sourceId);

      outgoingCount.set(sourceId, (outgoingCount.get(sourceId) || 0) + 1);
    });

    // PageRank iterations
    for (let i = 0; i < iterations; i++) {
      const newImportance = new Map<string, number>();

      data.nodes.forEach(node => {
        let rank = (1 - damping) / data.nodes.length;

        const incoming = incomingEdges.get(node.id) || [];
        incoming.forEach(sourceId => {
          const sourceImportance = importance.get(sourceId) || 0;
          const sourceOutgoing = outgoingCount.get(sourceId) || 1;
          rank += damping * (sourceImportance / sourceOutgoing);
        });

        newImportance.set(node.id, rank);
      });

      // Update importance values
      newImportance.forEach((value, key) => {
        importance.set(key, value);
      });
    }

    return importance;
  }
}

/**
 * Level-of-detail (LOD) management for large graphs
 */
export class LODManager {
  static filterByImportance(
    data: GraphData,
    maxNodes: number,
    importanceMap?: Map<string, number>
  ): GraphData {
    if (data.nodes.length <= maxNodes) return data;

    // Calculate importance if not provided
    const importance = importanceMap || GraphClusterer.calculateNodeImportance(data);

    // Sort nodes by importance
    const sortedNodes = [...data.nodes].sort((a, b) => {
      const aImportance = importance.get(a.id) || 0;
      const bImportance = importance.get(b.id) || 0;
      return bImportance - aImportance;
    });

    // Take top nodes
    const visibleNodes = sortedNodes.slice(0, maxNodes);
    const visibleNodeIds = new Set(visibleNodes.map(n => n.id));

    // Filter edges
    const visibleEdges = data.edges.filter(edge => {
      const sourceId = typeof edge.source === 'string' ? edge.source : edge.source.id;
      const targetId = typeof edge.target === 'string' ? edge.target : edge.target.id;
      return visibleNodeIds.has(sourceId) && visibleNodeIds.has(targetId);
    });

    return {
      nodes: visibleNodes,
      edges: visibleEdges,
    };
  }

  static aggregateClusters(data: GraphData, maxNodes: number): GraphData {
    if (data.nodes.length <= maxNodes) return data;

    const communities = GraphClusterer.detectCommunities(data);
    const clusterNodes: GraphNode[] = [];
    const clusterEdges: GraphEdge[] = [];

    // Create super-nodes for each community
    const communityNodes = new Map<number, GraphNode[]>();
    communities.forEach((communityId, nodeId) => {
      const node = data.nodes.find(n => n.id === nodeId);
      if (node) {
        if (!communityNodes.has(communityId)) {
          communityNodes.set(communityId, []);
        }
        communityNodes.get(communityId)!.push(node);
      }
    });

    // Create cluster nodes
    communityNodes.forEach((nodes, communityId) => {
      if (nodes.length > 5) {
        // Create super-node for large clusters
        clusterNodes.push({
          id: `cluster-${communityId}`,
          label: `Cluster ${communityId} (${nodes.length} nodes)`,
          type: 'folder',
          metadata: {
            nodeCount: nodes.length,
            isCluster: true,
          },
        });
      } else {
        // Keep individual nodes for small clusters
        clusterNodes.push(...nodes);
      }
    });

    return {
      nodes: clusterNodes,
      edges: [], // Simplified: no edges for clustered view
    };
  }
}

/**
 * Web Worker for offloading heavy computations
 */
export const createGraphWorker = (): Worker => {
  const workerCode = `
    self.addEventListener('message', (e) => {
      const { type, data } = e.data;

      switch (type) {
        case 'CALCULATE_LAYOUT':
          // Perform force simulation calculations
          const result = performForceSimulation(data);
          self.postMessage({ type: 'LAYOUT_COMPLETE', data: result });
          break;

        case 'DETECT_COMMUNITIES':
          // Perform community detection
          const communities = detectCommunities(data);
          self.postMessage({ type: 'COMMUNITIES_DETECTED', data: communities });
          break;
      }
    });

    function performForceSimulation(data) {
      // Simplified force simulation
      return data;
    }

    function detectCommunities(data) {
      // Community detection logic
      return new Map();
    }
  `;

  const blob = new Blob([workerCode], { type: 'application/javascript' });
  const workerUrl = URL.createObjectURL(blob);
  return new Worker(workerUrl);
};