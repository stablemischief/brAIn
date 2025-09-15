import React from 'react';
import { KnowledgeGraph } from '../components/KnowledgeGraph';

const KnowledgeGraphPage: React.FC = () => {
  const handleNodeSelect = (node: any) => {
    console.log('Node selected:', node);
  };

  const handleEdgeSelect = (edge: any) => {
    console.log('Edge selected:', edge);
  };

  return (
    <div className="h-screen bg-gray-900">
      <div className="h-full flex flex-col">
        {/* Header */}
        <div className="bg-gray-800 border-b border-gray-700 px-6 py-4">
          <h1 className="text-2xl font-bold text-white">Knowledge Graph Explorer</h1>
          <p className="text-sm text-gray-400 mt-1">
            Interactive visualization of document relationships and extracted entities
          </p>
        </div>

        {/* Graph Container */}
        <div className="flex-1 relative">
          <KnowledgeGraph
            className="w-full h-full"
            onNodeSelect={handleNodeSelect}
            onEdgeSelect={handleEdgeSelect}
            config={{
              showLabels: true,
              showLegend: true,
              nodeRadius: 8,
              chargeStrength: -400,
              linkDistance: 120,
            }}
          />
        </div>

        {/* Status Bar */}
        <div className="bg-gray-800 border-t border-gray-700 px-6 py-2">
          <div className="flex items-center justify-between text-sm text-gray-400">
            <div>
              Press <kbd className="px-2 py-1 bg-gray-700 rounded">Space</kbd> to pause simulation
            </div>
            <div>
              Use mouse wheel to zoom • Click and drag to pan • Click nodes to select
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default KnowledgeGraphPage;