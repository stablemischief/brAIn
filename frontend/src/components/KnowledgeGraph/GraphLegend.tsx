import React from 'react';
import { DEFAULT_COLOR_SCHEME } from '../../types/graph';

interface LegendItem {
  label: string;
  color: string;
  type: 'node' | 'edge';
}

const GraphLegend: React.FC = () => {
  const nodeTypes: LegendItem[] = [
    { label: 'Document', color: DEFAULT_COLOR_SCHEME.document, type: 'node' },
    { label: 'Entity', color: DEFAULT_COLOR_SCHEME.entity, type: 'node' },
    { label: 'Concept', color: DEFAULT_COLOR_SCHEME.concept, type: 'node' },
    { label: 'Tag', color: DEFAULT_COLOR_SCHEME.tag, type: 'node' },
    { label: 'Folder', color: DEFAULT_COLOR_SCHEME.folder, type: 'node' },
  ];

  const edgeTypes: LegendItem[] = [
    { label: 'References', color: DEFAULT_COLOR_SCHEME.edge, type: 'edge' },
    { label: 'Selected', color: DEFAULT_COLOR_SCHEME.selectedNode, type: 'node' },
  ];

  return (
    <div className="bg-gray-800 rounded-lg shadow-lg p-4 space-y-3">
      <h3 className="text-sm font-semibold text-gray-300 mb-2">Node Types</h3>
      <div className="space-y-2">
        {nodeTypes.map((item) => (
          <div key={item.label} className="flex items-center space-x-2">
            <div
              className="w-4 h-4 rounded-full border-2 border-white"
              style={{ backgroundColor: item.color }}
            />
            <span className="text-xs text-gray-400">{item.label}</span>
          </div>
        ))}
      </div>

      <div className="border-t border-gray-700 pt-3 mt-3">
        <h3 className="text-sm font-semibold text-gray-300 mb-2">Interactions</h3>
        <div className="space-y-2">
          {edgeTypes.map((item) => (
            <div key={item.label} className="flex items-center space-x-2">
              {item.type === 'edge' ? (
                <div className="w-4 h-0.5" style={{ backgroundColor: item.color }} />
              ) : (
                <div
                  className="w-4 h-4 rounded-full border-2"
                  style={{
                    backgroundColor: item.color,
                    borderColor: item.color
                  }}
                />
              )}
              <span className="text-xs text-gray-400">{item.label}</span>
            </div>
          ))}
        </div>
      </div>

      <div className="border-t border-gray-700 pt-3 mt-3">
        <h3 className="text-sm font-semibold text-gray-300 mb-2">Controls</h3>
        <div className="space-y-1 text-xs text-gray-400">
          <div>• Click: Select node</div>
          <div>• Drag: Move node</div>
          <div>• Scroll: Zoom</div>
          <div>• Shift+Drag: Pan</div>
        </div>
      </div>
    </div>
  );
};

export default GraphLegend;