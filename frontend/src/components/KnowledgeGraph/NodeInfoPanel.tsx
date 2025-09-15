import React from 'react';
import { XMarkIcon } from '@heroicons/react/24/outline';
import { GraphNode } from '../../types/graph';

interface NodeInfoPanelProps {
  node: GraphNode;
  onClose: () => void;
}

const NodeInfoPanel: React.FC<NodeInfoPanelProps> = ({ node, onClose }) => {
  return (
    <div className="bg-gray-800 rounded-lg shadow-lg p-4">
      <div className="flex items-start justify-between mb-3">
        <div>
          <h3 className="text-lg font-semibold text-white">{node.label}</h3>
          <span className="text-xs text-gray-400 capitalize">{node.type}</span>
        </div>
        <button
          onClick={onClose}
          className="p-1 hover:bg-gray-700 rounded transition-colors"
        >
          <XMarkIcon className="w-5 h-5 text-gray-400" />
        </button>
      </div>

      {node.metadata?.description && (
        <div className="mb-3">
          <h4 className="text-sm font-medium text-gray-300 mb-1">Description</h4>
          <p className="text-sm text-gray-400">{node.metadata.description}</p>
        </div>
      )}

      <div className="space-y-3">
        <div>
          <h4 className="text-sm font-medium text-gray-300 mb-1">Details</h4>
          <dl className="space-y-1">
            <div className="flex justify-between text-sm">
              <dt className="text-gray-400">ID:</dt>
              <dd className="text-gray-300 font-mono text-xs">{node.id.substring(0, 8)}...</dd>
            </div>
            {node.group && (
              <div className="flex justify-between text-sm">
                <dt className="text-gray-400">Group:</dt>
                <dd className="text-gray-300">{node.group}</dd>
              </div>
            )}
            {node.metadata?.confidence && (
              <div className="flex justify-between text-sm">
                <dt className="text-gray-400">Confidence:</dt>
                <dd className="text-gray-300">
                  {(node.metadata.confidence * 100).toFixed(1)}%
                </dd>
              </div>
            )}
            {node.metadata?.source && (
              <div className="flex justify-between text-sm">
                <dt className="text-gray-400">Source:</dt>
                <dd className="text-gray-300">{node.metadata.source}</dd>
              </div>
            )}
            {node.metadata?.extractedAt && (
              <div className="flex justify-between text-sm">
                <dt className="text-gray-400">Extracted:</dt>
                <dd className="text-gray-300">
                  {new Date(node.metadata.extractedAt).toLocaleDateString()}
                </dd>
              </div>
            )}
          </dl>
        </div>

        {node.metadata && Object.keys(node.metadata).length > 0 && (
          <div>
            <h4 className="text-sm font-medium text-gray-300 mb-1">Metadata</h4>
            <div className="bg-gray-900 rounded p-2 max-h-32 overflow-y-auto">
              <pre className="text-xs text-gray-400">
                {JSON.stringify(
                  Object.fromEntries(
                    Object.entries(node.metadata).filter(
                      ([key]) => !['description', 'confidence', 'source', 'extractedAt'].includes(key)
                    )
                  ),
                  null,
                  2
                )}
              </pre>
            </div>
          </div>
        )}

        <div className="pt-3 border-t border-gray-700">
          <button className="w-full px-3 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 transition-colors text-sm">
            View Related Documents
          </button>
        </div>
      </div>
    </div>
  );
};

export default NodeInfoPanel;