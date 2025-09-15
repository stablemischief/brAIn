import React from 'react';
import {
  MagnifyingGlassPlusIcon,
  MagnifyingGlassMinusIcon,
  ArrowPathIcon,
  ArrowsPointingOutIcon,
  Cog6ToothIcon,
} from '@heroicons/react/24/outline';

interface GraphControlsProps {
  onZoomIn: () => void;
  onZoomOut: () => void;
  onZoomReset: () => void;
  onRefresh: () => void;
  onSettingsClick?: () => void;
  currentZoom: number;
}

const GraphControls: React.FC<GraphControlsProps> = ({
  onZoomIn,
  onZoomOut,
  onZoomReset,
  onRefresh,
  onSettingsClick,
  currentZoom,
}) => {
  return (
    <div className="flex flex-col bg-gray-800 rounded-lg shadow-lg p-2 space-y-2">
      <button
        onClick={onZoomIn}
        className="p-2 hover:bg-gray-700 rounded-md transition-colors group relative"
        title="Zoom In"
      >
        <MagnifyingGlassPlusIcon className="w-5 h-5 text-gray-300 group-hover:text-white" />
      </button>

      <button
        onClick={onZoomOut}
        className="p-2 hover:bg-gray-700 rounded-md transition-colors group relative"
        title="Zoom Out"
      >
        <MagnifyingGlassMinusIcon className="w-5 h-5 text-gray-300 group-hover:text-white" />
      </button>

      <button
        onClick={onZoomReset}
        className="p-2 hover:bg-gray-700 rounded-md transition-colors group relative"
        title="Reset Zoom"
      >
        <ArrowsPointingOutIcon className="w-5 h-5 text-gray-300 group-hover:text-white" />
      </button>

      <div className="border-t border-gray-700 pt-2 mt-2">
        <div className="text-xs text-gray-400 text-center mb-2">
          {(currentZoom * 100).toFixed(0)}%
        </div>
      </div>

      <button
        onClick={onRefresh}
        className="p-2 hover:bg-gray-700 rounded-md transition-colors group relative"
        title="Refresh Graph"
      >
        <ArrowPathIcon className="w-5 h-5 text-gray-300 group-hover:text-white" />
      </button>

      {onSettingsClick && (
        <button
          onClick={onSettingsClick}
          className="p-2 hover:bg-gray-700 rounded-md transition-colors group relative"
          title="Graph Settings"
        >
          <Cog6ToothIcon className="w-5 h-5 text-gray-300 group-hover:text-white" />
        </button>
      )}
    </div>
  );
};

export default GraphControls;