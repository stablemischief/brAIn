import React, { useState, useCallback, useEffect } from 'react';
import { MagnifyingGlassIcon, XMarkIcon } from '@heroicons/react/24/outline';

interface GraphSearchProps {
  onSearch: (query: string) => void;
  placeholder?: string;
  debounceMs?: number;
}

const GraphSearch: React.FC<GraphSearchProps> = ({
  onSearch,
  placeholder = 'Search nodes and relationships...',
  debounceMs = 300,
}) => {
  const [query, setQuery] = useState('');
  const [isExpanded, setIsExpanded] = useState(false);

  // Debounced search
  useEffect(() => {
    const timer = setTimeout(() => {
      onSearch(query);
    }, debounceMs);

    return () => clearTimeout(timer);
  }, [query, onSearch, debounceMs]);

  const handleClear = useCallback(() => {
    setQuery('');
    onSearch('');
  }, [onSearch]);

  const handleToggle = useCallback(() => {
    setIsExpanded(!isExpanded);
    if (!isExpanded) {
      // Focus input when expanding
      setTimeout(() => {
        const input = document.getElementById('graph-search-input');
        input?.focus();
      }, 100);
    } else {
      // Clear when collapsing
      handleClear();
    }
  }, [isExpanded, handleClear]);

  return (
    <div className="relative">
      <div
        className={`flex items-center bg-gray-800 rounded-lg shadow-lg transition-all duration-300 ${
          isExpanded ? 'w-64' : 'w-auto'
        }`}
      >
        <button
          onClick={handleToggle}
          className="p-3 hover:bg-gray-700 rounded-l-lg transition-colors"
          title={isExpanded ? 'Close search' : 'Open search'}
        >
          <MagnifyingGlassIcon className="w-5 h-5 text-gray-300" />
        </button>

        {isExpanded && (
          <>
            <input
              id="graph-search-input"
              type="text"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder={placeholder}
              className="flex-1 px-3 py-2 bg-transparent text-white placeholder-gray-400 focus:outline-none"
            />

            {query && (
              <button
                onClick={handleClear}
                className="p-2 hover:bg-gray-700 rounded-r-lg transition-colors"
                title="Clear search"
              >
                <XMarkIcon className="w-4 h-4 text-gray-400" />
              </button>
            )}
          </>
        )}
      </div>

      {isExpanded && query && (
        <div className="absolute top-full mt-2 w-64 bg-gray-800 rounded-lg shadow-lg p-2 z-10">
          <div className="text-xs text-gray-400 mb-2">Search filters:</div>
          <div className="space-y-1">
            <label className="flex items-center text-sm text-gray-300 hover:bg-gray-700 p-1 rounded cursor-pointer">
              <input type="checkbox" className="mr-2" defaultChecked />
              Nodes
            </label>
            <label className="flex items-center text-sm text-gray-300 hover:bg-gray-700 p-1 rounded cursor-pointer">
              <input type="checkbox" className="mr-2" defaultChecked />
              Relationships
            </label>
            <label className="flex items-center text-sm text-gray-300 hover:bg-gray-700 p-1 rounded cursor-pointer">
              <input type="checkbox" className="mr-2" />
              Metadata
            </label>
          </div>
        </div>
      )}
    </div>
  );
};

export default GraphSearch;