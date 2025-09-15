import React from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import { Dashboard } from './components/Dashboard';
import { ErrorBoundary } from './components/ErrorBoundary';
import { LoadingSpinner } from './components/LoadingSpinner';
import { RealtimeProvider } from './context/RealtimeContext';

// Lazy load other pages when they're created
const SearchPage = React.lazy(() => import('./pages/SearchPage').catch(() => ({ default: () => <div>Search page coming soon</div> })));
const AnalyticsPage = React.lazy(() => import('./pages/AnalyticsPage').catch(() => ({ default: () => <div>Analytics page coming soon</div> })));
const ConfigurationPage = React.lazy(() => import('./pages/ConfigurationPage').catch(() => ({ default: () => <div>Configuration page coming soon</div> })));
const KnowledgeGraphPage = React.lazy(() => import('./pages/KnowledgeGraphPage'));

function App() {
  return (
    <ErrorBoundary>
      <RealtimeProvider>
        <Router>
          <div className="App">
            <Routes>
              {/* Main Dashboard Route */}
              <Route 
                path="/" 
                element={<Dashboard />} 
              />
              
              {/* Future Routes */}
              <Route 
                path="/search" 
                element={
                  <React.Suspense fallback={<LoadingSpinner />}>
                    <SearchPage />
                  </React.Suspense>
                } 
              />
              
              <Route 
                path="/analytics" 
                element={
                  <React.Suspense fallback={<LoadingSpinner />}>
                    <AnalyticsPage />
                  </React.Suspense>
                } 
              />
              
              <Route
                path="/config"
                element={
                  <React.Suspense fallback={<LoadingSpinner />}>
                    <ConfigurationPage />
                  </React.Suspense>
                }
              />

              <Route
                path="/knowledge-graph"
                element={
                  <React.Suspense fallback={<LoadingSpinner />}>
                    <KnowledgeGraphPage />
                  </React.Suspense>
                }
              />

              {/* Catch all - redirect to dashboard */}
              <Route 
                path="*" 
                element={<Navigate to="/" replace />} 
              />
            </Routes>
          </div>
        </Router>
      </RealtimeProvider>
    </ErrorBoundary>
  );
}

export default App;