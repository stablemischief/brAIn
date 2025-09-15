import React from 'react';
import { CostDashboard } from '../components/Cost/CostDashboard';

const AnalyticsPage: React.FC = () => {
  return (
    <div className="min-h-screen bg-gray-50 dark:bg-gray-900">
      <div className="container mx-auto px-4 py-8">
        <h1 className="text-3xl font-bold text-gray-900 dark:text-white mb-8">
          Cost Analytics Dashboard
        </h1>
        <CostDashboard />
      </div>
    </div>
  );
};

export default AnalyticsPage;