import React, { useState } from 'react';
import { motion } from 'framer-motion';
import {
  ChartBarIcon,
  CurrencyDollarIcon,
  DocumentArrowDownIcon,
  Cog6ToothIcon,
  InformationCircleIcon,
  ClockIcon,
  ExclamationTriangleIcon,
} from '@heroicons/react/24/outline';
import { CostAnalytics, BudgetInfo } from '../../types';
import CostChart from './CostChart';
import BudgetManager from './BudgetManager';
import CostProjections from './CostProjections';
import OptimizationCards from './OptimizationCards';
import ExportControls from './ExportControls';
import { useCostData, useCostInsights } from '../../hooks/useCostData';

interface CostDashboardProps {
  className?: string;
}

export const CostDashboard: React.FC<CostDashboardProps> = ({ 
  className = '' 
}) => {
  const [activeTab, setActiveTab] = useState<'overview' | 'charts' | 'budget' | 'projections' | 'recommendations'>('overview');
  const [timeframe, setTimeframe] = useState<'7d' | '30d' | '90d'>('30d');

  // Use real data hooks
  const {
    analytics,
    historicalData,
    budgetInfo,
    isLoading,
    error,
    updateFilters,
    updateBudget,
    refetch,
  } = useCostData({
    filters: { timeframe },
    autoRefresh: true,
  });

  const { insights } = useCostInsights();

  // Handle timeframe changes
  const handleTimeframeChange = (newTimeframe: '7d' | '30d' | '90d') => {
    setTimeframe(newTimeframe);
    updateFilters({ timeframe: newTimeframe });
  };

  const tabs = [
    { id: 'overview', label: 'Overview', icon: InformationCircleIcon },
    { id: 'charts', label: 'Charts', icon: ChartBarIcon },
    { id: 'budget', label: 'Budget', icon: CurrencyDollarIcon },
    { id: 'projections', label: 'Projections', icon: ClockIcon },
    { id: 'recommendations', label: 'Optimize', icon: Cog6ToothIcon },
  ] as const;

  const timeframes = [
    { value: '7d', label: '7 Days' },
    { value: '30d', label: '30 Days' },
    { value: '90d', label: '90 Days' },
  ] as const;

  // Show loading state
  if (isLoading) {
    return (
      <div className={`max-w-7xl mx-auto p-6 space-y-6 ${className}`}>
        <div className="flex items-center justify-center h-96">
          <div className="text-center">
            <div className="animate-spin rounded-full h-12 w-12 border-4 border-blue-600 border-t-transparent mx-auto mb-4" />
            <p className="text-gray-600 dark:text-gray-400">Loading cost analytics...</p>
          </div>
        </div>
      </div>
    );
  }

  // Show error state
  if (error && !analytics && !historicalData.length) {
    return (
      <div className={`max-w-7xl mx-auto p-6 space-y-6 ${className}`}>
        <div className="flex items-center justify-center h-96">
          <div className="text-center">
            <ExclamationTriangleIcon className="h-12 w-12 text-red-500 mx-auto mb-4" />
            <h3 className="text-lg font-medium text-gray-900 dark:text-white mb-2">
              Failed to load cost data
            </h3>
            <p className="text-gray-600 dark:text-gray-400 mb-4">{error}</p>
            <button
              onClick={() => refetch()}
              className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
            >
              Try Again
            </button>
          </div>
        </div>
      </div>
    );
  }

  // Use fallback data if no analytics available
  const costData = analytics || {
    total_cost: 0,
    daily_costs: [],
    cost_by_model: {},
    cost_by_operation: {},
    token_usage: { total_tokens: 0, input_tokens: 0, output_tokens: 0 },
    projected_monthly_cost: 0,
    timeframe: timeframe === '7d' ? '7 days' : timeframe === '90d' ? '90 days' : '30 days',
  } as CostAnalytics;

  const budget = budgetInfo || {
    monthly_limit: 0,
    current_month_spend: 0,
    remaining_budget: 0,
    budget_utilization_percentage: 0,
    projected_month_end_spend: 0,
    alerts: { approaching_limit: false, over_budget: false, high_daily_spend: false },
  } as BudgetInfo;

  return (
    <div className={`max-w-7xl mx-auto p-6 space-y-6 ${className}`}>
      {/* Header with Title and Controls */}
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
        <div>
          <h1 className="text-3xl font-bold text-gray-900 dark:text-white">
            Cost Analytics Dashboard
          </h1>
          <p className="text-gray-600 dark:text-gray-400 mt-1">
            Monitor AI usage costs, budget, and optimization opportunities
          </p>
        </div>

        <div className="flex items-center gap-3">
          {/* Timeframe Selector */}
          <div className="flex bg-gray-100 dark:bg-gray-700 rounded-lg p-1">
            {timeframes.map((tf) => (
              <button
                key={tf.value}
                onClick={() => handleTimeframeChange(tf.value)}
                className={`px-3 py-1 rounded-md text-sm font-medium transition-colors ${
                  timeframe === tf.value
                    ? 'bg-white dark:bg-gray-600 text-gray-900 dark:text-white shadow-sm'
                    : 'text-gray-600 dark:text-gray-300 hover:text-gray-900 dark:hover:text-white'
                }`}
              >
                {tf.label}
              </button>
            ))}
          </div>

          {/* Export Controls */}
          <ExportControls
            costData={costData}
            historicalData={historicalData}
            onExport={async (options) => {
              console.log('Export requested:', options);
              // Mock API call - would generate actual export
              return new Promise(resolve => {
                setTimeout(() => {
                  resolve({
                    url: 'blob:mock-export-url',
                    filename: `cost-analytics-${new Date().toISOString().split('T')[0]}.${options.format === 'excel' ? 'xlsx' : options.format}`
                  });
                }, 2000);
              });
            }}
          />
        </div>
      </div>

      {/* Quick Stats Overview */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0 }}
          className="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 p-6"
        >
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-gray-600 dark:text-gray-400">Total Spend</p>
              <p className="text-2xl font-bold text-gray-900 dark:text-white">
                ${costData.total_cost.toFixed(2)}
              </p>
              <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">
                {costData.timeframe}
              </p>
            </div>
            <div className="p-3 bg-blue-100 dark:bg-blue-900/30 rounded-lg">
              <CurrencyDollarIcon className="h-6 w-6 text-blue-600 dark:text-blue-400" />
            </div>
          </div>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.1 }}
          className="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 p-6"
        >
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-gray-600 dark:text-gray-400">Monthly Projection</p>
              <p className="text-2xl font-bold text-gray-900 dark:text-white">
                ${costData.projected_monthly_cost.toFixed(2)}
              </p>
              <p className="text-xs text-green-600 dark:text-green-400 mt-1">
                {budget.monthly_limit > 0 ? ((budget.projected_month_end_spend / budget.monthly_limit) * 100).toFixed(1) : 0}% of budget
              </p>
            </div>
            <div className="p-3 bg-green-100 dark:bg-green-900/30 rounded-lg">
              <ChartBarIcon className="h-6 w-6 text-green-600 dark:text-green-400" />
            </div>
          </div>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.2 }}
          className="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 p-6"
        >
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-gray-600 dark:text-gray-400">Total Tokens</p>
              <p className="text-2xl font-bold text-gray-900 dark:text-white">
                {(costData.token_usage.total_tokens / 1000).toFixed(0)}K
              </p>
              <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">
                {costData.token_usage.total_tokens > 0 ? ((costData.token_usage.output_tokens / costData.token_usage.total_tokens) * 100).toFixed(0) : 0}% output
              </p>
            </div>
            <div className="p-3 bg-purple-100 dark:bg-purple-900/30 rounded-lg">
              <InformationCircleIcon className="h-6 w-6 text-purple-600 dark:text-purple-400" />
            </div>
          </div>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.3 }}
          className="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 p-6"
        >
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-gray-600 dark:text-gray-400">Daily Average</p>
              <p className="text-2xl font-bold text-gray-900 dark:text-white">
                ${historicalData.length > 0 ? (costData.total_cost / historicalData.length).toFixed(2) : '0.00'}
              </p>
              <p className="text-xs text-blue-600 dark:text-blue-400 mt-1">
                per day
              </p>
            </div>
            <div className="p-3 bg-orange-100 dark:bg-orange-900/30 rounded-lg">
              <ClockIcon className="h-6 w-6 text-orange-600 dark:text-orange-400" />
            </div>
          </div>
        </motion.div>
      </div>

      {/* Tab Navigation */}
      <div className="border-b border-gray-200 dark:border-gray-700">
        <nav className="-mb-px flex space-x-8">
          {tabs.map((tab) => {
            const Icon = tab.icon;
            return (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id as typeof activeTab)}
                className={`group inline-flex items-center px-1 py-4 border-b-2 font-medium text-sm ${
                  activeTab === tab.id
                    ? 'border-blue-500 text-blue-600 dark:text-blue-400'
                    : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300 dark:text-gray-400 dark:hover:text-gray-300'
                }`}
              >
                <Icon
                  className={`mr-2 h-5 w-5 ${
                    activeTab === tab.id
                      ? 'text-blue-500 dark:text-blue-400'
                      : 'text-gray-400 group-hover:text-gray-500 dark:group-hover:text-gray-300'
                  }`}
                />
                {tab.label}
              </button>
            );
          })}
        </nav>
      </div>

      {/* Tab Content */}
      <motion.div
        key={activeTab}
        initial={{ opacity: 0, x: 20 }}
        animate={{ opacity: 1, x: 0 }}
        transition={{ duration: 0.3 }}
        className="mt-6"
      >
        {activeTab === 'overview' && (
          <div className="space-y-6">
            {/* Cost Breakdown Cards */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <div className="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 p-6">
                <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
                  Cost by Model
                </h3>
                <div className="space-y-3">
                  {Object.entries(costData.cost_by_model).map(([model, cost]) => (
                    <div key={model} className="flex items-center justify-between">
                      <div className="flex items-center">
                        <div className="w-3 h-3 rounded-full bg-blue-500 mr-3"></div>
                        <span className="text-sm font-medium text-gray-900 dark:text-white">
                          {model}
                        </span>
                      </div>
                      <div className="text-right">
                        <p className="text-sm font-semibold text-gray-900 dark:text-white">
                          ${cost.toFixed(2)}
                        </p>
                        <p className="text-xs text-gray-500 dark:text-gray-400">
                          {costData.total_cost > 0 ? ((cost / costData.total_cost) * 100).toFixed(1) : 0}%
                        </p>
                      </div>
                    </div>
                  ))}
                </div>
              </div>

              <div className="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 p-6">
                <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
                  Cost by Operation
                </h3>
                <div className="space-y-3">
                  {Object.entries(costData.cost_by_operation).map(([operation, cost]) => (
                    <div key={operation} className="flex items-center justify-between">
                      <div className="flex items-center">
                        <div className="w-3 h-3 rounded-full bg-green-500 mr-3"></div>
                        <span className="text-sm font-medium text-gray-900 dark:text-white capitalize">
                          {operation}
                        </span>
                      </div>
                      <div className="text-right">
                        <p className="text-sm font-semibold text-gray-900 dark:text-white">
                          ${cost.toFixed(2)}
                        </p>
                        <p className="text-xs text-gray-500 dark:text-gray-400">
                          {costData.total_cost > 0 ? ((cost / costData.total_cost) * 100).toFixed(1) : 0}%
                        </p>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>

            {/* Recent Activity */}
            <div className="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 p-6">
              <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
                Recent Daily Costs
              </h3>
              <div className="overflow-x-auto">
                <table className="min-w-full">
                  <thead>
                    <tr className="border-b border-gray-200 dark:border-gray-700">
                      <th className="text-left py-3 px-4 text-sm font-medium text-gray-900 dark:text-white">
                        Date
                      </th>
                      <th className="text-right py-3 px-4 text-sm font-medium text-gray-900 dark:text-white">
                        Cost
                      </th>
                      <th className="text-right py-3 px-4 text-sm font-medium text-gray-900 dark:text-white">
                        Tokens
                      </th>
                      <th className="text-right py-3 px-4 text-sm font-medium text-gray-900 dark:text-white">
                        Requests
                      </th>
                    </tr>
                  </thead>
                  <tbody>
                    {historicalData.slice(-5).map((day, index) => (
                      <tr key={day.date} className="border-b border-gray-100 dark:border-gray-700">
                        <td className="py-3 px-4 text-sm text-gray-900 dark:text-white">
                          {new Date(day.date).toLocaleDateString()}
                        </td>
                        <td className="py-3 px-4 text-sm font-medium text-gray-900 dark:text-white text-right">
                          ${day.cost.toFixed(2)}
                        </td>
                        <td className="py-3 px-4 text-sm text-gray-600 dark:text-gray-400 text-right">
                          {day.tokens.toLocaleString()}
                        </td>
                        <td className="py-3 px-4 text-sm text-gray-600 dark:text-gray-400 text-right">
                          {day.requests}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        )}

        {activeTab === 'charts' && (
          <CostChart 
            data={historicalData}
            timeframe={timeframe}
            className="w-full"
          />
        )}

        {activeTab === 'budget' && (
          <BudgetManager
            budgetInfo={budget}
            onUpdateBudget={updateBudget}
            className="w-full"
          />
        )}

        {activeTab === 'projections' && (
          <CostProjections
            historicalData={historicalData}
            monthlyBudget={budget.monthly_limit}
            className="w-full"
          />
        )}

        {activeTab === 'recommendations' && (
          <OptimizationCards
            costData={costData}
            historicalData={historicalData}
            onImplementRecommendation={async (recommendationId) => {
              console.log('Implementing recommendation:', recommendationId);
              // API call to implement recommendation would go here
              return new Promise(resolve => setTimeout(resolve, 2000)); // Simulate API delay
            }}
            className="w-full"
          />
        )}
      </motion.div>
    </div>
  );
};

export default CostDashboard;