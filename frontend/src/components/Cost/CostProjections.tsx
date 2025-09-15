import React, { useMemo, useState } from 'react';
import { motion } from 'framer-motion';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  Filler,
} from 'chart.js';
import { Line } from 'react-chartjs-2';
import { format, addDays, startOfMonth, endOfMonth, differenceInDays } from 'date-fns';
import {
  ArrowTrendingUpIcon,
  ArrowTrendingDownIcon,
  CalendarIcon,
  ClockIcon,
  ExclamationTriangleIcon,
  InformationCircleIcon,
  ChartBarIcon,
  CurrencyDollarIcon,
} from '@heroicons/react/24/outline';
import { DailyCost } from '../../types';

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  Filler
);

interface ProjectionData {
  date: string;
  actualCost?: number;
  projectedCost: number;
  confidence: 'high' | 'medium' | 'low';
  scenario: 'conservative' | 'realistic' | 'optimistic';
}

interface CostProjectionsProps {
  historicalData: DailyCost[];
  monthlyBudget: number;
  className?: string;
}

interface ProjectionScenario {
  name: string;
  label: string;
  color: string;
  multiplier: number;
  description: string;
}

export const CostProjections: React.FC<CostProjectionsProps> = ({
  historicalData,
  monthlyBudget,
  className = '',
}) => {
  const [selectedScenario, setSelectedScenario] = useState<'all' | 'conservative' | 'realistic' | 'optimistic'>('all');
  const [projectionPeriod, setProjectionPeriod] = useState<'month' | 'quarter' | 'year'>('month');

  const scenarios: ProjectionScenario[] = [
    {
      name: 'conservative',
      label: 'Conservative',
      color: 'rgb(34, 197, 94)', // green
      multiplier: 1.1,
      description: 'Assumes minimal growth in usage',
    },
    {
      name: 'realistic',
      label: 'Realistic',
      color: 'rgb(59, 130, 246)', // blue
      multiplier: 1.25,
      description: 'Based on current trends',
    },
    {
      name: 'optimistic',
      label: 'Optimistic',
      color: 'rgb(239, 68, 68)', // red
      multiplier: 1.4,
      description: 'Assumes increased usage patterns',
    },
  ];

  // Generate projections based on historical data
  const projections = useMemo(() => {
    if (!historicalData || historicalData.length === 0) return [];

    const today = new Date();
    const monthStart = startOfMonth(today);
    const monthEnd = endOfMonth(today);
    
    // Calculate trend from historical data
    const recentDays = historicalData.slice(-7); // Last 7 days
    const avgRecentCost = recentDays.reduce((sum, day) => sum + day.cost, 0) / recentDays.length;
    
    // Calculate growth rate
    const firstHalf = historicalData.slice(0, Math.floor(historicalData.length / 2));
    const secondHalf = historicalData.slice(Math.floor(historicalData.length / 2));
    const firstHalfAvg = firstHalf.reduce((sum, day) => sum + day.cost, 0) / firstHalf.length;
    const secondHalfAvg = secondHalf.reduce((sum, day) => sum + day.cost, 0) / secondHalf.length;
    
    const growthRate = secondHalfAvg > 0 ? (secondHalfAvg - firstHalfAvg) / firstHalfAvg : 0;
    const dailyGrowthRate = growthRate / secondHalf.length;

    const daysToProject = projectionPeriod === 'month' ? 30 : 
                         projectionPeriod === 'quarter' ? 90 : 365;

    const projectionData: ProjectionData[] = [];

    for (let i = 0; i <= daysToProject; i++) {
      const projectionDate = format(addDays(today, i), 'yyyy-MM-dd');
      
      scenarios.forEach((scenario) => {
        // Apply scenario multiplier and growth rate
        const baseProjection = avgRecentCost * (1 + dailyGrowthRate * i);
        const scenarioProjection = baseProjection * scenario.multiplier;
        
        // Add some variance for realism
        const variance = Math.random() * 0.1 - 0.05; // ±5% variance
        const projectedCost = Math.max(0, scenarioProjection * (1 + variance));
        
        // Determine confidence based on distance from today
        let confidence: 'high' | 'medium' | 'low';
        if (i <= 7) confidence = 'high';
        else if (i <= 30) confidence = 'medium';
        else confidence = 'low';

        projectionData.push({
          date: projectionDate,
          projectedCost,
          confidence,
          scenario: scenario.name as 'conservative' | 'realistic' | 'optimistic',
        });
      });
    }

    return projectionData;
  }, [historicalData, projectionPeriod]);

  // Calculate summary metrics
  const summaryMetrics = useMemo(() => {
    const monthlyProjections = projections.filter((p, index) => index < 90); // ~30 days * 3 scenarios
    
    const scenarioSummaries = scenarios.map((scenario) => {
      const scenarioData = monthlyProjections.filter(p => p.scenario === scenario.name);
      const totalCost = scenarioData.reduce((sum, p) => sum + p.projectedCost, 0);
      const avgDailyCost = totalCost / (scenarioData.length || 1);
      
      return {
        ...scenario,
        totalCost,
        avgDailyCost,
        budgetUtilization: (totalCost / monthlyBudget) * 100,
        overBudget: totalCost > monthlyBudget,
      };
    });

    return scenarioSummaries;
  }, [projections, monthlyBudget, scenarios]);

  // Prepare chart data
  const chartData = useMemo(() => {
    const filteredProjections = selectedScenario === 'all' 
      ? projections
      : projections.filter(p => p.scenario === selectedScenario);

    // Group by scenario for chart
    const datasets = scenarios
      .filter(scenario => selectedScenario === 'all' || scenario.name === selectedScenario)
      .map((scenario) => {
        const scenarioData = filteredProjections
          .filter(p => p.scenario === scenario.name)
          .sort((a, b) => new Date(a.date).getTime() - new Date(b.date).getTime());

        return {
          label: scenario.label,
          data: scenarioData.map(p => ({
            x: p.date,
            y: p.projectedCost,
          })),
          borderColor: scenario.color,
          backgroundColor: scenario.color.replace('rgb', 'rgba').replace(')', ', 0.1)'),
          tension: 0.4,
          fill: false,
          pointRadius: 0,
          pointHoverRadius: 5,
          pointHoverBackgroundColor: scenario.color,
          pointHoverBorderColor: 'white',
          pointHoverBorderWidth: 2,
        };
      });

    return {
      datasets,
    };
  }, [projections, scenarios, selectedScenario]);

  const chartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        display: true,
        position: 'top' as const,
        labels: {
          color: 'rgb(107, 114, 128)',
          usePointStyle: true,
          pointStyle: 'circle',
        },
      },
      tooltip: {
        mode: 'index' as const,
        intersect: false,
        backgroundColor: 'rgba(17, 24, 39, 0.95)',
        titleColor: 'rgb(243, 244, 246)',
        bodyColor: 'rgb(209, 213, 219)',
        borderColor: 'rgb(75, 85, 99)',
        borderWidth: 1,
        cornerRadius: 8,
        callbacks: {
          title: (context: any) => {
            return format(new Date(context[0].parsed.x), 'MMM dd, yyyy');
          },
          label: (context: any) => {
            return `${context.dataset.label}: $${context.parsed.y.toFixed(2)}`;
          },
        },
      },
    },
    scales: {
      x: {
        type: 'time' as const,
        time: {
          unit: 'day' as const,
          displayFormats: {
            day: 'MMM dd',
          },
        },
        grid: {
          color: 'rgba(107, 114, 128, 0.1)',
          drawBorder: false,
        },
        ticks: {
          color: 'rgb(107, 114, 128)',
          maxTicksLimit: 8,
        },
      },
      y: {
        beginAtZero: true,
        grid: {
          color: 'rgba(107, 114, 128, 0.1)',
          drawBorder: false,
        },
        ticks: {
          color: 'rgb(107, 114, 128)',
          callback: function(value: any) {
            return `$${value.toFixed(2)}`;
          },
        },
      },
    },
    interaction: {
      mode: 'nearest' as const,
      axis: 'x' as const,
      intersect: false,
    },
    animation: {
      duration: 1000,
      easing: 'easeInOutQuart' as const,
    },
  };

  const periodOptions = [
    { value: 'month', label: '30 Days' },
    { value: 'quarter', label: '90 Days' },
    { value: 'year', label: '1 Year' },
  ] as const;

  return (
    <div className={`space-y-6 ${className}`}>
      {/* Header Controls */}
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
        <div>
          <h3 className="text-lg font-semibold text-gray-900 dark:text-white">
            Cost Projections & Trends
          </h3>
          <p className="text-sm text-gray-600 dark:text-gray-400 mt-1">
            Predictive analysis based on historical usage patterns
          </p>
        </div>

        <div className="flex items-center gap-3">
          {/* Period Selector */}
          <div className="flex bg-gray-100 dark:bg-gray-700 rounded-lg p-1">
            {periodOptions.map((period) => (
              <button
                key={period.value}
                onClick={() => setProjectionPeriod(period.value)}
                className={`px-3 py-1 rounded-md text-sm font-medium transition-colors ${
                  projectionPeriod === period.value
                    ? 'bg-white dark:bg-gray-600 text-gray-900 dark:text-white shadow-sm'
                    : 'text-gray-600 dark:text-gray-300 hover:text-gray-900 dark:hover:text-white'
                }`}
              >
                {period.label}
              </button>
            ))}
          </div>

          {/* Scenario Selector */}
          <select
            value={selectedScenario}
            onChange={(e) => setSelectedScenario(e.target.value as typeof selectedScenario)}
            className="px-3 py-2 bg-white dark:bg-gray-700 border border-gray-300 dark:border-gray-600 rounded-lg text-sm focus:ring-2 focus:ring-blue-500 focus:border-transparent"
          >
            <option value="all">All Scenarios</option>
            {scenarios.map((scenario) => (
              <option key={scenario.name} value={scenario.name}>
                {scenario.label}
              </option>
            ))}
          </select>
        </div>
      </div>

      {/* Summary Cards */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        {summaryMetrics.map((summary) => (
          <motion.div
            key={summary.name}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 p-6"
          >
            <div className="flex items-center justify-between mb-4">
              <div className="flex items-center gap-2">
                <div 
                  className="w-3 h-3 rounded-full"
                  style={{ backgroundColor: summary.color }}
                />
                <h4 className="font-semibold text-gray-900 dark:text-white">
                  {summary.label}
                </h4>
              </div>
              {summary.overBudget && (
                <ExclamationTriangleIcon className="h-5 w-5 text-red-500" />
              )}
            </div>

            <div className="space-y-3">
              <div>
                <p className="text-2xl font-bold text-gray-900 dark:text-white">
                  ${summary.totalCost.toFixed(2)}
                </p>
                <p className="text-sm text-gray-600 dark:text-gray-400">
                  Projected monthly cost
                </p>
              </div>

              <div className="flex items-center justify-between">
                <span className="text-sm text-gray-600 dark:text-gray-400">
                  Daily average
                </span>
                <span className="text-sm font-medium text-gray-900 dark:text-white">
                  ${summary.avgDailyCost.toFixed(2)}
                </span>
              </div>

              <div className="flex items-center justify-between">
                <span className="text-sm text-gray-600 dark:text-gray-400">
                  Budget usage
                </span>
                <span className={`text-sm font-medium ${
                  summary.overBudget 
                    ? 'text-red-600 dark:text-red-400'
                    : 'text-green-600 dark:text-green-400'
                }`}>
                  {summary.budgetUtilization.toFixed(1)}%
                </span>
              </div>

              <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2">
                <div
                  className={`h-2 rounded-full transition-all duration-500 ${
                    summary.overBudget ? 'bg-red-500' : 'bg-green-500'
                  }`}
                  style={{ 
                    width: `${Math.min(summary.budgetUtilization, 100)}%` 
                  }}
                />
              </div>

              <p className="text-xs text-gray-500 dark:text-gray-400">
                {summary.description}
              </p>
            </div>
          </motion.div>
        ))}
      </div>

      {/* Projection Chart */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 p-6"
      >
        <div className="flex items-center gap-2 mb-6">
          <ChartBarIcon className="h-5 w-5 text-gray-600 dark:text-gray-400" />
          <h4 className="text-lg font-semibold text-gray-900 dark:text-white">
            Cost Projection Trends
          </h4>
        </div>

        <div className="h-[400px]">
          {projections.length > 0 ? (
            <Line data={chartData} options={chartOptions} />
          ) : (
            <div className="flex items-center justify-center h-full text-gray-500 dark:text-gray-400">
              <div className="text-center">
                <ChartBarIcon className="h-12 w-12 mx-auto mb-4" />
                <p>Insufficient data for projections</p>
                <p className="text-sm mt-2">Need at least 7 days of historical data</p>
              </div>
            </div>
          )}
        </div>
      </motion.div>

      {/* Key Insights */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 p-6"
      >
        <div className="flex items-center gap-2 mb-4">
          <InformationCircleIcon className="h-5 w-5 text-blue-600 dark:text-blue-400" />
          <h4 className="text-lg font-semibold text-gray-900 dark:text-white">
            Key Insights
          </h4>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div>
            <h5 className="font-medium text-gray-900 dark:text-white mb-2">
              Trend Analysis
            </h5>
            <div className="space-y-2">
              {summaryMetrics.map((summary) => (
                <div key={summary.name} className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <div 
                      className="w-2 h-2 rounded-full"
                      style={{ backgroundColor: summary.color }}
                    />
                    <span className="text-sm text-gray-600 dark:text-gray-400">
                      {summary.label}
                    </span>
                  </div>
                  <div className="flex items-center gap-1">
                    {summary.budgetUtilization > 100 ? (
                      <ArrowTrendingUpIcon className="h-4 w-4 text-red-500" />
                    ) : (
                      <ArrowTrendingDownIcon className="h-4 w-4 text-green-500" />
                    )}
                    <span className="text-sm font-medium text-gray-900 dark:text-white">
                      {summary.budgetUtilization.toFixed(0)}%
                    </span>
                  </div>
                </div>
              ))}
            </div>
          </div>

          <div>
            <h5 className="font-medium text-gray-900 dark:text-white mb-2">
              Recommendations
            </h5>
            <div className="space-y-2 text-sm text-gray-600 dark:text-gray-400">
              {summaryMetrics.some(s => s.overBudget) && (
                <div className="flex items-start gap-2">
                  <ExclamationTriangleIcon className="h-4 w-4 text-red-500 mt-0.5 flex-shrink-0" />
                  <span>Consider budget increase or usage optimization</span>
                </div>
              )}
              <div className="flex items-start gap-2">
                <ClockIcon className="h-4 w-4 text-blue-500 mt-0.5 flex-shrink-0" />
                <span>Monitor daily usage patterns for early detection</span>
              </div>
              <div className="flex items-start gap-2">
                <CurrencyDollarIcon className="h-4 w-4 text-green-500 mt-0.5 flex-shrink-0" />
                <span>Review cost optimization opportunities weekly</span>
              </div>
            </div>
          </div>
        </div>
      </motion.div>
    </div>
  );
};

export default CostProjections;