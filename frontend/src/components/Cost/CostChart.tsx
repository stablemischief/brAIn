import React, { useMemo } from 'react';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  Title,
  Tooltip,
  Legend,
  TimeScale,
  Filler,
} from 'chart.js';
import { Line, Bar } from 'react-chartjs-2';
import 'chartjs-adapter-date-fns';
import { format, parseISO } from 'date-fns';
import { DailyCost } from '../../types';
import { 
  ChartBarIcon, 
  PresentationChartLineIcon,
  CalendarIcon,
  ArrowTrendingUpIcon,
  ArrowTrendingDownIcon,
} from '@heroicons/react/24/outline';

// Register Chart.js components
ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  Title,
  Tooltip,
  Legend,
  TimeScale,
  Filler
);

interface CostChartProps {
  data: DailyCost[];
  chartType?: 'line' | 'bar' | 'area';
  timeframe: '7d' | '30d' | '90d';
  className?: string;
  showControls?: boolean;
}

interface ChartMetrics {
  trend: 'increasing' | 'decreasing' | 'stable';
  trendPercentage: number;
  averageCost: number;
  totalCost: number;
  peakDay: DailyCost | null;
  lowestDay: DailyCost | null;
}

export const CostChart: React.FC<CostChartProps> = ({
  data,
  chartType = 'line',
  timeframe,
  className = '',
  showControls = true,
}) => {
  const [selectedChartType, setSelectedChartType] = React.useState<'line' | 'bar' | 'area'>(chartType);
  const [metric, setMetric] = React.useState<'cost' | 'tokens' | 'requests'>('cost');

  // Calculate chart metrics
  const metrics = useMemo<ChartMetrics>(() => {
    if (!data || data.length === 0) {
      return {
        trend: 'stable',
        trendPercentage: 0,
        averageCost: 0,
        totalCost: 0,
        peakDay: null,
        lowestDay: null,
      };
    }

    const totalCost = data.reduce((sum, day) => sum + day.cost, 0);
    const averageCost = totalCost / data.length;
    
    const peakDay = data.reduce((peak, day) => 
      day.cost > (peak?.cost || 0) ? day : peak, data[0]
    );
    
    const lowestDay = data.reduce((lowest, day) => 
      day.cost < (lowest?.cost || Infinity) ? day : lowest, data[0]
    );

    // Calculate trend
    let trend: 'increasing' | 'decreasing' | 'stable' = 'stable';
    let trendPercentage = 0;

    if (data.length >= 2) {
      const firstHalf = data.slice(0, Math.floor(data.length / 2));
      const secondHalf = data.slice(Math.floor(data.length / 2));
      
      const firstHalfAvg = firstHalf.reduce((sum, day) => sum + day.cost, 0) / firstHalf.length;
      const secondHalfAvg = secondHalf.reduce((sum, day) => sum + day.cost, 0) / secondHalf.length;
      
      if (Math.abs(secondHalfAvg - firstHalfAvg) > 0.01) {
        trend = secondHalfAvg > firstHalfAvg ? 'increasing' : 'decreasing';
        trendPercentage = Math.abs(((secondHalfAvg - firstHalfAvg) / firstHalfAvg) * 100);
      }
    }

    return {
      trend,
      trendPercentage,
      averageCost,
      totalCost,
      peakDay,
      lowestDay,
    };
  }, [data]);

  // Prepare chart data
  const chartData = useMemo(() => {
    const labels = data.map(day => day.date);
    
    const getValue = (day: DailyCost) => {
      switch (metric) {
        case 'tokens':
          return day.tokens;
        case 'requests':
          return day.requests;
        default:
          return day.cost;
      }
    };

    const baseData = {
      labels,
      datasets: [
        {
          label: metric === 'cost' ? 'Cost ($)' : metric === 'tokens' ? 'Tokens' : 'Requests',
          data: data.map(getValue),
          borderColor: 'rgb(59, 130, 246)',
          backgroundColor: selectedChartType === 'area' 
            ? 'rgba(59, 130, 246, 0.1)' 
            : 'rgba(59, 130, 246, 0.8)',
          tension: 0.4,
          fill: selectedChartType === 'area',
          pointHoverRadius: 8,
          pointHoverBackgroundColor: 'rgb(59, 130, 246)',
          pointHoverBorderColor: 'rgb(255, 255, 255)',
          pointHoverBorderWidth: 2,
        },
      ],
    };

    return baseData;
  }, [data, metric, selectedChartType]);

  // Chart options
  const chartOptions = useMemo(() => ({
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
        displayColors: true,
        callbacks: {
          label: (context: any) => {
            const value = context.parsed.y;
            switch (metric) {
              case 'cost':
                return `${context.dataset.label}: $${value.toFixed(4)}`;
              case 'tokens':
                return `${context.dataset.label}: ${value.toLocaleString()}`;
              default:
                return `${context.dataset.label}: ${value}`;
            }
          },
          labelColor: () => ({
            borderColor: 'rgb(59, 130, 246)',
            backgroundColor: 'rgb(59, 130, 246)',
          }),
        },
      },
    },
    scales: {
      x: {
        type: 'category' as const,
        grid: {
          color: 'rgba(107, 114, 128, 0.1)',
          drawBorder: false,
        },
        ticks: {
          color: 'rgb(107, 114, 128)',
          callback: function(value: any, index: number) {
            const date = data[index]?.date;
            return date ? format(parseISO(date), 'MMM dd') : '';
          },
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
            switch (metric) {
              case 'cost':
                return `$${value.toFixed(2)}`;
              case 'tokens':
                return `${(value / 1000).toFixed(0)}K`;
              default:
                return value.toString();
            }
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
  }), [data, metric]);

  const chartTypes = [
    { value: 'line', label: 'Line', icon: PresentationChartLineIcon },
    { value: 'area', label: 'Area', icon: CalendarIcon },
    { value: 'bar', label: 'Bar', icon: ChartBarIcon },
  ] as const;

  const metricOptions = [
    { value: 'cost', label: 'Cost ($)', color: 'text-blue-600' },
    { value: 'tokens', label: 'Tokens', color: 'text-green-600' },
    { value: 'requests', label: 'Requests', color: 'text-purple-600' },
  ] as const;

  const renderChart = () => {
    const commonProps = {
      data: chartData,
      options: chartOptions,
      height: 300,
    };

    switch (selectedChartType) {
      case 'bar':
        return <Bar {...commonProps} />;
      default:
        return <Line {...commonProps} />;
    }
  };

  if (!data || data.length === 0) {
    return (
      <div className={`bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 p-6 ${className}`}>
        <div className="flex items-center justify-center h-64 text-gray-500 dark:text-gray-400">
          <div className="text-center">
            <ChartBarIcon className="h-12 w-12 mx-auto mb-4" />
            <p>No data available for the selected timeframe</p>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className={`bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 ${className}`}>
      {/* Chart Header */}
      <div className="p-6 pb-0">
        <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
          <div>
            <h3 className="text-lg font-semibold text-gray-900 dark:text-white">
              Cost Trends
            </h3>
            <p className="text-sm text-gray-600 dark:text-gray-400 mt-1">
              {timeframe === '7d' ? 'Last 7 days' : timeframe === '30d' ? 'Last 30 days' : 'Last 90 days'}
            </p>
          </div>

          {showControls && (
            <div className="flex items-center gap-3">
              {/* Metric Selector */}
              <div className="flex bg-gray-100 dark:bg-gray-700 rounded-lg p-1">
                {metricOptions.map((m) => (
                  <button
                    key={m.value}
                    onClick={() => setMetric(m.value as typeof metric)}
                    className={`px-3 py-1 rounded-md text-xs font-medium transition-colors ${
                      metric === m.value
                        ? 'bg-white dark:bg-gray-600 text-gray-900 dark:text-white shadow-sm'
                        : 'text-gray-600 dark:text-gray-300 hover:text-gray-900 dark:hover:text-white'
                    }`}
                  >
                    {m.label}
                  </button>
                ))}
              </div>

              {/* Chart Type Selector */}
              <div className="flex bg-gray-100 dark:bg-gray-700 rounded-lg p-1">
                {chartTypes.map((type) => {
                  const Icon = type.icon;
                  return (
                    <button
                      key={type.value}
                      onClick={() => setSelectedChartType(type.value as typeof selectedChartType)}
                      className={`p-2 rounded-md transition-colors ${
                        selectedChartType === type.value
                          ? 'bg-white dark:bg-gray-600 text-gray-900 dark:text-white shadow-sm'
                          : 'text-gray-600 dark:text-gray-300 hover:text-gray-900 dark:hover:text-white'
                      }`}
                      title={type.label}
                    >
                      <Icon className="h-4 w-4" />
                    </button>
                  );
                })}
              </div>
            </div>
          )}
        </div>

        {/* Key Metrics */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mt-6">
          <div className="text-center">
            <p className="text-xs text-gray-500 dark:text-gray-400">Total</p>
            <p className="text-lg font-semibold text-gray-900 dark:text-white">
              ${metrics.totalCost.toFixed(2)}
            </p>
          </div>
          <div className="text-center">
            <p className="text-xs text-gray-500 dark:text-gray-400">Average</p>
            <p className="text-lg font-semibold text-gray-900 dark:text-white">
              ${metrics.averageCost.toFixed(2)}
            </p>
          </div>
          <div className="text-center">
            <p className="text-xs text-gray-500 dark:text-gray-400">Peak Day</p>
            <p className="text-lg font-semibold text-gray-900 dark:text-white">
              ${metrics.peakDay?.cost.toFixed(2) || '0.00'}
            </p>
          </div>
          <div className="text-center flex items-center justify-center">
            <div className="flex items-center gap-1">
              {metrics.trend === 'increasing' ? (
                <ArrowTrendingUpIcon className="h-4 w-4 text-red-500" />
              ) : metrics.trend === 'decreasing' ? (
                <ArrowTrendingDownIcon className="h-4 w-4 text-green-500" />
              ) : null}
              <div>
                <p className="text-xs text-gray-500 dark:text-gray-400">Trend</p>
                <p className={`text-lg font-semibold ${
                  metrics.trend === 'increasing' 
                    ? 'text-red-600 dark:text-red-400'
                    : metrics.trend === 'decreasing'
                    ? 'text-green-600 dark:text-green-400'
                    : 'text-gray-600 dark:text-gray-400'
                }`}>
                  {metrics.trendPercentage > 0 ? `${metrics.trendPercentage.toFixed(1)}%` : 'Stable'}
                </p>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Chart */}
      <div className="p-6 pt-4">
        <div className="h-[300px]">
          {renderChart()}
        </div>
      </div>
    </div>
  );
};

export default CostChart;