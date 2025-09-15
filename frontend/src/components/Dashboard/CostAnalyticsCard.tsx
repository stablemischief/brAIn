import React from 'react';
import { CostAnalytics } from '../../types';
import {
  BanknotesIcon,
  ChartBarIcon,
  ExclamationTriangleIcon,
  ArrowTrendingUpIcon,
  ArrowTrendingDownIcon,
  ClockIcon
} from '@heroicons/react/24/outline';

interface CostAnalyticsCardProps {
  data: CostAnalytics | null;
  connected: boolean;
  className?: string;
}

interface QuickStatProps {
  label: string;
  value: string;
  trend?: 'up' | 'down' | 'stable';
  trendValue?: string;
  icon: React.ReactNode;
  color: string;
}

const QuickStat: React.FC<QuickStatProps> = ({ 
  label, 
  value, 
  trend, 
  trendValue, 
  icon, 
  color 
}) => {
  const getTrendIcon = () => {
    switch (trend) {
      case 'up':
        return <ArrowTrendingUpIcon className="h-3 w-3 text-red-500" />;
      case 'down':
        return <ArrowTrendingDownIcon className="h-3 w-3 text-green-500" />;
      default:
        return null;
    }
  };

  const getTrendColor = () => {
    switch (trend) {
      case 'up':
        return 'text-red-600 dark:text-red-400';
      case 'down':
        return 'text-green-600 dark:text-green-400';
      default:
        return 'text-gray-500 dark:text-gray-400';
    }
  };

  return (
    <div className="flex items-center justify-between p-3 bg-white dark:bg-gray-700 rounded-lg border border-gray-200 dark:border-gray-600">
      <div className="flex items-center space-x-3">
        <div className={color}>
          {icon}
        </div>
        <div>
          <p className="text-xs text-gray-500 dark:text-gray-400">{label}</p>
          <p className="text-lg font-semibold text-gray-900 dark:text-white">
            {value}
          </p>
        </div>
      </div>
      
      {trend && trendValue && (
        <div className={`flex items-center space-x-1 text-xs ${getTrendColor()}`}>
          {getTrendIcon()}
          <span>{trendValue}</span>
        </div>
      )}
    </div>
  );
};

export const CostAnalyticsCard: React.FC<CostAnalyticsCardProps> = ({ 
  data, 
  connected, 
  className = '' 
}) => {
  if (!connected) {
    return (
      <div className={`bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 p-6 ${className}`}>
        <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
          Cost Analytics
        </h3>
        <div className="text-center py-8">
          <ExclamationTriangleIcon className="h-12 w-12 text-gray-400 mx-auto mb-4" />
          <p className="text-gray-500 dark:text-gray-400">
            Connection lost - unable to retrieve cost data
          </p>
        </div>
      </div>
    );
  }

  if (!data) {
    return (
      <div className={`bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 p-6 ${className}`}>
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-lg font-semibold text-gray-900 dark:text-white">
            Cost Analytics
          </h3>
          <ClockIcon className="h-4 w-4 text-gray-400 animate-spin" />
        </div>
        
        <div className="space-y-3">
          <div className="animate-pulse">
            <div className="h-4 bg-gray-200 dark:bg-gray-700 rounded w-3/4 mb-2"></div>
            <div className="h-8 bg-gray-200 dark:bg-gray-700 rounded w-1/2"></div>
          </div>
        </div>
      </div>
    );
  }

  // Calculate trends and format data
  const formatCurrency = (amount: number) => `$${amount.toFixed(4)}`;
  const formatTokens = (tokens: number) => tokens.toLocaleString();

  const dailyAverage = data.daily_costs.length > 0 
    ? data.daily_costs.reduce((sum, day) => sum + day.cost, 0) / data.daily_costs.length 
    : 0;

  const recentTrend = data.daily_costs.length >= 2 
    ? data.daily_costs[data.daily_costs.length - 1].cost > data.daily_costs[data.daily_costs.length - 2].cost 
      ? 'up' : 'down'
    : 'stable';

  const topModel = Object.keys(data.cost_by_model).reduce((a, b) => 
    data.cost_by_model[a] > data.cost_by_model[b] ? a : b, 
    Object.keys(data.cost_by_model)[0]
  );

  return (
    <div className={`bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 p-6 ${className}`}>
      {/* Header */}
      <div className="flex items-center justify-between mb-6">
        <h3 className="text-lg font-semibold text-gray-900 dark:text-white">
          Cost Analytics
        </h3>
        <div className="text-xs text-gray-500 dark:text-gray-400">
          {data.timeframe}
        </div>
      </div>

      {/* Total Cost Highlight */}
      <div className="bg-gradient-to-r from-blue-50 to-indigo-50 dark:from-blue-900/20 dark:to-indigo-900/20 p-4 rounded-lg mb-6">
        <div className="flex items-center justify-between">
          <div>
            <p className="text-sm text-gray-600 dark:text-gray-400">Total Spend</p>
            <p className="text-3xl font-bold text-gray-900 dark:text-white">
              {formatCurrency(data.total_cost)}
            </p>
            <p className="text-sm text-gray-500 dark:text-gray-400">
              {formatTokens(data.token_usage.total_tokens)} tokens
            </p>
          </div>
          <div className="text-right">
            <p className="text-sm text-gray-600 dark:text-gray-400">Projected Monthly</p>
            <p className={`text-lg font-semibold ${
              data.projected_monthly_cost > data.total_cost * 2 
                ? 'text-yellow-600 dark:text-yellow-400'
                : 'text-gray-900 dark:text-white'
            }`}>
              {formatCurrency(data.projected_monthly_cost)}
            </p>
          </div>
        </div>
      </div>

      {/* Quick Stats Grid */}
      <div className="space-y-3 mb-6">
        <QuickStat
          label="Daily Average"
          value={formatCurrency(dailyAverage)}
          trend={recentTrend}
          trendValue={recentTrend !== 'stable' ? '12%' : undefined}
          icon={<ChartBarIcon className="h-5 w-5" />}
          color="text-blue-600 dark:text-blue-400"
        />
        
        <QuickStat
          label="Input Tokens"
          value={formatTokens(data.token_usage.input_tokens)}
          icon={<BanknotesIcon className="h-5 w-5" />}
          color="text-green-600 dark:text-green-400"
        />
        
        <QuickStat
          label="Output Tokens" 
          value={formatTokens(data.token_usage.output_tokens)}
          icon={<BanknotesIcon className="h-5 w-5" />}
          color="text-purple-600 dark:text-purple-400"
        />
      </div>

      {/* Top Model */}
      {topModel && (
        <div className="border-t border-gray-200 dark:border-gray-600 pt-4">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-gray-600 dark:text-gray-400">Top Model</p>
              <p className="font-medium text-gray-900 dark:text-white">{topModel}</p>
            </div>
            <div className="text-right">
              <p className="text-sm font-medium text-gray-900 dark:text-white">
                {formatCurrency(data.cost_by_model[topModel])}
              </p>
              <p className="text-xs text-gray-500 dark:text-gray-400">
                {((data.cost_by_model[topModel] / data.total_cost) * 100).toFixed(1)}%
              </p>
            </div>
          </div>
        </div>
      )}

      {/* Budget Warning */}
      {data.projected_monthly_cost > data.total_cost * 3 && (
        <div className="mt-4 p-3 bg-yellow-50 dark:bg-yellow-900/20 border border-yellow-200 dark:border-yellow-800 rounded-lg">
          <div className="flex items-center space-x-2">
            <ExclamationTriangleIcon className="h-5 w-5 text-yellow-500" />
            <p className="text-sm text-yellow-700 dark:text-yellow-300">
              Projected monthly cost is significantly higher than current spend
            </p>
          </div>
        </div>
      )}
    </div>
  );
};

export default CostAnalyticsCard;