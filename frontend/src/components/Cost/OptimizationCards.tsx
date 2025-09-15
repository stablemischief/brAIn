import React, { useState, useMemo } from 'react';
import { motion } from 'framer-motion';
import {
  LightBulbIcon,
  ChartBarIcon,
  CurrencyDollarIcon,
  ClockIcon,
  ExclamationTriangleIcon,
  CheckCircleIcon,
  ArrowRightIcon,
  SparklesIcon,
  Cog6ToothIcon,
  FireIcon,
  ShieldCheckIcon,
  BoltIcon,
} from '@heroicons/react/24/outline';
import { CostAnalytics, DailyCost } from '../../types';

interface OptimizationRecommendation {
  id: string;
  title: string;
  description: string;
  category: 'model' | 'usage' | 'scheduling' | 'caching' | 'architecture';
  priority: 'high' | 'medium' | 'low';
  potentialSavings: number;
  savingsPercentage: number;
  complexity: 'easy' | 'moderate' | 'complex';
  timeToImplement: string;
  impact: 'immediate' | 'short-term' | 'long-term';
  actionItems: string[];
  metrics: {
    currentCost: number;
    projectedCost: number;
    affectedRequests: number;
  };
  implemented?: boolean;
}

interface OptimizationCardsProps {
  costData: CostAnalytics;
  historicalData: DailyCost[];
  onImplementRecommendation: (recommendationId: string) => Promise<void>;
  className?: string;
}

export const OptimizationCards: React.FC<OptimizationCardsProps> = ({
  costData,
  historicalData,
  onImplementRecommendation,
  className = '',
}) => {
  const [filter, setFilter] = useState<'all' | 'high' | 'medium' | 'low'>('all');
  const [category, setCategory] = useState<'all' | OptimizationRecommendation['category']>('all');
  const [implementingIds, setImplementingIds] = useState<Set<string>>(new Set());

  // Generate recommendations based on cost data
  const recommendations = useMemo<OptimizationRecommendation[]>(() => {
    const recs: OptimizationRecommendation[] = [];

    // Model optimization recommendations
    if (costData.cost_by_model['gpt-4'] && costData.cost_by_model['gpt-4'] > costData.total_cost * 0.6) {
      recs.push({
        id: 'model-optimization-1',
        title: 'Switch to GPT-3.5-turbo for Simple Tasks',
        description: 'GPT-4 is being used for 60%+ of requests. Consider using GPT-3.5-turbo for simpler tasks like data extraction and basic analysis.',
        category: 'model',
        priority: 'high',
        potentialSavings: costData.cost_by_model['gpt-4'] * 0.3,
        savingsPercentage: 30,
        complexity: 'easy',
        timeToImplement: '1-2 hours',
        impact: 'immediate',
        actionItems: [
          'Audit current GPT-4 usage patterns',
          'Identify tasks suitable for GPT-3.5-turbo',
          'Implement model selection logic',
          'Monitor quality metrics after switch',
        ],
        metrics: {
          currentCost: costData.cost_by_model['gpt-4'],
          projectedCost: costData.cost_by_model['gpt-4'] * 0.7,
          affectedRequests: Math.floor(costData.token_usage.total_tokens * 0.3 / 1000),
        },
      });
    }

    // Token usage optimization
    if (costData.token_usage.output_tokens > costData.token_usage.input_tokens * 0.8) {
      recs.push({
        id: 'token-optimization-1',
        title: 'Optimize Prompt Length and Output',
        description: 'Output tokens are consuming significant costs. Optimize prompts to be more concise and limit response length.',
        category: 'usage',
        priority: 'medium',
        potentialSavings: costData.total_cost * 0.15,
        savingsPercentage: 15,
        complexity: 'moderate',
        timeToImplement: '3-5 hours',
        impact: 'short-term',
        actionItems: [
          'Analyze prompt patterns for redundancy',
          'Implement response length limits',
          'Add prompt optimization guidelines',
          'A/B test shorter prompts',
        ],
        metrics: {
          currentCost: costData.total_cost,
          projectedCost: costData.total_cost * 0.85,
          affectedRequests: Math.floor(costData.token_usage.total_tokens / 1000),
        },
      });
    }

    // Caching recommendations
    const repeatRequestRate = 0.25; // Mock calculation
    if (repeatRequestRate > 0.2) {
      recs.push({
        id: 'caching-1',
        title: 'Implement Response Caching',
        description: 'High repeat request rate detected. Implement caching for frequently requested analyses.',
        category: 'caching',
        priority: 'high',
        potentialSavings: costData.total_cost * 0.25,
        savingsPercentage: 25,
        complexity: 'moderate',
        timeToImplement: '1-2 days',
        impact: 'immediate',
        actionItems: [
          'Set up Redis caching layer',
          'Identify cacheable request patterns',
          'Implement cache invalidation strategy',
          'Monitor cache hit rates',
        ],
        metrics: {
          currentCost: costData.total_cost,
          projectedCost: costData.total_cost * 0.75,
          affectedRequests: Math.floor(costData.token_usage.total_tokens * 0.25 / 1000),
        },
      });
    }

    // Scheduling optimization
    const dailyVariance = historicalData.length > 1 ? 
      Math.max(...historicalData.map(d => d.cost)) / Math.min(...historicalData.map(d => d.cost)) : 1;
    
    if (dailyVariance > 2) {
      recs.push({
        id: 'scheduling-1',
        title: 'Implement Off-Peak Processing',
        description: 'High cost variance detected. Schedule batch processing during off-peak hours for cost savings.',
        category: 'scheduling',
        priority: 'medium',
        potentialSavings: costData.total_cost * 0.12,
        savingsPercentage: 12,
        complexity: 'complex',
        timeToImplement: '2-3 days',
        impact: 'long-term',
        actionItems: [
          'Analyze usage patterns by time of day',
          'Implement job queue system',
          'Set up scheduled batch processing',
          'Monitor cost patterns after implementation',
        ],
        metrics: {
          currentCost: costData.total_cost,
          projectedCost: costData.total_cost * 0.88,
          affectedRequests: Math.floor(costData.token_usage.total_tokens * 0.3 / 1000),
        },
      });
    }

    // Architecture optimization
    recs.push({
      id: 'architecture-1',
      title: 'Implement Request Batching',
      description: 'Process multiple similar requests in batches to reduce per-request overhead and improve efficiency.',
      category: 'architecture',
      priority: 'low',
      potentialSavings: costData.total_cost * 0.08,
      savingsPercentage: 8,
      complexity: 'complex',
      timeToImplement: '1 week',
      impact: 'long-term',
      actionItems: [
        'Design batch processing architecture',
        'Implement request queuing system',
        'Add batch size optimization logic',
        'Test and validate batch processing',
      ],
      metrics: {
        currentCost: costData.total_cost,
        projectedCost: costData.total_cost * 0.92,
        affectedRequests: Math.floor(costData.token_usage.total_tokens / 1000),
      },
    });

    return recs.sort((a, b) => {
      const priorityOrder = { high: 3, medium: 2, low: 1 };
      return priorityOrder[b.priority] - priorityOrder[a.priority] || b.potentialSavings - a.potentialSavings;
    });
  }, [costData, historicalData]);

  // Filter recommendations
  const filteredRecommendations = useMemo(() => {
    return recommendations.filter(rec => {
      const matchesPriority = filter === 'all' || rec.priority === filter;
      const matchesCategory = category === 'all' || rec.category === category;
      return matchesPriority && matchesCategory;
    });
  }, [recommendations, filter, category]);

  // Calculate total potential savings
  const totalSavings = useMemo(() => {
    return filteredRecommendations.reduce((sum, rec) => sum + rec.potentialSavings, 0);
  }, [filteredRecommendations]);

  const handleImplement = async (recommendationId: string) => {
    setImplementingIds(prev => new Set(Array.from(prev).concat(recommendationId)));
    try {
      await onImplementRecommendation(recommendationId);
      // Update recommendation as implemented
    } catch (error) {
      console.error('Failed to implement recommendation:', error);
    } finally {
      setImplementingIds(prev => {
        const next = new Set(prev);
        next.delete(recommendationId);
        return next;
      });
    }
  };

  const getPriorityColor = (priority: OptimizationRecommendation['priority']) => {
    switch (priority) {
      case 'high':
        return 'bg-red-100 text-red-700 dark:bg-red-900/20 dark:text-red-300';
      case 'medium':
        return 'bg-yellow-100 text-yellow-700 dark:bg-yellow-900/20 dark:text-yellow-300';
      case 'low':
        return 'bg-green-100 text-green-700 dark:bg-green-900/20 dark:text-green-300';
    }
  };

  const getCategoryIcon = (category: OptimizationRecommendation['category']) => {
    switch (category) {
      case 'model':
        return SparklesIcon;
      case 'usage':
        return ChartBarIcon;
      case 'scheduling':
        return ClockIcon;
      case 'caching':
        return BoltIcon;
      case 'architecture':
        return Cog6ToothIcon;
      default:
        return LightBulbIcon;
    }
  };

  const getComplexityColor = (complexity: OptimizationRecommendation['complexity']) => {
    switch (complexity) {
      case 'easy':
        return 'text-green-600 dark:text-green-400';
      case 'moderate':
        return 'text-yellow-600 dark:text-yellow-400';
      case 'complex':
        return 'text-red-600 dark:text-red-400';
    }
  };

  return (
    <div className={`space-y-6 ${className}`}>
      {/* Header and Filters */}
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
        <div>
          <h3 className="text-lg font-semibold text-gray-900 dark:text-white">
            Cost Optimization Recommendations
          </h3>
          <p className="text-sm text-gray-600 dark:text-gray-400 mt-1">
            Potential savings: <span className="font-medium text-green-600 dark:text-green-400">
              ${totalSavings.toFixed(2)}
            </span> ({filteredRecommendations.length} recommendations)
          </p>
        </div>

        <div className="flex items-center gap-3">
          {/* Priority Filter */}
          <select
            value={filter}
            onChange={(e) => setFilter(e.target.value as typeof filter)}
            className="px-3 py-2 bg-white dark:bg-gray-700 border border-gray-300 dark:border-gray-600 rounded-lg text-sm focus:ring-2 focus:ring-blue-500 focus:border-transparent"
          >
            <option value="all">All Priorities</option>
            <option value="high">High Priority</option>
            <option value="medium">Medium Priority</option>
            <option value="low">Low Priority</option>
          </select>

          {/* Category Filter */}
          <select
            value={category}
            onChange={(e) => setCategory(e.target.value as typeof category)}
            className="px-3 py-2 bg-white dark:bg-gray-700 border border-gray-300 dark:border-gray-600 rounded-lg text-sm focus:ring-2 focus:ring-blue-500 focus:border-transparent"
          >
            <option value="all">All Categories</option>
            <option value="model">Model Optimization</option>
            <option value="usage">Usage Optimization</option>
            <option value="scheduling">Scheduling</option>
            <option value="caching">Caching</option>
            <option value="architecture">Architecture</option>
          </select>
        </div>
      </div>

      {/* Summary Stats */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <div className="bg-blue-50 dark:bg-blue-900/20 rounded-lg p-4">
          <div className="flex items-center gap-2">
            <CurrencyDollarIcon className="h-5 w-5 text-blue-600 dark:text-blue-400" />
            <span className="text-sm font-medium text-blue-700 dark:text-blue-300">
              Total Savings
            </span>
          </div>
          <p className="text-2xl font-bold text-blue-900 dark:text-blue-100 mt-1">
            ${totalSavings.toFixed(2)}
          </p>
        </div>

        <div className="bg-red-50 dark:bg-red-900/20 rounded-lg p-4">
          <div className="flex items-center gap-2">
            <FireIcon className="h-5 w-5 text-red-600 dark:text-red-400" />
            <span className="text-sm font-medium text-red-700 dark:text-red-300">
              High Priority
            </span>
          </div>
          <p className="text-2xl font-bold text-red-900 dark:text-red-100 mt-1">
            {recommendations.filter(r => r.priority === 'high').length}
          </p>
        </div>

        <div className="bg-green-50 dark:bg-green-900/20 rounded-lg p-4">
          <div className="flex items-center gap-2">
            <CheckCircleIcon className="h-5 w-5 text-green-600 dark:text-green-400" />
            <span className="text-sm font-medium text-green-700 dark:text-green-300">
              Quick Wins
            </span>
          </div>
          <p className="text-2xl font-bold text-green-900 dark:text-green-100 mt-1">
            {recommendations.filter(r => r.complexity === 'easy').length}
          </p>
        </div>

        <div className="bg-purple-50 dark:bg-purple-900/20 rounded-lg p-4">
          <div className="flex items-center gap-2">
            <BoltIcon className="h-5 w-5 text-purple-600 dark:text-purple-400" />
            <span className="text-sm font-medium text-purple-700 dark:text-purple-300">
              Immediate Impact
            </span>
          </div>
          <p className="text-2xl font-bold text-purple-900 dark:text-purple-100 mt-1">
            {recommendations.filter(r => r.impact === 'immediate').length}
          </p>
        </div>
      </div>

      {/* Recommendation Cards */}
      <div className="grid grid-cols-1 gap-6">
        {filteredRecommendations.map((recommendation, index) => {
          const Icon = getCategoryIcon(recommendation.category);
          const isImplementing = implementingIds.has(recommendation.id);

          return (
            <motion.div
              key={recommendation.id}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: index * 0.1 }}
              className="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 p-6"
            >
              <div className="flex items-start justify-between mb-4">
                <div className="flex items-start gap-4">
                  <div className="p-3 bg-blue-100 dark:bg-blue-900/30 rounded-lg">
                    <Icon className="h-6 w-6 text-blue-600 dark:text-blue-400" />
                  </div>
                  
                  <div className="flex-1">
                    <div className="flex items-center gap-2 mb-2">
                      <h4 className="text-lg font-semibold text-gray-900 dark:text-white">
                        {recommendation.title}
                      </h4>
                      <span className={`px-2 py-1 rounded-full text-xs font-medium ${getPriorityColor(recommendation.priority)}`}>
                        {recommendation.priority}
                      </span>
                    </div>
                    
                    <p className="text-gray-600 dark:text-gray-400 mb-4">
                      {recommendation.description}
                    </p>

                    <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-4">
                      <div>
                        <p className="text-sm font-medium text-gray-900 dark:text-white">
                          Potential Savings
                        </p>
                        <p className="text-lg font-bold text-green-600 dark:text-green-400">
                          ${recommendation.potentialSavings.toFixed(2)}
                        </p>
                        <p className="text-xs text-gray-500 dark:text-gray-400">
                          {recommendation.savingsPercentage}% reduction
                        </p>
                      </div>

                      <div>
                        <p className="text-sm font-medium text-gray-900 dark:text-white">
                          Complexity
                        </p>
                        <p className={`text-sm font-medium ${getComplexityColor(recommendation.complexity)}`}>
                          {recommendation.complexity}
                        </p>
                        <p className="text-xs text-gray-500 dark:text-gray-400">
                          {recommendation.timeToImplement}
                        </p>
                      </div>

                      <div>
                        <p className="text-sm font-medium text-gray-900 dark:text-white">
                          Impact
                        </p>
                        <p className="text-sm text-gray-600 dark:text-gray-400 capitalize">
                          {recommendation.impact}
                        </p>
                        <p className="text-xs text-gray-500 dark:text-gray-400">
                          {recommendation.metrics.affectedRequests.toLocaleString()} requests
                        </p>
                      </div>

                      <div>
                        <p className="text-sm font-medium text-gray-900 dark:text-white">
                          Cost Projection
                        </p>
                        <p className="text-sm text-gray-600 dark:text-gray-400">
                          ${recommendation.metrics.currentCost.toFixed(2)} → ${recommendation.metrics.projectedCost.toFixed(2)}
                        </p>
                        <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-1 mt-1">
                          <div 
                            className="bg-green-500 h-1 rounded-full"
                            style={{ width: `${(1 - recommendation.metrics.projectedCost / recommendation.metrics.currentCost) * 100}%` }}
                          />
                        </div>
                      </div>
                    </div>
                  </div>
                </div>

                <button
                  onClick={() => handleImplement(recommendation.id)}
                  disabled={isImplementing || recommendation.implemented}
                  className={`flex items-center gap-2 px-4 py-2 rounded-lg font-medium transition-colors ${
                    recommendation.implemented
                      ? 'bg-green-100 text-green-700 dark:bg-green-900/20 dark:text-green-300'
                      : 'bg-blue-600 text-white hover:bg-blue-700 disabled:opacity-50'
                  }`}
                >
                  {isImplementing ? (
                    <div className="animate-spin rounded-full h-4 w-4 border-2 border-white border-t-transparent" />
                  ) : recommendation.implemented ? (
                    <CheckCircleIcon className="h-4 w-4" />
                  ) : (
                    <ArrowRightIcon className="h-4 w-4" />
                  )}
                  {isImplementing ? 'Implementing...' : recommendation.implemented ? 'Implemented' : 'Implement'}
                </button>
              </div>

              {/* Action Items */}
              <div className="border-t border-gray-200 dark:border-gray-700 pt-4">
                <h5 className="text-sm font-medium text-gray-900 dark:text-white mb-3">
                  Implementation Steps:
                </h5>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-2">
                  {recommendation.actionItems.map((item, itemIndex) => (
                    <div key={itemIndex} className="flex items-center gap-2">
                      <div className="w-1.5 h-1.5 rounded-full bg-blue-600 dark:bg-blue-400 flex-shrink-0 mt-2" />
                      <span className="text-sm text-gray-600 dark:text-gray-400">
                        {item}
                      </span>
                    </div>
                  ))}
                </div>
              </div>
            </motion.div>
          );
        })}
      </div>

      {filteredRecommendations.length === 0 && (
        <div className="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 p-12 text-center">
          <LightBulbIcon className="h-12 w-12 text-gray-400 mx-auto mb-4" />
          <h3 className="text-lg font-medium text-gray-900 dark:text-white mb-2">
            No recommendations match your filters
          </h3>
          <p className="text-gray-600 dark:text-gray-400">
            Try adjusting your priority or category filters to see more recommendations.
          </p>
        </div>
      )}
    </div>
  );
};

export default OptimizationCards;