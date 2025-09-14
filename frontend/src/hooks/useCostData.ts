import { useState, useEffect, useCallback, useMemo } from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { CostAnalytics, DailyCost, BudgetInfo, ApiResponse } from '@/types';
import { format, subDays, startOfDay } from 'date-fns';

interface CostDataFilters {
  timeframe: '7d' | '30d' | '90d';
  startDate?: Date;
  endDate?: Date;
  models?: string[];
  operations?: string[];
}

interface UseCostDataOptions {
  autoRefresh?: boolean;
  refreshInterval?: number;
  filters?: CostDataFilters;
}

interface CostDataState {
  analytics: CostAnalytics | null;
  historicalData: DailyCost[];
  budgetInfo: BudgetInfo | null;
  filters: CostDataFilters;
  isLoading: boolean;
  error: string | null;
}

interface CostDataActions {
  refetch: () => Promise<void>;
  updateFilters: (newFilters: Partial<CostDataFilters>) => void;
  updateBudget: (budgetData: any) => Promise<void>;
  clearCache: () => void;
}

const API_BASE_URL = '/api';

// API functions
const fetchCostAnalytics = async (filters: CostDataFilters): Promise<CostAnalytics> => {
  const params = new URLSearchParams();
  
  if (filters.timeframe) params.append('timeframe', filters.timeframe);
  if (filters.startDate) params.append('start_date', format(filters.startDate, 'yyyy-MM-dd'));
  if (filters.endDate) params.append('end_date', format(filters.endDate, 'yyyy-MM-dd'));
  if (filters.models?.length) params.append('models', filters.models.join(','));
  if (filters.operations?.length) params.append('operations', filters.operations.join(','));

  const response = await fetch(`${API_BASE_URL}/analytics/costs?${params}`);
  if (!response.ok) {
    throw new Error(`Failed to fetch cost analytics: ${response.statusText}`);
  }
  
  const data: ApiResponse<CostAnalytics> = await response.json();
  if (!data.success) {
    throw new Error(data.error?.message || 'Failed to fetch cost analytics');
  }
  
  return data.data!;
};

const fetchHistoricalData = async (filters: CostDataFilters): Promise<DailyCost[]> => {
  const params = new URLSearchParams();
  
  if (filters.timeframe) params.append('timeframe', filters.timeframe);
  if (filters.startDate) params.append('start_date', format(filters.startDate, 'yyyy-MM-dd'));
  if (filters.endDate) params.append('end_date', format(filters.endDate, 'yyyy-MM-dd'));

  const response = await fetch(`${API_BASE_URL}/analytics/historical?${params}`);
  if (!response.ok) {
    throw new Error(`Failed to fetch historical data: ${response.statusText}`);
  }
  
  const data: ApiResponse<DailyCost[]> = await response.json();
  if (!data.success) {
    throw new Error(data.error?.message || 'Failed to fetch historical data');
  }
  
  return data.data!;
};

const fetchBudgetInfo = async (): Promise<BudgetInfo> => {
  const response = await fetch(`${API_BASE_URL}/budget/info`);
  if (!response.ok) {
    throw new Error(`Failed to fetch budget info: ${response.statusText}`);
  }
  
  const data: ApiResponse<BudgetInfo> = await response.json();
  if (!data.success) {
    throw new Error(data.error?.message || 'Failed to fetch budget info');
  }
  
  return data.data!;
};

const updateBudgetSettings = async (budgetData: any): Promise<BudgetInfo> => {
  const response = await fetch(`${API_BASE_URL}/budget/update`, {
    method: 'PUT',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(budgetData),
  });
  
  if (!response.ok) {
    throw new Error(`Failed to update budget: ${response.statusText}`);
  }
  
  const data: ApiResponse<BudgetInfo> = await response.json();
  if (!data.success) {
    throw new Error(data.error?.message || 'Failed to update budget');
  }
  
  return data.data!;
};

// Generate query keys
const getCostQueryKeys = {
  all: ['cost'] as const,
  analytics: (filters: CostDataFilters) => [...getCostQueryKeys.all, 'analytics', filters] as const,
  historical: (filters: CostDataFilters) => [...getCostQueryKeys.all, 'historical', filters] as const,
  budget: () => [...getCostQueryKeys.all, 'budget'] as const,
};

// Default filters
const defaultFilters: CostDataFilters = {
  timeframe: '30d',
  models: undefined,
  operations: undefined,
};

export const useCostData = (options: UseCostDataOptions = {}): CostDataState & CostDataActions => {
  const {
    autoRefresh = true,
    refreshInterval = 30000, // 30 seconds
    filters: initialFilters = defaultFilters,
  } = options;

  const queryClient = useQueryClient();
  const [filters, setFilters] = useState<CostDataFilters>(initialFilters);
  const [error, setError] = useState<string | null>(null);

  // Calculate date range based on timeframe
  const dateRange = useMemo(() => {
    const now = new Date();
    let start: Date, end: Date;

    if (filters.startDate && filters.endDate) {
      start = startOfDay(filters.startDate);
      end = startOfDay(filters.endDate);
    } else {
      end = startOfDay(now);
      switch (filters.timeframe) {
        case '7d':
          start = startOfDay(subDays(now, 7));
          break;
        case '90d':
          start = startOfDay(subDays(now, 90));
          break;
        default: // '30d'
          start = startOfDay(subDays(now, 30));
          break;
      }
    }

    return { start, end };
  }, [filters]);

  // Update filters with calculated date range
  const effectiveFilters = useMemo(() => ({
    ...filters,
    startDate: dateRange.start,
    endDate: dateRange.end,
  }), [filters, dateRange]);

  // Cost analytics query
  const analyticsQuery = useQuery({
    queryKey: getCostQueryKeys.analytics(effectiveFilters),
    queryFn: () => fetchCostAnalytics(effectiveFilters),
    staleTime: autoRefresh ? refreshInterval : Infinity,
    refetchInterval: autoRefresh ? refreshInterval : false,
    retry: (failureCount, error) => {
      // Don't retry on 4xx errors
      if (error instanceof Error && error.message.includes('4')) {
        return false;
      }
      return failureCount < 3;
    },
    onError: (error: Error) => {
      setError(error.message);
    },
  });

  // Historical data query
  const historicalQuery = useQuery({
    queryKey: getCostQueryKeys.historical(effectiveFilters),
    queryFn: () => fetchHistoricalData(effectiveFilters),
    staleTime: autoRefresh ? refreshInterval : Infinity,
    refetchInterval: autoRefresh ? refreshInterval : false,
    retry: (failureCount, error) => {
      if (error instanceof Error && error.message.includes('4')) {
        return false;
      }
      return failureCount < 3;
    },
    onError: (error: Error) => {
      setError(error.message);
    },
  });

  // Budget info query
  const budgetQuery = useQuery({
    queryKey: getCostQueryKeys.budget(),
    queryFn: fetchBudgetInfo,
    staleTime: autoRefresh ? refreshInterval * 2 : Infinity, // Budget changes less frequently
    refetchInterval: autoRefresh ? refreshInterval * 2 : false,
    retry: (failureCount, error) => {
      if (error instanceof Error && error.message.includes('4')) {
        return false;
      }
      return failureCount < 3;
    },
    onError: (error: Error) => {
      setError(error.message);
    },
  });

  // Budget update mutation
  const budgetMutation = useMutation({
    mutationFn: updateBudgetSettings,
    onSuccess: (data) => {
      // Update the budget query cache
      queryClient.setQueryData(getCostQueryKeys.budget(), data);
      setError(null);
    },
    onError: (error: Error) => {
      setError(error.message);
    },
  });

  // Clear any existing error when data loads successfully
  useEffect(() => {
    if (analyticsQuery.data || historicalQuery.data || budgetQuery.data) {
      setError(null);
    }
  }, [analyticsQuery.data, historicalQuery.data, budgetQuery.data]);

  // Actions
  const updateFilters = useCallback((newFilters: Partial<CostDataFilters>) => {
    setFilters(prev => ({ ...prev, ...newFilters }));
  }, []);

  const refetch = useCallback(async () => {
    setError(null);
    await Promise.all([
      analyticsQuery.refetch(),
      historicalQuery.refetch(),
      budgetQuery.refetch(),
    ]);
  }, [analyticsQuery, historicalQuery, budgetQuery]);

  const updateBudget = useCallback(async (budgetData: any) => {
    await budgetMutation.mutateAsync(budgetData);
  }, [budgetMutation]);

  const clearCache = useCallback(() => {
    queryClient.removeQueries({ queryKey: getCostQueryKeys.all });
    setError(null);
  }, [queryClient]);

  // Determine loading state
  const isLoading = analyticsQuery.isLoading || historicalQuery.isLoading || budgetQuery.isLoading;

  // Combine errors
  const combinedError = error || 
    analyticsQuery.error?.message || 
    historicalQuery.error?.message || 
    budgetQuery.error?.message || 
    budgetMutation.error?.message || 
    null;

  return {
    // State
    analytics: analyticsQuery.data || null,
    historicalData: historicalQuery.data || [],
    budgetInfo: budgetQuery.data || null,
    filters: effectiveFilters,
    isLoading,
    error: combinedError,

    // Actions
    refetch,
    updateFilters,
    updateBudget,
    clearCache,
  };
};

// Additional hooks for specific use cases

export const useCostAnalytics = (filters?: Partial<CostDataFilters>) => {
  const { analytics, isLoading, error, refetch } = useCostData({
    filters: { ...defaultFilters, ...filters },
  });

  return {
    data: analytics,
    isLoading,
    error,
    refetch,
  };
};

export const useBudgetInfo = () => {
  const { budgetInfo, isLoading, error, updateBudget, refetch } = useCostData();

  return {
    data: budgetInfo,
    isLoading,
    error,
    updateBudget,
    refetch,
  };
};

export const useHistoricalCosts = (timeframe: '7d' | '30d' | '90d' = '30d') => {
  const { historicalData, isLoading, error, refetch } = useCostData({
    filters: { timeframe },
  });

  return {
    data: historicalData,
    isLoading,
    error,
    refetch,
  };
};

// Hook for cost optimization insights
export const useCostInsights = () => {
  const { analytics, historicalData, isLoading, error } = useCostData();

  const insights = useMemo(() => {
    if (!analytics || !historicalData || historicalData.length === 0) {
      return null;
    }

    // Calculate trends
    const recentDays = historicalData.slice(-7);
    const previousDays = historicalData.slice(-14, -7);
    
    const recentAvg = recentDays.reduce((sum, day) => sum + day.cost, 0) / recentDays.length;
    const previousAvg = previousDays.reduce((sum, day) => sum + day.cost, 0) / previousDays.length || recentAvg;
    
    const trend = recentAvg > previousAvg ? 'increasing' : recentAvg < previousAvg ? 'decreasing' : 'stable';
    const trendPercentage = previousAvg > 0 ? Math.abs((recentAvg - previousAvg) / previousAvg * 100) : 0;

    // Find peak usage day
    const peakDay = historicalData.reduce((peak, day) => day.cost > peak.cost ? day : peak);
    
    // Cost efficiency
    const avgCostPerToken = analytics.token_usage.total_tokens > 0 
      ? analytics.total_cost / analytics.token_usage.total_tokens 
      : 0;

    return {
      trend,
      trendPercentage,
      peakDay,
      avgCostPerToken,
      dailyAverage: recentAvg,
      weeklyChange: recentAvg - previousAvg,
      mostExpensiveModel: Object.keys(analytics.cost_by_model).reduce((a, b) => 
        analytics.cost_by_model[a] > analytics.cost_by_model[b] ? a : b
      ),
    };
  }, [analytics, historicalData]);

  return {
    insights,
    isLoading,
    error,
  };
};

export default useCostData;