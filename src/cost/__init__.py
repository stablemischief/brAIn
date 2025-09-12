"""
Cost Optimization System for brAIn v2.0
Comprehensive cost tracking, budget management, and optimization

This module provides:
- Accurate token counting across multiple LLM providers
- Real-time cost calculation and tracking
- Budget enforcement with configurable rules and alerts
- Advanced cost forecasting and optimization algorithms
- Comprehensive analytics and reporting capabilities

Main Components:
- TokenCounter: Accurate token counting for all supported models
- CostCalculator: Real-time cost calculation with detailed breakdowns
- BudgetEnforcer: Budget management with enforcement rules and alerts
- CostOptimizer: Intelligent optimization recommendations and forecasting
- CostAnalytics: Comprehensive reporting and insights system
"""

from .token_counter import (
    TokenCounter, TokenCount, ModelProvider, ModelConfig,
    get_token_counter, count_tokens, count_message_tokens,
    estimate_processing_cost
)

from .cost_calculator import (
    CostCalculator, CostBreakdown, CostCategory, CostForecast, BudgetPeriod,
    get_cost_calculator, calculate_cost, estimate_batch_cost
)

from .budget_manager import (
    BudgetEnforcer, BudgetRule, BudgetViolation, BudgetConfig, BudgetAlert,
    EnforcementAction, AlertSeverity, ApprovalRequest,
    get_budget_enforcer
)

from .optimizer import (
    CostOptimizer, OptimizationStrategy, OptimizationRecommendation,
    BatchOptimizationResult, CostTrend, ForecastingMethod,
    get_cost_optimizer
)

from .analytics import (
    CostAnalytics, AnalyticsReport, CostAlert, ReportType, AlertType,
    get_cost_analytics
)

# Main system interface
from .cost_system import CostSystem, get_cost_system

__all__ = [
    # Token counting
    'TokenCounter', 'TokenCount', 'ModelProvider', 'ModelConfig',
    'get_token_counter', 'count_tokens', 'count_message_tokens', 'estimate_processing_cost',
    
    # Cost calculation  
    'CostCalculator', 'CostBreakdown', 'CostCategory', 'CostForecast', 'BudgetPeriod',
    'get_cost_calculator', 'calculate_cost', 'estimate_batch_cost',
    
    # Budget management
    'BudgetEnforcer', 'BudgetRule', 'BudgetViolation', 'BudgetConfig', 'BudgetAlert',
    'EnforcementAction', 'AlertSeverity', 'ApprovalRequest', 'get_budget_enforcer',
    
    # Optimization
    'CostOptimizer', 'OptimizationStrategy', 'OptimizationRecommendation',
    'BatchOptimizationResult', 'CostTrend', 'ForecastingMethod', 'get_cost_optimizer',
    
    # Analytics
    'CostAnalytics', 'AnalyticsReport', 'CostAlert', 'ReportType', 'AlertType',
    'get_cost_analytics',
    
    # Main system
    'CostSystem', 'get_cost_system'
]