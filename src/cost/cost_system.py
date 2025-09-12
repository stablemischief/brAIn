"""
Unified Cost Management System for brAIn v2.0
Integrates all cost optimization components into a single, cohesive system

This module provides the main interface for all cost-related operations,
combining token counting, cost calculation, budget enforcement, optimization,
and analytics into a unified system.
"""

import logging
import asyncio
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from decimal import Decimal
import json

from .token_counter import TokenCounter, TokenCount, get_token_counter
from .cost_calculator import (
    CostCalculator, CostBreakdown, CostCategory, BudgetPeriod,
    get_cost_calculator
)
from .budget_manager import (
    BudgetEnforcer, BudgetConfig, BudgetRule, BudgetViolation,
    EnforcementAction, get_budget_enforcer
)
from .optimizer import (
    CostOptimizer, OptimizationStrategy, OptimizationRecommendation,
    BatchOptimizationResult, get_cost_optimizer
)
from .analytics import (
    CostAnalytics, AnalyticsReport, CostAlert, ReportType,
    get_cost_analytics
)

logger = logging.getLogger(__name__)


@dataclass
class CostSystemConfig:
    """Configuration for the cost management system"""
    # Budget settings
    daily_budget_limit: Optional[Decimal] = None
    monthly_budget_limit: Optional[Decimal] = None
    enable_budget_enforcement: bool = True
    
    # Optimization settings
    default_optimization_strategy: OptimizationStrategy = OptimizationStrategy.BALANCED
    enable_auto_optimization: bool = True
    quality_threshold: float = 0.8
    
    # Analytics settings
    enable_real_time_monitoring: bool = True
    alert_thresholds: Dict[str, float] = field(default_factory=lambda: {
        'cost_spike_multiplier': 2.0,
        'efficiency_threshold': 0.7,
        'budget_warning_threshold': 0.8
    })
    
    # System settings
    enable_caching: bool = True
    cache_ttl_minutes: int = 15
    max_history_days: int = 90


class CostSystem:
    """
    Unified Cost Management System
    
    Provides a single interface for all cost-related operations including:
    - Token counting and cost calculation
    - Budget enforcement and monitoring
    - Cost optimization and recommendations  
    - Analytics and reporting
    - Real-time cost tracking
    """
    
    def __init__(self, config: Optional[CostSystemConfig] = None):
        self.config = config or CostSystemConfig()
        
        # Initialize core components
        self.token_counter = get_token_counter()
        self.cost_calculator = get_cost_calculator()
        
        # Initialize budget system if enabled
        if self.config.enable_budget_enforcement:
            budget_config = self._create_budget_config()
            self.budget_enforcer = get_budget_enforcer(
                self.cost_calculator, 
                alert_callbacks=[self._handle_budget_alert]
            )
            if budget_config:
                self._setup_default_budget_rules(budget_config)
        else:
            self.budget_enforcer = None
        
        # Initialize optimizer
        self.optimizer = get_cost_optimizer(self.cost_calculator)
        
        # Initialize analytics
        self.analytics = get_cost_analytics(
            self.cost_calculator, 
            self.budget_enforcer, 
            self.optimizer
        )
        
        # System state
        self._operation_cache = {} if self.config.enable_caching else None
        self._alert_callbacks: List[Callable] = []
        self._optimization_cache = {}
        
        logger.info("Cost Management System initialized")
    
    async def process_operation(
        self,
        content: str,
        operation_type: str,
        operation_id: Optional[str] = None,
        model: Optional[str] = None,
        category: Optional[CostCategory] = None,
        session_id: Optional[str] = None,
        enable_optimization: bool = True,
        quality_threshold: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Process a single operation with full cost management
        
        Returns comprehensive results including costs, optimization, and compliance
        """
        
        operation_id = operation_id or f"op_{int(datetime.now().timestamp())}"
        
        try:
            # Step 1: Optimization (if enabled)
            recommended_model = model
            optimization_result = None
            
            if enable_optimization and self.config.enable_auto_optimization:
                try:
                    optimization_result = self.optimizer.optimize_operation(
                        content=content,
                        operation_type=operation_type,
                        strategy=self.config.default_optimization_strategy,
                        quality_threshold=quality_threshold or self.config.quality_threshold
                    )
                    recommended_model = optimization_result.recommended_model
                    logger.info(f"Optimized model selection: {recommended_model} for operation {operation_id}")
                except Exception as e:
                    logger.warning(f"Optimization failed for operation {operation_id}: {e}")
                    recommended_model = model or "gpt-4o-mini"  # Fallback
            
            # Step 2: Token counting and cost calculation
            token_count = self.token_counter.estimate_processing_tokens(
                content, operation_type, recommended_model or "gpt-4o-mini"
            )
            
            cost_category = category or self._infer_category_from_operation_type(operation_type)
            cost_breakdown = self.cost_calculator.calculate_operation_cost(
                token_count, cost_category, operation_id
            )
            
            # Step 3: Budget compliance check
            budget_compliance = {"allowed": True, "warnings": [], "violations": []}
            
            if self.budget_enforcer:
                allowed, violations, approval_requests = self.budget_enforcer.check_budget_compliance(
                    cost_breakdown, session_id
                )
                
                budget_compliance = {
                    "allowed": allowed,
                    "violations": [asdict(v) for v in violations] if violations else [],
                    "approval_requests": [asdict(req) for req in approval_requests] if approval_requests else []
                }
                
                if not allowed:
                    logger.warning(f"Operation {operation_id} blocked by budget enforcement")
            
            # Step 4: Real-time monitoring and alerts
            alerts = []
            if self.config.enable_real_time_monitoring:
                alerts = self.analytics.detect_cost_anomalies()
            
            # Step 5: Cache results if enabled
            if self._operation_cache is not None:
                cache_key = f"{operation_type}_{hash(content)}_{recommended_model}"
                self._operation_cache[cache_key] = {
                    'cost_breakdown': cost_breakdown,
                    'optimization': optimization_result,
                    'timestamp': datetime.now()
                }
            
            # Return comprehensive results
            return {
                'operation_id': operation_id,
                'status': 'completed' if budget_compliance['allowed'] else 'blocked',
                'model_used': recommended_model,
                'cost_breakdown': {
                    'total_cost': float(cost_breakdown.total_cost),
                    'input_tokens': cost_breakdown.input_tokens,
                    'output_tokens': cost_breakdown.output_tokens,
                    'input_cost': float(cost_breakdown.input_cost),
                    'output_cost': float(cost_breakdown.output_cost),
                    'model': cost_breakdown.model,
                    'provider': cost_breakdown.provider.value,
                    'category': cost_breakdown.category.value
                },
                'optimization': {
                    'enabled': enable_optimization,
                    'recommendation': asdict(optimization_result) if optimization_result else None,
                    'model_optimized': recommended_model != model if model else False
                },
                'budget_compliance': budget_compliance,
                'alerts': [asdict(alert) for alert in alerts] if alerts else [],
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error processing operation {operation_id}: {e}")
            return {
                'operation_id': operation_id,
                'status': 'error',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    async def process_batch(
        self,
        operations: List[Dict[str, Any]],
        batch_id: Optional[str] = None,
        optimization_strategy: OptimizationStrategy = OptimizationStrategy.BATCH_OPTIMIZED,
        enable_optimization: bool = True
    ) -> Dict[str, Any]:
        """
        Process multiple operations as a batch with optimization
        
        operations: List of {'content': str, 'type': str, 'priority': int?, ...}
        """
        
        batch_id = batch_id or f"batch_{int(datetime.now().timestamp())}"
        
        try:
            # Step 1: Batch optimization
            optimization_result = None
            if enable_optimization:
                try:
                    optimization_result = self.optimizer.optimize_batch_processing(
                        operations, optimization_strategy
                    )
                    logger.info(f"Batch optimization completed: {optimization_result.savings_percentage:.1f}% savings")
                except Exception as e:
                    logger.warning(f"Batch optimization failed: {e}")
            
            # Step 2: Process individual operations
            operation_results = []
            total_cost = Decimal('0')
            
            for i, op in enumerate(operations):
                op_id = f"{batch_id}_op_{i}"
                
                # Use optimized model if available
                recommended_model = None
                if optimization_result:
                    recommended_model = optimization_result.model_assignments.get(f"op_{i}")
                
                result = await self.process_operation(
                    content=op['content'],
                    operation_type=op['type'],
                    operation_id=op_id,
                    model=recommended_model,
                    enable_optimization=False  # Already optimized at batch level
                )
                
                operation_results.append(result)
                if result['status'] == 'completed':
                    total_cost += Decimal(str(result['cost_breakdown']['total_cost']))
            
            # Step 3: Batch analytics
            successful_operations = [r for r in operation_results if r['status'] == 'completed']
            failed_operations = [r for r in operation_results if r['status'] != 'completed']
            
            return {
                'batch_id': batch_id,
                'status': 'completed',
                'summary': {
                    'total_operations': len(operations),
                    'successful_operations': len(successful_operations),
                    'failed_operations': len(failed_operations),
                    'total_cost': float(total_cost),
                    'average_cost_per_operation': float(total_cost / len(successful_operations)) if successful_operations else 0
                },
                'optimization': {
                    'enabled': enable_optimization,
                    'result': asdict(optimization_result) if optimization_result else None
                },
                'operations': operation_results,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error processing batch {batch_id}: {e}")
            return {
                'batch_id': batch_id,
                'status': 'error',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def get_real_time_dashboard(self) -> Dict[str, Any]:
        """Get real-time cost metrics for dashboard display"""
        
        try:
            # Get core metrics
            metrics = self.analytics.get_real_time_metrics()
            
            # Get budget status
            budget_status = {}
            if self.budget_enforcer:
                budget_status = self.budget_enforcer.get_budget_status()
            
            # Get recent optimization insights
            optimization_insights = self.optimizer.get_optimization_insights()
            
            # Get active alerts
            active_alerts = [
                asdict(alert) for alert in self.analytics.alerts.values()
                if not alert.resolved
            ]
            
            return {
                'timestamp': datetime.now().isoformat(),
                'status': 'operational',
                'metrics': metrics,
                'budget_status': budget_status,
                'optimization_insights': optimization_insights,
                'active_alerts': active_alerts,
                'system_health': {
                    'components_active': 4,  # token_counter, cost_calculator, budget_enforcer, optimizer
                    'cache_size': len(self._operation_cache) if self._operation_cache else 0,
                    'history_size': len(self.cost_calculator.cost_history)
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting dashboard data: {e}")
            return {
                'timestamp': datetime.now().isoformat(),
                'status': 'error',
                'error': str(e)
            }
    
    def generate_cost_report(
        self,
        report_type: ReportType,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> AnalyticsReport:
        """Generate comprehensive cost report"""
        
        return self.analytics.generate_report(report_type, start_date, end_date)
    
    def add_budget_rule(self, rule: BudgetRule) -> bool:
        """Add a new budget rule to the system"""
        
        if not self.budget_enforcer:
            logger.warning("Budget enforcement is disabled")
            return False
        
        return self.budget_enforcer.add_budget_rule(rule)
    
    def get_cost_forecast(
        self,
        period: BudgetPeriod,
        confidence_threshold: float = 0.7
    ) -> Optional[Dict[str, Any]]:
        """Get cost forecast for specified period"""
        
        forecast = self.optimizer.forecast_costs(period, confidence_threshold=confidence_threshold)
        
        if forecast:
            return {
                'period': forecast.period.value,
                'projected_cost': float(forecast.projected_cost),
                'confidence': forecast.confidence,
                'trend': forecast.trend,
                'factors': forecast.factors,
                'generated_at': forecast.generated_at.isoformat()
            }
        
        return None
    
    def register_alert_callback(self, callback: Callable[[Dict[str, Any]], None]):
        """Register callback for cost alerts"""
        
        self._alert_callbacks.append(callback)
        logger.info("Alert callback registered")
    
    def cleanup_old_data(self, days_to_keep: Optional[int] = None):
        """Clean up old cost data beyond retention period"""
        
        days = days_to_keep or self.config.max_history_days
        cutoff = datetime.now() - timedelta(days=days)
        
        # Clean up cost history
        original_count = len(self.cost_calculator.cost_history)
        self.cost_calculator.cost_history = [
            cost for cost in self.cost_calculator.cost_history
            if cost.timestamp >= cutoff
        ]
        
        cleaned_count = original_count - len(self.cost_calculator.cost_history)
        
        # Clean up cache
        if self._operation_cache:
            cache_cutoff = datetime.now() - timedelta(minutes=self.config.cache_ttl_minutes)
            old_keys = [
                key for key, data in self._operation_cache.items()
                if data['timestamp'] < cache_cutoff
            ]
            for key in old_keys:
                del self._operation_cache[key]
        
        logger.info(f"Cleaned up {cleaned_count} old cost records")
    
    def export_data(
        self,
        start_date: datetime,
        end_date: datetime,
        format: str = "json"
    ) -> Union[str, Dict[str, Any]]:
        """Export cost data for external analysis"""
        
        return self.analytics.export_cost_data(start_date, end_date, format)
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        
        return {
            'timestamp': datetime.now().isoformat(),
            'version': '2.0',
            'config': {
                'budget_enforcement_enabled': self.config.enable_budget_enforcement,
                'auto_optimization_enabled': self.config.enable_auto_optimization,
                'real_time_monitoring_enabled': self.config.enable_real_time_monitoring,
                'caching_enabled': self.config.enable_caching
            },
            'components': {
                'token_counter': 'active',
                'cost_calculator': 'active',
                'budget_enforcer': 'active' if self.budget_enforcer else 'disabled',
                'optimizer': 'active',
                'analytics': 'active'
            },
            'statistics': {
                'total_operations': len(self.cost_calculator.cost_history),
                'total_cost': float(sum(c.total_cost for c in self.cost_calculator.cost_history)),
                'cache_size': len(self._operation_cache) if self._operation_cache else 0,
                'active_alerts': len([a for a in self.analytics.alerts.values() if not a.resolved])
            }
        }
    
    def _create_budget_config(self) -> Optional[BudgetConfig]:
        """Create budget configuration from system config"""
        
        if not any([self.config.daily_budget_limit, self.config.monthly_budget_limit]):
            return None
        
        return BudgetConfig(
            daily_limit=self.config.daily_budget_limit,
            monthly_limit=self.config.monthly_budget_limit,
            hard_stop_enabled=True,
            grace_percentage=0.1
        )
    
    def _setup_default_budget_rules(self, budget_config: BudgetConfig):
        """Set up default budget rules"""
        
        if budget_config.daily_limit or budget_config.monthly_limit:
            self.budget_enforcer.create_default_rules(
                daily_limit=budget_config.daily_limit or Decimal('10.00'),
                monthly_limit=budget_config.monthly_limit or Decimal('300.00')
            )
    
    def _handle_budget_alert(self, alert_data: Dict[str, Any]):
        """Handle budget alerts from the enforcer"""
        
        logger.warning(f"Budget alert: {alert_data}")
        
        # Notify registered callbacks
        for callback in self._alert_callbacks:
            try:
                callback(alert_data)
            except Exception as e:
                logger.error(f"Alert callback failed: {e}")
    
    def _infer_category_from_operation_type(self, operation_type: str) -> CostCategory:
        """Infer cost category from operation type"""
        
        mapping = {
            'extraction': CostCategory.EXTRACTION,
            'summarization': CostCategory.SUMMARIZATION,
            'classification': CostCategory.CLASSIFICATION,
            'embedding': CostCategory.EMBEDDING,
            'chat': CostCategory.CHAT,
            'analysis': CostCategory.ANALYSIS,
        }
        
        return mapping.get(operation_type.lower(), CostCategory.PROCESSING)


# Global system instance
_cost_system: Optional[CostSystem] = None


def get_cost_system(config: Optional[CostSystemConfig] = None) -> CostSystem:
    """Get the global cost system instance"""
    global _cost_system
    if _cost_system is None:
        _cost_system = CostSystem(config)
    return _cost_system


# Convenience functions for common operations
async def process_document(
    content: str,
    operation_type: str = "processing",
    enable_optimization: bool = True,
    **kwargs
) -> Dict[str, Any]:
    """Process a single document with cost management"""
    
    system = get_cost_system()
    return await system.process_operation(
        content=content,
        operation_type=operation_type,
        enable_optimization=enable_optimization,
        **kwargs
    )


async def process_document_batch(
    documents: List[Dict[str, Any]],
    enable_optimization: bool = True,
    **kwargs
) -> Dict[str, Any]:
    """Process multiple documents as an optimized batch"""
    
    system = get_cost_system()
    return await system.process_batch(
        operations=documents,
        enable_optimization=enable_optimization,
        **kwargs
    )


def get_dashboard_data() -> Dict[str, Any]:
    """Get real-time dashboard data"""
    
    system = get_cost_system()
    return system.get_real_time_dashboard()


def setup_cost_system(
    daily_budget: Optional[float] = None,
    monthly_budget: Optional[float] = None,
    enable_optimization: bool = True,
    **kwargs
) -> CostSystem:
    """Set up the cost system with specified configuration"""
    
    config = CostSystemConfig(
        daily_budget_limit=Decimal(str(daily_budget)) if daily_budget else None,
        monthly_budget_limit=Decimal(str(monthly_budget)) if monthly_budget else None,
        enable_auto_optimization=enable_optimization,
        **kwargs
    )
    
    return get_cost_system(config)