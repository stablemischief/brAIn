"""
brAIn v2.0 Cost Analytics and Dashboard Data
Advanced cost analysis and dashboard data preparation for visualization.
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, date, timedelta
from enum import Enum
from uuid import UUID
from dataclasses import dataclass

from pydantic import BaseModel, Field
from ..monitoring.cost_calculator import CostCalculator, TokenUsage
from ..monitoring.budget_manager import BudgetManager, BudgetPeriod

logger = logging.getLogger(__name__)


class TimeGranularity(str, Enum):
    """Time granularity for analytics"""
    HOURLY = "hourly"
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"


class CostMetric(str, Enum):
    """Cost metric types"""
    TOTAL_COST = "total_cost"
    AVERAGE_COST = "average_cost"
    COST_PER_TOKEN = "cost_per_token"
    TOKEN_COUNT = "token_count"
    OPERATION_COUNT = "operation_count"


@dataclass
class TimeSeriesPoint:
    """Single point in a time series"""
    timestamp: datetime
    value: float
    metadata: Dict[str, Any]


class CostBreakdown(BaseModel):
    """Cost breakdown by different dimensions"""
    
    total_cost: float = Field(
        description="Total cost in USD",
        ge=0.0
    )
    
    by_model: Dict[str, float] = Field(
        default_factory=dict,
        description="Cost breakdown by model"
    )
    
    by_operation_type: Dict[str, float] = Field(
        default_factory=dict,
        description="Cost breakdown by operation type"
    )
    
    by_user: Dict[str, float] = Field(
        default_factory=dict,
        description="Cost breakdown by user"
    )
    
    by_time_period: Dict[str, float] = Field(
        default_factory=dict,
        description="Cost breakdown by time period"
    )
    
    token_usage: Dict[str, int] = Field(
        default_factory=dict,
        description="Token usage breakdown"
    )


class AnalyticsQuery(BaseModel):
    """Query parameters for cost analytics"""
    
    start_date: datetime = Field(
        description="Start date for analysis"
    )
    
    end_date: datetime = Field(
        description="End date for analysis"
    )
    
    user_ids: Optional[List[UUID]] = Field(
        default=None,
        description="Filter by specific users"
    )
    
    operation_types: Optional[List[str]] = Field(
        default=None,
        description="Filter by operation types"
    )
    
    model_names: Optional[List[str]] = Field(
        default=None,
        description="Filter by model names"
    )
    
    granularity: TimeGranularity = Field(
        default=TimeGranularity.DAILY,
        description="Time granularity for aggregation"
    )
    
    metrics: List[CostMetric] = Field(
        default_factory=lambda: [CostMetric.TOTAL_COST],
        description="Metrics to include in analysis"
    )


class DashboardData(BaseModel):
    """Dashboard data structure for frontend"""
    
    summary: Dict[str, Any] = Field(
        description="Summary statistics"
    )
    
    time_series: Dict[str, List[TimeSeriesPoint]] = Field(
        description="Time series data for charts"
    )
    
    breakdowns: CostBreakdown = Field(
        description="Cost breakdowns for pie charts"
    )
    
    trends: Dict[str, Any] = Field(
        description="Trend analysis"
    )
    
    alerts: List[Dict[str, Any]] = Field(
        description="Active cost alerts"
    )
    
    recommendations: List[Dict[str, Any]] = Field(
        description="Cost optimization recommendations"
    )


class CostAnalytics:
    """
    Advanced cost analytics engine for dashboard data preparation.
    
    Features:
    - Multi-dimensional cost analysis
    - Time series generation for charts
    - Trend analysis and forecasting
    - Real-time dashboard data
    - Cost optimization insights
    """
    
    def __init__(self):
        self.cost_calculator = CostCalculator()
        self.budget_manager = BudgetManager()
    
    def generate_dashboard_data(
        self,
        query: AnalyticsQuery,
        user_id: Optional[UUID] = None
    ) -> DashboardData:
        """
        Generate complete dashboard data for cost visualization.
        
        Args:
            query: Analytics query parameters
            user_id: Optional user ID for user-specific dashboard
            
        Returns:
            Complete dashboard data structure
        """
        # Generate summary statistics
        summary = self._generate_summary(query, user_id)
        
        # Generate time series data
        time_series = self._generate_time_series(query, user_id)
        
        # Generate cost breakdowns
        breakdowns = self._generate_cost_breakdowns(query, user_id)
        
        # Generate trend analysis
        trends = self._generate_trends(query, user_id)
        
        # Get active alerts
        alerts = self._get_active_alerts(user_id)
        
        # Get recommendations
        recommendations = self._get_recommendations(user_id)
        
        return DashboardData(
            summary=summary,
            time_series=time_series,
            breakdowns=breakdowns,
            trends=trends,
            alerts=alerts,
            recommendations=recommendations
        )
    
    def _generate_summary(
        self,
        query: AnalyticsQuery,
        user_id: Optional[UUID]
    ) -> Dict[str, Any]:
        """Generate summary statistics"""
        # Placeholder implementation - would integrate with database
        return {
            "total_cost": 156.75,
            "total_tokens": 1250000,
            "total_operations": 450,
            "average_cost_per_operation": 0.348,
            "average_tokens_per_operation": 2778,
            "cost_change_percentage": 12.5,
            "token_change_percentage": 8.3,
            "most_expensive_model": "gpt-4-turbo",
            "most_used_operation": "document_processing",
            "peak_usage_hour": 14,
            "current_month_spending": 425.30,
            "monthly_budget": 500.00,
            "budget_remaining": 74.70,
            "days_remaining_in_month": 8
        }
    
    def _generate_time_series(
        self,
        query: AnalyticsQuery,
        user_id: Optional[UUID]
    ) -> Dict[str, List[TimeSeriesPoint]]:
        """Generate time series data for charts"""
        time_series = {}
        
        # Generate sample time series data
        current = query.start_date
        daily_cost_series = []
        daily_token_series = []
        hourly_cost_series = []
        
        # Daily cost time series
        while current <= query.end_date:
            # Simulate daily cost data
            base_cost = 15.0
            variation = (current.weekday() + 1) * 2.5  # Weekday variation
            daily_cost = base_cost + variation
            
            daily_cost_series.append(TimeSeriesPoint(
                timestamp=current,
                value=daily_cost,
                metadata={
                    "operations": 25,
                    "primary_model": "gpt-4-turbo"
                }
            ))
            
            # Daily token series
            daily_token_series.append(TimeSeriesPoint(
                timestamp=current,
                value=daily_cost * 2500,  # Approximate tokens
                metadata={
                    "input_tokens": int(daily_cost * 1500),
                    "output_tokens": int(daily_cost * 1000)
                }
            ))
            
            current += timedelta(days=1)
        
        # Hourly cost series (last 24 hours)
        if query.granularity == TimeGranularity.HOURLY:
            current_hour = query.end_date.replace(minute=0, second=0, microsecond=0)
            for hour in range(24):
                hour_cost = 1.5 + (hour % 6) * 0.8  # Simulate hourly variation
                hourly_cost_series.append(TimeSeriesPoint(
                    timestamp=current_hour - timedelta(hours=23-hour),
                    value=hour_cost,
                    metadata={
                        "operations": max(1, hour % 5),
                        "peak_hour": hour in [9, 10, 14, 15]
                    }
                ))
        
        time_series["daily_cost"] = daily_cost_series
        time_series["daily_tokens"] = daily_token_series
        
        if hourly_cost_series:
            time_series["hourly_cost"] = hourly_cost_series
        
        return time_series
    
    def _generate_cost_breakdowns(
        self,
        query: AnalyticsQuery,
        user_id: Optional[UUID]
    ) -> CostBreakdown:
        """Generate cost breakdowns for pie charts"""
        return CostBreakdown(
            total_cost=156.75,
            by_model={
                "gpt-4-turbo": 89.25,
                "gpt-3.5-turbo": 45.50,
                "text-embedding-3-small": 12.00,
                "claude-3-sonnet-20240229": 10.00
            },
            by_operation_type={
                "document_processing": 95.30,
                "search": 25.75,
                "embedding_generation": 20.50,
                "ai_configuration": 15.20
            },
            by_user={
                str(user_id) if user_id else "system": 156.75
            },
            by_time_period={
                "week_1": 35.25,
                "week_2": 42.50,
                "week_3": 38.75,
                "week_4": 40.25
            },
            token_usage={
                "total_input_tokens": 750000,
                "total_output_tokens": 500000,
                "total_tokens": 1250000
            }
        )
    
    def _generate_trends(
        self,
        query: AnalyticsQuery,
        user_id: Optional[UUID]
    ) -> Dict[str, Any]:
        """Generate trend analysis"""
        return {
            "cost_trend": {
                "direction": "increasing",
                "percentage_change": 12.5,
                "trend_strength": "moderate",
                "period_comparison": "vs_last_month"
            },
            "token_trend": {
                "direction": "stable",
                "percentage_change": 2.3,
                "trend_strength": "weak",
                "period_comparison": "vs_last_month"
            },
            "efficiency_trend": {
                "cost_per_token": 0.000125,
                "change": -3.5,  # Improvement in efficiency
                "direction": "improving"
            },
            "usage_patterns": {
                "peak_days": ["Tuesday", "Wednesday", "Thursday"],
                "peak_hours": [9, 10, 14, 15],
                "seasonal_factor": 1.15
            },
            "forecasts": {
                "next_week_cost": 185.50,
                "next_month_cost": 742.30,
                "confidence_interval": [650.0, 850.0]
            }
        }
    
    def _get_active_alerts(self, user_id: Optional[UUID]) -> List[Dict[str, Any]]:
        """Get active cost alerts"""
        if not user_id:
            return []
        
        # Get budget status
        budget_status = self.budget_manager.get_budget_status(
            user_id, BudgetPeriod.MONTHLY
        )
        
        alerts = []
        
        if budget_status:
            if budget_status.percentage_used > 0.8:
                alerts.append({
                    "id": "budget_80_percent",
                    "type": "warning",
                    "title": "Budget 80% Used",
                    "message": f"You've used {budget_status.percentage_used*100:.1f}% of your monthly budget",
                    "severity": "warning",
                    "timestamp": datetime.utcnow().isoformat()
                })
            
            if budget_status.is_over_budget:
                alerts.append({
                    "id": "over_budget",
                    "type": "critical",
                    "title": "Over Budget",
                    "message": f"Current spending (${budget_status.current_spending:.2f}) exceeds budget limit (${budget_status.budget_limit:.2f})",
                    "severity": "critical",
                    "timestamp": datetime.utcnow().isoformat()
                })
            
            if budget_status.projected_spending and budget_status.projected_spending > budget_status.budget_limit:
                alerts.append({
                    "id": "projected_overage",
                    "type": "warning",
                    "title": "Projected Budget Overage",
                    "message": f"Projected spending (${budget_status.projected_spending:.2f}) may exceed budget",
                    "severity": "warning",
                    "timestamp": datetime.utcnow().isoformat()
                })
        
        return alerts
    
    def _get_recommendations(self, user_id: Optional[UUID]) -> List[Dict[str, Any]]:
        """Get cost optimization recommendations"""
        if not user_id:
            return []
        
        recommendations = self.budget_manager.get_cost_optimization_recommendations(user_id)
        
        return [
            {
                "id": f"rec_{i}",
                "category": rec.category,
                "title": rec.title,
                "description": rec.description,
                "potential_savings": rec.potential_savings,
                "implementation_effort": rec.implementation_effort,
                "priority": rec.priority,
                "status": "pending"
            }
            for i, rec in enumerate(recommendations)
        ]
    
    def generate_cost_report(
        self,
        query: AnalyticsQuery,
        user_id: Optional[UUID] = None,
        include_details: bool = True
    ) -> Dict[str, Any]:
        """
        Generate detailed cost report for export.
        
        Args:
            query: Analytics query parameters
            user_id: Optional user ID filter
            include_details: Whether to include detailed breakdowns
            
        Returns:
            Detailed cost report
        """
        report = {
            "report_metadata": {
                "generated_at": datetime.utcnow().isoformat(),
                "period": {
                    "start": query.start_date.isoformat(),
                    "end": query.end_date.isoformat()
                },
                "user_id": str(user_id) if user_id else None,
                "filters": {
                    "operation_types": query.operation_types,
                    "model_names": query.model_names
                }
            }
        }
        
        # Add summary
        report["summary"] = self._generate_summary(query, user_id)
        
        # Add detailed breakdowns if requested
        if include_details:
            report["detailed_breakdown"] = self._generate_cost_breakdowns(query, user_id)
            report["trends"] = self._generate_trends(query, user_id)
            
            # Add operation-level details (placeholder)
            report["operation_details"] = self._generate_operation_details(query, user_id)
        
        return report
    
    def _generate_operation_details(
        self,
        query: AnalyticsQuery,
        user_id: Optional[UUID]
    ) -> List[Dict[str, Any]]:
        """Generate detailed operation-level cost data"""
        # Placeholder - would integrate with actual database
        return [
            {
                "operation_id": "op_001",
                "timestamp": "2025-09-11T10:30:00Z",
                "operation_type": "document_processing",
                "model_name": "gpt-4-turbo",
                "input_tokens": 2500,
                "output_tokens": 800,
                "cost": 0.098,
                "duration_ms": 3500,
                "success": True
            },
            {
                "operation_id": "op_002",
                "timestamp": "2025-09-11T11:15:00Z",
                "operation_type": "search",
                "model_name": "text-embedding-3-small",
                "input_tokens": 150,
                "output_tokens": 0,
                "cost": 0.003,
                "duration_ms": 250,
                "success": True
            }
        ]
    
    def calculate_roi_metrics(
        self,
        query: AnalyticsQuery,
        user_id: Optional[UUID] = None
    ) -> Dict[str, Any]:
        """
        Calculate ROI and efficiency metrics.
        
        Args:
            query: Analytics query parameters
            user_id: Optional user ID filter
            
        Returns:
            ROI and efficiency metrics
        """
        # Placeholder implementation
        total_cost = 156.75
        total_operations = 450
        total_documents_processed = 125
        
        return {
            "cost_efficiency": {
                "cost_per_operation": total_cost / total_operations,
                "cost_per_document": total_cost / total_documents_processed,
                "tokens_per_dollar": 1250000 / total_cost,
                "operations_per_dollar": total_operations / total_cost
            },
            "productivity_metrics": {
                "documents_per_hour": 5.2,
                "average_processing_time": 45.0,  # seconds
                "success_rate": 0.98,
                "retry_rate": 0.02
            },
            "quality_metrics": {
                "average_quality_score": 0.89,
                "high_quality_operations": 0.75,
                "quality_improvement_trend": 0.05
            },
            "roi_analysis": {
                "automation_savings": 2500.0,  # USD saved through automation
                "efficiency_gain": 3.2,  # multiplier vs manual process
                "payback_period_months": 2.5
            }
        }


# Global analytics instance
_cost_analytics = CostAnalytics()


def get_cost_analytics() -> CostAnalytics:
    """Get the global cost analytics instance"""
    return _cost_analytics


# Convenience functions for common analytics operations
def generate_daily_cost_summary(
    user_id: Optional[UUID] = None,
    days: int = 30
) -> Dict[str, Any]:
    """Generate daily cost summary for the last N days"""
    end_date = datetime.utcnow()
    start_date = end_date - timedelta(days=days)
    
    query = AnalyticsQuery(
        start_date=start_date,
        end_date=end_date,
        user_ids=[user_id] if user_id else None,
        granularity=TimeGranularity.DAILY,
        metrics=[CostMetric.TOTAL_COST, CostMetric.TOKEN_COUNT]
    )
    
    analytics = get_cost_analytics()
    return analytics.generate_dashboard_data(query, user_id).model_dump()


def generate_model_comparison_report(
    user_id: Optional[UUID] = None,
    days: int = 30
) -> Dict[str, Any]:
    """Generate model comparison report"""
    end_date = datetime.utcnow()
    start_date = end_date - timedelta(days=days)
    
    query = AnalyticsQuery(
        start_date=start_date,
        end_date=end_date,
        user_ids=[user_id] if user_id else None,
        granularity=TimeGranularity.DAILY
    )
    
    analytics = get_cost_analytics()
    dashboard_data = analytics.generate_dashboard_data(query, user_id)
    
    return {
        "model_costs": dashboard_data.breakdowns.by_model,
        "model_efficiency": {
            model: {
                "total_cost": cost,
                "estimated_tokens": int(cost * 8000),  # Rough estimate
                "cost_per_1k_tokens": cost / (cost * 8000 / 1000)
            }
            for model, cost in dashboard_data.breakdowns.by_model.items()
        },
        "recommendations": dashboard_data.recommendations
    }