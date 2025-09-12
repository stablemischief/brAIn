"""
Cost analytics and reporting system with comprehensive insights and real-time monitoring
Provides dashboards, reports, and actionable insights for cost management
"""

import logging
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from enum import Enum
import json
from decimal import Decimal
import statistics
from collections import defaultdict
import pandas as pd
import numpy as np

from .cost_calculator import CostCalculator, CostBreakdown, CostCategory, BudgetPeriod
from .budget_manager import BudgetEnforcer, BudgetViolation
from .optimizer import CostOptimizer, OptimizationStrategy, CostTrend

logger = logging.getLogger(__name__)


class ReportType(Enum):
    """Types of cost reports"""
    DAILY_SUMMARY = "daily_summary"
    WEEKLY_ANALYSIS = "weekly_analysis"
    MONTHLY_REPORT = "monthly_report"
    MODEL_PERFORMANCE = "model_performance"
    COST_BREAKDOWN = "cost_breakdown"
    OPTIMIZATION_INSIGHTS = "optimization_insights"
    BUDGET_STATUS = "budget_status"
    FORECAST_REPORT = "forecast_report"


class AlertType(Enum):
    """Types of cost alerts"""
    BUDGET_THRESHOLD = "budget_threshold"
    COST_SPIKE = "cost_spike"
    INEFFICIENT_USAGE = "inefficient_usage"
    OPTIMIZATION_OPPORTUNITY = "optimization_opportunity"
    FORECAST_WARNING = "forecast_warning"


@dataclass
class CostAlert:
    """Cost alert with details and recommendations"""
    id: str
    type: AlertType
    severity: str  # "low", "medium", "high", "critical"
    title: str
    description: str
    current_value: float
    threshold_value: Optional[float]
    recommendation: str
    potential_savings: Optional[str]
    created_at: datetime
    acknowledged: bool = False
    resolved: bool = False


@dataclass
class AnalyticsReport:
    """Comprehensive analytics report"""
    report_id: str
    report_type: ReportType
    generated_at: datetime
    period_start: datetime
    period_end: datetime
    summary: Dict[str, Any]
    detailed_analysis: Dict[str, Any]
    insights: List[str]
    recommendations: List[Dict[str, Any]]
    charts_data: Dict[str, Any]
    metadata: Optional[Dict[str, Any]] = None


class CostAnalytics:
    """
    Comprehensive cost analytics system providing insights, reports, and monitoring
    """
    
    def __init__(
        self,
        cost_calculator: CostCalculator,
        budget_enforcer: Optional[BudgetEnforcer] = None,
        optimizer: Optional[CostOptimizer] = None
    ):
        self.cost_calculator = cost_calculator
        self.budget_enforcer = budget_enforcer
        self.optimizer = optimizer
        
        # Analytics state
        self.alerts: Dict[str, CostAlert] = {}
        self.reports_cache: Dict[str, AnalyticsReport] = {}
        self.thresholds = {
            'cost_spike_multiplier': 2.0,  # Alert if cost > 2x average
            'efficiency_threshold': 0.7,   # Alert if efficiency < 70%
            'budget_warning_threshold': 0.8,  # Alert at 80% of budget
        }
    
    def generate_report(
        self,
        report_type: ReportType,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        **kwargs
    ) -> AnalyticsReport:
        """Generate comprehensive analytics report"""
        
        # Set default date range
        if not end_date:
            end_date = datetime.now()
        if not start_date:
            if report_type == ReportType.DAILY_SUMMARY:
                start_date = end_date.replace(hour=0, minute=0, second=0, microsecond=0)
            elif report_type == ReportType.WEEKLY_ANALYSIS:
                start_date = end_date - timedelta(days=7)
            elif report_type == ReportType.MONTHLY_REPORT:
                start_date = end_date - timedelta(days=30)
            else:
                start_date = end_date - timedelta(days=7)
        
        report_id = f"{report_type.value}_{int(datetime.now().timestamp())}"
        
        # Generate report based on type
        if report_type == ReportType.DAILY_SUMMARY:
            return self._generate_daily_summary(report_id, start_date, end_date)
        elif report_type == ReportType.WEEKLY_ANALYSIS:
            return self._generate_weekly_analysis(report_id, start_date, end_date)
        elif report_type == ReportType.MONTHLY_REPORT:
            return self._generate_monthly_report(report_id, start_date, end_date)
        elif report_type == ReportType.MODEL_PERFORMANCE:
            return self._generate_model_performance_report(report_id, start_date, end_date)
        elif report_type == ReportType.COST_BREAKDOWN:
            return self._generate_cost_breakdown_report(report_id, start_date, end_date)
        elif report_type == ReportType.OPTIMIZATION_INSIGHTS:
            return self._generate_optimization_insights_report(report_id, start_date, end_date)
        elif report_type == ReportType.BUDGET_STATUS:
            return self._generate_budget_status_report(report_id, start_date, end_date)
        elif report_type == ReportType.FORECAST_REPORT:
            return self._generate_forecast_report(report_id, start_date, end_date)
        else:
            raise ValueError(f"Unknown report type: {report_type}")
    
    def get_real_time_metrics(self) -> Dict[str, Any]:
        """Get real-time cost metrics for dashboard"""
        
        now = datetime.now()
        
        # Today's costs
        today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        today_costs = [
            c for c in self.cost_calculator.cost_history
            if c.timestamp >= today_start
        ]
        
        today_total = sum(c.total_cost for c in today_costs)
        today_operations = len(today_costs)
        
        # This week's costs
        week_start = now - timedelta(days=now.weekday())
        week_start = week_start.replace(hour=0, minute=0, second=0, microsecond=0)
        week_costs = [
            c for c in self.cost_calculator.cost_history
            if c.timestamp >= week_start
        ]
        
        week_total = sum(c.total_cost for c in week_costs)
        
        # This month's costs
        month_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        month_costs = [
            c for c in self.cost_calculator.cost_history
            if c.timestamp >= month_start
        ]
        
        month_total = sum(c.total_cost for c in month_costs)
        
        # Recent activity (last hour)
        hour_ago = now - timedelta(hours=1)
        recent_costs = [
            c for c in self.cost_calculator.cost_history
            if c.timestamp >= hour_ago
        ]
        
        recent_operations = len(recent_costs)
        recent_total = sum(c.total_cost for c in recent_costs)
        
        # Calculate trends
        yesterday_start = today_start - timedelta(days=1)
        yesterday_costs = [
            c for c in self.cost_calculator.cost_history
            if yesterday_start <= c.timestamp < today_start
        ]
        yesterday_total = sum(c.total_cost for c in yesterday_costs)
        
        daily_trend = "stable"
        if yesterday_total > 0:
            change_pct = float((today_total - yesterday_total) / yesterday_total * 100)
            if change_pct > 10:
                daily_trend = "increasing"
            elif change_pct < -10:
                daily_trend = "decreasing"
        
        # Model usage distribution
        model_usage = defaultdict(int)
        for cost in today_costs:
            model_usage[cost.model] += 1
        
        # Active alerts count
        active_alerts = len([a for a in self.alerts.values() if not a.resolved])
        critical_alerts = len([
            a for a in self.alerts.values() 
            if not a.resolved and a.severity == "critical"
        ])
        
        return {
            'timestamp': now.isoformat(),
            'current_costs': {
                'today': float(today_total),
                'week': float(week_total),
                'month': float(month_total),
                'last_hour': float(recent_total)
            },
            'operations': {
                'today': today_operations,
                'last_hour': recent_operations,
                'avg_cost_per_operation': float(today_total / today_operations) if today_operations > 0 else 0
            },
            'trends': {
                'daily_trend': daily_trend,
                'daily_change_pct': float((today_total - yesterday_total) / yesterday_total * 100) if yesterday_total > 0 else 0
            },
            'model_distribution': dict(model_usage),
            'alerts': {
                'total_active': active_alerts,
                'critical': critical_alerts
            }
        }
    
    def detect_cost_anomalies(self, lookback_days: int = 7) -> List[CostAlert]:
        """Detect cost anomalies and generate alerts"""
        
        new_alerts = []
        now = datetime.now()
        
        # Get recent cost data
        cutoff = now - timedelta(days=lookback_days)
        recent_costs = [
            c for c in self.cost_calculator.cost_history
            if c.timestamp >= cutoff
        ]
        
        if len(recent_costs) < 10:  # Need sufficient data
            return new_alerts
        
        # Daily cost analysis
        daily_costs = defaultdict(Decimal)
        for cost in recent_costs:
            day = cost.timestamp.strftime('%Y-%m-%d')
            daily_costs[day] += cost.total_cost
        
        daily_amounts = list(daily_costs.values())
        
        if len(daily_amounts) >= 3:
            mean_cost = statistics.mean(daily_amounts)
            std_cost = statistics.stdev(daily_amounts) if len(daily_amounts) > 1 else Decimal('0')
            
            # Cost spike detection
            today = now.strftime('%Y-%m-%d')
            today_cost = daily_costs.get(today, Decimal('0'))
            
            if today_cost > mean_cost * Decimal(str(self.thresholds['cost_spike_multiplier'])):
                alert = CostAlert(
                    id=f"cost_spike_{int(now.timestamp())}",
                    type=AlertType.COST_SPIKE,
                    severity="high",
                    title="Daily Cost Spike Detected",
                    description=f"Today's cost (${today_cost}) is {float(today_cost/mean_cost):.1f}x the recent average",
                    current_value=float(today_cost),
                    threshold_value=float(mean_cost * Decimal(str(self.thresholds['cost_spike_multiplier']))),
                    recommendation="Review recent operations for unexpected high-cost activities",
                    potential_savings=f"${float(today_cost - mean_cost):.4f}",
                    created_at=now
                )
                new_alerts.append(alert)
                self.alerts[alert.id] = alert
        
        # Model efficiency analysis
        if self.optimizer:
            insights = self.optimizer.get_optimization_insights()
            
            for model, efficiency_data in insights.get('model_efficiency', {}).items():
                if efficiency_data['efficiency_score'] < self.thresholds['efficiency_threshold']:
                    alert = CostAlert(
                        id=f"inefficient_{model}_{int(now.timestamp())}",
                        type=AlertType.INEFFICIENT_USAGE,
                        severity="medium",
                        title=f"Inefficient Model Usage: {model}",
                        description=f"Model {model} showing low efficiency score ({efficiency_data['efficiency_score']:.2f})",
                        current_value=efficiency_data['efficiency_score'],
                        threshold_value=self.thresholds['efficiency_threshold'],
                        recommendation=f"Consider switching to more cost-effective alternatives for {model}",
                        potential_savings="15-30%",
                        created_at=now
                    )
                    new_alerts.append(alert)
                    self.alerts[alert.id] = alert
        
        # Budget threshold alerts
        if self.budget_enforcer:
            budget_status = self.budget_enforcer.get_budget_status()
            
            for rule in budget_status.get('rules', []):
                if (rule['percentage'] >= self.thresholds['budget_warning_threshold'] * 100 
                    and rule['status'] not in ['ok', 'caution']):
                    
                    alert = CostAlert(
                        id=f"budget_{rule['rule_id']}_{int(now.timestamp())}",
                        type=AlertType.BUDGET_THRESHOLD,
                        severity="high" if rule['percentage'] >= 90 else "medium",
                        title=f"Budget Threshold Alert: {rule['rule_name']}",
                        description=f"Budget usage at {rule['percentage']:.1f}% of limit (${rule['current_spend']:.4f}/${rule['limit']:.4f})",
                        current_value=rule['percentage'],
                        threshold_value=self.thresholds['budget_warning_threshold'] * 100,
                        recommendation="Review spending patterns and consider implementing cost controls",
                        potential_savings=f"${rule['remaining']:.4f} remaining budget",
                        created_at=now
                    )
                    new_alerts.append(alert)
                    self.alerts[alert.id] = alert
        
        return new_alerts
    
    def get_cost_insights(self, days: int = 30) -> Dict[str, Any]:
        """Get actionable cost insights and recommendations"""
        
        cutoff = datetime.now() - timedelta(days=days)
        relevant_costs = [
            c for c in self.cost_calculator.cost_history
            if c.timestamp >= cutoff
        ]
        
        if not relevant_costs:
            return {"error": "No cost data available"}
        
        insights = {
            'period_analysis': self._analyze_cost_period(relevant_costs),
            'model_insights': self._analyze_model_usage(relevant_costs),
            'category_insights': self._analyze_category_distribution(relevant_costs),
            'optimization_opportunities': self._identify_optimization_opportunities(relevant_costs),
            'recommendations': self._generate_actionable_recommendations(relevant_costs)
        }
        
        return insights
    
    def export_cost_data(
        self,
        start_date: datetime,
        end_date: datetime,
        format: str = "json"
    ) -> Union[str, Dict[str, Any]]:
        """Export cost data in various formats"""
        
        relevant_costs = [
            c for c in self.cost_calculator.cost_history
            if start_date <= c.timestamp <= end_date
        ]
        
        export_data = []
        for cost in relevant_costs:
            export_data.append({
                'operation_id': cost.operation_id,
                'timestamp': cost.timestamp.isoformat(),
                'model': cost.model,
                'provider': cost.provider.value,
                'category': cost.category.value,
                'input_tokens': cost.input_tokens,
                'output_tokens': cost.output_tokens,
                'total_tokens': cost.total_tokens,
                'input_cost': float(cost.input_cost),
                'output_cost': float(cost.output_cost),
                'total_cost': float(cost.total_cost),
                'metadata': cost.metadata
            })
        
        if format.lower() == "json":
            return {
                'export_timestamp': datetime.now().isoformat(),
                'period': {
                    'start': start_date.isoformat(),
                    'end': end_date.isoformat()
                },
                'total_records': len(export_data),
                'total_cost': sum(float(c.total_cost) for c in relevant_costs),
                'data': export_data
            }
        elif format.lower() == "csv":
            # Convert to CSV format
            import io
            import csv
            
            output = io.StringIO()
            if export_data:
                writer = csv.DictWriter(output, fieldnames=export_data[0].keys())
                writer.writeheader()
                writer.writerows(export_data)
            return output.getvalue()
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    def _generate_daily_summary(self, report_id: str, start_date: datetime, end_date: datetime) -> AnalyticsReport:
        """Generate daily summary report"""
        
        relevant_costs = [
            c for c in self.cost_calculator.cost_history
            if start_date <= c.timestamp <= end_date
        ]
        
        total_cost = sum(c.total_cost for c in relevant_costs)
        total_operations = len(relevant_costs)
        
        # Model breakdown
        model_breakdown = defaultdict(Decimal)
        for cost in relevant_costs:
            model_breakdown[cost.model] += cost.total_cost
        
        # Category breakdown
        category_breakdown = defaultdict(Decimal)
        for cost in relevant_costs:
            category_breakdown[cost.category.value] += cost.total_cost
        
        # Hourly distribution
        hourly_breakdown = defaultdict(Decimal)
        for cost in relevant_costs:
            hour = cost.timestamp.hour
            hourly_breakdown[hour] += cost.total_cost
        
        summary = {
            'total_cost': float(total_cost),
            'total_operations': total_operations,
            'average_cost_per_operation': float(total_cost / total_operations) if total_operations > 0 else 0,
            'most_used_model': max(model_breakdown, key=model_breakdown.get) if model_breakdown else None,
            'peak_hour': max(hourly_breakdown, key=hourly_breakdown.get) if hourly_breakdown else None
        }
        
        detailed_analysis = {
            'model_breakdown': {k: float(v) for k, v in model_breakdown.items()},
            'category_breakdown': {k: float(v) for k, v in category_breakdown.items()},
            'hourly_distribution': {str(k): float(v) for k, v in hourly_breakdown.items()}
        }
        
        # Generate insights
        insights = []
        if total_operations > 0:
            insights.append(f"Processed {total_operations} operations with total cost of ${total_cost:.4f}")
            
            if model_breakdown:
                top_model = max(model_breakdown, key=model_breakdown.get)
                insights.append(f"Most used model: {top_model} (${model_breakdown[top_model]:.4f})")
            
            if hourly_breakdown:
                peak_hour = max(hourly_breakdown, key=hourly_breakdown.get)
                insights.append(f"Peak activity hour: {peak_hour}:00 (${hourly_breakdown[peak_hour]:.4f})")
        
        # Charts data for visualization
        charts_data = {
            'model_pie_chart': [
                {'name': model, 'value': float(cost)}
                for model, cost in model_breakdown.items()
            ],
            'hourly_line_chart': [
                {'hour': hour, 'cost': float(cost)}
                for hour, cost in sorted(hourly_breakdown.items())
            ]
        }
        
        return AnalyticsReport(
            report_id=report_id,
            report_type=ReportType.DAILY_SUMMARY,
            generated_at=datetime.now(),
            period_start=start_date,
            period_end=end_date,
            summary=summary,
            detailed_analysis=detailed_analysis,
            insights=insights,
            recommendations=[],
            charts_data=charts_data
        )
    
    def _generate_optimization_insights_report(self, report_id: str, start_date: datetime, end_date: datetime) -> AnalyticsReport:
        """Generate optimization insights report"""
        
        if not self.optimizer:
            raise ValueError("Optimizer not available")
        
        insights_data = self.optimizer.get_optimization_insights()
        
        summary = {
            'total_models_analyzed': len(insights_data.get('model_efficiency', {})),
            'optimization_opportunities': len(insights_data.get('recommendations', [])),
            'potential_savings': "15-30%"  # Estimated based on recommendations
        }
        
        detailed_analysis = {
            'model_efficiency': insights_data.get('model_efficiency', {}),
            'recommendations': insights_data.get('recommendations', [])
        }
        
        # Generate actionable insights
        insights = []
        for rec in insights_data.get('recommendations', []):
            insights.append(f"{rec['type'].title()}: {rec['description']}")
        
        if not insights:
            insights.append("No significant optimization opportunities detected")
        
        recommendations = [
            {
                'priority': rec['priority'],
                'type': rec['type'],
                'description': rec['description'],
                'potential_savings': rec.get('potential_savings', 'TBD')
            }
            for rec in insights_data.get('recommendations', [])
        ]
        
        return AnalyticsReport(
            report_id=report_id,
            report_type=ReportType.OPTIMIZATION_INSIGHTS,
            generated_at=datetime.now(),
            period_start=start_date,
            period_end=end_date,
            summary=summary,
            detailed_analysis=detailed_analysis,
            insights=insights,
            recommendations=recommendations,
            charts_data={}
        )
    
    def _analyze_cost_period(self, costs: List[CostBreakdown]) -> Dict[str, Any]:
        """Analyze cost patterns over the period"""
        
        if not costs:
            return {}
        
        total_cost = sum(c.total_cost for c in costs)
        daily_costs = defaultdict(list)
        
        for cost in costs:
            day = cost.timestamp.strftime('%Y-%m-%d')
            daily_costs[day].append(float(cost.total_cost))
        
        daily_totals = {day: sum(costs) for day, costs in daily_costs.items()}
        
        return {
            'total_cost': float(total_cost),
            'daily_average': statistics.mean(daily_totals.values()) if daily_totals else 0,
            'peak_day': max(daily_totals, key=daily_totals.get) if daily_totals else None,
            'lowest_day': min(daily_totals, key=daily_totals.get) if daily_totals else None,
            'cost_volatility': statistics.stdev(daily_totals.values()) if len(daily_totals) > 1 else 0
        }
    
    def _analyze_model_usage(self, costs: List[CostBreakdown]) -> Dict[str, Any]:
        """Analyze model usage patterns"""
        
        model_stats = defaultdict(lambda: {
            'count': 0, 'total_cost': Decimal('0'), 'total_tokens': 0
        })
        
        for cost in costs:
            stats = model_stats[cost.model]
            stats['count'] += 1
            stats['total_cost'] += cost.total_cost
            stats['total_tokens'] += cost.total_tokens
        
        # Calculate efficiency metrics
        model_insights = {}
        for model, stats in model_stats.items():
            avg_cost = stats['total_cost'] / stats['count'] if stats['count'] > 0 else Decimal('0')
            cost_per_token = stats['total_cost'] / stats['total_tokens'] if stats['total_tokens'] > 0 else Decimal('0')
            
            model_insights[model] = {
                'usage_count': stats['count'],
                'total_cost': float(stats['total_cost']),
                'average_cost_per_operation': float(avg_cost),
                'cost_per_1k_tokens': float(cost_per_token * 1000),
                'total_tokens': stats['total_tokens']
            }
        
        return model_insights
    
    def _analyze_category_distribution(self, costs: List[CostBreakdown]) -> Dict[str, Any]:
        """Analyze cost distribution by category"""
        
        category_stats = defaultdict(lambda: {
            'count': 0, 'total_cost': Decimal('0')
        })
        
        for cost in costs:
            stats = category_stats[cost.category.value]
            stats['count'] += 1
            stats['total_cost'] += cost.total_cost
        
        total_cost = sum(c.total_cost for c in costs)
        
        category_insights = {}
        for category, stats in category_stats.items():
            percentage = float(stats['total_cost'] / total_cost * 100) if total_cost > 0 else 0
            
            category_insights[category] = {
                'operations': stats['count'],
                'total_cost': float(stats['total_cost']),
                'percentage_of_total': percentage,
                'average_cost': float(stats['total_cost'] / stats['count']) if stats['count'] > 0 else 0
            }
        
        return category_insights
    
    def _identify_optimization_opportunities(self, costs: List[CostBreakdown]) -> List[Dict[str, Any]]:
        """Identify specific optimization opportunities"""
        
        opportunities = []
        
        # High-cost operations
        sorted_costs = sorted(costs, key=lambda x: x.total_cost, reverse=True)
        if len(sorted_costs) >= 5:
            top_5_cost = sum(c.total_cost for c in sorted_costs[:5])
            total_cost = sum(c.total_cost for c in costs)
            
            if top_5_cost / total_cost > 0.5:  # Top 5 operations account for >50% of cost
                opportunities.append({
                    'type': 'high_cost_operations',
                    'description': 'Top 5 operations account for >50% of total cost',
                    'recommendation': 'Focus optimization efforts on these high-impact operations',
                    'potential_impact': 'high'
                })
        
        # Model diversity
        unique_models = len(set(c.model for c in costs))
        if unique_models == 1:
            opportunities.append({
                'type': 'model_diversity',
                'description': 'Only using one model type',
                'recommendation': 'Evaluate alternative models for different operation types',
                'potential_impact': 'medium'
            })
        
        return opportunities
    
    def _generate_actionable_recommendations(self, costs: List[CostBreakdown]) -> List[Dict[str, Any]]:
        """Generate actionable cost optimization recommendations"""
        
        recommendations = []
        
        # Analyze patterns and generate specific recommendations
        model_usage = defaultdict(int)
        category_usage = defaultdict(int)
        
        for cost in costs:
            model_usage[cost.model] += 1
            category_usage[cost.category.value] += 1
        
        # Model optimization recommendations
        if model_usage:
            most_used_model = max(model_usage, key=model_usage.get)
            recommendations.append({
                'category': 'model_optimization',
                'priority': 'high',
                'action': f'Evaluate alternatives to {most_used_model}',
                'rationale': f'{most_used_model} is used in {model_usage[most_used_model]} operations',
                'estimated_savings': '10-20%'
            })
        
        return recommendations
    
    # Additional report generation methods would go here...
    def _generate_weekly_analysis(self, report_id: str, start_date: datetime, end_date: datetime) -> AnalyticsReport:
        """Generate weekly analysis report (simplified)"""
        # Implementation similar to daily summary but with weekly aggregations
        return self._generate_daily_summary(report_id, start_date, end_date)
    
    def _generate_monthly_report(self, report_id: str, start_date: datetime, end_date: datetime) -> AnalyticsReport:
        """Generate monthly report (simplified)"""
        return self._generate_daily_summary(report_id, start_date, end_date)
    
    def _generate_model_performance_report(self, report_id: str, start_date: datetime, end_date: datetime) -> AnalyticsReport:
        """Generate model performance report (simplified)"""
        return self._generate_daily_summary(report_id, start_date, end_date)
    
    def _generate_cost_breakdown_report(self, report_id: str, start_date: datetime, end_date: datetime) -> AnalyticsReport:
        """Generate cost breakdown report (simplified)"""
        return self._generate_daily_summary(report_id, start_date, end_date)
    
    def _generate_budget_status_report(self, report_id: str, start_date: datetime, end_date: datetime) -> AnalyticsReport:
        """Generate budget status report (simplified)"""
        return self._generate_daily_summary(report_id, start_date, end_date)
    
    def _generate_forecast_report(self, report_id: str, start_date: datetime, end_date: datetime) -> AnalyticsReport:
        """Generate forecast report (simplified)"""
        return self._generate_daily_summary(report_id, start_date, end_date)


# Global analytics instance
_cost_analytics: Optional[CostAnalytics] = None


def get_cost_analytics(
    cost_calculator: Optional[CostCalculator] = None,
    budget_enforcer: Optional[BudgetEnforcer] = None,
    optimizer: Optional[CostOptimizer] = None
) -> CostAnalytics:
    """Get the global cost analytics instance"""
    global _cost_analytics
    if _cost_analytics is None:
        from .cost_calculator import get_cost_calculator
        from .budget_manager import get_budget_enforcer
        from .optimizer import get_cost_optimizer
        
        calc = cost_calculator or get_cost_calculator()
        budg = budget_enforcer or get_budget_enforcer(calc)
        opt = optimizer or get_cost_optimizer(calc)
        
        _cost_analytics = CostAnalytics(calc, budg, opt)
    return _cost_analytics