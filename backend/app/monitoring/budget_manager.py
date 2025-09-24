"""
brAIn v2.0 Budget Management and Alerting System
Intelligent budget monitoring with predictive alerts and cost optimization.
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, date, timedelta
from enum import Enum
from uuid import UUID

from pydantic import BaseModel, Field
from .cost_calculator import TokenUsage, CostCalculator

logger = logging.getLogger(__name__)


class BudgetPeriod(str, Enum):
    """Budget period types"""

    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    YEARLY = "yearly"


class AlertSeverity(str, Enum):
    """Alert severity levels"""

    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class BudgetAlert(BaseModel):
    """Budget alert configuration"""

    id: str = Field(description="Unique alert identifier")

    user_id: Optional[UUID] = Field(
        default=None, description="User ID for user-specific alerts"
    )

    name: str = Field(description="Human-readable alert name")

    threshold_percentage: float = Field(
        description="Budget percentage threshold (0.0-1.0)", ge=0.0, le=1.0
    )

    budget_period: BudgetPeriod = Field(description="Budget period for the alert")

    severity: AlertSeverity = Field(description="Alert severity level")

    enabled: bool = Field(default=True, description="Whether the alert is active")

    notification_channels: List[str] = Field(
        default_factory=list,
        description="Notification channels (email, webhook, dashboard)",
        examples=[["email", "dashboard"]],
    )

    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional alert metadata"
    )


class BudgetStatus(BaseModel):
    """Current budget status for a user"""

    user_id: UUID = Field(description="User ID")

    period: BudgetPeriod = Field(description="Budget period")

    period_start: datetime = Field(description="Start of the current period")

    period_end: datetime = Field(description="End of the current period")

    budget_limit: float = Field(description="Budget limit for the period", ge=0.0)

    current_spending: float = Field(
        description="Current spending in the period", ge=0.0
    )

    projected_spending: Optional[float] = Field(
        default=None, description="Projected spending for the full period"
    )

    remaining_budget: float = Field(description="Remaining budget for the period")

    percentage_used: float = Field(
        description="Percentage of budget used (0.0-1.0)", ge=0.0
    )

    days_remaining: int = Field(description="Days remaining in the period", ge=0)

    is_over_budget: bool = Field(description="Whether currently over budget")

    active_alerts: List[str] = Field(
        default_factory=list, description="List of active alert IDs"
    )


class SpendingForecast(BaseModel):
    """Spending forecast based on usage patterns"""

    period: BudgetPeriod = Field(description="Forecast period")

    current_spending: float = Field(description="Current spending", ge=0.0)

    projected_spending: float = Field(description="Projected total spending", ge=0.0)

    confidence_interval: Tuple[float, float] = Field(
        description="95% confidence interval for projection"
    )

    spending_trend: str = Field(
        description="Spending trend (increasing, stable, decreasing)"
    )

    days_to_budget_exhaustion: Optional[int] = Field(
        default=None, description="Days until budget is exhausted (if trending over)"
    )

    recommendations: List[str] = Field(
        default_factory=list, description="Cost optimization recommendations"
    )


class CostOptimizationRecommendation(BaseModel):
    """Cost optimization recommendation"""

    category: str = Field(
        description="Recommendation category",
        examples=["model_selection", "batching", "caching"],
    )

    title: str = Field(description="Recommendation title")

    description: str = Field(description="Detailed description of the recommendation")

    potential_savings: float = Field(
        description="Potential monthly savings in USD", ge=0.0
    )

    implementation_effort: str = Field(
        description="Implementation effort level", examples=["low", "medium", "high"]
    )

    priority: int = Field(
        description="Priority score (1-10, higher is more important)", ge=1, le=10
    )


class BudgetManager:
    """
    Comprehensive budget management with intelligent alerting and forecasting.

    Features:
    - Multi-period budget tracking
    - Predictive spending forecasts
    - Intelligent alert management
    - Cost optimization recommendations
    - Real-time budget monitoring
    """

    def __init__(self):
        self.cost_calculator = CostCalculator()
        self._user_budgets: Dict[UUID, Dict[BudgetPeriod, float]] = {}
        self._alerts: Dict[str, BudgetAlert] = {}
        self._spending_history: Dict[UUID, List[Dict[str, Any]]] = {}
        self._triggered_alerts: Dict[str, datetime] = {}

    def set_user_budget(
        self, user_id: UUID, budget_limit: float, period: BudgetPeriod
    ) -> bool:
        """
        Set budget limit for a user and period.

        Args:
            user_id: User ID
            budget_limit: Budget limit in USD
            period: Budget period

        Returns:
            True if set successfully
        """
        try:
            if user_id not in self._user_budgets:
                self._user_budgets[user_id] = {}

            self._user_budgets[user_id][period] = budget_limit

            logger.info(
                f"Set {period.value} budget limit of ${budget_limit} for user {user_id}"
            )
            return True

        except Exception as e:
            logger.error(f"Failed to set user budget: {e}")
            return False

    def get_budget_status(
        self,
        user_id: UUID,
        period: BudgetPeriod,
        current_date: Optional[datetime] = None,
    ) -> Optional[BudgetStatus]:
        """
        Get current budget status for a user.

        Args:
            user_id: User ID
            period: Budget period
            current_date: Current date (defaults to now)

        Returns:
            BudgetStatus or None if no budget set
        """
        if current_date is None:
            current_date = datetime.utcnow()

        # Get budget limit
        budget_limit = self._user_budgets.get(user_id, {}).get(period)
        if budget_limit is None:
            return None

        # Calculate period boundaries
        period_start, period_end = self._get_period_boundaries(current_date, period)

        # Get current spending (placeholder - would integrate with database)
        current_spending = self._get_current_spending(user_id, period_start, period_end)

        # Calculate remaining budget
        remaining_budget = budget_limit - current_spending

        # Calculate percentage used
        percentage_used = min(
            current_spending / budget_limit if budget_limit > 0 else 0, 1.0
        )

        # Calculate days remaining
        days_remaining = max((period_end.date() - current_date.date()).days, 0)

        # Check if over budget
        is_over_budget = current_spending > budget_limit

        # Calculate projected spending
        projected_spending = self._calculate_projected_spending(
            user_id, current_spending, period_start, period_end, current_date
        )

        # Get active alerts
        active_alerts = self._get_active_alerts(user_id, percentage_used, period)

        return BudgetStatus(
            user_id=user_id,
            period=period,
            period_start=period_start,
            period_end=period_end,
            budget_limit=budget_limit,
            current_spending=current_spending,
            projected_spending=projected_spending,
            remaining_budget=remaining_budget,
            percentage_used=percentage_used,
            days_remaining=days_remaining,
            is_over_budget=is_over_budget,
            active_alerts=active_alerts,
        )

    def create_budget_alert(
        self,
        alert_id: str,
        user_id: Optional[UUID],
        name: str,
        threshold_percentage: float,
        budget_period: BudgetPeriod,
        severity: AlertSeverity,
        notification_channels: List[str],
    ) -> BudgetAlert:
        """
        Create a new budget alert.

        Args:
            alert_id: Unique alert identifier
            user_id: User ID (None for system-wide alerts)
            name: Alert name
            threshold_percentage: Threshold as percentage (0.0-1.0)
            budget_period: Budget period
            severity: Alert severity
            notification_channels: List of notification channels

        Returns:
            Created BudgetAlert
        """
        alert = BudgetAlert(
            id=alert_id,
            user_id=user_id,
            name=name,
            threshold_percentage=threshold_percentage,
            budget_period=budget_period,
            severity=severity,
            notification_channels=notification_channels,
        )

        self._alerts[alert_id] = alert

        logger.info(f"Created budget alert: {name} at {threshold_percentage*100}%")
        return alert

    def check_budget_alerts(
        self,
        user_id: UUID,
        current_spending: float,
        budget_limit: float,
        period: BudgetPeriod,
    ) -> List[BudgetAlert]:
        """
        Check for triggered budget alerts.

        Args:
            user_id: User ID
            current_spending: Current spending amount
            budget_limit: Budget limit
            period: Budget period

        Returns:
            List of triggered alerts
        """
        if budget_limit <= 0:
            return []

        percentage_used = min(current_spending / budget_limit, 1.0)
        triggered_alerts = []

        for alert in self._alerts.values():
            # Check if alert applies to this user and period
            if alert.user_id and alert.user_id != user_id:
                continue

            if alert.budget_period != period:
                continue

            if not alert.enabled:
                continue

            # Check if threshold is exceeded
            if percentage_used >= alert.threshold_percentage:
                # Check if alert was already triggered recently
                last_triggered = self._triggered_alerts.get(alert.id)
                if (
                    last_triggered
                    and (datetime.utcnow() - last_triggered).seconds < 3600
                ):
                    continue  # Don't trigger same alert within 1 hour

                triggered_alerts.append(alert)
                self._triggered_alerts[alert.id] = datetime.utcnow()

        return triggered_alerts

    def generate_spending_forecast(
        self,
        user_id: UUID,
        period: BudgetPeriod,
        current_date: Optional[datetime] = None,
    ) -> Optional[SpendingForecast]:
        """
        Generate spending forecast based on historical data.

        Args:
            user_id: User ID
            period: Budget period
            current_date: Current date (defaults to now)

        Returns:
            SpendingForecast or None if insufficient data
        """
        if current_date is None:
            current_date = datetime.utcnow()

        period_start, period_end = self._get_period_boundaries(current_date, period)
        current_spending = self._get_current_spending(user_id, period_start, period_end)

        # Calculate projection based on spending velocity
        days_elapsed = (current_date.date() - period_start.date()).days + 1
        total_days = (period_end.date() - period_start.date()).days + 1

        if days_elapsed <= 0:
            return None

        # Simple linear projection (could be enhanced with ML models)
        daily_average = current_spending / days_elapsed
        projected_spending = daily_average * total_days

        # Calculate confidence interval (±20% for simple model)
        confidence_margin = projected_spending * 0.2
        confidence_interval = (
            max(0, projected_spending - confidence_margin),
            projected_spending + confidence_margin,
        )

        # Determine trend
        recent_spending = self._get_recent_spending_trend(user_id, current_date)
        trend = self._analyze_spending_trend(recent_spending)

        # Calculate days to budget exhaustion
        budget_limit = self._user_budgets.get(user_id, {}).get(period, 0)
        days_to_exhaustion = None
        if budget_limit > 0 and daily_average > 0:
            remaining_budget = budget_limit - current_spending
            if remaining_budget > 0:
                days_to_exhaustion = int(remaining_budget / daily_average)

        # Generate recommendations
        recommendations = self._generate_cost_recommendations(
            user_id, current_spending, projected_spending, budget_limit
        )

        return SpendingForecast(
            period=period,
            current_spending=current_spending,
            projected_spending=projected_spending,
            confidence_interval=confidence_interval,
            spending_trend=trend,
            days_to_budget_exhaustion=days_to_exhaustion,
            recommendations=[rec.title for rec in recommendations],
        )

    def get_cost_optimization_recommendations(
        self, user_id: UUID, period: BudgetPeriod = BudgetPeriod.MONTHLY
    ) -> List[CostOptimizationRecommendation]:
        """
        Generate cost optimization recommendations for a user.

        Args:
            user_id: User ID
            period: Budget period for analysis

        Returns:
            List of optimization recommendations
        """
        # Analyze spending patterns (placeholder - would integrate with actual data)
        spending_analysis = self._analyze_user_spending_patterns(user_id, period)

        recommendations = []

        # Model selection recommendations
        if spending_analysis.get("expensive_models_usage", 0) > 0.3:
            recommendations.append(
                CostOptimizationRecommendation(
                    category="model_selection",
                    title="Switch to Cost-Effective Models",
                    description="Consider using GPT-3.5-turbo instead of GPT-4 for simple tasks to reduce costs by up to 90%",
                    potential_savings=spending_analysis.get(
                        "model_switching_savings", 50.0
                    ),
                    implementation_effort="low",
                    priority=8,
                )
            )

        # Batching recommendations
        if spending_analysis.get("small_requests_ratio", 0) > 0.5:
            recommendations.append(
                CostOptimizationRecommendation(
                    category="batching",
                    title="Implement Request Batching",
                    description="Batch multiple small requests together to reduce per-request overhead and improve cost efficiency",
                    potential_savings=spending_analysis.get("batching_savings", 25.0),
                    implementation_effort="medium",
                    priority=6,
                )
            )

        # Caching recommendations
        if spending_analysis.get("duplicate_requests_ratio", 0) > 0.2:
            recommendations.append(
                CostOptimizationRecommendation(
                    category="caching",
                    title="Enable Response Caching",
                    description="Implement caching for frequently repeated queries to avoid redundant API calls",
                    potential_savings=spending_analysis.get("caching_savings", 40.0),
                    implementation_effort="medium",
                    priority=7,
                )
            )

        # Token optimization
        if spending_analysis.get("average_tokens_per_request", 0) > 2000:
            recommendations.append(
                CostOptimizationRecommendation(
                    category="token_optimization",
                    title="Optimize Prompt Length",
                    description="Reduce prompt length and use more concise instructions to lower token usage",
                    potential_savings=spending_analysis.get(
                        "token_optimization_savings", 20.0
                    ),
                    implementation_effort="low",
                    priority=5,
                )
            )

        # Sort by priority
        recommendations.sort(key=lambda x: x.priority, reverse=True)

        return recommendations

    def _get_period_boundaries(
        self, current_date: datetime, period: BudgetPeriod
    ) -> Tuple[datetime, datetime]:
        """Calculate period start and end dates"""
        if period == BudgetPeriod.DAILY:
            start = current_date.replace(hour=0, minute=0, second=0, microsecond=0)
            end = start + timedelta(days=1) - timedelta(microseconds=1)
        elif period == BudgetPeriod.WEEKLY:
            days_since_monday = current_date.weekday()
            start = (current_date - timedelta(days=days_since_monday)).replace(
                hour=0, minute=0, second=0, microsecond=0
            )
            end = start + timedelta(days=7) - timedelta(microseconds=1)
        elif period == BudgetPeriod.MONTHLY:
            start = current_date.replace(
                day=1, hour=0, minute=0, second=0, microsecond=0
            )
            if start.month == 12:
                end = start.replace(year=start.year + 1, month=1) - timedelta(
                    microseconds=1
                )
            else:
                end = start.replace(month=start.month + 1) - timedelta(microseconds=1)
        elif period == BudgetPeriod.YEARLY:
            start = current_date.replace(
                month=1, day=1, hour=0, minute=0, second=0, microsecond=0
            )
            end = start.replace(year=start.year + 1) - timedelta(microseconds=1)
        else:
            raise ValueError(f"Unsupported period: {period}")

        return start, end

    def _get_current_spending(
        self, user_id: UUID, period_start: datetime, period_end: datetime
    ) -> float:
        """Get current spending for user in period (placeholder)"""
        # This would integrate with the actual database
        # For now, return a placeholder value
        return 0.0

    def _calculate_projected_spending(
        self,
        user_id: UUID,
        current_spending: float,
        period_start: datetime,
        period_end: datetime,
        current_date: datetime,
    ) -> float:
        """Calculate projected spending for the period"""
        days_elapsed = (current_date.date() - period_start.date()).days + 1
        total_days = (period_end.date() - period_start.date()).days + 1

        if days_elapsed <= 0:
            return current_spending

        daily_average = current_spending / days_elapsed
        return daily_average * total_days

    def _get_active_alerts(
        self, user_id: UUID, percentage_used: float, period: BudgetPeriod
    ) -> List[str]:
        """Get list of active alert IDs"""
        active_alerts = []

        for alert_id, alert in self._alerts.items():
            if alert.user_id and alert.user_id != user_id:
                continue

            if alert.budget_period != period:
                continue

            if not alert.enabled:
                continue

            if percentage_used >= alert.threshold_percentage:
                active_alerts.append(alert_id)

        return active_alerts

    def _get_recent_spending_trend(
        self, user_id: UUID, current_date: datetime, days: int = 7
    ) -> List[float]:
        """Get recent daily spending for trend analysis"""
        # Placeholder - would integrate with database
        return [10.0, 12.0, 8.0, 15.0, 11.0, 9.0, 13.0]

    def _analyze_spending_trend(self, daily_spending: List[float]) -> str:
        """Analyze spending trend from daily data"""
        if len(daily_spending) < 2:
            return "stable"

        # Simple trend analysis
        recent_avg = (
            sum(daily_spending[-3:]) / 3
            if len(daily_spending) >= 3
            else daily_spending[-1]
        )
        earlier_avg = (
            sum(daily_spending[:-3]) / (len(daily_spending) - 3)
            if len(daily_spending) > 3
            else daily_spending[0]
        )

        if recent_avg > earlier_avg * 1.1:
            return "increasing"
        elif recent_avg < earlier_avg * 0.9:
            return "decreasing"
        else:
            return "stable"

    def _generate_cost_recommendations(
        self,
        user_id: UUID,
        current_spending: float,
        projected_spending: float,
        budget_limit: float,
    ) -> List[CostOptimizationRecommendation]:
        """Generate cost recommendations based on spending analysis"""
        # Placeholder implementation
        return []

    def _analyze_user_spending_patterns(
        self, user_id: UUID, period: BudgetPeriod
    ) -> Dict[str, float]:
        """Analyze user spending patterns for recommendations"""
        # Placeholder analysis - would integrate with actual data
        return {
            "expensive_models_usage": 0.4,
            "small_requests_ratio": 0.6,
            "duplicate_requests_ratio": 0.3,
            "average_tokens_per_request": 2500,
            "model_switching_savings": 75.0,
            "batching_savings": 30.0,
            "caching_savings": 45.0,
            "token_optimization_savings": 25.0,
        }


# Global budget manager instance
_budget_manager = BudgetManager()


def get_budget_manager() -> BudgetManager:
    """Get the global budget manager instance"""
    return _budget_manager
