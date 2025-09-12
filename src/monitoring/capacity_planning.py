"""
Capacity planning and resource forecasting system.

This module provides intelligent capacity planning capabilities including resource
usage forecasting, scaling recommendations, cost optimization, and proactive
capacity management based on historical trends and predictive analytics.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import asyncio
import logging
import statistics
from collections import defaultdict, deque
import json

# Optional ML imports
try:
    from sklearn.linear_model import LinearRegression, Ridge
    from sklearn.preprocessing import PolynomialFeatures, StandardScaler
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_absolute_error, r2_score
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False

from .anomaly_detection import MetricPoint
from .predictive import PredictionResult, PredictionConfidence


class ResourceType(Enum):
    """Types of resources for capacity planning"""
    CPU = "cpu"
    MEMORY = "memory"
    DISK = "disk"
    NETWORK = "network"
    DATABASE = "database"
    CONTAINERS = "containers"
    CUSTOM = "custom"


class ScalingDirection(Enum):
    """Direction of scaling recommendation"""
    UP = "up"
    DOWN = "down"
    MAINTAIN = "maintain"


class CapacityStatus(Enum):
    """Status of capacity utilization"""
    OPTIMAL = "optimal"
    WARNING = "warning"
    CRITICAL = "critical"
    OVER_PROVISIONED = "over_provisioned"


@dataclass
class ResourceCapacity:
    """Current capacity information for a resource"""
    resource_type: ResourceType
    current_usage: float
    total_capacity: float
    utilization_percent: float
    available_capacity: float
    unit: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CapacityForecast:
    """Forecast of future capacity needs"""
    resource_type: ResourceType
    forecast_horizon_days: int
    predicted_usage: List[Tuple[datetime, float]]
    predicted_peak: float
    predicted_peak_time: datetime
    growth_rate: float
    seasonal_pattern: bool
    confidence: PredictionConfidence
    model_accuracy: float


@dataclass
class ScalingRecommendation:
    """Recommendation for scaling resources"""
    resource_type: ResourceType
    scaling_direction: ScalingDirection
    recommended_capacity: float
    current_capacity: float
    urgency: str  # low, medium, high, critical
    estimated_cost_impact: Optional[float] = None
    implementation_timeline: str = ""
    reasoning: str = ""
    prerequisites: List[str] = field(default_factory=list)


@dataclass
class CapacityAlert:
    """Alert for capacity-related issues"""
    resource_type: ResourceType
    status: CapacityStatus
    current_utilization: float
    threshold_breached: str
    projected_exhaustion: Optional[datetime] = None
    recommended_actions: List[str] = field(default_factory=list)
    priority: str = "medium"


@dataclass
class CapacityPlanningConfig:
    """Configuration for capacity planning system"""
    # Forecasting parameters
    forecast_horizon_days: int = 30
    min_historical_data_points: int = 100
    seasonal_analysis_period: int = 7  # days
    
    # Threshold settings
    warning_threshold: float = 0.7  # 70%
    critical_threshold: float = 0.85  # 85%
    over_provision_threshold: float = 0.3  # 30%
    
    # Growth analysis
    growth_analysis_window_days: int = 14
    significant_growth_rate: float = 0.1  # 10% growth
    
    # Scaling recommendations
    scaling_buffer: float = 0.2  # 20% buffer for scaling
    cost_optimization_enabled: bool = True
    auto_scaling_suggestions: bool = True
    
    # Model parameters
    forecasting_models: List[str] = field(default_factory=lambda: ["linear", "polynomial", "seasonal"])
    model_retraining_interval_hours: int = 24


class ResourceForecaster:
    """
    Forecasts future resource usage based on historical trends and patterns.
    """
    
    def __init__(self, config: CapacityPlanningConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.models = {}
        self.historical_accuracy = defaultdict(list)
    
    async def forecast_resource_usage(
        self,
        resource_type: ResourceType,
        historical_data: List[MetricPoint],
        forecast_days: Optional[int] = None
    ) -> CapacityForecast:
        """
        Generate capacity forecast for a resource.
        
        Args:
            resource_type: Type of resource to forecast
            historical_data: Historical usage data
            forecast_days: Days ahead to forecast (defaults to config value)
            
        Returns:
            Capacity forecast with predictions and confidence
        """
        forecast_days = forecast_days or self.config.forecast_horizon_days
        
        if len(historical_data) < self.config.min_historical_data_points:
            return self._create_basic_forecast(resource_type, historical_data, forecast_days)
        
        # Prepare data
        timestamps, values = self._prepare_forecast_data(historical_data)
        
        # Try different forecasting models
        forecasts = {}
        accuracies = {}
        
        # Linear trend forecasting
        if "linear" in self.config.forecasting_models:
            linear_forecast, linear_accuracy = await self._linear_forecast(
                timestamps, values, forecast_days
            )
            forecasts["linear"] = linear_forecast
            accuracies["linear"] = linear_accuracy
        
        # Polynomial trend forecasting
        if "polynomial" in self.config.forecasting_models:
            poly_forecast, poly_accuracy = await self._polynomial_forecast(
                timestamps, values, forecast_days
            )
            forecasts["polynomial"] = poly_forecast
            accuracies["polynomial"] = poly_accuracy
        
        # Seasonal forecasting
        if "seasonal" in self.config.forecasting_models and len(historical_data) > 14:
            seasonal_forecast, seasonal_accuracy = await self._seasonal_forecast(
                timestamps, values, forecast_days
            )
            forecasts["seasonal"] = seasonal_forecast
            accuracies["seasonal"] = seasonal_accuracy
        
        # Select best forecast
        if not forecasts:
            return self._create_basic_forecast(resource_type, historical_data, forecast_days)
        
        best_model = max(accuracies.items(), key=lambda x: x[1])[0]
        best_forecast = forecasts[best_model]
        best_accuracy = accuracies[best_model]
        
        # Analyze forecast characteristics
        forecast_values = [value for _, value in best_forecast]
        peak_value = max(forecast_values)
        peak_index = forecast_values.index(peak_value)
        peak_time = best_forecast[peak_index][0]
        
        # Calculate growth rate
        growth_rate = self._calculate_growth_rate(values, forecast_values)
        
        # Detect seasonal patterns
        seasonal_pattern = await self._detect_seasonal_pattern(values)
        
        # Convert accuracy to confidence
        confidence = self._accuracy_to_confidence(best_accuracy)
        
        return CapacityForecast(
            resource_type=resource_type,
            forecast_horizon_days=forecast_days,
            predicted_usage=best_forecast,
            predicted_peak=peak_value,
            predicted_peak_time=peak_time,
            growth_rate=growth_rate,
            seasonal_pattern=seasonal_pattern,
            confidence=confidence,
            model_accuracy=best_accuracy
        )
    
    async def _linear_forecast(
        self,
        timestamps: List[float],
        values: List[float],
        forecast_days: int
    ) -> Tuple[List[Tuple[datetime, float]], float]:
        """Generate linear trend forecast"""
        try:
            # Fit linear regression
            X = np.array(timestamps).reshape(-1, 1)
            y = np.array(values)
            
            if ML_AVAILABLE:
                model = LinearRegression()
                model.fit(X, y)
                
                # Calculate accuracy using R²
                y_pred = model.predict(X)
                accuracy = max(r2_score(y, y_pred), 0.0)
            else:
                # Simple linear regression without sklearn
                slope, intercept = np.polyfit(timestamps, values, 1)
                accuracy = 0.5  # Default accuracy
                
                # Create model-like object for prediction
                class SimpleModel:
                    def predict(self, X):
                        return slope * X.flatten() + intercept
                
                model = SimpleModel()
            
            # Generate forecast points
            last_timestamp = timestamps[-1]
            forecast_points = []
            
            for i in range(1, forecast_days + 1):
                future_timestamp = last_timestamp + (i * 24 * 3600)  # Add days in seconds
                predicted_value = model.predict([[future_timestamp]])[0]
                
                # Ensure non-negative values for usage metrics
                predicted_value = max(predicted_value, 0)
                
                forecast_points.append((
                    datetime.fromtimestamp(future_timestamp),
                    predicted_value
                ))
            
            return forecast_points, accuracy
            
        except Exception as e:
            self.logger.error(f"Linear forecasting error: {e}")
            return [], 0.1
    
    async def _polynomial_forecast(
        self,
        timestamps: List[float],
        values: List[float],
        forecast_days: int
    ) -> Tuple[List[Tuple[datetime, float]], float]:
        """Generate polynomial trend forecast"""
        if not ML_AVAILABLE:
            return await self._linear_forecast(timestamps, values, forecast_days)
        
        try:
            # Prepare polynomial features
            X = np.array(timestamps).reshape(-1, 1)
            y = np.array(values)
            
            poly_features = PolynomialFeatures(degree=2)
            X_poly = poly_features.fit_transform(X)
            
            # Fit polynomial regression
            model = Ridge(alpha=1.0)  # Ridge for stability
            model.fit(X_poly, y)
            
            # Calculate accuracy
            y_pred = model.predict(X_poly)
            accuracy = max(r2_score(y, y_pred), 0.0)
            
            # Generate forecast
            last_timestamp = timestamps[-1]
            forecast_points = []
            
            for i in range(1, forecast_days + 1):
                future_timestamp = last_timestamp + (i * 24 * 3600)
                X_future = poly_features.transform([[future_timestamp]])
                predicted_value = model.predict(X_future)[0]
                
                # Ensure reasonable bounds
                predicted_value = max(predicted_value, 0)
                predicted_value = min(predicted_value, max(values) * 3)  # Cap at 3x max historical
                
                forecast_points.append((
                    datetime.fromtimestamp(future_timestamp),
                    predicted_value
                ))
            
            return forecast_points, accuracy
            
        except Exception as e:
            self.logger.error(f"Polynomial forecasting error: {e}")
            return await self._linear_forecast(timestamps, values, forecast_days)
    
    async def _seasonal_forecast(
        self,
        timestamps: List[float],
        values: List[float],
        forecast_days: int
    ) -> Tuple[List[Tuple[datetime, float]], float]:
        """Generate seasonal forecast"""
        try:
            # Simple seasonal forecasting using day-of-week patterns
            df_data = []
            for i, (ts, val) in enumerate(zip(timestamps, values)):
                dt = datetime.fromtimestamp(ts)
                df_data.append({
                    'timestamp': ts,
                    'value': val,
                    'hour': dt.hour,
                    'day_of_week': dt.weekday(),
                    'day_of_month': dt.day
                })
            
            # Group by hour and day of week to find patterns
            hourly_patterns = defaultdict(list)
            daily_patterns = defaultdict(list)
            
            for data_point in df_data:
                hourly_patterns[data_point['hour']].append(data_point['value'])
                daily_patterns[data_point['day_of_week']].append(data_point['value'])
            
            # Calculate averages for patterns
            hourly_avg = {hour: statistics.mean(vals) for hour, vals in hourly_patterns.items()}
            daily_avg = {day: statistics.mean(vals) for day, vals in daily_patterns.items()}
            
            # Calculate overall trend
            trend_slope = (values[-10:] if len(values) >= 10 else values[-len(values)//2:])
            trend_growth = statistics.mean(trend_slope) if trend_slope else values[-1]
            
            # Generate seasonal forecast
            last_timestamp = timestamps[-1]
            forecast_points = []
            
            for i in range(1, forecast_days + 1):
                future_timestamp = last_timestamp + (i * 24 * 3600)
                future_dt = datetime.fromtimestamp(future_timestamp)
                
                # Combine trend with seasonal patterns
                base_value = trend_growth
                hourly_factor = hourly_avg.get(future_dt.hour, base_value) / base_value if base_value > 0 else 1.0
                daily_factor = daily_avg.get(future_dt.weekday(), base_value) / base_value if base_value > 0 else 1.0
                
                # Weighted combination
                predicted_value = base_value * (0.5 + 0.3 * hourly_factor + 0.2 * daily_factor)
                predicted_value = max(predicted_value, 0)
                
                forecast_points.append((future_dt, predicted_value))
            
            # Calculate accuracy based on pattern consistency
            accuracy = 0.6  # Moderate accuracy for seasonal
            if len(hourly_patterns) > 12 and len(daily_patterns) > 5:
                # Higher accuracy if we have good seasonal data
                accuracy = 0.75
            
            return forecast_points, accuracy
            
        except Exception as e:
            self.logger.error(f"Seasonal forecasting error: {e}")
            return await self._linear_forecast(timestamps, values, forecast_days)
    
    def _prepare_forecast_data(self, data: List[MetricPoint]) -> Tuple[List[float], List[float]]:
        """Prepare data for forecasting"""
        timestamps = [point.timestamp.timestamp() for point in data]
        values = [point.value for point in data]
        return timestamps, values
    
    def _create_basic_forecast(
        self,
        resource_type: ResourceType,
        historical_data: List[MetricPoint],
        forecast_days: int
    ) -> CapacityForecast:
        """Create a basic forecast when insufficient data is available"""
        if not historical_data:
            current_value = 0.0
        else:
            current_value = historical_data[-1].value
        
        # Simple flat forecast
        base_time = datetime.utcnow()
        predicted_usage = [
            (base_time + timedelta(days=i), current_value)
            for i in range(1, forecast_days + 1)
        ]
        
        return CapacityForecast(
            resource_type=resource_type,
            forecast_horizon_days=forecast_days,
            predicted_usage=predicted_usage,
            predicted_peak=current_value,
            predicted_peak_time=base_time,
            growth_rate=0.0,
            seasonal_pattern=False,
            confidence=PredictionConfidence.LOW,
            model_accuracy=0.1
        )
    
    def _calculate_growth_rate(self, historical: List[float], forecast: List[float]) -> float:
        """Calculate growth rate from historical to forecast"""
        if not historical or not forecast:
            return 0.0
        
        historical_avg = statistics.mean(historical[-7:]) if len(historical) >= 7 else historical[-1]
        forecast_avg = statistics.mean(forecast[:7]) if len(forecast) >= 7 else forecast[0]
        
        if historical_avg > 0:
            return (forecast_avg - historical_avg) / historical_avg
        return 0.0
    
    async def _detect_seasonal_pattern(self, values: List[float]) -> bool:
        """Detect if there are seasonal patterns in the data"""
        if len(values) < 14:
            return False
        
        # Simple pattern detection using autocorrelation
        try:
            # Check for daily patterns (assuming hourly data)
            if len(values) >= 24:
                daily_corr = np.corrcoef(values[:-24], values[24:])[0, 1]
                if not np.isnan(daily_corr) and daily_corr > 0.5:
                    return True
            
            # Check for weekly patterns
            if len(values) >= 168:  # 7 days of hourly data
                weekly_corr = np.corrcoef(values[:-168], values[168:])[0, 1]
                if not np.isnan(weekly_corr) and weekly_corr > 0.3:
                    return True
            
        except Exception:
            pass
        
        return False
    
    def _accuracy_to_confidence(self, accuracy: float) -> PredictionConfidence:
        """Convert numerical accuracy to confidence enum"""
        if accuracy >= 0.8:
            return PredictionConfidence.VERY_HIGH
        elif accuracy >= 0.6:
            return PredictionConfidence.HIGH
        elif accuracy >= 0.4:
            return PredictionConfidence.MEDIUM
        else:
            return PredictionConfidence.LOW


class CapacityAnalyzer:
    """
    Analyzes current capacity utilization and generates recommendations.
    """
    
    def __init__(self, config: CapacityPlanningConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def analyze_current_capacity(
        self,
        resource_capacities: Dict[ResourceType, ResourceCapacity]
    ) -> Dict[str, Any]:
        """
        Analyze current capacity utilization across all resources.
        
        Args:
            resource_capacities: Current capacity data for each resource type
            
        Returns:
            Comprehensive capacity analysis
        """
        analysis = {
            "timestamp": datetime.utcnow(),
            "overall_status": CapacityStatus.OPTIMAL,
            "resource_analysis": {},
            "alerts": [],
            "recommendations": []
        }
        
        critical_count = 0
        warning_count = 0
        
        for resource_type, capacity in resource_capacities.items():
            resource_analysis = self._analyze_resource_capacity(capacity)
            analysis["resource_analysis"][resource_type.value] = resource_analysis
            
            # Generate alerts if needed
            alert = self._generate_capacity_alert(capacity, resource_analysis)
            if alert:
                analysis["alerts"].append(alert)
                
                if alert.status == CapacityStatus.CRITICAL:
                    critical_count += 1
                elif alert.status == CapacityStatus.WARNING:
                    warning_count += 1
        
        # Determine overall status
        if critical_count > 0:
            analysis["overall_status"] = CapacityStatus.CRITICAL
        elif warning_count > 0:
            analysis["overall_status"] = CapacityStatus.WARNING
        
        # Generate cross-resource recommendations
        cross_recommendations = self._generate_cross_resource_recommendations(resource_capacities)
        analysis["recommendations"].extend(cross_recommendations)
        
        return analysis
    
    def _analyze_resource_capacity(self, capacity: ResourceCapacity) -> Dict[str, Any]:
        """Analyze individual resource capacity"""
        utilization = capacity.utilization_percent / 100.0
        
        # Determine status
        if utilization >= self.config.critical_threshold:
            status = CapacityStatus.CRITICAL
        elif utilization >= self.config.warning_threshold:
            status = CapacityStatus.WARNING
        elif utilization <= self.config.over_provision_threshold:
            status = CapacityStatus.OVER_PROVISIONED
        else:
            status = CapacityStatus.OPTIMAL
        
        # Calculate headroom
        headroom_percent = (1.0 - utilization) * 100
        headroom_absolute = capacity.total_capacity - capacity.current_usage
        
        return {
            "status": status.value,
            "utilization_percent": capacity.utilization_percent,
            "headroom_percent": headroom_percent,
            "headroom_absolute": headroom_absolute,
            "unit": capacity.unit,
            "thresholds": {
                "warning": self.config.warning_threshold * 100,
                "critical": self.config.critical_threshold * 100,
                "over_provision": self.config.over_provision_threshold * 100
            }
        }
    
    def _generate_capacity_alert(
        self,
        capacity: ResourceCapacity,
        analysis: Dict[str, Any]
    ) -> Optional[CapacityAlert]:
        """Generate capacity alert if thresholds are breached"""
        status_str = analysis["status"]
        
        if status_str in ["warning", "critical", "over_provisioned"]:
            status = CapacityStatus(status_str)
            
            # Determine threshold breached
            utilization = capacity.utilization_percent / 100.0
            if utilization >= self.config.critical_threshold:
                threshold = f"Critical ({self.config.critical_threshold * 100:.0f}%)"
                priority = "high"
            elif utilization >= self.config.warning_threshold:
                threshold = f"Warning ({self.config.warning_threshold * 100:.0f}%)"
                priority = "medium"
            else:
                threshold = f"Over-provisioned (<{self.config.over_provision_threshold * 100:.0f}%)"
                priority = "low"
            
            # Generate recommended actions
            actions = self._get_recommended_actions(capacity, status)
            
            return CapacityAlert(
                resource_type=capacity.resource_type,
                status=status,
                current_utilization=capacity.utilization_percent,
                threshold_breached=threshold,
                recommended_actions=actions,
                priority=priority
            )
        
        return None
    
    def _get_recommended_actions(
        self,
        capacity: ResourceCapacity,
        status: CapacityStatus
    ) -> List[str]:
        """Get recommended actions for capacity status"""
        actions = []
        resource_name = capacity.resource_type.value
        
        if status == CapacityStatus.CRITICAL:
            actions.extend([
                f"Immediately scale up {resource_name} capacity",
                f"Investigate high {resource_name} usage patterns",
                f"Consider emergency load balancing for {resource_name}",
                f"Set up auto-scaling if not already configured"
            ])
        
        elif status == CapacityStatus.WARNING:
            actions.extend([
                f"Plan to scale up {resource_name} capacity soon",
                f"Monitor {resource_name} usage trends closely",
                f"Review {resource_name} optimization opportunities",
                f"Prepare scaling procedures"
            ])
        
        elif status == CapacityStatus.OVER_PROVISIONED:
            actions.extend([
                f"Consider scaling down {resource_name} to optimize costs",
                f"Analyze {resource_name} usage patterns for right-sizing",
                f"Review {resource_name} allocation efficiency"
            ])
        
        return actions
    
    def _generate_cross_resource_recommendations(
        self,
        capacities: Dict[ResourceType, ResourceCapacity]
    ) -> List[str]:
        """Generate recommendations based on cross-resource analysis"""
        recommendations = []
        
        # Check for resource imbalances
        cpu_util = capacities.get(ResourceType.CPU)
        memory_util = capacities.get(ResourceType.MEMORY)
        
        if cpu_util and memory_util:
            cpu_pct = cpu_util.utilization_percent
            mem_pct = memory_util.utilization_percent
            
            # Detect imbalances
            if cpu_pct > 80 and mem_pct < 40:
                recommendations.append("CPU is high while memory is low - consider CPU-optimized instances")
            elif mem_pct > 80 and cpu_pct < 40:
                recommendations.append("Memory is high while CPU is low - consider memory-optimized instances")
        
        # Check for disk and database correlation
        disk_util = capacities.get(ResourceType.DISK)
        db_util = capacities.get(ResourceType.DATABASE)
        
        if disk_util and db_util and disk_util.utilization_percent > 70 and db_util.utilization_percent > 70:
            recommendations.append("Both disk and database utilization are high - consider database optimization and storage scaling")
        
        return recommendations


class ScalingAdvisor:
    """
    Provides intelligent scaling recommendations based on capacity analysis and forecasts.
    """
    
    def __init__(self, config: CapacityPlanningConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def generate_scaling_recommendations(
        self,
        current_capacities: Dict[ResourceType, ResourceCapacity],
        forecasts: Dict[ResourceType, CapacityForecast]
    ) -> List[ScalingRecommendation]:
        """
        Generate scaling recommendations based on current state and forecasts.
        
        Args:
            current_capacities: Current capacity utilization
            forecasts: Capacity forecasts for each resource
            
        Returns:
            List of scaling recommendations
        """
        recommendations = []
        
        for resource_type in current_capacities.keys():
            capacity = current_capacities[resource_type]
            forecast = forecasts.get(resource_type)
            
            recommendation = self._generate_resource_recommendation(capacity, forecast)
            if recommendation:
                recommendations.append(recommendation)
        
        # Sort by urgency
        urgency_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
        recommendations.sort(key=lambda r: urgency_order.get(r.urgency, 3))
        
        return recommendations
    
    def _generate_resource_recommendation(
        self,
        capacity: ResourceCapacity,
        forecast: Optional[CapacityForecast]
    ) -> Optional[ScalingRecommendation]:
        """Generate scaling recommendation for a specific resource"""
        current_util = capacity.utilization_percent / 100.0
        
        # Current state analysis
        if current_util >= self.config.critical_threshold:
            return self._create_urgent_scale_up_recommendation(capacity, forecast)
        elif current_util >= self.config.warning_threshold:
            return self._create_planned_scale_up_recommendation(capacity, forecast)
        elif current_util <= self.config.over_provision_threshold:
            return self._create_scale_down_recommendation(capacity, forecast)
        
        # Future state analysis with forecast
        if forecast and forecast.confidence in [PredictionConfidence.HIGH, PredictionConfidence.VERY_HIGH]:
            return self._analyze_forecast_scaling(capacity, forecast)
        
        return None
    
    def _create_urgent_scale_up_recommendation(
        self,
        capacity: ResourceCapacity,
        forecast: Optional[CapacityForecast]
    ) -> ScalingRecommendation:
        """Create urgent scale-up recommendation"""
        # Calculate recommended capacity with buffer
        current_usage = capacity.current_usage
        recommended_capacity = current_usage / (1.0 - self.config.scaling_buffer)
        
        # Consider forecast if available
        if forecast and forecast.predicted_peak > current_usage:
            recommended_capacity = max(
                recommended_capacity,
                forecast.predicted_peak / (1.0 - self.config.scaling_buffer)
            )
        
        return ScalingRecommendation(
            resource_type=capacity.resource_type,
            scaling_direction=ScalingDirection.UP,
            recommended_capacity=recommended_capacity,
            current_capacity=capacity.total_capacity,
            urgency="critical",
            implementation_timeline="Immediate (within 1 hour)",
            reasoning=f"Current utilization ({capacity.utilization_percent:.1f}%) exceeds critical threshold",
            prerequisites=["Verify scaling resources are available", "Prepare monitoring for scaled resources"]
        )
    
    def _create_planned_scale_up_recommendation(
        self,
        capacity: ResourceCapacity,
        forecast: Optional[CapacityForecast]
    ) -> ScalingRecommendation:
        """Create planned scale-up recommendation"""
        current_usage = capacity.current_usage
        recommended_capacity = current_usage / (1.0 - self.config.scaling_buffer)
        
        # Consider forecast growth
        if forecast:
            if forecast.growth_rate > self.config.significant_growth_rate:
                # Aggressive scaling for high growth
                recommended_capacity = forecast.predicted_peak / (1.0 - self.config.scaling_buffer)
                urgency = "high"
                timeline = "Within 24 hours"
            else:
                urgency = "medium"
                timeline = "Within 3-5 days"
        else:
            urgency = "medium"
            timeline = "Within 3-5 days"
        
        return ScalingRecommendation(
            resource_type=capacity.resource_type,
            scaling_direction=ScalingDirection.UP,
            recommended_capacity=recommended_capacity,
            current_capacity=capacity.total_capacity,
            urgency=urgency,
            implementation_timeline=timeline,
            reasoning=f"Current utilization ({capacity.utilization_percent:.1f}%) approaching limits",
            prerequisites=["Plan scaling window", "Prepare rollback plan", "Notify stakeholders"]
        )
    
    def _create_scale_down_recommendation(
        self,
        capacity: ResourceCapacity,
        forecast: Optional[CapacityForecast]
    ) -> Optional[ScalingRecommendation]:
        """Create scale-down recommendation for cost optimization"""
        if not self.config.cost_optimization_enabled:
            return None
        
        current_usage = capacity.current_usage
        
        # Check if forecast shows future growth
        if forecast and forecast.growth_rate > 0.05:  # 5% growth
            return None  # Don't scale down if growth is expected
        
        # Calculate optimal capacity
        peak_usage = current_usage
        if forecast:
            peak_usage = max(current_usage, forecast.predicted_peak)
        
        # Add safety buffer
        recommended_capacity = peak_usage / (1.0 - self.config.scaling_buffer)
        
        # Only recommend if significant savings
        if recommended_capacity < capacity.total_capacity * 0.8:  # At least 20% reduction
            return ScalingRecommendation(
                resource_type=capacity.resource_type,
                scaling_direction=ScalingDirection.DOWN,
                recommended_capacity=recommended_capacity,
                current_capacity=capacity.total_capacity,
                urgency="low",
                implementation_timeline="During next maintenance window",
                reasoning=f"Low utilization ({capacity.utilization_percent:.1f}%) presents cost optimization opportunity",
                prerequisites=["Validate minimal usage pattern", "Ensure auto-scaling can handle spikes", "Plan monitoring"]
            )
        
        return None
    
    def _analyze_forecast_scaling(
        self,
        capacity: ResourceCapacity,
        forecast: CapacityForecast
    ) -> Optional[ScalingRecommendation]:
        """Analyze forecast to determine proactive scaling needs"""
        current_util = capacity.utilization_percent / 100.0
        predicted_peak_util = forecast.predicted_peak / capacity.total_capacity
        
        # Check if forecast shows future capacity issues
        if predicted_peak_util >= self.config.critical_threshold:
            days_to_critical = None
            
            # Find when we'll hit critical threshold
            for date, predicted_value in forecast.predicted_usage:
                predicted_util = predicted_value / capacity.total_capacity
                if predicted_util >= self.config.critical_threshold:
                    days_to_critical = (date - datetime.utcnow()).days
                    break
            
            if days_to_critical and days_to_critical <= 7:  # Within a week
                recommended_capacity = forecast.predicted_peak / (1.0 - self.config.scaling_buffer)
                
                return ScalingRecommendation(
                    resource_type=capacity.resource_type,
                    scaling_direction=ScalingDirection.UP,
                    recommended_capacity=recommended_capacity,
                    current_capacity=capacity.total_capacity,
                    urgency="high" if days_to_critical <= 3 else "medium",
                    implementation_timeline=f"Within {days_to_critical} days",
                    reasoning=f"Forecast predicts capacity exhaustion in {days_to_critical} days",
                    prerequisites=["Validate forecast accuracy", "Prepare scaling procedure", "Set up monitoring"]
                )
        
        return None


class CapacityPlanningEngine:
    """
    Main capacity planning engine that orchestrates forecasting, analysis, and recommendations.
    """
    
    def __init__(self, config: Optional[CapacityPlanningConfig] = None):
        self.config = config or CapacityPlanningConfig()
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.forecaster = ResourceForecaster(self.config)
        self.analyzer = CapacityAnalyzer(self.config)
        self.scaling_advisor = ScalingAdvisor(self.config)
        
        # Caching
        self.forecast_cache = {}
        self.last_analysis_time = None
    
    async def run_capacity_planning(
        self,
        resource_data: Dict[ResourceType, List[MetricPoint]],
        current_capacities: Dict[ResourceType, ResourceCapacity]
    ) -> Dict[str, Any]:
        """
        Run comprehensive capacity planning analysis.
        
        Args:
            resource_data: Historical resource usage data
            current_capacities: Current capacity information
            
        Returns:
            Comprehensive capacity planning report
        """
        planning_report = {
            "timestamp": datetime.utcnow(),
            "current_analysis": {},
            "forecasts": {},
            "scaling_recommendations": [],
            "capacity_alerts": [],
            "summary": {}
        }
        
        try:
            # Analyze current capacity
            current_analysis = self.analyzer.analyze_current_capacity(current_capacities)
            planning_report["current_analysis"] = current_analysis
            planning_report["capacity_alerts"] = current_analysis["alerts"]
            
            # Generate forecasts
            forecasts = {}
            for resource_type, historical_data in resource_data.items():
                forecast = await self.forecaster.forecast_resource_usage(
                    resource_type, historical_data
                )
                forecasts[resource_type] = forecast
                
                # Convert to serializable format
                planning_report["forecasts"][resource_type.value] = {
                    "forecast_horizon_days": forecast.forecast_horizon_days,
                    "predicted_peak": forecast.predicted_peak,
                    "predicted_peak_time": forecast.predicted_peak_time.isoformat(),
                    "growth_rate": forecast.growth_rate,
                    "seasonal_pattern": forecast.seasonal_pattern,
                    "confidence": forecast.confidence.value,
                    "model_accuracy": forecast.model_accuracy
                }
            
            # Generate scaling recommendations
            recommendations = self.scaling_advisor.generate_scaling_recommendations(
                current_capacities, forecasts
            )
            
            planning_report["scaling_recommendations"] = [
                {
                    "resource_type": rec.resource_type.value,
                    "scaling_direction": rec.scaling_direction.value,
                    "recommended_capacity": rec.recommended_capacity,
                    "current_capacity": rec.current_capacity,
                    "urgency": rec.urgency,
                    "implementation_timeline": rec.implementation_timeline,
                    "reasoning": rec.reasoning,
                    "prerequisites": rec.prerequisites
                }
                for rec in recommendations
            ]
            
            # Generate summary
            planning_report["summary"] = self._generate_summary(
                current_analysis, forecasts, recommendations
            )
            
            self.last_analysis_time = datetime.utcnow()
            
        except Exception as e:
            self.logger.error(f"Capacity planning analysis failed: {e}")
            planning_report["error"] = str(e)
        
        return planning_report
    
    def _generate_summary(
        self,
        current_analysis: Dict[str, Any],
        forecasts: Dict[ResourceType, CapacityForecast],
        recommendations: List[ScalingRecommendation]
    ) -> Dict[str, Any]:
        """Generate executive summary of capacity planning results"""
        # Count issues by severity
        critical_alerts = len([a for a in current_analysis["alerts"] if a.priority == "high"])
        warning_alerts = len([a for a in current_analysis["alerts"] if a.priority == "medium"])
        
        # Count recommendations by urgency
        critical_recs = len([r for r in recommendations if r.urgency == "critical"])
        high_recs = len([r for r in recommendations if r.urgency == "high"])
        
        # Analyze forecast trends
        growing_resources = []
        stable_resources = []
        
        for resource_type, forecast in forecasts.items():
            if forecast.growth_rate > 0.1:  # 10% growth
                growing_resources.append(resource_type.value)
            elif abs(forecast.growth_rate) < 0.05:  # Stable within 5%
                stable_resources.append(resource_type.value)
        
        return {
            "overall_status": current_analysis["overall_status"],
            "total_alerts": len(current_analysis["alerts"]),
            "critical_alerts": critical_alerts,
            "warning_alerts": warning_alerts,
            "total_recommendations": len(recommendations),
            "urgent_recommendations": critical_recs + high_recs,
            "growing_resources": growing_resources,
            "stable_resources": stable_resources,
            "forecast_horizon_days": self.config.forecast_horizon_days,
            "next_analysis_recommended": datetime.utcnow() + timedelta(hours=self.config.model_retraining_interval_hours)
        }