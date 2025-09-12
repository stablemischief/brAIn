"""
Predictive analysis engine for failure prediction and system health forecasting.

This module provides advanced predictive analytics capabilities for anticipating
system failures, resource exhaustion, performance degradation, and other issues
before they occur.
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
from collections import deque, defaultdict
import json

# Optional ML imports
try:
    from sklearn.linear_model import LinearRegression, Ridge
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.preprocessing import StandardScaler, PolynomialFeatures
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    from sklearn.model_selection import train_test_split
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False

from .anomaly_detection import AnomalyResult, MetricPoint, AnomalyType, AnomalySeverity


class PredictionType(Enum):
    """Types of predictions the system can make"""
    FAILURE = "failure"
    CAPACITY = "capacity"
    PERFORMANCE = "performance"
    ANOMALY = "anomaly"
    TREND = "trend"


class PredictionConfidence(Enum):
    """Confidence levels for predictions"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"


@dataclass
class PredictionResult:
    """Result of a predictive analysis"""
    metric_name: str
    prediction_type: PredictionType
    predicted_value: float
    prediction_time: datetime
    confidence: PredictionConfidence
    confidence_score: float
    time_to_event: Optional[timedelta] = None
    probability: Optional[float] = None
    context: Dict[str, Any] = field(default_factory=dict)
    description: str = ""
    recommendations: List[str] = field(default_factory=list)


@dataclass
class FailurePrediction:
    """Specific failure prediction result"""
    failure_type: str
    probability: float
    estimated_time: datetime
    confidence: PredictionConfidence
    contributing_factors: List[str]
    impact_assessment: str
    mitigation_strategies: List[str]


@dataclass
class PredictiveConfig:
    """Configuration for predictive analysis"""
    # Time series parameters
    min_history_points: int = 100
    prediction_horizon: int = 24  # Hours ahead
    confidence_threshold: float = 0.7
    
    # Model parameters
    models_to_use: List[str] = field(default_factory=lambda: ["linear", "polynomial", "ensemble"])
    feature_window: int = 50
    polynomial_degree: int = 2
    
    # Failure prediction
    failure_threshold_multiplier: float = 2.0
    critical_metrics: List[str] = field(default_factory=lambda: [
        "cpu_usage", "memory_usage", "disk_usage", "error_rate", "response_time"
    ])
    
    # Capacity planning
    capacity_warning_threshold: float = 0.8
    capacity_critical_threshold: float = 0.9
    growth_rate_window: int = 30  # Days
    
    # Model retraining
    retrain_interval: timedelta = timedelta(hours=6)
    min_accuracy_threshold: float = 0.6


class FailurePredictor:
    """
    Predicts system failures based on historical patterns and current trends.
    """
    
    def __init__(self, config: PredictiveConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Failure pattern storage
        self.failure_patterns = defaultdict(list)
        self.failure_history = []
        
        # Model storage
        self.failure_models = {}
        self.last_retrain_time = {}
    
    async def predict_failures(
        self,
        metric_data: Dict[str, List[MetricPoint]],
        recent_anomalies: List[AnomalyResult]
    ) -> List[FailurePrediction]:
        """
        Predict potential system failures based on metrics and anomaly patterns.
        
        Args:
            metric_data: Dictionary of metric names to time series data
            recent_anomalies: Recent anomaly detections for pattern analysis
            
        Returns:
            List of failure predictions with probabilities and timing
        """
        failure_predictions = []
        
        # Analyze critical metrics for failure patterns
        for metric_name, data in metric_data.items():
            if metric_name in self.config.critical_metrics and len(data) >= self.config.min_history_points:
                
                # Check for rapid degradation patterns
                degradation_prediction = await self._predict_degradation_failure(metric_name, data)
                if degradation_prediction:
                    failure_predictions.append(degradation_prediction)
                
                # Check for capacity exhaustion
                capacity_prediction = await self._predict_capacity_failure(metric_name, data)
                if capacity_prediction:
                    failure_predictions.append(capacity_prediction)
        
        # Analyze anomaly patterns for failure prediction
        anomaly_predictions = await self._predict_anomaly_based_failures(recent_anomalies)
        failure_predictions.extend(anomaly_predictions)
        
        # Cross-metric correlation analysis
        correlation_predictions = await self._predict_correlation_failures(metric_data)
        failure_predictions.extend(correlation_predictions)
        
        # Sort by probability and time to failure
        failure_predictions.sort(key=lambda p: (p.probability, -p.estimated_time.timestamp()), reverse=True)
        
        return failure_predictions
    
    async def _predict_degradation_failure(
        self,
        metric_name: str,
        data: List[MetricPoint]
    ) -> Optional[FailurePrediction]:
        """Predict failure due to rapid metric degradation"""
        if len(data) < 20:
            return None
        
        # Calculate trend over recent data
        recent_data = data[-20:]  # Last 20 points
        values = [point.value for point in recent_data]
        timestamps = [point.timestamp.timestamp() for point in recent_data]
        
        # Calculate slope
        if len(values) < 2:
            return None
        
        x = np.array(timestamps)
        y = np.array(values)
        
        try:
            # Linear regression for trend
            slope = np.polyfit(x, y, 1)[0]
            
            # Check if slope indicates rapid degradation
            threshold_slope = self._get_degradation_threshold(metric_name)
            
            if abs(slope) > threshold_slope:
                # Estimate time to failure
                current_value = values[-1]
                failure_threshold = self._get_failure_threshold(metric_name, current_value)
                
                if slope != 0:
                    time_to_threshold = (failure_threshold - current_value) / slope
                    estimated_failure_time = datetime.utcnow() + timedelta(seconds=time_to_threshold)
                    
                    # Calculate probability based on trend strength
                    trend_strength = abs(slope) / threshold_slope
                    probability = min(trend_strength * 0.3, 0.9)  # Cap at 90%
                    
                    if probability > 0.2:  # Only predict if reasonably probable
                        return FailurePrediction(
                            failure_type=f"{metric_name}_degradation",
                            probability=probability,
                            estimated_time=estimated_failure_time,
                            confidence=self._probability_to_confidence(probability),
                            contributing_factors=[f"Rapid {metric_name} degradation", f"Slope: {slope:.4f}"],
                            impact_assessment=self._assess_impact(metric_name),
                            mitigation_strategies=self._get_mitigation_strategies(metric_name)
                        )
            
        except Exception as e:
            self.logger.error(f"Error predicting degradation failure for {metric_name}: {e}")
        
        return None
    
    async def _predict_capacity_failure(
        self,
        metric_name: str,
        data: List[MetricPoint]
    ) -> Optional[FailurePrediction]:
        """Predict failure due to capacity exhaustion"""
        if "usage" not in metric_name.lower():
            return None
        
        values = [point.value for point in data[-50:]]  # Last 50 points
        
        # Calculate growth rate
        if len(values) < 10:
            return None
        
        # Simple exponential growth check
        recent_avg = statistics.mean(values[-10:])
        older_avg = statistics.mean(values[-20:-10]) if len(values) >= 20 else recent_avg
        
        if older_avg <= 0:
            return None
        
        growth_rate = (recent_avg - older_avg) / older_avg
        
        # Predict capacity exhaustion
        if growth_rate > 0.05:  # 5% growth rate threshold
            current_usage = values[-1]
            capacity_limit = 100.0 if "percent" in metric_name.lower() else 1.0
            
            if current_usage > capacity_limit * 0.5:  # Already above 50% usage
                # Estimate time to capacity exhaustion
                time_to_exhaustion = (capacity_limit - current_usage) / (growth_rate * older_avg)
                estimated_failure_time = datetime.utcnow() + timedelta(hours=time_to_exhaustion)
                
                # Calculate probability based on current usage and growth rate
                usage_factor = current_usage / capacity_limit
                growth_factor = min(growth_rate * 10, 1.0)  # Normalize growth rate
                probability = (usage_factor + growth_factor) / 2
                
                if probability > 0.3:
                    return FailurePrediction(
                        failure_type=f"{metric_name}_capacity_exhaustion",
                        probability=probability,
                        estimated_time=estimated_failure_time,
                        confidence=self._probability_to_confidence(probability),
                        contributing_factors=[
                            f"High {metric_name} usage: {current_usage:.1f}%",
                            f"Growth rate: {growth_rate*100:.1f}%"
                        ],
                        impact_assessment="System capacity exhaustion may cause service degradation",
                        mitigation_strategies=[
                            "Scale up resources",
                            "Optimize resource usage",
                            "Implement auto-scaling"
                        ]
                    )
        
        return None
    
    async def _predict_anomaly_based_failures(
        self,
        recent_anomalies: List[AnomalyResult]
    ) -> List[FailurePrediction]:
        """Predict failures based on anomaly patterns"""
        predictions = []
        
        # Group anomalies by metric and analyze patterns
        metric_anomalies = defaultdict(list)
        for anomaly in recent_anomalies:
            metric_anomalies[anomaly.metric_name].append(anomaly)
        
        for metric_name, anomalies in metric_anomalies.items():
            # Check for increasing anomaly frequency
            recent_count = len([a for a in anomalies if a.timestamp > datetime.utcnow() - timedelta(hours=2)])
            older_count = len([a for a in anomalies if 
                              datetime.utcnow() - timedelta(hours=4) < a.timestamp <= datetime.utcnow() - timedelta(hours=2)])
            
            if recent_count > older_count * 1.5 and recent_count >= 3:
                # Anomaly frequency is increasing
                probability = min(recent_count / 10.0, 0.8)  # Scale with frequency
                estimated_time = datetime.utcnow() + timedelta(hours=1)  # Short-term prediction
                
                predictions.append(FailurePrediction(
                    failure_type=f"{metric_name}_anomaly_cascade",
                    probability=probability,
                    estimated_time=estimated_time,
                    confidence=self._probability_to_confidence(probability),
                    contributing_factors=[
                        f"Increasing anomaly frequency in {metric_name}",
                        f"Recent anomalies: {recent_count}, Previous: {older_count}"
                    ],
                    impact_assessment="Cascading anomalies may indicate impending system failure",
                    mitigation_strategies=[
                        "Investigate root cause of anomalies",
                        "Implement circuit breakers",
                        "Scale affected components"
                    ]
                ))
        
        return predictions
    
    async def _predict_correlation_failures(
        self,
        metric_data: Dict[str, List[MetricPoint]]
    ) -> List[FailurePrediction]:
        """Predict failures based on cross-metric correlations"""
        predictions = []
        
        # Check for dangerous metric combinations
        if "error_rate" in metric_data and "response_time" in metric_data:
            error_data = metric_data["error_rate"]
            response_data = metric_data["response_time"]
            
            if len(error_data) >= 10 and len(response_data) >= 10:
                recent_errors = statistics.mean([p.value for p in error_data[-10:]])
                recent_response = statistics.mean([p.value for p in response_data[-10:]])
                
                # High error rate + high response time = likely system overload
                if recent_errors > 0.05 and recent_response > 1000:  # 5% error rate, 1s response time
                    probability = min((recent_errors * 10) + (recent_response / 5000), 0.9)
                    
                    predictions.append(FailurePrediction(
                        failure_type="system_overload",
                        probability=probability,
                        estimated_time=datetime.utcnow() + timedelta(minutes=30),
                        confidence=self._probability_to_confidence(probability),
                        contributing_factors=[
                            f"High error rate: {recent_errors*100:.1f}%",
                            f"High response time: {recent_response:.0f}ms"
                        ],
                        impact_assessment="System overload may cause service outage",
                        mitigation_strategies=[
                            "Implement load balancing",
                            "Scale up compute resources",
                            "Implement request throttling"
                        ]
                    ))
        
        return predictions
    
    def _get_degradation_threshold(self, metric_name: str) -> float:
        """Get degradation threshold for a metric"""
        thresholds = {
            "cpu_usage": 5.0,  # 5% per time unit
            "memory_usage": 10.0,
            "disk_usage": 2.0,
            "error_rate": 0.01,  # 1% per time unit
            "response_time": 100.0  # 100ms per time unit
        }
        return thresholds.get(metric_name, 1.0)
    
    def _get_failure_threshold(self, metric_name: str, current_value: float) -> float:
        """Get failure threshold for a metric"""
        if "usage" in metric_name:
            return 100.0  # 100% usage
        elif "error_rate" in metric_name:
            return 0.5  # 50% error rate
        elif "response_time" in metric_name:
            return current_value * self.config.failure_threshold_multiplier
        else:
            return current_value * self.config.failure_threshold_multiplier
    
    def _assess_impact(self, metric_name: str) -> str:
        """Assess the impact of a metric failure"""
        impact_map = {
            "cpu_usage": "High CPU usage may cause performance degradation and timeouts",
            "memory_usage": "Memory exhaustion may cause application crashes and instability",
            "disk_usage": "Disk space exhaustion may prevent data storage and cause failures",
            "error_rate": "High error rates indicate service degradation and user impact",
            "response_time": "Slow response times affect user experience and may trigger timeouts"
        }
        return impact_map.get(metric_name, f"Failure in {metric_name} may affect system stability")
    
    def _get_mitigation_strategies(self, metric_name: str) -> List[str]:
        """Get mitigation strategies for a metric failure"""
        strategies_map = {
            "cpu_usage": [
                "Scale up CPU resources",
                "Optimize CPU-intensive operations",
                "Implement caching to reduce CPU load"
            ],
            "memory_usage": [
                "Increase memory allocation",
                "Optimize memory usage patterns",
                "Implement memory cleanup routines"
            ],
            "disk_usage": [
                "Add more storage capacity",
                "Implement log rotation",
                "Archive or compress old data"
            ],
            "error_rate": [
                "Investigate and fix underlying issues",
                "Implement better error handling",
                "Add circuit breakers"
            ],
            "response_time": [
                "Optimize database queries",
                "Implement caching strategies",
                "Scale up infrastructure"
            ]
        }
        return strategies_map.get(metric_name, ["Investigate root cause", "Scale resources"])
    
    def _probability_to_confidence(self, probability: float) -> PredictionConfidence:
        """Convert probability to confidence level"""
        if probability >= 0.8:
            return PredictionConfidence.VERY_HIGH
        elif probability >= 0.6:
            return PredictionConfidence.HIGH
        elif probability >= 0.4:
            return PredictionConfidence.MEDIUM
        else:
            return PredictionConfidence.LOW


class TrendPredictor:
    """
    Predicts future trends in metrics using various forecasting models.
    """
    
    def __init__(self, config: PredictiveConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.models = {}
        self.scalers = {}
    
    async def predict_trend(
        self,
        data: List[MetricPoint],
        metric_name: str,
        prediction_hours: int = 24
    ) -> PredictionResult:
        """
        Predict future trend for a metric.
        
        Args:
            data: Historical metric data
            metric_name: Name of the metric
            prediction_hours: Hours ahead to predict
            
        Returns:
            Trend prediction result
        """
        if len(data) < self.config.min_history_points:
            return PredictionResult(
                metric_name=metric_name,
                prediction_type=PredictionType.TREND,
                predicted_value=data[-1].value if data else 0.0,
                prediction_time=datetime.utcnow() + timedelta(hours=prediction_hours),
                confidence=PredictionConfidence.LOW,
                confidence_score=0.1,
                description="Insufficient data for trend prediction"
            )
        
        # Prepare data for prediction
        values = [point.value for point in data]
        timestamps = [point.timestamp.timestamp() for point in data]
        
        predictions = {}
        confidences = {}
        
        # Try different models
        if "linear" in self.config.models_to_use:
            linear_pred, linear_conf = await self._linear_prediction(timestamps, values, prediction_hours)
            predictions["linear"] = linear_pred
            confidences["linear"] = linear_conf
        
        if "polynomial" in self.config.models_to_use:
            poly_pred, poly_conf = await self._polynomial_prediction(timestamps, values, prediction_hours)
            predictions["polynomial"] = poly_pred
            confidences["polynomial"] = poly_conf
        
        if "ensemble" in self.config.models_to_use and ML_AVAILABLE:
            ensemble_pred, ensemble_conf = await self._ensemble_prediction(timestamps, values, prediction_hours)
            predictions["ensemble"] = ensemble_pred
            confidences["ensemble"] = ensemble_conf
        
        # Select best prediction based on confidence
        if not predictions:
            return PredictionResult(
                metric_name=metric_name,
                prediction_type=PredictionType.TREND,
                predicted_value=values[-1],
                prediction_time=datetime.utcnow() + timedelta(hours=prediction_hours),
                confidence=PredictionConfidence.LOW,
                confidence_score=0.1,
                description="No prediction models available"
            )
        
        best_model = max(confidences.items(), key=lambda x: x[1])[0]
        best_prediction = predictions[best_model]
        best_confidence = confidences[best_model]
        
        # Convert confidence score to enum
        confidence_level = self._score_to_confidence(best_confidence)
        
        return PredictionResult(
            metric_name=metric_name,
            prediction_type=PredictionType.TREND,
            predicted_value=best_prediction,
            prediction_time=datetime.utcnow() + timedelta(hours=prediction_hours),
            confidence=confidence_level,
            confidence_score=best_confidence,
            context={
                "model_used": best_model,
                "all_predictions": predictions,
                "all_confidences": confidences,
                "data_points": len(data)
            },
            description=f"Trend prediction using {best_model} model with {best_confidence:.2f} confidence"
        )
    
    async def _linear_prediction(
        self,
        timestamps: List[float],
        values: List[float],
        prediction_hours: int
    ) -> Tuple[float, float]:
        """Linear regression prediction"""
        try:
            if len(timestamps) < 2:
                return values[-1] if values else 0.0, 0.1
            
            # Fit linear model
            slope, intercept = np.polyfit(timestamps, values, 1)
            
            # Predict future value
            future_timestamp = timestamps[-1] + (prediction_hours * 3600)  # Convert hours to seconds
            predicted_value = slope * future_timestamp + intercept
            
            # Calculate confidence based on R-squared
            predicted_values = [slope * t + intercept for t in timestamps]
            ss_res = sum((values[i] - predicted_values[i])**2 for i in range(len(values)))
            ss_tot = sum((values[i] - statistics.mean(values))**2 for i in range(len(values)))
            
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            confidence = max(r_squared, 0.1)  # Minimum confidence of 10%
            
            return predicted_value, confidence
            
        except Exception as e:
            self.logger.error(f"Linear prediction error: {e}")
            return values[-1] if values else 0.0, 0.1
    
    async def _polynomial_prediction(
        self,
        timestamps: List[float],
        values: List[float],
        prediction_hours: int
    ) -> Tuple[float, float]:
        """Polynomial regression prediction"""
        try:
            if len(timestamps) < self.config.polynomial_degree + 1:
                return await self._linear_prediction(timestamps, values, prediction_hours)
            
            # Fit polynomial model
            coeffs = np.polyfit(timestamps, values, self.config.polynomial_degree)
            poly_func = np.poly1d(coeffs)
            
            # Predict future value
            future_timestamp = timestamps[-1] + (prediction_hours * 3600)
            predicted_value = poly_func(future_timestamp)
            
            # Calculate confidence based on fit quality
            predicted_values = [poly_func(t) for t in timestamps]
            mse = mean_squared_error(values, predicted_values)
            value_range = max(values) - min(values)
            
            # Normalize MSE to get confidence
            if value_range > 0:
                normalized_mse = mse / (value_range ** 2)
                confidence = max(1 - normalized_mse, 0.1)
            else:
                confidence = 0.5
            
            return predicted_value, confidence
            
        except Exception as e:
            self.logger.error(f"Polynomial prediction error: {e}")
            return await self._linear_prediction(timestamps, values, prediction_hours)
    
    async def _ensemble_prediction(
        self,
        timestamps: List[float],
        values: List[float],
        prediction_hours: int
    ) -> Tuple[float, float]:
        """Ensemble model prediction using RandomForest"""
        if not ML_AVAILABLE:
            return await self._linear_prediction(timestamps, values, prediction_hours)
        
        try:
            # Create features (timestamp, lag features, moving averages)
            features = []
            targets = []
            
            window_size = min(10, len(values) // 2)
            
            for i in range(window_size, len(values)):
                feature_row = [
                    timestamps[i],  # Current timestamp
                    values[i-1],  # Previous value
                    statistics.mean(values[max(0, i-window_size):i]),  # Moving average
                    max(values[max(0, i-window_size):i]) - min(values[max(0, i-window_size):i])  # Range
                ]
                features.append(feature_row)
                targets.append(values[i])
            
            if len(features) < 5:
                return await self._linear_prediction(timestamps, values, prediction_hours)
            
            # Train model
            X = np.array(features)
            y = np.array(targets)
            
            model = RandomForestRegressor(n_estimators=50, random_state=42)
            model.fit(X, y)
            
            # Predict future value
            future_timestamp = timestamps[-1] + (prediction_hours * 3600)
            last_value = values[-1]
            recent_avg = statistics.mean(values[-window_size:])
            recent_range = max(values[-window_size:]) - min(values[-window_size:])
            
            future_features = np.array([[future_timestamp, last_value, recent_avg, recent_range]])
            predicted_value = model.predict(future_features)[0]
            
            # Calculate confidence using out-of-bag score or cross-validation approximation
            if len(features) > 10:
                # Simple train-test split for confidence estimation
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
                model_test = RandomForestRegressor(n_estimators=50, random_state=42)
                model_test.fit(X_train, y_train)
                predictions = model_test.predict(X_test)
                
                mse = mean_squared_error(y_test, predictions)
                value_range = max(values) - min(values)
                
                if value_range > 0:
                    normalized_mse = mse / (value_range ** 2)
                    confidence = max(1 - normalized_mse, 0.1)
                else:
                    confidence = 0.5
            else:
                confidence = 0.5
            
            return predicted_value, confidence
            
        except Exception as e:
            self.logger.error(f"Ensemble prediction error: {e}")
            return await self._linear_prediction(timestamps, values, prediction_hours)
    
    def _score_to_confidence(self, score: float) -> PredictionConfidence:
        """Convert numerical confidence score to enum"""
        if score >= 0.8:
            return PredictionConfidence.VERY_HIGH
        elif score >= 0.65:
            return PredictionConfidence.HIGH
        elif score >= 0.4:
            return PredictionConfidence.MEDIUM
        else:
            return PredictionConfidence.LOW


class PredictiveEngine:
    """
    Main predictive analysis engine that coordinates different prediction modules.
    """
    
    def __init__(self, config: Optional[PredictiveConfig] = None):
        self.config = config or PredictiveConfig()
        self.logger = logging.getLogger(__name__)
        
        # Initialize prediction modules
        self.failure_predictor = FailurePredictor(self.config)
        self.trend_predictor = TrendPredictor(self.config)
        
        # Prediction history
        self.prediction_history = defaultdict(list)
        
    async def run_comprehensive_prediction(
        self,
        metric_data: Dict[str, List[MetricPoint]],
        recent_anomalies: List[AnomalyResult]
    ) -> Dict[str, Any]:
        """
        Run comprehensive predictive analysis across all modules.
        
        Args:
            metric_data: Historical metric data
            recent_anomalies: Recent anomaly detections
            
        Returns:
            Comprehensive prediction results
        """
        results = {
            "timestamp": datetime.utcnow(),
            "failure_predictions": [],
            "trend_predictions": {},
            "capacity_forecasts": {},
            "health_score": 0.0,
            "recommendations": []
        }
        
        try:
            # Failure predictions
            failure_predictions = await self.failure_predictor.predict_failures(
                metric_data, recent_anomalies
            )
            results["failure_predictions"] = [
                {
                    "failure_type": fp.failure_type,
                    "probability": fp.probability,
                    "estimated_time": fp.estimated_time.isoformat(),
                    "confidence": fp.confidence.value,
                    "contributing_factors": fp.contributing_factors,
                    "impact_assessment": fp.impact_assessment,
                    "mitigation_strategies": fp.mitigation_strategies
                }
                for fp in failure_predictions
            ]
            
            # Trend predictions for each metric
            for metric_name, data in metric_data.items():
                if len(data) >= self.config.min_history_points:
                    trend_prediction = await self.trend_predictor.predict_trend(
                        data, metric_name, self.config.prediction_horizon
                    )
                    results["trend_predictions"][metric_name] = {
                        "predicted_value": trend_prediction.predicted_value,
                        "confidence": trend_prediction.confidence.value,
                        "confidence_score": trend_prediction.confidence_score,
                        "prediction_time": trend_prediction.prediction_time.isoformat(),
                        "context": trend_prediction.context
                    }
            
            # Calculate overall health score
            health_score = await self._calculate_health_score(failure_predictions, metric_data)
            results["health_score"] = health_score
            
            # Generate recommendations
            recommendations = await self._generate_recommendations(
                failure_predictions, results["trend_predictions"]
            )
            results["recommendations"] = recommendations
            
            # Store prediction history
            for metric_name in metric_data.keys():
                self.prediction_history[metric_name].append({
                    "timestamp": datetime.utcnow(),
                    "results": results
                })
                
                # Keep only recent history
                cutoff_time = datetime.utcnow() - timedelta(days=7)
                self.prediction_history[metric_name] = [
                    p for p in self.prediction_history[metric_name]
                    if p["timestamp"] > cutoff_time
                ]
            
        except Exception as e:
            self.logger.error(f"Error in comprehensive prediction: {e}")
            results["error"] = str(e)
        
        return results
    
    async def _calculate_health_score(
        self,
        failure_predictions: List[FailurePrediction],
        metric_data: Dict[str, List[MetricPoint]]
    ) -> float:
        """Calculate overall system health score (0-100)"""
        base_score = 100.0
        
        # Deduct points for failure predictions
        for prediction in failure_predictions:
            deduction = prediction.probability * 20  # Up to 20 points per prediction
            base_score -= deduction
        
        # Deduct points for concerning metric trends
        for metric_name, data in metric_data.items():
            if len(data) >= 10:
                recent_values = [p.value for p in data[-10:]]
                
                # Check for concerning patterns
                if "error_rate" in metric_name:
                    avg_error_rate = statistics.mean(recent_values)
                    if avg_error_rate > 0.05:  # 5% error rate
                        base_score -= min(avg_error_rate * 100, 15)
                
                elif "usage" in metric_name:
                    avg_usage = statistics.mean(recent_values)
                    if avg_usage > 80:  # 80% usage
                        base_score -= min((avg_usage - 80) / 2, 10)
        
        return max(base_score, 0.0)
    
    async def _generate_recommendations(
        self,
        failure_predictions: List[FailurePrediction],
        trend_predictions: Dict[str, Any]
    ) -> List[str]:
        """Generate actionable recommendations based on predictions"""
        recommendations = []
        
        # Recommendations based on failure predictions
        for prediction in failure_predictions[:3]:  # Top 3 failure risks
            recommendations.extend(prediction.mitigation_strategies)
        
        # Recommendations based on trends
        for metric_name, trend_data in trend_predictions.items():
            if trend_data["confidence_score"] > 0.7:
                predicted_value = trend_data["predicted_value"]
                
                if "usage" in metric_name and predicted_value > 90:
                    recommendations.append(f"Scale up {metric_name} - predicted to reach {predicted_value:.1f}%")
                
                elif "response_time" in metric_name and predicted_value > 1000:
                    recommendations.append(f"Optimize performance - response time predicted to reach {predicted_value:.0f}ms")
        
        # Remove duplicates while preserving order
        unique_recommendations = []
        seen = set()
        for rec in recommendations:
            if rec not in seen:
                unique_recommendations.append(rec)
                seen.add(rec)
        
        return unique_recommendations[:10]  # Return top 10 recommendations
    
    def get_prediction_accuracy(self, metric_name: str) -> Optional[Dict[str, float]]:
        """Calculate prediction accuracy for a metric over recent history"""
        history = self.prediction_history.get(metric_name, [])
        
        if len(history) < 5:
            return None
        
        # This is a simplified accuracy calculation
        # In a real implementation, you would compare predictions with actual outcomes
        recent_predictions = history[-5:]
        
        # Calculate consistency of predictions as a proxy for accuracy
        confidence_scores = []
        for prediction in recent_predictions:
            trend_pred = prediction["results"]["trend_predictions"].get(metric_name)
            if trend_pred:
                confidence_scores.append(trend_pred["confidence_score"])
        
        if confidence_scores:
            return {
                "average_confidence": statistics.mean(confidence_scores),
                "confidence_consistency": 1.0 - statistics.stdev(confidence_scores) if len(confidence_scores) > 1 else 1.0,
                "prediction_count": len(confidence_scores)
            }
        
        return None