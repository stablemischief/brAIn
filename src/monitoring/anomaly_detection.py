"""
Anomaly detection algorithms for predictive monitoring.

This module provides statistical and machine learning-based anomaly detection
algorithms for identifying unusual patterns in system metrics, processing
performance, and user behavior.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import asyncio
import logging
from collections import deque, defaultdict
import statistics
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

# Optional ML imports - gracefully handle if not available
try:
    from sklearn.ensemble import IsolationForest
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False


class AnomalyType(Enum):
    """Types of anomalies that can be detected"""
    STATISTICAL = "statistical"
    TREND = "trend"
    SEASONAL = "seasonal"
    POINT = "point"
    CONTEXTUAL = "contextual"
    COLLECTIVE = "collective"


class AnomalySeverity(Enum):
    """Severity levels for detected anomalies"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class AnomalyResult:
    """Result of anomaly detection analysis"""
    timestamp: datetime
    metric_name: str
    value: float
    anomaly_type: AnomalyType
    severity: AnomalySeverity
    confidence: float
    threshold: float
    deviation: float
    context: Dict[str, Any] = field(default_factory=dict)
    description: str = ""


@dataclass
class MetricPoint:
    """Single metric data point"""
    timestamp: datetime
    value: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AnomalyDetectionConfig:
    """Configuration for anomaly detection algorithms"""
    # Statistical thresholds
    z_score_threshold: float = 3.0
    iqr_multiplier: float = 1.5
    mad_threshold: float = 3.5  # Median Absolute Deviation
    
    # Time series parameters
    window_size: int = 50
    seasonal_period: int = 24  # Hours for daily seasonality
    trend_sensitivity: float = 0.1
    
    # Machine learning parameters
    contamination_rate: float = 0.1  # Expected anomaly rate
    n_estimators: int = 100
    
    # Detection sensitivity
    sensitivity_level: str = "medium"  # low, medium, high
    min_data_points: int = 10
    
    # Contextual detection
    enable_contextual: bool = True
    context_window: int = 100


class StatisticalAnomalyDetector:
    """
    Statistical anomaly detection using various threshold-based methods.
    """
    
    def __init__(self, config: AnomalyDetectionConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def detect_z_score_anomalies(
        self,
        data: List[MetricPoint],
        metric_name: str
    ) -> List[AnomalyResult]:
        """Detect anomalies using Z-score method"""
        if len(data) < self.config.min_data_points:
            return []
        
        values = [point.value for point in data]
        mean = statistics.mean(values)
        std_dev = statistics.stdev(values) if len(values) > 1 else 0
        
        if std_dev == 0:
            return []
        
        anomalies = []
        threshold = self.config.z_score_threshold
        
        for point in data:
            z_score = abs((point.value - mean) / std_dev)
            
            if z_score > threshold:
                severity = self._calculate_severity(z_score, threshold)
                confidence = min(z_score / threshold, 5.0) / 5.0  # Normalize to 0-1
                
                anomaly = AnomalyResult(
                    timestamp=point.timestamp,
                    metric_name=metric_name,
                    value=point.value,
                    anomaly_type=AnomalyType.STATISTICAL,
                    severity=severity,
                    confidence=confidence,
                    threshold=threshold,
                    deviation=z_score,
                    context={
                        "mean": mean,
                        "std_dev": std_dev,
                        "z_score": z_score
                    },
                    description=f"Value {point.value:.2f} is {z_score:.2f} standard deviations from mean"
                )
                anomalies.append(anomaly)
        
        return anomalies
    
    def detect_iqr_anomalies(
        self,
        data: List[MetricPoint],
        metric_name: str
    ) -> List[AnomalyResult]:
        """Detect anomalies using Interquartile Range (IQR) method"""
        if len(data) < self.config.min_data_points:
            return []
        
        values = [point.value for point in data]
        q1 = np.percentile(values, 25)
        q3 = np.percentile(values, 75)
        iqr = q3 - q1
        
        lower_bound = q1 - (self.config.iqr_multiplier * iqr)
        upper_bound = q3 + (self.config.iqr_multiplier * iqr)
        
        anomalies = []
        
        for point in data:
            if point.value < lower_bound or point.value > upper_bound:
                # Calculate how far outside the bounds
                if point.value < lower_bound:
                    deviation = (lower_bound - point.value) / iqr if iqr > 0 else 0
                else:
                    deviation = (point.value - upper_bound) / iqr if iqr > 0 else 0
                
                severity = self._calculate_severity(deviation, 1.0)
                confidence = min(deviation, 3.0) / 3.0  # Normalize to 0-1
                
                anomaly = AnomalyResult(
                    timestamp=point.timestamp,
                    metric_name=metric_name,
                    value=point.value,
                    anomaly_type=AnomalyType.STATISTICAL,
                    severity=severity,
                    confidence=confidence,
                    threshold=self.config.iqr_multiplier,
                    deviation=deviation,
                    context={
                        "q1": q1,
                        "q3": q3,
                        "iqr": iqr,
                        "lower_bound": lower_bound,
                        "upper_bound": upper_bound
                    },
                    description=f"Value {point.value:.2f} is outside IQR bounds [{lower_bound:.2f}, {upper_bound:.2f}]"
                )
                anomalies.append(anomaly)
        
        return anomalies
    
    def detect_mad_anomalies(
        self,
        data: List[MetricPoint],
        metric_name: str
    ) -> List[AnomalyResult]:
        """Detect anomalies using Median Absolute Deviation (MAD) method"""
        if len(data) < self.config.min_data_points:
            return []
        
        values = [point.value for point in data]
        median = np.median(values)
        mad = np.median([abs(v - median) for v in values])
        
        if mad == 0:
            return []
        
        anomalies = []
        threshold = self.config.mad_threshold
        
        for point in data:
            modified_z_score = 0.6745 * (point.value - median) / mad
            
            if abs(modified_z_score) > threshold:
                severity = self._calculate_severity(abs(modified_z_score), threshold)
                confidence = min(abs(modified_z_score) / threshold, 3.0) / 3.0
                
                anomaly = AnomalyResult(
                    timestamp=point.timestamp,
                    metric_name=metric_name,
                    value=point.value,
                    anomaly_type=AnomalyType.STATISTICAL,
                    severity=severity,
                    confidence=confidence,
                    threshold=threshold,
                    deviation=abs(modified_z_score),
                    context={
                        "median": median,
                        "mad": mad,
                        "modified_z_score": modified_z_score
                    },
                    description=f"Value {point.value:.2f} has modified Z-score of {modified_z_score:.2f}"
                )
                anomalies.append(anomaly)
        
        return anomalies
    
    def _calculate_severity(self, deviation: float, threshold: float) -> AnomalySeverity:
        """Calculate anomaly severity based on deviation magnitude"""
        ratio = deviation / threshold
        
        if ratio >= 3.0:
            return AnomalySeverity.CRITICAL
        elif ratio >= 2.0:
            return AnomalySeverity.HIGH
        elif ratio >= 1.5:
            return AnomalySeverity.MEDIUM
        else:
            return AnomalySeverity.LOW


class TimeSeriesAnomalyDetector:
    """
    Time series anomaly detection for trend and seasonal pattern analysis.
    """
    
    def __init__(self, config: AnomalyDetectionConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self._trend_history: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=config.window_size)
        )
    
    def detect_trend_anomalies(
        self,
        data: List[MetricPoint],
        metric_name: str
    ) -> List[AnomalyResult]:
        """Detect anomalies in trend patterns"""
        if len(data) < self.config.min_data_points:
            return []
        
        # Calculate moving averages for trend detection
        values = [point.value for point in data]
        timestamps = [point.timestamp for point in data]
        
        # Simple trend detection using slope
        anomalies = []
        window = min(len(data), self.config.window_size)
        
        for i in range(window, len(data)):
            # Calculate slope over window
            window_data = data[i-window:i]
            x = np.arange(len(window_data))
            y = [p.value for p in window_data]
            
            if len(y) < 2:
                continue
            
            # Simple linear regression for slope
            slope = np.polyfit(x, y, 1)[0]
            
            # Compare with historical trend
            historical_slopes = self._trend_history[metric_name]
            if len(historical_slopes) > 5:
                mean_slope = statistics.mean(historical_slopes)
                slope_std = statistics.stdev(historical_slopes) if len(historical_slopes) > 1 else 0
                
                if slope_std > 0:
                    slope_z_score = abs((slope - mean_slope) / slope_std)
                    
                    if slope_z_score > self.config.z_score_threshold:
                        severity = self._calculate_trend_severity(slope_z_score)
                        confidence = min(slope_z_score / self.config.z_score_threshold, 3.0) / 3.0
                        
                        anomaly = AnomalyResult(
                            timestamp=data[i].timestamp,
                            metric_name=metric_name,
                            value=data[i].value,
                            anomaly_type=AnomalyType.TREND,
                            severity=severity,
                            confidence=confidence,
                            threshold=self.config.z_score_threshold,
                            deviation=slope_z_score,
                            context={
                                "current_slope": slope,
                                "mean_slope": mean_slope,
                                "slope_std": slope_std,
                                "window_size": window
                            },
                            description=f"Trend change detected: slope {slope:.4f} vs historical mean {mean_slope:.4f}"
                        )
                        anomalies.append(anomaly)
            
            # Update trend history
            historical_slopes.append(slope)
        
        return anomalies
    
    def detect_seasonal_anomalies(
        self,
        data: List[MetricPoint],
        metric_name: str
    ) -> List[AnomalyResult]:
        """Detect anomalies in seasonal patterns"""
        if len(data) < self.config.seasonal_period * 2:
            return []
        
        anomalies = []
        period = self.config.seasonal_period
        
        # Group data by seasonal position
        seasonal_groups = defaultdict(list)
        
        for i, point in enumerate(data):
            season_pos = i % period
            seasonal_groups[season_pos].append(point.value)
        
        # Detect anomalies within each seasonal group
        for i, point in enumerate(data):
            season_pos = i % period
            
            if len(seasonal_groups[season_pos]) < 3:
                continue
            
            seasonal_values = seasonal_groups[season_pos]
            seasonal_mean = statistics.mean(seasonal_values)
            seasonal_std = statistics.stdev(seasonal_values) if len(seasonal_values) > 1 else 0
            
            if seasonal_std > 0:
                z_score = abs((point.value - seasonal_mean) / seasonal_std)
                
                if z_score > self.config.z_score_threshold:
                    severity = self._calculate_trend_severity(z_score)
                    confidence = min(z_score / self.config.z_score_threshold, 3.0) / 3.0
                    
                    anomaly = AnomalyResult(
                        timestamp=point.timestamp,
                        metric_name=metric_name,
                        value=point.value,
                        anomaly_type=AnomalyType.SEASONAL,
                        severity=severity,
                        confidence=confidence,
                        threshold=self.config.z_score_threshold,
                        deviation=z_score,
                        context={
                            "seasonal_position": season_pos,
                            "seasonal_mean": seasonal_mean,
                            "seasonal_std": seasonal_std,
                            "seasonal_period": period
                        },
                        description=f"Seasonal anomaly at position {season_pos}: {point.value:.2f} vs seasonal mean {seasonal_mean:.2f}"
                    )
                    anomalies.append(anomaly)
        
        return anomalies
    
    def _calculate_trend_severity(self, z_score: float) -> AnomalySeverity:
        """Calculate severity for trend anomalies"""
        if z_score >= 4.0:
            return AnomalySeverity.CRITICAL
        elif z_score >= 3.0:
            return AnomalySeverity.HIGH
        elif z_score >= 2.0:
            return AnomalySeverity.MEDIUM
        else:
            return AnomalySeverity.LOW


class MLAnomalyDetector:
    """
    Machine learning-based anomaly detection using isolation forest and other ML techniques.
    """
    
    def __init__(self, config: AnomalyDetectionConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.models = {}
        self.scalers = {}
        
        if not ML_AVAILABLE:
            self.logger.warning("Scikit-learn not available. ML-based detection disabled.")
    
    def detect_isolation_forest_anomalies(
        self,
        data: List[MetricPoint],
        metric_name: str,
        features: Optional[List[str]] = None
    ) -> List[AnomalyResult]:
        """Detect anomalies using Isolation Forest algorithm"""
        if not ML_AVAILABLE or len(data) < self.config.min_data_points:
            return []
        
        try:
            # Prepare feature matrix
            feature_matrix = self._prepare_features(data, features)
            
            if feature_matrix.shape[0] < self.config.min_data_points:
                return []
            
            # Scale features
            scaler_key = f"{metric_name}_isolation"
            if scaler_key not in self.scalers:
                self.scalers[scaler_key] = StandardScaler()
                scaled_features = self.scalers[scaler_key].fit_transform(feature_matrix)
            else:
                scaled_features = self.scalers[scaler_key].transform(feature_matrix)
            
            # Train or use existing model
            model_key = f"{metric_name}_isolation"
            if model_key not in self.models:
                self.models[model_key] = IsolationForest(
                    contamination=self.config.contamination_rate,
                    n_estimators=self.config.n_estimators,
                    random_state=42
                )
                self.models[model_key].fit(scaled_features)
            
            # Predict anomalies
            anomaly_scores = self.models[model_key].decision_function(scaled_features)
            predictions = self.models[model_key].predict(scaled_features)
            
            anomalies = []
            for i, (point, score, prediction) in enumerate(zip(data, anomaly_scores, predictions)):
                if prediction == -1:  # Anomaly detected
                    # Convert score to confidence (higher negative score = more anomalous)
                    confidence = min(abs(score), 1.0)
                    severity = self._score_to_severity(score)
                    
                    anomaly = AnomalyResult(
                        timestamp=point.timestamp,
                        metric_name=metric_name,
                        value=point.value,
                        anomaly_type=AnomalyType.POINT,
                        severity=severity,
                        confidence=confidence,
                        threshold=self.config.contamination_rate,
                        deviation=abs(score),
                        context={
                            "isolation_score": score,
                            "features_used": features or ["value"],
                            "model_type": "isolation_forest"
                        },
                        description=f"Isolation Forest detected anomaly with score {score:.3f}"
                    )
                    anomalies.append(anomaly)
            
            return anomalies
            
        except Exception as e:
            self.logger.error(f"Error in isolation forest detection: {e}")
            return []
    
    def _prepare_features(
        self,
        data: List[MetricPoint],
        features: Optional[List[str]] = None
    ) -> np.ndarray:
        """Prepare feature matrix for ML algorithms"""
        if features is None:
            # Use basic features: value, hour of day, day of week
            feature_matrix = []
            
            for point in data:
                feature_row = [
                    point.value,
                    point.timestamp.hour,
                    point.timestamp.weekday(),
                    point.timestamp.minute
                ]
                feature_matrix.append(feature_row)
            
            return np.array(feature_matrix)
        else:
            # Use custom features from metadata
            feature_matrix = []
            
            for point in data:
                feature_row = [point.value]  # Always include value
                
                for feature in features:
                    if feature in point.metadata:
                        feature_row.append(point.metadata[feature])
                    else:
                        feature_row.append(0.0)  # Default value
                
                feature_matrix.append(feature_row)
            
            return np.array(feature_matrix)
    
    def _score_to_severity(self, score: float) -> AnomalySeverity:
        """Convert isolation forest score to severity level"""
        # Isolation forest scores are typically between -1 and 1
        # More negative scores indicate stronger anomalies
        abs_score = abs(score)
        
        if abs_score >= 0.8:
            return AnomalySeverity.CRITICAL
        elif abs_score >= 0.6:
            return AnomalySeverity.HIGH
        elif abs_score >= 0.4:
            return AnomalySeverity.MEDIUM
        else:
            return AnomalySeverity.LOW


class CompositeAnomalyDetector:
    """
    Composite anomaly detector that combines multiple detection methods
    for improved accuracy and reduced false positives.
    """
    
    def __init__(self, config: AnomalyDetectionConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize individual detectors
        self.statistical_detector = StatisticalAnomalyDetector(config)
        self.timeseries_detector = TimeSeriesAnomalyDetector(config)
        self.ml_detector = MLAnomalyDetector(config)
        
        # Detection history for consensus
        self.detection_history: Dict[str, List[AnomalyResult]] = defaultdict(list)
    
    async def detect_anomalies(
        self,
        data: List[MetricPoint],
        metric_name: str,
        methods: Optional[List[str]] = None
    ) -> List[AnomalyResult]:
        """
        Perform comprehensive anomaly detection using multiple methods.
        
        Args:
            data: Time series data points
            metric_name: Name of the metric being analyzed
            methods: List of methods to use. If None, uses all available methods.
            
        Returns:
            List of detected anomalies with consensus scoring
        """
        if not data:
            return []
        
        # Default methods based on configuration
        if methods is None:
            methods = ["z_score", "iqr", "trend"]
            if ML_AVAILABLE:
                methods.append("isolation_forest")
        
        all_anomalies = []
        
        # Run statistical detections
        if "z_score" in methods:
            z_anomalies = self.statistical_detector.detect_z_score_anomalies(data, metric_name)
            all_anomalies.extend(z_anomalies)
        
        if "iqr" in methods:
            iqr_anomalies = self.statistical_detector.detect_iqr_anomalies(data, metric_name)
            all_anomalies.extend(iqr_anomalies)
        
        if "mad" in methods:
            mad_anomalies = self.statistical_detector.detect_mad_anomalies(data, metric_name)
            all_anomalies.extend(mad_anomalies)
        
        # Run time series detections
        if "trend" in methods:
            trend_anomalies = self.timeseries_detector.detect_trend_anomalies(data, metric_name)
            all_anomalies.extend(trend_anomalies)
        
        if "seasonal" in methods:
            seasonal_anomalies = self.timeseries_detector.detect_seasonal_anomalies(data, metric_name)
            all_anomalies.extend(seasonal_anomalies)
        
        # Run ML detections
        if "isolation_forest" in methods and ML_AVAILABLE:
            ml_anomalies = self.ml_detector.detect_isolation_forest_anomalies(data, metric_name)
            all_anomalies.extend(ml_anomalies)
        
        # Apply consensus filtering
        consensus_anomalies = self._apply_consensus(all_anomalies, metric_name)
        
        # Update detection history
        self.detection_history[metric_name].extend(consensus_anomalies)
        
        # Keep only recent history
        cutoff_time = datetime.utcnow() - timedelta(hours=24)
        self.detection_history[metric_name] = [
            a for a in self.detection_history[metric_name] 
            if a.timestamp > cutoff_time
        ]
        
        return consensus_anomalies
    
    def _apply_consensus(
        self,
        anomalies: List[AnomalyResult],
        metric_name: str
    ) -> List[AnomalyResult]:
        """Apply consensus logic to reduce false positives"""
        if not anomalies:
            return []
        
        # Group anomalies by time window (5-minute windows)
        time_groups = defaultdict(list)
        
        for anomaly in anomalies:
            # Round timestamp to 5-minute intervals
            rounded_time = anomaly.timestamp.replace(
                minute=(anomaly.timestamp.minute // 5) * 5,
                second=0,
                microsecond=0
            )
            time_groups[rounded_time].append(anomaly)
        
        consensus_anomalies = []
        
        for timestamp, group_anomalies in time_groups.items():
            if len(group_anomalies) == 1:
                # Single detection - apply stricter threshold
                anomaly = group_anomalies[0]
                if anomaly.confidence >= 0.7:  # Higher confidence required
                    consensus_anomalies.append(anomaly)
            
            elif len(group_anomalies) >= 2:
                # Multiple detections - create consensus anomaly
                consensus_anomaly = self._create_consensus_anomaly(group_anomalies, timestamp)
                consensus_anomalies.append(consensus_anomaly)
        
        return consensus_anomalies
    
    def _create_consensus_anomaly(
        self,
        anomalies: List[AnomalyResult],
        timestamp: datetime
    ) -> AnomalyResult:
        """Create a consensus anomaly from multiple detections"""
        # Use the anomaly with highest confidence as base
        base_anomaly = max(anomalies, key=lambda a: a.confidence)
        
        # Calculate consensus metrics
        avg_confidence = statistics.mean([a.confidence for a in anomalies])
        max_deviation = max([a.deviation for a in anomalies])
        consensus_severity = self._consensus_severity([a.severity for a in anomalies])
        
        # Combine context from all methods
        combined_context = {
            "detection_methods": [a.anomaly_type.value for a in anomalies],
            "method_count": len(anomalies),
            "individual_confidences": [a.confidence for a in anomalies],
            "consensus_confidence": avg_confidence
        }
        
        # Add context from all anomalies
        for anomaly in anomalies:
            combined_context.update(anomaly.context)
        
        return AnomalyResult(
            timestamp=timestamp,
            metric_name=base_anomaly.metric_name,
            value=base_anomaly.value,
            anomaly_type=AnomalyType.COLLECTIVE,
            severity=consensus_severity,
            confidence=min(avg_confidence * 1.2, 1.0),  # Boost consensus confidence
            threshold=base_anomaly.threshold,
            deviation=max_deviation,
            context=combined_context,
            description=f"Consensus anomaly detected by {len(anomalies)} methods"
        )
    
    def _consensus_severity(self, severities: List[AnomalySeverity]) -> AnomalySeverity:
        """Calculate consensus severity from multiple detections"""
        severity_values = {
            AnomalySeverity.LOW: 1,
            AnomalySeverity.MEDIUM: 2,
            AnomalySeverity.HIGH: 3,
            AnomalySeverity.CRITICAL: 4
        }
        
        # Use maximum severity but require at least medium consensus
        max_severity = max(severities, key=lambda s: severity_values[s])
        
        # If we have multiple high/critical detections, escalate
        high_count = sum(1 for s in severities if severity_values[s] >= 3)
        if high_count >= 2:
            return AnomalySeverity.CRITICAL
        
        return max_severity
    
    def get_anomaly_summary(
        self,
        metric_name: str,
        hours_back: int = 24
    ) -> Dict[str, Any]:
        """Get summary of recent anomalies for a metric"""
        cutoff_time = datetime.utcnow() - timedelta(hours=hours_back)
        recent_anomalies = [
            a for a in self.detection_history.get(metric_name, [])
            if a.timestamp > cutoff_time
        ]
        
        if not recent_anomalies:
            return {
                "metric_name": metric_name,
                "total_anomalies": 0,
                "anomaly_rate": 0.0,
                "severity_distribution": {},
                "most_recent": None
            }
        
        severity_counts = defaultdict(int)
        for anomaly in recent_anomalies:
            severity_counts[anomaly.severity.value] += 1
        
        return {
            "metric_name": metric_name,
            "total_anomalies": len(recent_anomalies),
            "anomaly_rate": len(recent_anomalies) / hours_back,
            "severity_distribution": dict(severity_counts),
            "most_recent": recent_anomalies[-1] if recent_anomalies else None,
            "time_range": f"Last {hours_back} hours"
        }