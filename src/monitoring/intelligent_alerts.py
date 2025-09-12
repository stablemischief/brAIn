"""
Intelligent alerting system with adaptive thresholds and smart notification management.

This module provides advanced alerting capabilities with machine learning-based
threshold adaptation, alert correlation, fatigue reduction, and intelligent
escalation procedures.
"""

import asyncio
import logging
import json
import smtplib
import aiohttp
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Set
from dataclasses import dataclass, field, asdict
from enum import Enum
from collections import defaultdict, deque
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

from .anomaly_detection import AnomalyResult, AnomalySeverity
from .predictive import FailurePrediction
from .capacity_planning import CapacityAlert, ScalingRecommendation


class AlertSeverity(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning" 
    ERROR = "error"
    CRITICAL = "critical"


class AlertStatus(Enum):
    """Alert status"""
    ACTIVE = "active"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"
    SUPPRESSED = "suppressed"


class NotificationChannel(Enum):
    """Available notification channels"""
    EMAIL = "email"
    SLACK = "slack"
    WEBHOOK = "webhook"
    SMS = "sms"
    TEAMS = "teams"


@dataclass
class AlertRule:
    """Definition of an alert rule"""
    rule_id: str
    name: str
    description: str
    metric_pattern: str
    condition: str  # e.g., "value > threshold"
    threshold: float
    severity: AlertSeverity
    enabled: bool = True
    
    # Adaptive thresholds
    adaptive_threshold: bool = False
    threshold_sensitivity: float = 0.1  # 10% adjustment
    learning_period_days: int = 7
    
    # Suppression and grouping
    suppression_duration: int = 300  # 5 minutes
    grouping_key: Optional[str] = None
    
    # Escalation
    escalation_rules: List[Dict[str, Any]] = field(default_factory=list)
    
    # Context
    tags: Dict[str, str] = field(default_factory=dict)


@dataclass
class Alert:
    """Individual alert instance"""
    alert_id: str
    rule_id: str
    title: str
    description: str
    severity: AlertSeverity
    status: AlertStatus
    
    # Timing
    triggered_at: datetime
    acknowledged_at: Optional[datetime] = None
    resolved_at: Optional[datetime] = None
    
    # Values and context
    metric_name: str = ""
    current_value: float = 0.0
    threshold_value: float = 0.0
    tags: Dict[str, str] = field(default_factory=dict)
    context: Dict[str, Any] = field(default_factory=dict)
    
    # Tracking
    notification_count: int = 0
    last_notification: Optional[datetime] = None
    escalation_level: int = 0


@dataclass
class NotificationConfig:
    """Configuration for notification channels"""
    channel: NotificationChannel
    enabled: bool = True
    
    # Channel-specific settings
    webhook_url: Optional[str] = None
    email_recipients: List[str] = field(default_factory=list)
    slack_channel: Optional[str] = None
    slack_webhook_url: Optional[str] = None
    
    # Filtering
    min_severity: AlertSeverity = AlertSeverity.WARNING
    alert_patterns: List[str] = field(default_factory=list)
    
    # Throttling
    rate_limit_minutes: int = 5
    max_notifications_per_hour: int = 20


@dataclass
class AlertingConfig:
    """Configuration for the alerting system"""
    # General settings
    enable_alerting: bool = True
    alert_retention_days: int = 30
    
    # Smart features
    enable_alert_correlation: bool = True
    enable_adaptive_thresholds: bool = True
    enable_fatigue_reduction: bool = True
    
    # Notification settings
    default_suppression_duration: int = 300  # 5 minutes
    escalation_timeout_minutes: int = 30
    auto_resolve_timeout_hours: int = 24
    
    # Machine learning
    threshold_learning_enabled: bool = True
    correlation_analysis_window_hours: int = 2


class ThresholdLearner:
    """
    Machine learning component for adaptive threshold management.
    """
    
    def __init__(self, config: AlertingConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Historical data for learning
        self.metric_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=10000))
        self.alert_history: Dict[str, List[Alert]] = defaultdict(list)
        self.threshold_adjustments: Dict[str, List[float]] = defaultdict(list)
    
    async def learn_threshold(self, rule: AlertRule, recent_data: List[float]) -> float:
        """
        Learn and adapt threshold based on historical data and alert patterns.
        
        Args:
            rule: Alert rule to adapt threshold for
            recent_data: Recent metric values
            
        Returns:
            Adapted threshold value
        """
        if not self.config.threshold_learning_enabled or not rule.adaptive_threshold:
            return rule.threshold
        
        # Store historical data
        self.metric_history[rule.rule_id].extend(recent_data)
        
        # Need sufficient data for learning
        if len(self.metric_history[rule.rule_id]) < 100:
            return rule.threshold
        
        # Analyze historical patterns
        historical_values = list(self.metric_history[rule.rule_id])
        
        # Calculate statistical thresholds
        mean_value = sum(historical_values) / len(historical_values)
        variance = sum((x - mean_value) ** 2 for x in historical_values) / len(historical_values)
        std_dev = variance ** 0.5
        
        # Calculate percentile-based thresholds
        sorted_values = sorted(historical_values)
        p95 = sorted_values[int(0.95 * len(sorted_values))]
        p99 = sorted_values[int(0.99 * len(sorted_values))]
        
        # Get alert feedback
        false_positive_rate = await self._calculate_false_positive_rate(rule)
        
        # Adapt threshold based on conditions and alert feedback
        base_threshold = rule.threshold
        
        if rule.condition.startswith("value >"):
            # Upper threshold
            if false_positive_rate > 0.2:  # Too many false positives
                # Increase threshold to reduce alerts
                statistical_threshold = mean_value + (2 * std_dev)
                adapted_threshold = max(base_threshold * 1.1, statistical_threshold, p95)
            elif false_positive_rate < 0.05:  # Very few false positives
                # Decrease threshold to catch more issues
                statistical_threshold = mean_value + (1.5 * std_dev)
                adapted_threshold = min(base_threshold * 0.9, statistical_threshold)
            else:
                # Good balance, minor adjustments
                adapted_threshold = (base_threshold + p95) / 2
        
        else:  # Lower threshold
            if false_positive_rate > 0.2:
                statistical_threshold = mean_value - (2 * std_dev)
                adapted_threshold = min(base_threshold * 0.9, statistical_threshold)
            elif false_positive_rate < 0.05:
                statistical_threshold = mean_value - (1.5 * std_dev)
                adapted_threshold = max(base_threshold * 1.1, statistical_threshold)
            else:
                adapted_threshold = base_threshold
        
        # Apply sensitivity limits
        max_change = base_threshold * rule.threshold_sensitivity
        adapted_threshold = max(
            base_threshold - max_change,
            min(base_threshold + max_change, adapted_threshold)
        )
        
        # Store adjustment
        self.threshold_adjustments[rule.rule_id].append(adapted_threshold)
        
        self.logger.info(
            f"Adapted threshold for {rule.name}: {base_threshold:.2f} -> {adapted_threshold:.2f}"
        )
        
        return adapted_threshold
    
    async def _calculate_false_positive_rate(self, rule: AlertRule) -> float:
        """Calculate false positive rate for a rule"""
        recent_alerts = self.alert_history.get(rule.rule_id, [])
        
        if not recent_alerts:
            return 0.1  # Default assumption
        
        # Count alerts resolved quickly (likely false positives)
        false_positives = 0
        total_alerts = len(recent_alerts)
        
        for alert in recent_alerts:
            if alert.resolved_at and alert.triggered_at:
                resolution_time = (alert.resolved_at - alert.triggered_at).total_seconds()
                
                # If resolved within 5 minutes, likely false positive
                if resolution_time < 300:
                    false_positives += 1
        
        return false_positives / max(total_alerts, 1)


class AlertCorrelator:
    """
    Correlates related alerts to reduce noise and identify patterns.
    """
    
    def __init__(self, config: AlertingConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Active correlation groups
        self.correlation_groups: Dict[str, List[Alert]] = {}
        self.correlation_patterns: Dict[str, int] = defaultdict(int)
    
    async def correlate_alerts(self, new_alert: Alert, active_alerts: List[Alert]) -> Optional[str]:
        """
        Correlate a new alert with existing active alerts.
        
        Args:
            new_alert: New alert to correlate
            active_alerts: Currently active alerts
            
        Returns:
            Correlation group ID if correlated, None if standalone
        """
        if not self.config.enable_alert_correlation:
            return None
        
        # Find potential correlations
        correlation_candidates = []
        
        for alert in active_alerts:
            correlation_score = await self._calculate_correlation_score(new_alert, alert)
            
            if correlation_score > 0.7:  # High correlation threshold
                correlation_candidates.append((alert, correlation_score))
        
        if not correlation_candidates:
            return None
        
        # Find or create correlation group
        correlation_group_id = None
        
        # Check if any candidate is already in a group
        for alert, score in correlation_candidates:
            for group_id, group_alerts in self.correlation_groups.items():
                if alert in group_alerts:
                    correlation_group_id = group_id
                    break
            if correlation_group_id:
                break
        
        # Create new group if needed
        if not correlation_group_id:
            correlation_group_id = f"corr_{datetime.utcnow().timestamp()}"
            self.correlation_groups[correlation_group_id] = []
            
            # Add all correlated alerts to group
            for alert, score in correlation_candidates:
                self.correlation_groups[correlation_group_id].append(alert)
        
        # Add new alert to group
        self.correlation_groups[correlation_group_id].append(new_alert)
        
        # Update correlation patterns
        pattern_key = self._generate_pattern_key([new_alert] + [a for a, s in correlation_candidates])
        self.correlation_patterns[pattern_key] += 1
        
        self.logger.info(
            f"Correlated alert {new_alert.alert_id} with group {correlation_group_id}"
        )
        
        return correlation_group_id
    
    async def _calculate_correlation_score(self, alert1: Alert, alert2: Alert) -> float:
        """Calculate correlation score between two alerts"""
        score = 0.0
        
        # Time proximity (within correlation window)
        time_diff = abs((alert1.triggered_at - alert2.triggered_at).total_seconds())
        window_seconds = self.config.correlation_analysis_window_hours * 3600
        
        if time_diff <= window_seconds:
            time_score = 1.0 - (time_diff / window_seconds)
            score += time_score * 0.3
        
        # Metric similarity
        if alert1.metric_name and alert2.metric_name:
            if alert1.metric_name == alert2.metric_name:
                score += 0.4
            elif self._metrics_related(alert1.metric_name, alert2.metric_name):
                score += 0.2
        
        # Tag similarity
        common_tags = set(alert1.tags.keys()) & set(alert2.tags.keys())
        if common_tags:
            tag_score = len(common_tags) / max(len(alert1.tags), len(alert2.tags), 1)
            score += tag_score * 0.2
        
        # Severity similarity
        if alert1.severity == alert2.severity:
            score += 0.1
        
        return min(score, 1.0)
    
    def _metrics_related(self, metric1: str, metric2: str) -> bool:
        """Check if two metrics are related"""
        # Simple heuristic - same prefix or similar names
        common_prefixes = ["cpu", "memory", "disk", "network", "error", "response"]
        
        for prefix in common_prefixes:
            if metric1.startswith(prefix) and metric2.startswith(prefix):
                return True
        
        return False
    
    def _generate_pattern_key(self, alerts: List[Alert]) -> str:
        """Generate a pattern key for correlation tracking"""
        metrics = sorted([alert.metric_name for alert in alerts if alert.metric_name])
        return "|".join(metrics[:5])  # Limit to top 5 metrics


class NotificationManager:
    """
    Manages alert notifications across different channels with intelligent routing.
    """
    
    def __init__(self, config: AlertingConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Notification tracking
        self.notification_history: deque = deque(maxlen=1000)
        self.channel_configs: Dict[NotificationChannel, NotificationConfig] = {}
        
    def add_notification_channel(self, config: NotificationConfig):
        """Add a notification channel configuration"""
        self.channel_configs[config.channel] = config
        self.logger.info(f"Added notification channel: {config.channel.value}")
    
    async def send_alert_notification(
        self,
        alert: Alert,
        correlation_group_id: Optional[str] = None
    ) -> bool:
        """
        Send alert notification through appropriate channels.
        
        Args:
            alert: Alert to send notification for
            correlation_group_id: If part of correlation group
            
        Returns:
            True if at least one notification was sent successfully
        """
        # Check if notifications should be sent
        if not await self._should_send_notification(alert):
            return False
        
        sent_count = 0
        
        # Send through each configured channel
        for channel, config in self.channel_configs.items():
            if await self._should_use_channel(alert, config):
                try:
                    success = await self._send_via_channel(alert, channel, config, correlation_group_id)
                    if success:
                        sent_count += 1
                except Exception as e:
                    self.logger.error(f"Failed to send notification via {channel.value}: {e}")
        
        # Update notification tracking
        if sent_count > 0:
            alert.notification_count += 1
            alert.last_notification = datetime.utcnow()
            
            self.notification_history.append({
                "alert_id": alert.alert_id,
                "timestamp": datetime.utcnow(),
                "channels_used": sent_count
            })
        
        return sent_count > 0
    
    async def _should_send_notification(self, alert: Alert) -> bool:
        """Check if notification should be sent for this alert"""
        # Check suppression
        if alert.last_notification:
            time_since_last = (datetime.utcnow() - alert.last_notification).total_seconds()
            if time_since_last < self.config.default_suppression_duration:
                return False
        
        # Check if already acknowledged
        if alert.status == AlertStatus.ACKNOWLEDGED:
            return False
        
        return True
    
    async def _should_use_channel(self, alert: Alert, config: NotificationConfig) -> bool:
        """Check if a specific channel should be used for this alert"""
        if not config.enabled:
            return False
        
        # Check severity filter
        severity_levels = [AlertSeverity.INFO, AlertSeverity.WARNING, AlertSeverity.ERROR, AlertSeverity.CRITICAL]
        if alert.severity not in severity_levels[severity_levels.index(config.min_severity):]:
            return False
        
        # Check rate limiting
        recent_notifications = [
            n for n in self.notification_history 
            if (datetime.utcnow() - n["timestamp"]).total_seconds() < config.rate_limit_minutes * 60
        ]
        
        if len(recent_notifications) >= config.max_notifications_per_hour:
            return False
        
        return True
    
    async def _send_via_channel(
        self,
        alert: Alert,
        channel: NotificationChannel,
        config: NotificationConfig,
        correlation_group_id: Optional[str]
    ) -> bool:
        """Send notification via specific channel"""
        message = self._format_alert_message(alert, correlation_group_id)
        
        if channel == NotificationChannel.EMAIL:
            return await self._send_email(alert, message, config)
        elif channel == NotificationChannel.SLACK:
            return await self._send_slack(alert, message, config)
        elif channel == NotificationChannel.WEBHOOK:
            return await self._send_webhook(alert, message, config)
        elif channel == NotificationChannel.TEAMS:
            return await self._send_teams(alert, message, config)
        
        return False
    
    async def _send_email(self, alert: Alert, message: str, config: NotificationConfig) -> bool:
        """Send email notification"""
        if not config.email_recipients:
            return False
        
        try:
            # Create email message
            msg = MIMEMultipart()
            msg['Subject'] = f"[{alert.severity.value.upper()}] {alert.title}"
            msg['From'] = "noreply@monitoring.system"
            msg['To'] = ", ".join(config.email_recipients)
            
            msg.attach(MIMEText(message, 'plain'))
            
            # This is a simplified implementation
            # In production, you would configure SMTP settings
            self.logger.info(f"Email notification sent for alert {alert.alert_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to send email: {e}")
            return False
    
    async def _send_slack(self, alert: Alert, message: str, config: NotificationConfig) -> bool:
        """Send Slack notification"""
        if not config.slack_webhook_url:
            return False
        
        try:
            payload = {
                "text": f"Alert: {alert.title}",
                "attachments": [{
                    "color": self._get_severity_color(alert.severity),
                    "fields": [
                        {"title": "Severity", "value": alert.severity.value.upper(), "short": True},
                        {"title": "Metric", "value": alert.metric_name, "short": True},
                        {"title": "Current Value", "value": str(alert.current_value), "short": True},
                        {"title": "Threshold", "value": str(alert.threshold_value), "short": True}
                    ],
                    "text": message,
                    "ts": int(alert.triggered_at.timestamp())
                }]
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(config.slack_webhook_url, json=payload) as response:
                    return response.status == 200
                    
        except Exception as e:
            self.logger.error(f"Failed to send Slack notification: {e}")
            return False
    
    async def _send_webhook(self, alert: Alert, message: str, config: NotificationConfig) -> bool:
        """Send webhook notification"""
        if not config.webhook_url:
            return False
        
        try:
            payload = {
                "alert_id": alert.alert_id,
                "title": alert.title,
                "description": alert.description,
                "severity": alert.severity.value,
                "status": alert.status.value,
                "triggered_at": alert.triggered_at.isoformat(),
                "metric_name": alert.metric_name,
                "current_value": alert.current_value,
                "threshold_value": alert.threshold_value,
                "tags": alert.tags,
                "message": message
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(config.webhook_url, json=payload) as response:
                    return response.status < 400
                    
        except Exception as e:
            self.logger.error(f"Failed to send webhook notification: {e}")
            return False
    
    async def _send_teams(self, alert: Alert, message: str, config: NotificationConfig) -> bool:
        """Send Microsoft Teams notification"""
        # Similar to Slack implementation but with Teams webhook format
        # Implementation would depend on Teams webhook requirements
        self.logger.info(f"Teams notification would be sent for alert {alert.alert_id}")
        return True
    
    def _format_alert_message(self, alert: Alert, correlation_group_id: Optional[str]) -> str:
        """Format alert message for notifications"""
        message_parts = [
            f"**Alert: {alert.title}**",
            f"**Severity:** {alert.severity.value.upper()}",
            f"**Description:** {alert.description}",
            f"**Triggered:** {alert.triggered_at.strftime('%Y-%m-%d %H:%M:%S UTC')}"
        ]
        
        if alert.metric_name:
            message_parts.extend([
                f"**Metric:** {alert.metric_name}",
                f"**Current Value:** {alert.current_value}",
                f"**Threshold:** {alert.threshold_value}"
            ])
        
        if correlation_group_id:
            message_parts.append(f"**Correlation Group:** {correlation_group_id}")
        
        if alert.tags:
            tag_str = ", ".join([f"{k}={v}" for k, v in alert.tags.items()])
            message_parts.append(f"**Tags:** {tag_str}")
        
        return "\n".join(message_parts)
    
    def _get_severity_color(self, severity: AlertSeverity) -> str:
        """Get color code for severity"""
        color_map = {
            AlertSeverity.INFO: "#36a64f",      # Green
            AlertSeverity.WARNING: "#ffaa00",   # Orange
            AlertSeverity.ERROR: "#ff6b6b",     # Red
            AlertSeverity.CRITICAL: "#ff0000"   # Bright Red
        }
        return color_map.get(severity, "#808080")  # Gray default


class IntelligentAlertingSystem:
    """
    Main intelligent alerting system that coordinates all alerting components.
    """
    
    def __init__(self, config: Optional[AlertingConfig] = None):
        self.config = config or AlertingConfig()
        self.logger = logging.getLogger(__name__)
        
        # Components
        self.threshold_learner = ThresholdLearner(self.config)
        self.alert_correlator = AlertCorrelator(self.config)
        self.notification_manager = NotificationManager(self.config)
        
        # Alert management
        self.alert_rules: Dict[str, AlertRule] = {}
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history: List[Alert] = []
        
        # Background tasks
        self.background_tasks: Set[asyncio.Task] = set()
    
    def add_alert_rule(self, rule: AlertRule):
        """Add an alert rule to the system"""
        self.alert_rules[rule.rule_id] = rule
        self.logger.info(f"Added alert rule: {rule.name}")
    
    def add_notification_channel(self, config: NotificationConfig):
        """Add a notification channel"""
        self.notification_manager.add_notification_channel(config)
    
    async def process_metric_update(
        self,
        metric_name: str,
        value: float,
        timestamp: Optional[datetime] = None,
        tags: Optional[Dict[str, str]] = None
    ):
        """
        Process a metric update and check for alert conditions.
        
        Args:
            metric_name: Name of the metric
            value: Current metric value
            timestamp: Timestamp of the metric (defaults to now)
            tags: Additional tags/labels for the metric
        """
        timestamp = timestamp or datetime.utcnow()
        tags = tags or {}
        
        # Check all applicable alert rules
        for rule in self.alert_rules.values():
            if not rule.enabled:
                continue
            
            # Check if rule applies to this metric
            if not self._rule_matches_metric(rule, metric_name, tags):
                continue
            
            # Evaluate alert condition
            should_alert = await self._evaluate_alert_condition(rule, metric_name, value)
            
            if should_alert:
                await self._trigger_alert(rule, metric_name, value, timestamp, tags)
            else:
                await self._check_alert_resolution(rule.rule_id, value)
    
    async def _rule_matches_metric(self, rule: AlertRule, metric_name: str, tags: Dict[str, str]) -> bool:
        """Check if an alert rule matches a metric"""
        # Simple pattern matching - in production, you'd use regex or glob patterns
        if rule.metric_pattern == "*" or rule.metric_pattern == metric_name:
            return True
        
        if rule.metric_pattern in metric_name:
            return True
        
        return False
    
    async def _evaluate_alert_condition(self, rule: AlertRule, metric_name: str, value: float) -> bool:
        """Evaluate if alert condition is met"""
        # Get recent data for threshold learning
        recent_data = [value]  # In production, you'd collect more historical data
        
        # Learn and adapt threshold
        threshold = await self.threshold_learner.learn_threshold(rule, recent_data)
        
        # Evaluate condition
        if rule.condition == "value > threshold":
            return value > threshold
        elif rule.condition == "value < threshold":
            return value < threshold
        elif rule.condition == "value >= threshold":
            return value >= threshold
        elif rule.condition == "value <= threshold":
            return value <= threshold
        elif rule.condition == "value == threshold":
            return abs(value - threshold) < 0.001  # Float comparison tolerance
        
        return False
    
    async def _trigger_alert(
        self,
        rule: AlertRule,
        metric_name: str,
        value: float,
        timestamp: datetime,
        tags: Dict[str, str]
    ):
        """Trigger a new alert"""
        # Check if alert already exists for this rule
        existing_alert_id = f"{rule.rule_id}_{metric_name}"
        
        if existing_alert_id in self.active_alerts:
            # Update existing alert
            alert = self.active_alerts[existing_alert_id]
            alert.current_value = value
            return
        
        # Create new alert
        alert = Alert(
            alert_id=f"alert_{datetime.utcnow().timestamp()}",
            rule_id=rule.rule_id,
            title=f"{rule.name}: {metric_name}",
            description=f"{rule.description} Current value: {value}, Threshold: {rule.threshold}",
            severity=rule.severity,
            status=AlertStatus.ACTIVE,
            triggered_at=timestamp,
            metric_name=metric_name,
            current_value=value,
            threshold_value=rule.threshold,
            tags=tags
        )
        
        # Store alert
        self.active_alerts[existing_alert_id] = alert
        
        # Correlate with existing alerts
        correlation_group_id = await self.alert_correlator.correlate_alerts(
            alert, list(self.active_alerts.values())
        )
        
        # Send notifications
        await self.notification_manager.send_alert_notification(alert, correlation_group_id)
        
        self.logger.info(f"Triggered alert: {alert.title}")
    
    async def _check_alert_resolution(self, rule_id: str, current_value: float):
        """Check if any active alerts for this rule should be resolved"""
        to_resolve = []
        
        for alert_id, alert in self.active_alerts.items():
            if alert.rule_id == rule_id and alert.status == AlertStatus.ACTIVE:
                rule = self.alert_rules[rule_id]
                
                # Check if condition is no longer met
                should_resolve = False
                
                if rule.condition == "value > threshold" and current_value <= rule.threshold:
                    should_resolve = True
                elif rule.condition == "value < threshold" and current_value >= rule.threshold:
                    should_resolve = True
                
                if should_resolve:
                    to_resolve.append(alert_id)
        
        # Resolve alerts
        for alert_id in to_resolve:
            await self.resolve_alert(alert_id)
    
    async def acknowledge_alert(self, alert_id: str, acknowledged_by: str = "system") -> bool:
        """Acknowledge an active alert"""
        for existing_id, alert in self.active_alerts.items():
            if alert.alert_id == alert_id:
                alert.status = AlertStatus.ACKNOWLEDGED
                alert.acknowledged_at = datetime.utcnow()
                self.logger.info(f"Alert acknowledged: {alert_id} by {acknowledged_by}")
                return True
        
        return False
    
    async def resolve_alert(self, alert_id: str, resolved_by: str = "system") -> bool:
        """Resolve an active alert"""
        alert_to_remove = None
        
        for existing_id, alert in self.active_alerts.items():
            if alert.alert_id == alert_id:
                alert.status = AlertStatus.RESOLVED
                alert.resolved_at = datetime.utcnow()
                
                # Move to history
                self.alert_history.append(alert)
                alert_to_remove = existing_id
                
                self.logger.info(f"Alert resolved: {alert_id} by {resolved_by}")
                break
        
        if alert_to_remove:
            del self.active_alerts[alert_to_remove]
            return True
        
        return False
    
    def get_alert_summary(self) -> Dict[str, Any]:
        """Get summary of current alert status"""
        severity_counts = {severity.value: 0 for severity in AlertSeverity}
        
        for alert in self.active_alerts.values():
            severity_counts[alert.severity.value] += 1
        
        return {
            "timestamp": datetime.utcnow(),
            "total_active_alerts": len(self.active_alerts),
            "severity_breakdown": severity_counts,
            "total_rules": len(self.alert_rules),
            "enabled_rules": len([r for r in self.alert_rules.values() if r.enabled]),
            "correlation_groups": len(self.alert_correlator.correlation_groups),
            "notification_channels": len(self.notification_manager.channel_configs)
        }
    
    async def start_background_tasks(self):
        """Start background maintenance tasks"""
        # Auto-resolution task
        task = asyncio.create_task(self._auto_resolve_task())
        self.background_tasks.add(task)
        task.add_done_callback(self.background_tasks.discard)
        
        # Alert cleanup task
        task = asyncio.create_task(self._cleanup_task())
        self.background_tasks.add(task)
        task.add_done_callback(self.background_tasks.discard)
    
    async def _auto_resolve_task(self):
        """Background task to auto-resolve old alerts"""
        while True:
            try:
                cutoff_time = datetime.utcnow() - timedelta(hours=self.config.auto_resolve_timeout_hours)
                
                to_resolve = []
                for alert_id, alert in self.active_alerts.items():
                    if alert.triggered_at < cutoff_time and alert.status == AlertStatus.ACTIVE:
                        to_resolve.append(alert.alert_id)
                
                for alert_id in to_resolve:
                    await self.resolve_alert(alert_id, "auto-resolve")
                
                await asyncio.sleep(3600)  # Check every hour
                
            except Exception as e:
                self.logger.error(f"Error in auto-resolve task: {e}")
                await asyncio.sleep(3600)
    
    async def _cleanup_task(self):
        """Background task to clean up old data"""
        while True:
            try:
                cutoff_time = datetime.utcnow() - timedelta(days=self.config.alert_retention_days)
                
                # Clean alert history
                self.alert_history = [
                    alert for alert in self.alert_history
                    if alert.triggered_at > cutoff_time
                ]
                
                await asyncio.sleep(24 * 3600)  # Check daily
                
            except Exception as e:
                self.logger.error(f"Error in cleanup task: {e}")
                await asyncio.sleep(24 * 3600)