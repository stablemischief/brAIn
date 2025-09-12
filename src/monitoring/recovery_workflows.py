"""
Recovery workflow definitions and management.

This module provides predefined recovery workflows for common failure scenarios,
workflow templates, and a system for creating custom recovery procedures.
"""

from datetime import timedelta
from typing import Dict, List
from .auto_recovery import RecoveryWorkflow, RecoveryActionDef, RecoveryAction, EscalationLevel


# Predefined Recovery Workflows

HIGH_CPU_USAGE_WORKFLOW = RecoveryWorkflow(
    name="high_cpu_usage",
    description="Recovery workflow for high CPU usage scenarios",
    trigger_conditions=[
        {"type": "metric", "metric_name": "cpu_usage", "operator": ">", "threshold": 90},
        {"type": "event", "event_pattern": "cpu_overload"}
    ],
    actions=[
        RecoveryActionDef(
            action_type=RecoveryAction.KILL_PROCESSES,
            name="kill_cpu_intensive_processes",
            description="Kill processes consuming excessive CPU",
            parameters={
                "process_pattern": r"(stress|cpuburn|.*high-cpu.*)",
                "signal": "TERM"
            },
            timeout_seconds=60,
            risk_level="medium",
            prerequisites=["cpu_monitoring_available"]
        ),
        RecoveryActionDef(
            action_type=RecoveryAction.SCALE_UP,
            name="scale_up_cpu",
            description="Scale up CPU resources",
            parameters={
                "target": "cpu",
                "factor": 2
            },
            timeout_seconds=300,
            risk_level="low"
        ),
        RecoveryActionDef(
            action_type=RecoveryAction.RESTART_SERVICE,
            name="restart_application",
            description="Restart main application service",
            parameters={
                "service_name": "app-service"
            },
            timeout_seconds=180,
            risk_level="medium",
            prerequisites=["service_running"]
        )
    ],
    max_retries=2,
    retry_delay=60,
    escalation_level=EscalationLevel.AUTOMATIC,
    success_threshold=0.7
)

HIGH_MEMORY_USAGE_WORKFLOW = RecoveryWorkflow(
    name="high_memory_usage",
    description="Recovery workflow for memory exhaustion",
    trigger_conditions=[
        {"type": "metric", "metric_name": "memory_usage", "operator": ">", "threshold": 95},
        {"type": "event", "event_pattern": "memory_exhaustion"}
    ],
    actions=[
        RecoveryActionDef(
            action_type=RecoveryAction.CLEAR_CACHE,
            name="clear_system_cache",
            description="Clear system memory caches",
            parameters={
                "cache_type": "system"
            },
            timeout_seconds=30,
            risk_level="low"
        ),
        RecoveryActionDef(
            action_type=RecoveryAction.KILL_PROCESSES,
            name="kill_memory_intensive_processes",
            description="Kill processes with high memory usage",
            parameters={
                "process_pattern": r"(chrome|firefox|.*memory-hog.*)",
                "signal": "TERM"
            },
            timeout_seconds=60,
            risk_level="medium"
        ),
        RecoveryActionDef(
            action_type=RecoveryAction.SCALE_UP,
            name="scale_up_memory",
            description="Scale up memory resources",
            parameters={
                "target": "memory",
                "factor": 1.5
            },
            timeout_seconds=300,
            risk_level="low"
        ),
        RecoveryActionDef(
            action_type=RecoveryAction.RESTART_SERVICE,
            name="restart_memory_intensive_services",
            description="Restart services with memory leaks",
            parameters={
                "service_name": "app-service"
            },
            timeout_seconds=180,
            risk_level="high"
        )
    ],
    max_retries=3,
    retry_delay=45,
    escalation_level=EscalationLevel.AUTOMATIC,
    success_threshold=0.8
)

DISK_SPACE_EXHAUSTION_WORKFLOW = RecoveryWorkflow(
    name="disk_space_exhaustion",
    description="Recovery workflow for disk space issues",
    trigger_conditions=[
        {"type": "metric", "metric_name": "disk_usage", "operator": ">", "threshold": 95},
        {"type": "event", "event_pattern": "disk_full"}
    ],
    actions=[
        RecoveryActionDef(
            action_type=RecoveryAction.DISK_CLEANUP,
            name="cleanup_temp_files",
            description="Clean up temporary files and logs",
            parameters={
                "target_path": "/tmp",
                "max_age_days": 1
            },
            timeout_seconds=120,
            risk_level="low"
        ),
        RecoveryActionDef(
            action_type=RecoveryAction.ROTATE_LOGS,
            name="rotate_application_logs",
            description="Force log rotation",
            parameters={
                "log_path": "/var/log"
            },
            timeout_seconds=60,
            risk_level="low"
        ),
        RecoveryActionDef(
            action_type=RecoveryAction.CUSTOM_SCRIPT,
            name="archive_old_data",
            description="Archive old application data",
            script_path="/opt/scripts/archive_data.sh",
            parameters={
                "archive_age_days": "30",
                "compression": "gzip"
            },
            timeout_seconds=600,
            risk_level="medium",
            prerequisites=["disk_space_available"]
        )
    ],
    max_retries=2,
    retry_delay=30,
    escalation_level=EscalationLevel.AUTOMATIC,
    success_threshold=0.6
)

HIGH_ERROR_RATE_WORKFLOW = RecoveryWorkflow(
    name="high_error_rate",
    description="Recovery workflow for application error spikes",
    trigger_conditions=[
        {"type": "metric", "metric_name": "error_rate", "operator": ">", "threshold": 0.1},
        {"type": "event", "event_pattern": "error_spike"}
    ],
    actions=[
        RecoveryActionDef(
            action_type=RecoveryAction.RESTART_SERVICE,
            name="restart_failing_service",
            description="Restart the failing application service",
            parameters={
                "service_name": "app-service"
            },
            timeout_seconds=120,
            risk_level="medium",
            prerequisites=["service_running"]
        ),
        RecoveryActionDef(
            action_type=RecoveryAction.CLEAR_CACHE,
            name="clear_application_cache",
            description="Clear application cache to reset state",
            parameters={
                "cache_type": "redis"
            },
            timeout_seconds=30,
            risk_level="low"
        ),
        RecoveryActionDef(
            action_type=RecoveryAction.SCALE_UP,
            name="scale_out_application",
            description="Scale out application instances",
            parameters={
                "target": "containers",
                "factor": 2
            },
            timeout_seconds=300,
            risk_level="low"
        ),
        RecoveryActionDef(
            action_type=RecoveryAction.CUSTOM_SCRIPT,
            name="check_database_connectivity",
            description="Verify database connectivity and repair if needed",
            script_path="/opt/scripts/db_health_check.sh",
            timeout_seconds=180,
            risk_level="low"
        )
    ],
    max_retries=3,
    retry_delay=60,
    escalation_level=EscalationLevel.AUTOMATIC,
    success_threshold=0.75
)

SLOW_RESPONSE_TIME_WORKFLOW = RecoveryWorkflow(
    name="slow_response_time",
    description="Recovery workflow for performance degradation",
    trigger_conditions=[
        {"type": "metric", "metric_name": "response_time", "operator": ">", "threshold": 5000},
        {"type": "event", "event_pattern": "performance_degradation"}
    ],
    actions=[
        RecoveryActionDef(
            action_type=RecoveryAction.CLEAR_CACHE,
            name="clear_all_caches",
            description="Clear all application caches",
            parameters={
                "cache_type": "redis"
            },
            timeout_seconds=60,
            risk_level="low"
        ),
        RecoveryActionDef(
            action_type=RecoveryAction.CUSTOM_SCRIPT,
            name="optimize_database_queries",
            description="Run database optimization",
            script_path="/opt/scripts/db_optimize.sh",
            timeout_seconds=300,
            risk_level="medium",
            prerequisites=["database_accessible"]
        ),
        RecoveryActionDef(
            action_type=RecoveryAction.SCALE_UP,
            name="scale_up_resources",
            description="Scale up CPU and memory",
            parameters={
                "target": "application",
                "factor": 1.5
            },
            timeout_seconds=300,
            risk_level="low"
        ),
        RecoveryActionDef(
            action_type=RecoveryAction.RESTART_SERVICE,
            name="restart_web_server",
            description="Restart web server",
            parameters={
                "service_name": "nginx"
            },
            timeout_seconds=60,
            risk_level="medium"
        )
    ],
    max_retries=2,
    retry_delay=120,
    escalation_level=EscalationLevel.NOTIFICATION,
    success_threshold=0.8
)

DATABASE_CONNECTION_FAILURE_WORKFLOW = RecoveryWorkflow(
    name="database_connection_failure",
    description="Recovery workflow for database connectivity issues",
    trigger_conditions=[
        {"type": "event", "event_pattern": "database_connection_failed"},
        {"type": "event", "event_pattern": "db_timeout"}
    ],
    actions=[
        RecoveryActionDef(
            action_type=RecoveryAction.CUSTOM_SCRIPT,
            name="test_database_connectivity",
            description="Test database connection",
            script_path="/opt/scripts/test_db_connection.sh",
            timeout_seconds=30,
            risk_level="low"
        ),
        RecoveryActionDef(
            action_type=RecoveryAction.RESTART_SERVICE,
            name="restart_database_service",
            description="Restart database service",
            parameters={
                "service_name": "postgresql"
            },
            timeout_seconds=180,
            risk_level="high",
            prerequisites=["database_accessible"]
        ),
        RecoveryActionDef(
            action_type=RecoveryAction.NETWORK_RESET,
            name="reset_network_connections",
            description="Reset network connections to database",
            command="sudo systemctl restart networking",
            timeout_seconds=60,
            risk_level="medium"
        ),
        RecoveryActionDef(
            action_type=RecoveryAction.CUSTOM_SCRIPT,
            name="failover_to_backup_database",
            description="Switch to backup database instance",
            script_path="/opt/scripts/db_failover.sh",
            timeout_seconds=120,
            risk_level="high",
            prerequisites=["backup_database_available"]
        )
    ],
    max_retries=2,
    retry_delay=90,
    escalation_level=EscalationLevel.MANUAL,
    success_threshold=0.9
)

NETWORK_CONNECTIVITY_ISSUES_WORKFLOW = RecoveryWorkflow(
    name="network_connectivity_issues",
    description="Recovery workflow for network problems",
    trigger_conditions=[
        {"type": "event", "event_pattern": "network_timeout"},
        {"type": "event", "event_pattern": "connection_refused"},
        {"type": "metric", "metric_name": "network_errors", "operator": ">", "threshold": 10}
    ],
    actions=[
        RecoveryActionDef(
            action_type=RecoveryAction.CUSTOM_SCRIPT,
            name="diagnose_network_issues",
            description="Run network diagnostics",
            script_path="/opt/scripts/network_diagnostics.sh",
            timeout_seconds=60,
            risk_level="low"
        ),
        RecoveryActionDef(
            action_type=RecoveryAction.NETWORK_RESET,
            name="restart_network_interfaces",
            description="Restart network interfaces",
            command="sudo systemctl restart NetworkManager",
            timeout_seconds=120,
            risk_level="medium"
        ),
        RecoveryActionDef(
            action_type=RecoveryAction.RESTART_SERVICE,
            name="restart_network_services",
            description="Restart network-dependent services",
            parameters={
                "service_name": "nginx"
            },
            timeout_seconds=60,
            risk_level="medium"
        ),
        RecoveryActionDef(
            action_type=RecoveryAction.CUSTOM_SCRIPT,
            name="switch_to_backup_network",
            description="Switch to backup network configuration",
            script_path="/opt/scripts/network_failover.sh",
            timeout_seconds=180,
            risk_level="high"
        )
    ],
    max_retries=3,
    retry_delay=60,
    escalation_level=EscalationLevel.NOTIFICATION,
    success_threshold=0.7
)

CONTAINER_FAILURE_WORKFLOW = RecoveryWorkflow(
    name="container_failure",
    description="Recovery workflow for container orchestration issues",
    trigger_conditions=[
        {"type": "event", "event_pattern": "container_crashed"},
        {"type": "event", "event_pattern": "pod_failed"},
        {"type": "metric", "metric_name": "container_restarts", "operator": ">", "threshold": 5}
    ],
    actions=[
        RecoveryActionDef(
            action_type=RecoveryAction.CUSTOM_SCRIPT,
            name="restart_failed_containers",
            description="Restart failed containers",
            command="docker restart $(docker ps -q --filter 'status=exited')",
            timeout_seconds=120,
            risk_level="low"
        ),
        RecoveryActionDef(
            action_type=RecoveryAction.SCALE_UP,
            name="scale_up_healthy_containers",
            description="Scale up healthy container instances",
            parameters={
                "target": "containers",
                "factor": 2
            },
            timeout_seconds=180,
            risk_level="low"
        ),
        RecoveryActionDef(
            action_type=RecoveryAction.CUSTOM_SCRIPT,
            name="recreate_containers_from_images",
            description="Recreate containers from fresh images",
            script_path="/opt/scripts/recreate_containers.sh",
            timeout_seconds=300,
            risk_level="medium"
        ),
        RecoveryActionDef(
            action_type=RecoveryAction.CUSTOM_SCRIPT,
            name="check_container_resources",
            description="Check and adjust container resource limits",
            script_path="/opt/scripts/adjust_container_resources.sh",
            timeout_seconds=60,
            risk_level="low"
        )
    ],
    max_retries=2,
    retry_delay=90,
    escalation_level=EscalationLevel.AUTOMATIC,
    success_threshold=0.8
)

SSL_CERTIFICATE_EXPIRATION_WORKFLOW = RecoveryWorkflow(
    name="ssl_certificate_expiration",
    description="Recovery workflow for SSL certificate issues",
    trigger_conditions=[
        {"type": "event", "event_pattern": "ssl_certificate_expired"},
        {"type": "event", "event_pattern": "ssl_certificate_expiring"},
        {"type": "metric", "metric_name": "ssl_cert_days_remaining", "operator": "<", "threshold": 7}
    ],
    actions=[
        RecoveryActionDef(
            action_type=RecoveryAction.CUSTOM_SCRIPT,
            name="renew_ssl_certificate",
            description="Attempt to renew SSL certificate automatically",
            script_path="/opt/scripts/renew_ssl_cert.sh",
            timeout_seconds=300,
            risk_level="low"
        ),
        RecoveryActionDef(
            action_type=RecoveryAction.RESTART_SERVICE,
            name="restart_web_server",
            description="Restart web server to load new certificate",
            parameters={
                "service_name": "nginx"
            },
            timeout_seconds=60,
            risk_level="medium"
        ),
        RecoveryActionDef(
            action_type=RecoveryAction.CUSTOM_SCRIPT,
            name="verify_ssl_certificate",
            description="Verify SSL certificate is valid and properly installed",
            script_path="/opt/scripts/verify_ssl_cert.sh",
            timeout_seconds=30,
            risk_level="low"
        )
    ],
    max_retries=1,
    retry_delay=300,
    escalation_level=EscalationLevel.NOTIFICATION,
    success_threshold=1.0
)

# Workflow Collection
PREDEFINED_WORKFLOWS = {
    "high_cpu_usage": HIGH_CPU_USAGE_WORKFLOW,
    "high_memory_usage": HIGH_MEMORY_USAGE_WORKFLOW,
    "disk_space_exhaustion": DISK_SPACE_EXHAUSTION_WORKFLOW,
    "high_error_rate": HIGH_ERROR_RATE_WORKFLOW,
    "slow_response_time": SLOW_RESPONSE_TIME_WORKFLOW,
    "database_connection_failure": DATABASE_CONNECTION_FAILURE_WORKFLOW,
    "network_connectivity_issues": NETWORK_CONNECTIVITY_ISSUES_WORKFLOW,
    "container_failure": CONTAINER_FAILURE_WORKFLOW,
    "ssl_certificate_expiration": SSL_CERTIFICATE_EXPIRATION_WORKFLOW
}


class WorkflowManager:
    """
    Manages recovery workflows and provides templates for creating custom workflows.
    """
    
    def __init__(self):
        self.workflows = PREDEFINED_WORKFLOWS.copy()
    
    def get_workflow(self, name: str) -> RecoveryWorkflow:
        """Get a workflow by name"""
        return self.workflows.get(name)
    
    def list_workflows(self) -> List[str]:
        """List all available workflow names"""
        return list(self.workflows.keys())
    
    def add_custom_workflow(self, workflow: RecoveryWorkflow):
        """Add a custom workflow"""
        self.workflows[workflow.name] = workflow
    
    def create_workflow_template(
        self,
        name: str,
        description: str,
        trigger_patterns: List[str]
    ) -> RecoveryWorkflow:
        """Create a basic workflow template"""
        trigger_conditions = [
            {"type": "event", "event_pattern": pattern}
            for pattern in trigger_patterns
        ]
        
        return RecoveryWorkflow(
            name=name,
            description=description,
            trigger_conditions=trigger_conditions,
            actions=[],  # To be filled in
            max_retries=2,
            retry_delay=60,
            escalation_level=EscalationLevel.AUTOMATIC,
            success_threshold=0.7
        )
    
    def get_workflow_for_event(self, event: str) -> List[RecoveryWorkflow]:
        """Get workflows that match a specific event"""
        matching_workflows = []
        
        for workflow in self.workflows.values():
            for condition in workflow.trigger_conditions:
                if condition.get("type") == "event":
                    pattern = condition.get("event_pattern", "")
                    if pattern in event or event in pattern:
                        matching_workflows.append(workflow)
                        break
        
        return matching_workflows
    
    def get_workflow_for_metric(
        self,
        metric_name: str,
        metric_value: float
    ) -> List[RecoveryWorkflow]:
        """Get workflows that match a specific metric condition"""
        matching_workflows = []
        
        for workflow in self.workflows.values():
            for condition in workflow.trigger_conditions:
                if condition.get("type") == "metric":
                    if condition.get("metric_name") == metric_name:
                        operator = condition.get("operator", ">")
                        threshold = condition.get("threshold", 0)
                        
                        if self._evaluate_condition(metric_value, operator, threshold):
                            matching_workflows.append(workflow)
                            break
        
        return matching_workflows
    
    def _evaluate_condition(self, value: float, operator: str, threshold: float) -> bool:
        """Evaluate a metric condition"""
        if operator == ">":
            return value > threshold
        elif operator == "<":
            return value < threshold
        elif operator == ">=":
            return value >= threshold
        elif operator == "<=":
            return value <= threshold
        elif operator == "==":
            return abs(value - threshold) < 0.001
        
        return False


# Example custom workflow creation
def create_custom_application_restart_workflow(
    service_name: str,
    health_check_url: str = None
) -> RecoveryWorkflow:
    """Create a custom workflow for application restart with health checks"""
    
    actions = [
        RecoveryActionDef(
            action_type=RecoveryAction.RESTART_SERVICE,
            name=f"restart_{service_name}",
            description=f"Restart {service_name} service",
            parameters={
                "service_name": service_name
            },
            timeout_seconds=120,
            risk_level="medium"
        )
    ]
    
    if health_check_url:
        actions.append(
            RecoveryActionDef(
                action_type=RecoveryAction.CUSTOM_SCRIPT,
                name="health_check",
                description="Perform health check after restart",
                command=f"curl -f {health_check_url}",
                timeout_seconds=30,
                risk_level="low"
            )
        )
    
    return RecoveryWorkflow(
        name=f"restart_{service_name}_workflow",
        description=f"Custom restart workflow for {service_name}",
        trigger_conditions=[
            {"type": "event", "event_pattern": f"{service_name}_failed"},
            {"type": "event", "event_pattern": f"{service_name}_unhealthy"}
        ],
        actions=actions,
        max_retries=2,
        retry_delay=60,
        escalation_level=EscalationLevel.NOTIFICATION,
        success_threshold=0.8
    )


# Workflow execution helpers
def validate_workflow(workflow: RecoveryWorkflow) -> List[str]:
    """Validate a workflow configuration and return any issues"""
    issues = []
    
    if not workflow.name:
        issues.append("Workflow name is required")
    
    if not workflow.trigger_conditions:
        issues.append("At least one trigger condition is required")
    
    if not workflow.actions:
        issues.append("At least one recovery action is required")
    
    for i, action in enumerate(workflow.actions):
        if not action.name:
            issues.append(f"Action {i} is missing a name")
        
        if action.action_type == RecoveryAction.CUSTOM_SCRIPT and not action.script_path:
            issues.append(f"Action {i} ({action.name}) needs script_path for CUSTOM_SCRIPT type")
        
        if action.timeout_seconds <= 0:
            issues.append(f"Action {i} ({action.name}) needs positive timeout_seconds")
    
    if workflow.success_threshold < 0 or workflow.success_threshold > 1:
        issues.append("Success threshold must be between 0 and 1")
    
    return issues