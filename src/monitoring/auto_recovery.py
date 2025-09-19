"""
Automated recovery system for handling system failures and degradations.

This module provides intelligent automated recovery mechanisms that can respond
to various failure scenarios with appropriate recovery actions, fallback strategies,
and escalation procedures.
"""

import asyncio
import logging
import json
import re
import shlex
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Awaitable
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import subprocess
import os

from .anomaly_detection import AnomalyResult, AnomalySeverity
from .predictive import FailurePrediction, PredictionConfidence


class CommandValidator:
    """
    Secure command validation and sanitization utilities.
    Prevents command injection attacks through input validation.
    """

    # Allowed commands for different operation types
    ALLOWED_COMMANDS = {
        'service_management': {
            'systemctl': ['start', 'stop', 'restart', 'reload', 'enable', 'disable', 'is-active', 'status'],
        },
        'container_management': {
            'docker': ['update', 'restart', 'stop', 'start', 'ps', 'inspect'],
        },
        'system_monitoring': {
            'top': ['-bn1'],
            'free': [],
            'df': ['/'],
            'ps': ['aux'],
        },
        'cache_management': {
            'redis-cli': ['-h', 'FLUSHALL'],
        },
        'log_management': {
            'logrotate': ['-f'],
        },
        'process_management': {
            'pgrep': ['-f'],
            'kill': ['-TERM', '-KILL', '-HUP'],
        },
        'disk_management': {
            'find': ['-type', 'f', '-mtime', '-delete'],
            'apt-get': ['clean'],
            'yum': ['clean', 'all'],
        }
    }

    # Dangerous characters and patterns
    DANGEROUS_PATTERNS = [
        r'[;&|`$()]',  # Shell metacharacters
        r'\.\./',      # Directory traversal
        r'/etc/',      # System directories
        r'/proc/',     # Process filesystem
        r'rm\s+-rf',   # Destructive commands
        r'dd\s+if=',   # Disk operations
        r'mkfs\.',     # Filesystem creation
        r'fdisk',      # Disk partitioning
        r'mount',      # Mount operations
        r'umount',     # Unmount operations
    ]

    @staticmethod
    def validate_service_name(service_name: str) -> bool:
        """Validate service name for systemctl operations."""
        if not service_name or len(service_name) > 50:
            return False

        # Only allow alphanumeric, hyphens, underscores, and dots
        if not re.match(r'^[a-zA-Z0-9._-]+$', service_name):
            return False

        # Prevent path traversal
        if '..' in service_name or '/' in service_name:
            return False

        return True

    @staticmethod
    def validate_container_name(container_name: str) -> bool:
        """Validate Docker container name."""
        if not container_name or len(container_name) > 100:
            return False

        # Docker container names: alphanumeric, hyphens, underscores
        if not re.match(r'^[a-zA-Z0-9._-]+$', container_name):
            return False

        return True

    @staticmethod
    def validate_file_path(file_path: str) -> bool:
        """Validate file paths to prevent directory traversal."""
        if not file_path or len(file_path) > 500:
            return False

        # Prevent directory traversal
        if '..' in file_path:
            return False

        # Only allow specific safe directories
        safe_prefixes = ['/tmp/', '/var/log/', '/opt/', '/home/', '/usr/local/']
        if not any(file_path.startswith(prefix) for prefix in safe_prefixes):
            return False

        return True

    @staticmethod
    def validate_process_pattern(pattern: str) -> bool:
        """Validate process search patterns."""
        if not pattern or len(pattern) > 100:
            return False

        # Check for dangerous patterns
        for dangerous in CommandValidator.DANGEROUS_PATTERNS:
            if re.search(dangerous, pattern, re.IGNORECASE):
                return False

        return True

    @staticmethod
    def validate_signal(signal: str) -> bool:
        """Validate process signals."""
        valid_signals = ['TERM', 'HUP', 'INT', 'QUIT', 'KILL', 'USR1', 'USR2']
        return signal in valid_signals

    @staticmethod
    def sanitize_command_args(args: List[str]) -> List[str]:
        """Sanitize command arguments."""
        sanitized = []
        for arg in args:
            # Remove dangerous characters
            sanitized_arg = re.sub(r'[;&|`$()]', '', str(arg))
            # Limit length
            if len(sanitized_arg) > 200:
                sanitized_arg = sanitized_arg[:200]
            sanitized.append(sanitized_arg)
        return sanitized

    @staticmethod
    def build_safe_command(base_cmd: str, args: List[str]) -> List[str]:
        """Build a safe command list for subprocess execution."""
        # Validate base command
        if base_cmd not in ['systemctl', 'docker', 'redis-cli', 'logrotate', 'pgrep', 'kill', 'find', 'apt-get', 'yum', 'chmod']:
            raise ValueError(f"Command not allowed: {base_cmd}")

        # Sanitize arguments
        safe_args = CommandValidator.sanitize_command_args(args)

        # Build command list (no shell injection possible)
        cmd_list = [base_cmd] + safe_args

        return cmd_list


class RecoveryAction(Enum):
    """Types of recovery actions available"""
    RESTART_SERVICE = "restart_service"
    SCALE_UP = "scale_up"
    SCALE_DOWN = "scale_down"
    CLEAR_CACHE = "clear_cache"
    ROTATE_LOGS = "rotate_logs"
    KILL_PROCESSES = "kill_processes"
    RESTORE_BACKUP = "restore_backup"
    NETWORK_RESET = "network_reset"
    DATABASE_REPAIR = "database_repair"
    DISK_CLEANUP = "disk_cleanup"
    CUSTOM_SCRIPT = "custom_script"


class RecoveryResult(Enum):
    """Results of recovery actions"""
    SUCCESS = "success"
    PARTIAL_SUCCESS = "partial_success"
    FAILED = "failed"
    SKIPPED = "skipped"
    TIMEOUT = "timeout"


class EscalationLevel(Enum):
    """Escalation levels for recovery procedures"""
    AUTOMATIC = "automatic"
    NOTIFICATION = "notification"
    MANUAL = "manual"
    EMERGENCY = "emergency"


@dataclass
class RecoveryActionDef:
    """Definition of a recovery action"""
    action_type: RecoveryAction
    name: str
    description: str
    command: Optional[str] = None
    script_path: Optional[str] = None
    parameters: Dict[str, Any] = field(default_factory=dict)
    timeout_seconds: int = 300
    required_permissions: List[str] = field(default_factory=list)
    prerequisites: List[str] = field(default_factory=list)
    risk_level: str = "low"  # low, medium, high
    success_criteria: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RecoveryExecution:
    """Record of a recovery action execution"""
    action_def: RecoveryActionDef
    trigger_event: str
    start_time: datetime
    end_time: Optional[datetime] = None
    result: Optional[RecoveryResult] = None
    output: str = ""
    error: str = ""
    metrics_before: Dict[str, float] = field(default_factory=dict)
    metrics_after: Dict[str, float] = field(default_factory=dict)
    success_rate: float = 0.0


@dataclass
class RecoveryWorkflow:
    """A workflow of recovery actions for a specific failure scenario"""
    name: str
    description: str
    trigger_conditions: List[Dict[str, Any]]
    actions: List[RecoveryActionDef]
    max_retries: int = 3
    retry_delay: int = 30  # seconds
    escalation_level: EscalationLevel = EscalationLevel.AUTOMATIC
    success_threshold: float = 0.8
    enabled: bool = True


@dataclass
class AutoRecoveryConfig:
    """Configuration for automated recovery system"""
    # General settings
    enable_auto_recovery: bool = True
    max_concurrent_recoveries: int = 3
    recovery_cooldown: int = 300  # seconds between recovery attempts
    
    # Safety settings
    max_recoveries_per_hour: int = 10
    require_confirmation_for_high_risk: bool = True
    enable_rollback: bool = True
    
    # Monitoring settings
    health_check_interval: int = 60  # seconds
    metric_collection_interval: int = 30  # seconds
    
    # Notification settings
    notify_on_recovery: bool = True
    notify_on_failure: bool = True
    notification_webhook: Optional[str] = None


class RecoveryExecutor:
    """
    Executes individual recovery actions with proper error handling and monitoring.
    """
    
    def __init__(self, config: AutoRecoveryConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Execution tracking
        self.active_executions: Dict[str, RecoveryExecution] = {}
        self.execution_history: List[RecoveryExecution] = []
        
    async def execute_action(
        self,
        action_def: RecoveryActionDef,
        trigger_event: str,
        context: Optional[Dict[str, Any]] = None
    ) -> RecoveryExecution:
        """
        Execute a single recovery action.
        
        Args:
            action_def: Definition of the action to execute
            trigger_event: Event that triggered this recovery
            context: Additional context for the execution
            
        Returns:
            Recovery execution record
        """
        execution = RecoveryExecution(
            action_def=action_def,
            trigger_event=trigger_event,
            start_time=datetime.utcnow()
        )
        
        execution_id = f"{action_def.name}_{execution.start_time.timestamp()}"
        self.active_executions[execution_id] = execution
        
        try:
            self.logger.info(f"Starting recovery action: {action_def.name} for {trigger_event}")
            
            # Pre-execution checks
            if not await self._check_prerequisites(action_def):
                execution.result = RecoveryResult.SKIPPED
                execution.error = "Prerequisites not met"
                return execution
            
            # Collect metrics before execution
            execution.metrics_before = await self._collect_metrics(action_def)
            
            # Execute the action
            if action_def.action_type == RecoveryAction.RESTART_SERVICE:
                result = await self._restart_service(action_def, context)
            elif action_def.action_type == RecoveryAction.SCALE_UP:
                result = await self._scale_up(action_def, context)
            elif action_def.action_type == RecoveryAction.SCALE_DOWN:
                result = await self._scale_down(action_def, context)
            elif action_def.action_type == RecoveryAction.CLEAR_CACHE:
                result = await self._clear_cache(action_def, context)
            elif action_def.action_type == RecoveryAction.ROTATE_LOGS:
                result = await self._rotate_logs(action_def, context)
            elif action_def.action_type == RecoveryAction.KILL_PROCESSES:
                result = await self._kill_processes(action_def, context)
            elif action_def.action_type == RecoveryAction.DISK_CLEANUP:
                result = await self._disk_cleanup(action_def, context)
            elif action_def.action_type == RecoveryAction.CUSTOM_SCRIPT:
                result = await self._execute_custom_script(action_def, context)
            else:
                result = await self._execute_command(action_def, context)
            
            execution.result = result.result
            execution.output = result.output
            execution.error = result.error
            
            # Collect metrics after execution
            execution.metrics_after = await self._collect_metrics(action_def)
            
            # Calculate success rate
            execution.success_rate = await self._calculate_success_rate(execution)
            
            self.logger.info(
                f"Recovery action {action_def.name} completed with result: {execution.result}"
            )
            
        except asyncio.TimeoutError:
            execution.result = RecoveryResult.TIMEOUT
            execution.error = f"Action timed out after {action_def.timeout_seconds} seconds"
            self.logger.error(f"Recovery action {action_def.name} timed out")
            
        except Exception as e:
            execution.result = RecoveryResult.FAILED
            execution.error = str(e)
            self.logger.error(f"Recovery action {action_def.name} failed: {e}")
        
        finally:
            execution.end_time = datetime.utcnow()
            
            # Move to history
            if execution_id in self.active_executions:
                del self.active_executions[execution_id]
            self.execution_history.append(execution)
            
            # Keep only recent history
            cutoff_time = datetime.utcnow() - timedelta(hours=24)
            self.execution_history = [
                e for e in self.execution_history if e.start_time > cutoff_time
            ]
        
        return execution
    
    async def _check_prerequisites(self, action_def: RecoveryActionDef) -> bool:
        """Check if prerequisites for action execution are met"""
        for prerequisite in action_def.prerequisites:
            if prerequisite == "disk_space_available":
                if not await self._check_disk_space():
                    return False
            elif prerequisite == "service_running":
                service_name = action_def.parameters.get("service_name")
                if service_name and not await self._is_service_running(service_name):
                    return False
            elif prerequisite == "database_accessible":
                if not await self._check_database_connection():
                    return False
        
        return True
    
    async def _restart_service(
        self,
        action_def: RecoveryActionDef,
        context: Optional[Dict[str, Any]]
    ) -> 'CommandResult':
        """Restart a system service"""
        service_name = action_def.parameters.get("service_name", "unknown")

        # Validate service name
        if not CommandValidator.validate_service_name(service_name):
            return CommandResult(
                result=RecoveryResult.FAILED,
                output="",
                error=f"Invalid service name: {service_name}"
            )

        try:
            # Use systemctl for service management - build safe command
            restart_cmd = CommandValidator.build_safe_command('systemctl', ['restart', service_name])
            result = await self._run_safe_command(restart_cmd, action_def.timeout_seconds)

            # Verify service is running
            await asyncio.sleep(5)  # Wait for service to start
            check_cmd = CommandValidator.build_safe_command('systemctl', ['is-active', service_name])
            check_result = await self._run_safe_command(check_cmd, 10)

            if check_result.return_code == 0 and "active" in check_result.output:
                return CommandResult(
                    result=RecoveryResult.SUCCESS,
                    output=f"Service {service_name} restarted successfully",
                    error=""
                )
            else:
                return CommandResult(
                    result=RecoveryResult.FAILED,
                    output=result.output,
                    error=f"Service {service_name} failed to start: {check_result.output}"
                )

        except Exception as e:
            return CommandResult(
                result=RecoveryResult.FAILED,
                output="",
                error=f"Failed to restart service {service_name}: {str(e)}"
            )
    
    async def _scale_up(
        self,
        action_def: RecoveryActionDef,
        context: Optional[Dict[str, Any]]
    ) -> 'CommandResult':
        """Scale up resources (Docker containers, processes, etc.)"""
        scale_target = action_def.parameters.get("target", "unknown")
        scale_factor = action_def.parameters.get("factor", 1)

        try:
            if "docker" in scale_target.lower():
                # Docker container scaling
                container_name = action_def.parameters.get("container_name")
                if not container_name:
                    return CommandResult(
                        result=RecoveryResult.FAILED,
                        output="",
                        error="Container name not specified for Docker scaling"
                    )

                # Validate container name
                if not CommandValidator.validate_container_name(container_name):
                    return CommandResult(
                        result=RecoveryResult.FAILED,
                        output="",
                        error=f"Invalid container name: {container_name}"
                    )

                # Validate scale factor
                try:
                    scale_value = float(scale_factor)
                    if scale_value <= 0 or scale_value > 16:  # Reasonable limits
                        raise ValueError("Scale factor out of range")
                except ValueError:
                    return CommandResult(
                        result=RecoveryResult.FAILED,
                        output="",
                        error=f"Invalid scale factor: {scale_factor}"
                    )

                # Build safe Docker command
                docker_cmd = CommandValidator.build_safe_command('docker', [
                    'update',
                    f'--cpus={scale_value}',
                    f'--memory={scale_value}g',
                    container_name
                ])
                result = await self._run_safe_command(docker_cmd, action_def.timeout_seconds)

                if result.return_code == 0:
                    return CommandResult(
                        result=RecoveryResult.SUCCESS,
                        output=f"Scaled up {container_name} by factor {scale_factor}",
                        error=""
                    )
                else:
                    return CommandResult(
                        result=RecoveryResult.FAILED,
                        output=result.output,
                        error=result.error
                    )

            # Default: log the scale up action (no actual operation for safety)
            return CommandResult(
                result=RecoveryResult.SUCCESS,
                output=f"Scale up action logged for {scale_target} (no operation performed for safety)",
                error=""
            )

        except Exception as e:
            return CommandResult(
                result=RecoveryResult.FAILED,
                output="",
                error=f"Failed to scale up {scale_target}: {str(e)}"
            )
    
    async def _scale_down(
        self,
        action_def: RecoveryActionDef,
        context: Optional[Dict[str, Any]]
    ) -> 'CommandResult':
        """Scale down resources"""
        scale_target = action_def.parameters.get("target", "unknown")
        scale_factor = action_def.parameters.get("factor", 0.5)
        
        # Implementation similar to scale_up but with reduction
        return CommandResult(
            result=RecoveryResult.SUCCESS,
            output=f"Scale down action logged for {scale_target}",
            error=""
        )
    
    async def _clear_cache(
        self,
        action_def: RecoveryActionDef,
        context: Optional[Dict[str, Any]]
    ) -> 'CommandResult':
        """Clear various types of caches"""
        cache_type = action_def.parameters.get("cache_type", "system")

        try:
            if cache_type == "redis":
                redis_host = action_def.parameters.get("redis_host", "localhost")

                # Validate redis host
                if not re.match(r'^[a-zA-Z0-9.-]+$', redis_host) or len(redis_host) > 100:
                    return CommandResult(
                        result=RecoveryResult.FAILED,
                        output="",
                        error=f"Invalid Redis host: {redis_host}"
                    )

                # Build safe Redis command
                redis_cmd = CommandValidator.build_safe_command('redis-cli', ['-h', redis_host, 'FLUSHALL'])
                result = await self._run_safe_command(redis_cmd, action_def.timeout_seconds)

            elif cache_type == "system":
                # System cache clearing - too dangerous, just log
                self.logger.warning("System cache clearing requested - skipped for security")
                return CommandResult(
                    result=RecoveryResult.SUCCESS,
                    output="System cache clearing skipped for security reasons",
                    error=""
                )
            else:
                # Custom cache clearing - not allowed for security
                return CommandResult(
                    result=RecoveryResult.FAILED,
                    output="",
                    error=f"Custom cache clearing not allowed for security: {cache_type}"
                )

            if result.return_code == 0:
                return CommandResult(
                    result=RecoveryResult.SUCCESS,
                    output=f"Cleared {cache_type} cache successfully",
                    error=""
                )
            else:
                return CommandResult(
                    result=RecoveryResult.FAILED,
                    output=result.output,
                    error=result.error
                )

        except Exception as e:
            return CommandResult(
                result=RecoveryResult.FAILED,
                output="",
                error=f"Failed to clear {cache_type} cache: {str(e)}"
            )
    
    async def _rotate_logs(
        self,
        action_def: RecoveryActionDef,
        context: Optional[Dict[str, Any]]
    ) -> 'CommandResult':
        """Rotate log files to free up disk space"""
        log_path = action_def.parameters.get("log_path", "/var/log")

        try:
            # Validate log path (not actually used in current implementation for security)
            if not CommandValidator.validate_file_path(log_path):
                self.logger.warning(f"Invalid log path provided: {log_path}")

            # Force log rotation - use standard config file only
            logrotate_cmd = CommandValidator.build_safe_command('logrotate', ['-f', '/etc/logrotate.conf'])
            result = await self._run_safe_command(logrotate_cmd, action_def.timeout_seconds)

            if result.return_code == 0:
                return CommandResult(
                    result=RecoveryResult.SUCCESS,
                    output="Log rotation completed successfully",
                    error=""
                )
            else:
                return CommandResult(
                    result=RecoveryResult.PARTIAL_SUCCESS,
                    output=result.output,
                    error=result.error
                )

        except Exception as e:
            return CommandResult(
                result=RecoveryResult.FAILED,
                output="",
                error=f"Failed to rotate logs: {str(e)}"
            )
    
    async def _kill_processes(
        self,
        action_def: RecoveryActionDef,
        context: Optional[Dict[str, Any]]
    ) -> 'CommandResult':
        """Kill problematic processes"""
        process_pattern = action_def.parameters.get("process_pattern")
        signal = action_def.parameters.get("signal", "TERM")

        if not process_pattern:
            return CommandResult(
                result=RecoveryResult.SKIPPED,
                output="No process pattern specified",
                error=""
            )

        # Validate process pattern
        if not CommandValidator.validate_process_pattern(process_pattern):
            return CommandResult(
                result=RecoveryResult.FAILED,
                output="",
                error=f"Invalid process pattern: {process_pattern}"
            )

        # Validate signal
        if not CommandValidator.validate_signal(signal):
            return CommandResult(
                result=RecoveryResult.FAILED,
                output="",
                error=f"Invalid signal: {signal}"
            )

        try:
            # Find processes matching pattern - build safe command
            pgrep_cmd = CommandValidator.build_safe_command('pgrep', ['-f', process_pattern])
            result = await self._run_safe_command(pgrep_cmd, 10)

            if result.return_code == 0 and result.output.strip():
                pids = result.output.strip().split('\n')

                # Validate PIDs are numeric
                valid_pids = []
                for pid in pids:
                    if pid.isdigit() and int(pid) > 1:  # Don't allow killing PID 1 (init)
                        valid_pids.append(pid)

                if not valid_pids:
                    return CommandResult(
                        result=RecoveryResult.SUCCESS,
                        output="No valid processes found to kill",
                        error=""
                    )

                # Build kill command safely
                kill_args = [f'-{signal}'] + valid_pids
                kill_cmd = CommandValidator.build_safe_command('kill', kill_args)
                kill_result = await self._run_safe_command(kill_cmd, action_def.timeout_seconds)

                return CommandResult(
                    result=RecoveryResult.SUCCESS if kill_result.return_code == 0 else RecoveryResult.PARTIAL_SUCCESS,
                    output=f"Killed {len(valid_pids)} processes matching '{process_pattern}'",
                    error=kill_result.error
                )
            else:
                return CommandResult(
                    result=RecoveryResult.SUCCESS,
                    output=f"No processes found matching '{process_pattern}'",
                    error=""
                )

        except Exception as e:
            return CommandResult(
                result=RecoveryResult.FAILED,
                output="",
                error=f"Failed to kill processes: {str(e)}"
            )
    
    async def _disk_cleanup(
        self,
        action_def: RecoveryActionDef,
        context: Optional[Dict[str, Any]]
    ) -> 'CommandResult':
        """Clean up disk space"""
        target_path = action_def.parameters.get("target_path", "/tmp")
        max_age_days = action_def.parameters.get("max_age_days", 7)

        # Validate target path
        if not CommandValidator.validate_file_path(target_path):
            return CommandResult(
                result=RecoveryResult.FAILED,
                output="",
                error=f"Invalid target path: {target_path}"
            )

        # Validate max_age_days
        try:
            age_days = int(max_age_days)
            if age_days < 1 or age_days > 365:
                raise ValueError("Age out of range")
        except ValueError:
            return CommandResult(
                result=RecoveryResult.FAILED,
                output="",
                error=f"Invalid max_age_days: {max_age_days}"
            )

        try:
            # Clean old files - build safe find command
            find_cmd = CommandValidator.build_safe_command('find', [
                target_path,
                '-type', 'f',
                '-mtime', f'+{age_days}',
                '-delete'
            ])
            result = await self._run_safe_command(find_cmd, action_def.timeout_seconds)

            cleanup_messages = []
            if result.return_code == 0:
                cleanup_messages.append(f"Cleaned old files from {target_path}")
            else:
                cleanup_messages.append(f"File cleanup had issues: {result.error}")

            # Clean package cache if applicable
            try:
                if os.path.exists("/usr/bin/apt-get"):
                    apt_cmd = CommandValidator.build_safe_command('apt-get', ['clean'])
                    apt_result = await self._run_safe_command(apt_cmd, 60)
                    if apt_result.return_code == 0:
                        cleanup_messages.append("APT cache cleaned")
                elif os.path.exists("/usr/bin/yum"):
                    yum_cmd = CommandValidator.build_safe_command('yum', ['clean', 'all'])
                    yum_result = await self._run_safe_command(yum_cmd, 60)
                    if yum_result.return_code == 0:
                        cleanup_messages.append("YUM cache cleaned")
            except Exception as cache_e:
                cleanup_messages.append(f"Package cache cleanup failed: {cache_e}")

            return CommandResult(
                result=RecoveryResult.SUCCESS,
                output="; ".join(cleanup_messages),
                error=""
            )

        except Exception as e:
            return CommandResult(
                result=RecoveryResult.FAILED,
                output="",
                error=f"Failed to cleanup disk: {str(e)}"
            )
    
    async def _execute_custom_script(
        self,
        action_def: RecoveryActionDef,
        context: Optional[Dict[str, Any]]
    ) -> 'CommandResult':
        """Execute a custom recovery script"""
        script_path = action_def.script_path

        if not script_path:
            return CommandResult(
                result=RecoveryResult.FAILED,
                output="",
                error="No script path specified"
            )

        # Validate script path
        if not CommandValidator.validate_file_path(script_path):
            return CommandResult(
                result=RecoveryResult.FAILED,
                output="",
                error=f"Invalid script path: {script_path}"
            )

        if not os.path.exists(script_path):
            return CommandResult(
                result=RecoveryResult.FAILED,
                output="",
                error=f"Script not found: {script_path}"
            )

        # Additional security check - only allow scripts in specific directories
        allowed_script_dirs = ['/opt/recovery/', '/usr/local/recovery/', '/home/recovery/']
        if not any(script_path.startswith(allowed_dir) for allowed_dir in allowed_script_dirs):
            return CommandResult(
                result=RecoveryResult.FAILED,
                output="",
                error=f"Script not in allowed directory: {script_path}"
            )

        try:
            # Make script executable - build safe command
            chmod_cmd = CommandValidator.build_safe_command('chmod', ['+x', script_path])
            chmod_result = await self._run_safe_command(chmod_cmd, 10)

            if chmod_result.return_code != 0:
                return CommandResult(
                    result=RecoveryResult.FAILED,
                    output="",
                    error=f"Failed to make script executable: {chmod_result.error}"
                )

            # Validate and sanitize parameters
            script_args = []
            for key, value in action_def.parameters.items():
                # Validate parameter key
                if not re.match(r'^[a-zA-Z_][a-zA-Z0-9_]{0,31}$', key):
                    return CommandResult(
                        result=RecoveryResult.FAILED,
                        output="",
                        error=f"Invalid parameter key: {key}"
                    )

                # Sanitize parameter value
                safe_value = re.sub(r'[;&|`$()]', '', str(value))
                if len(safe_value) > 200:
                    safe_value = safe_value[:200]

                script_args.append(f"{key}={safe_value}")

            # Execute script with parameters
            script_cmd = [script_path] + script_args
            result = await self._run_safe_command(script_cmd, action_def.timeout_seconds)

            return CommandResult(
                result=RecoveryResult.SUCCESS if result.return_code == 0 else RecoveryResult.FAILED,
                output=result.output,
                error=result.error
            )

        except Exception as e:
            return CommandResult(
                result=RecoveryResult.FAILED,
                output="",
                error=f"Failed to execute custom script: {str(e)}"
            )
    
    async def _execute_command(
        self,
        action_def: RecoveryActionDef,
        context: Optional[Dict[str, Any]]
    ) -> 'CommandResult':
        """Execute a generic command - DEPRECATED: Use specific action methods instead"""
        command = action_def.command

        if not command:
            return CommandResult(
                result=RecoveryResult.SKIPPED,
                output="",
                error="No command specified"
            )

        # Generic command execution is disabled for security
        # All commands must go through specific action methods with validation
        return CommandResult(
            result=RecoveryResult.FAILED,
            output="",
            error="Generic command execution disabled for security. Use specific action methods."
        )
    
    async def _run_command(self, command: str, timeout: int) -> 'ShellResult':
        """Run a shell command with timeout - DEPRECATED: Use _run_safe_command instead"""
        # This method is deprecated and should only be used for system monitoring commands
        # All other operations should use _run_safe_command
        self.logger.warning(f"Using deprecated _run_command: {command[:50]}...")

        # Check if this is a safe monitoring command
        safe_monitoring_commands = [
            'top -bn1', 'free', 'df /', 'systemctl is-active'
        ]

        command_safe = any(command.startswith(safe_cmd) for safe_cmd in safe_monitoring_commands)

        if not command_safe:
            raise ValueError(f"Unsafe command blocked: {command}")

        try:
            # Split command safely for monitoring operations only
            cmd_parts = shlex.split(command)

            process = await asyncio.create_subprocess_exec(
                *cmd_parts,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )

            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=timeout
            )

            return ShellResult(
                return_code=process.returncode,
                output=stdout.decode('utf-8'),
                error=stderr.decode('utf-8')
            )

        except asyncio.TimeoutError:
            if 'process' in locals():
                process.kill()
                await process.wait()
            raise
        except Exception as e:
            self.logger.error(f"Command execution failed: {e}")
            raise

    async def _run_safe_command(self, cmd_list: List[str], timeout: int) -> 'ShellResult':
        """Run a command safely using argument list (no shell injection possible)"""
        try:
            self.logger.info(f"Executing safe command: {' '.join(cmd_list)}")

            process = await asyncio.create_subprocess_exec(
                *cmd_list,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )

            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=timeout
            )

            return ShellResult(
                return_code=process.returncode,
                output=stdout.decode('utf-8', errors='replace'),
                error=stderr.decode('utf-8', errors='replace')
            )

        except asyncio.TimeoutError:
            self.logger.warning(f"Command timed out: {' '.join(cmd_list)}")
            process.kill()
            await process.wait()
            raise
        except Exception as e:
            self.logger.error(f"Safe command execution failed: {e}")
            raise
    
    async def _collect_metrics(self, action_def: RecoveryActionDef) -> Dict[str, float]:
        """Collect system metrics before/after recovery actions"""
        metrics = {}
        
        try:
            # CPU usage
            cpu_result = await self._run_command("top -bn1 | grep 'Cpu(s)' | awk '{print $2}' | cut -d'%' -f1", 10)
            if cpu_result.return_code == 0:
                try:
                    metrics["cpu_usage"] = float(cpu_result.output.strip())
                except ValueError:
                    pass
            
            # Memory usage
            mem_result = await self._run_command("free | grep Mem | awk '{printf \"%.1f\", $3/$2 * 100.0}'", 10)
            if mem_result.return_code == 0:
                try:
                    metrics["memory_usage"] = float(mem_result.output.strip())
                except ValueError:
                    pass
            
            # Disk usage
            disk_result = await self._run_command("df / | tail -1 | awk '{print $5}' | cut -d'%' -f1", 10)
            if disk_result.return_code == 0:
                try:
                    metrics["disk_usage"] = float(disk_result.output.strip())
                except ValueError:
                    pass
        
        except Exception as e:
            self.logger.error(f"Failed to collect metrics: {e}")
        
        return metrics
    
    async def _calculate_success_rate(self, execution: RecoveryExecution) -> float:
        """Calculate success rate based on metrics and result"""
        if execution.result == RecoveryResult.SUCCESS:
            base_score = 1.0
        elif execution.result == RecoveryResult.PARTIAL_SUCCESS:
            base_score = 0.7
        else:
            base_score = 0.0
        
        # Adjust based on metric improvements
        if execution.metrics_before and execution.metrics_after:
            improvements = 0
            total_metrics = 0
            
            for metric in execution.metrics_before:
                if metric in execution.metrics_after:
                    before = execution.metrics_before[metric]
                    after = execution.metrics_after[metric]
                    
                    # For usage metrics, lower is better
                    if "usage" in metric:
                        if after < before:
                            improvements += 1
                    total_metrics += 1
            
            if total_metrics > 0:
                improvement_rate = improvements / total_metrics
                base_score = (base_score + improvement_rate) / 2
        
        return base_score
    
    async def _check_disk_space(self) -> bool:
        """Check if sufficient disk space is available"""
        try:
            result = await self._run_command("df / | tail -1 | awk '{print $5}' | cut -d'%' -f1", 10)
            if result.return_code == 0:
                usage = int(result.output.strip())
                return usage < 95  # Less than 95% usage
        except Exception:
            pass
        return False
    
    async def _is_service_running(self, service_name: str) -> bool:
        """Check if a service is running"""
        try:
            # Validate service name
            if not CommandValidator.validate_service_name(service_name):
                return False

            # Build safe systemctl command
            systemctl_cmd = CommandValidator.build_safe_command('systemctl', ['is-active', service_name])
            result = await self._run_safe_command(systemctl_cmd, 10)
            return result.return_code == 0 and "active" in result.output
        except Exception:
            return False
    
    async def _check_database_connection(self) -> bool:
        """Check database connectivity"""
        # This is a placeholder - implement based on your database setup
        return True


@dataclass
class ShellResult:
    """Result of shell command execution"""
    return_code: int
    output: str
    error: str


@dataclass
class CommandResult:
    """Result of recovery command execution"""
    result: RecoveryResult
    output: str
    error: str


class AutoRecoverySystem:
    """
    Main automated recovery system that coordinates workflow execution
    and manages recovery strategies.
    """
    
    def __init__(self, config: Optional[AutoRecoveryConfig] = None):
        self.config = config or AutoRecoveryConfig()
        self.logger = logging.getLogger(__name__)
        
        # Components
        self.executor = RecoveryExecutor(self.config)
        
        # Workflows and tracking
        self.workflows: Dict[str, RecoveryWorkflow] = {}
        self.active_recoveries: Dict[str, Dict[str, Any]] = {}
        self.recovery_history: List[Dict[str, Any]] = []
        self.recovery_stats: Dict[str, Any] = defaultdict(int)
        
        # Rate limiting
        self.recent_recoveries: deque = deque(maxlen=100)
        
    def register_workflow(self, workflow: RecoveryWorkflow):
        """Register a recovery workflow"""
        self.workflows[workflow.name] = workflow
        self.logger.info(f"Registered recovery workflow: {workflow.name}")
    
    async def trigger_recovery(
        self,
        trigger_event: str,
        context: Optional[Dict[str, Any]] = None,
        manual_workflow: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Trigger automated recovery based on an event.
        
        Args:
            trigger_event: Event that triggered recovery (e.g., "high_cpu_usage")
            context: Additional context about the event
            manual_workflow: Specific workflow to run (overrides automatic selection)
            
        Returns:
            Recovery execution summary
        """
        if not self.config.enable_auto_recovery:
            return {"status": "disabled", "message": "Auto-recovery is disabled"}
        
        # Rate limiting check
        if not await self._check_rate_limits():
            return {"status": "rate_limited", "message": "Too many recent recovery attempts"}
        
        # Find appropriate workflow
        if manual_workflow:
            workflow = self.workflows.get(manual_workflow)
            if not workflow:
                return {"status": "error", "message": f"Workflow {manual_workflow} not found"}
        else:
            workflow = await self._select_workflow(trigger_event, context)
            if not workflow:
                return {"status": "no_workflow", "message": "No suitable recovery workflow found"}
        
        recovery_id = f"{workflow.name}_{datetime.utcnow().timestamp()}"
        
        # Check if already running similar recovery
        if await self._is_similar_recovery_running(workflow.name):
            return {"status": "already_running", "message": f"Similar recovery for {workflow.name} already in progress"}
        
        # Start recovery execution
        recovery_summary = {
            "recovery_id": recovery_id,
            "workflow_name": workflow.name,
            "trigger_event": trigger_event,
            "start_time": datetime.utcnow(),
            "status": "running",
            "actions_completed": 0,
            "actions_total": len(workflow.actions),
            "executions": []
        }
        
        self.active_recoveries[recovery_id] = recovery_summary
        
        try:
            # Execute workflow
            recovery_summary = await self._execute_workflow(workflow, trigger_event, context, recovery_summary)
            
            # Update statistics
            self.recovery_stats[f"workflow_{workflow.name}"] += 1
            self.recovery_stats[f"trigger_{trigger_event}"] += 1
            
            # Track for rate limiting
            self.recent_recoveries.append(datetime.utcnow())
            
        except Exception as e:
            recovery_summary["status"] = "error"
            recovery_summary["error"] = str(e)
            self.logger.error(f"Recovery workflow {workflow.name} failed: {e}")
        
        finally:
            recovery_summary["end_time"] = datetime.utcnow()
            recovery_summary["duration"] = (
                recovery_summary["end_time"] - recovery_summary["start_time"]
            ).total_seconds()
            
            # Move to history
            if recovery_id in self.active_recoveries:
                del self.active_recoveries[recovery_id]
            self.recovery_history.append(recovery_summary)
            
            # Keep only recent history
            cutoff_time = datetime.utcnow() - timedelta(hours=24)
            self.recovery_history = [
                r for r in self.recovery_history if r["start_time"] > cutoff_time
            ]
        
        return recovery_summary
    
    async def _select_workflow(
        self,
        trigger_event: str,
        context: Optional[Dict[str, Any]]
    ) -> Optional[RecoveryWorkflow]:
        """Select the most appropriate workflow for the trigger event"""
        matching_workflows = []
        
        for workflow in self.workflows.values():
            if not workflow.enabled:
                continue
            
            # Check trigger conditions
            for condition in workflow.trigger_conditions:
                if await self._evaluate_condition(condition, trigger_event, context):
                    matching_workflows.append(workflow)
                    break
        
        if not matching_workflows:
            return None
        
        # Select workflow with highest specificity (most conditions)
        return max(matching_workflows, key=lambda w: len(w.trigger_conditions))
    
    async def _evaluate_condition(
        self,
        condition: Dict[str, Any],
        trigger_event: str,
        context: Optional[Dict[str, Any]]
    ) -> bool:
        """Evaluate if a trigger condition matches the event"""
        condition_type = condition.get("type", "event")
        
        if condition_type == "event":
            return condition.get("event_pattern", "") in trigger_event
        
        elif condition_type == "metric":
            if not context:
                return False
            
            metric_name = condition.get("metric_name")
            operator = condition.get("operator", ">")
            threshold = condition.get("threshold", 0)
            
            if metric_name in context:
                value = context[metric_name]
                
                if operator == ">":
                    return value > threshold
                elif operator == "<":
                    return value < threshold
                elif operator == ">=":
                    return value >= threshold
                elif operator == "<=":
                    return value <= threshold
                elif operator == "==":
                    return value == threshold
        
        return False
    
    async def _execute_workflow(
        self,
        workflow: RecoveryWorkflow,
        trigger_event: str,
        context: Optional[Dict[str, Any]],
        recovery_summary: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute a recovery workflow"""
        self.logger.info(f"Executing recovery workflow: {workflow.name}")
        
        successful_actions = 0
        
        for i, action_def in enumerate(workflow.actions):
            try:
                # Execute action
                execution = await self.executor.execute_action(action_def, trigger_event, context)
                
                recovery_summary["executions"].append({
                    "action_name": action_def.name,
                    "result": execution.result.value,
                    "success_rate": execution.success_rate,
                    "duration": (
                        (execution.end_time - execution.start_time).total_seconds()
                        if execution.end_time else 0
                    ),
                    "output": execution.output[:200],  # Truncate for summary
                    "error": execution.error[:200] if execution.error else ""
                })
                
                if execution.result in [RecoveryResult.SUCCESS, RecoveryResult.PARTIAL_SUCCESS]:
                    successful_actions += 1
                
                recovery_summary["actions_completed"] = i + 1
                
                # Check if workflow should continue
                if execution.result == RecoveryResult.FAILED and action_def.risk_level == "high":
                    self.logger.warning(f"High-risk action {action_def.name} failed, stopping workflow")
                    break
                
            except Exception as e:
                self.logger.error(f"Error executing action {action_def.name}: {e}")
                recovery_summary["executions"].append({
                    "action_name": action_def.name,
                    "result": "error",
                    "error": str(e)
                })
        
        # Calculate overall success
        total_actions = len(workflow.actions)
        success_rate = successful_actions / total_actions if total_actions > 0 else 0
        
        if success_rate >= workflow.success_threshold:
            recovery_summary["status"] = "success"
        elif success_rate > 0:
            recovery_summary["status"] = "partial_success"
        else:
            recovery_summary["status"] = "failed"
        
        recovery_summary["success_rate"] = success_rate
        recovery_summary["successful_actions"] = successful_actions
        
        return recovery_summary
    
    async def _check_rate_limits(self) -> bool:
        """Check if recovery rate limits allow execution"""
        now = datetime.utcnow()
        hour_ago = now - timedelta(hours=1)
        
        # Count recoveries in last hour
        recent_count = sum(
            1 for recovery_time in self.recent_recoveries
            if recovery_time > hour_ago
        )
        
        return recent_count < self.config.max_recoveries_per_hour
    
    async def _is_similar_recovery_running(self, workflow_name: str) -> bool:
        """Check if a similar recovery is already running"""
        return any(
            recovery["workflow_name"] == workflow_name
            for recovery in self.active_recoveries.values()
        )
    
    def get_recovery_status(self, recovery_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a specific recovery"""
        # Check active recoveries first
        if recovery_id in self.active_recoveries:
            return self.active_recoveries[recovery_id]
        
        # Check history
        for recovery in self.recovery_history:
            if recovery["recovery_id"] == recovery_id:
                return recovery
        
        return None
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get overall recovery system status"""
        now = datetime.utcnow()
        hour_ago = now - timedelta(hours=1)
        day_ago = now - timedelta(hours=24)
        
        recent_recoveries_hour = sum(
            1 for recovery_time in self.recent_recoveries
            if recovery_time > hour_ago
        )
        
        recent_recoveries_day = len(self.recovery_history)
        
        return {
            "enabled": self.config.enable_auto_recovery,
            "active_recoveries": len(self.active_recoveries),
            "registered_workflows": len(self.workflows),
            "recent_recoveries_hour": recent_recoveries_hour,
            "recent_recoveries_day": recent_recoveries_day,
            "rate_limit_remaining": max(0, self.config.max_recoveries_per_hour - recent_recoveries_hour),
            "statistics": dict(self.recovery_stats)
        }