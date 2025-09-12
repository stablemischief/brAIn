"""
Budget management system with enforcement, alerts, and spending controls
Works with cost calculator to provide comprehensive budget oversight
"""

import logging
import json
from typing import Dict, List, Optional, Tuple, Callable, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from enum import Enum
import asyncio
from decimal import Decimal
import sqlite3
from pathlib import Path

from .cost_calculator import (
    CostCalculator, BudgetConfig, BudgetAlert, BudgetPeriod, 
    CostBreakdown, CostCategory
)

logger = logging.getLogger(__name__)


class AlertSeverity(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class EnforcementAction(Enum):
    """Budget enforcement actions"""
    LOG_ONLY = "log_only"
    ALERT = "alert"
    THROTTLE = "throttle"
    BLOCK = "block"
    APPROVE_REQUIRED = "approve_required"


@dataclass
class BudgetRule:
    """Budget rule configuration"""
    id: str
    name: str
    description: str
    period: BudgetPeriod
    limit: Decimal
    categories: Optional[List[CostCategory]] = None  # None means all categories
    models: Optional[List[str]] = None  # None means all models
    priority: int = 0  # Higher priority rules evaluated first
    enforcement_action: EnforcementAction = EnforcementAction.ALERT
    alert_thresholds: List[float] = field(default_factory=lambda: [0.5, 0.75, 0.9])  # % of limit
    enabled: bool = True
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)


@dataclass
class BudgetViolation:
    """Budget violation record"""
    id: str
    rule_id: str
    rule_name: str
    current_spend: Decimal
    limit: Decimal
    percentage: float
    period: BudgetPeriod
    severity: AlertSeverity
    action_taken: EnforcementAction
    cost_breakdown: CostBreakdown
    timestamp: datetime
    resolved: bool = False
    resolution_notes: Optional[str] = None


@dataclass
class ApprovalRequest:
    """Budget approval request for operations exceeding limits"""
    id: str
    operation_id: str
    proposed_cost: Decimal
    current_spend: Decimal
    limit: Decimal
    rule_id: str
    requester: Optional[str] = None
    justification: Optional[str] = None
    status: str = "pending"  # pending, approved, denied
    requested_at: datetime = field(default_factory=datetime.now)
    reviewed_at: Optional[datetime] = None
    reviewer: Optional[str] = None
    review_notes: Optional[str] = None


class BudgetEnforcer:
    """
    Budget enforcement engine with configurable rules and actions
    """
    
    def __init__(
        self,
        cost_calculator: CostCalculator,
        db_path: Optional[str] = None,
        alert_callbacks: Optional[List[Callable]] = None
    ):
        self.cost_calculator = cost_calculator
        self.db_path = db_path or "budget_manager.db"
        self.alert_callbacks = alert_callbacks or []
        
        # In-memory state
        self.rules: Dict[str, BudgetRule] = {}
        self.violations: Dict[str, BudgetViolation] = {}
        self.approval_requests: Dict[str, ApprovalRequest] = {}
        self.spending_cache: Dict[str, Dict[str, Decimal]] = {}  # period -> category -> amount
        
        # Initialize database
        self._init_database()
        self._load_rules()
    
    def _init_database(self):
        """Initialize SQLite database for persistent storage"""
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        
        with sqlite3.connect(self.db_path) as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS budget_rules (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    description TEXT,
                    period TEXT NOT NULL,
                    limit_amount DECIMAL NOT NULL,
                    categories JSON,
                    models JSON,
                    priority INTEGER DEFAULT 0,
                    enforcement_action TEXT DEFAULT 'alert',
                    alert_thresholds JSON,
                    enabled BOOLEAN DEFAULT 1,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                
                CREATE TABLE IF NOT EXISTS budget_violations (
                    id TEXT PRIMARY KEY,
                    rule_id TEXT NOT NULL,
                    rule_name TEXT NOT NULL,
                    current_spend DECIMAL NOT NULL,
                    limit_amount DECIMAL NOT NULL,
                    percentage REAL NOT NULL,
                    period TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    action_taken TEXT NOT NULL,
                    cost_breakdown JSON NOT NULL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    resolved BOOLEAN DEFAULT 0,
                    resolution_notes TEXT,
                    FOREIGN KEY (rule_id) REFERENCES budget_rules (id)
                );
                
                CREATE TABLE IF NOT EXISTS approval_requests (
                    id TEXT PRIMARY KEY,
                    operation_id TEXT NOT NULL,
                    proposed_cost DECIMAL NOT NULL,
                    current_spend DECIMAL NOT NULL,
                    limit_amount DECIMAL NOT NULL,
                    rule_id TEXT NOT NULL,
                    requester TEXT,
                    justification TEXT,
                    status TEXT DEFAULT 'pending',
                    requested_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    reviewed_at TIMESTAMP,
                    reviewer TEXT,
                    review_notes TEXT,
                    FOREIGN KEY (rule_id) REFERENCES budget_rules (id)
                );
                
                CREATE INDEX IF NOT EXISTS idx_violations_timestamp ON budget_violations(timestamp);
                CREATE INDEX IF NOT EXISTS idx_violations_rule ON budget_violations(rule_id);
                CREATE INDEX IF NOT EXISTS idx_approvals_status ON approval_requests(status);
            """)
    
    def add_budget_rule(self, rule: BudgetRule) -> bool:
        """Add a new budget rule"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO budget_rules 
                    (id, name, description, period, limit_amount, categories, models, 
                     priority, enforcement_action, alert_thresholds, enabled, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    rule.id, rule.name, rule.description, rule.period.value,
                    float(rule.limit),
                    json.dumps([c.value for c in rule.categories]) if rule.categories else None,
                    json.dumps(rule.models) if rule.models else None,
                    rule.priority, rule.enforcement_action.value,
                    json.dumps(rule.alert_thresholds), rule.enabled,
                    datetime.now().isoformat()
                ))
            
            self.rules[rule.id] = rule
            logger.info(f"Added budget rule: {rule.name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add budget rule {rule.id}: {e}")
            return False
    
    def remove_budget_rule(self, rule_id: str) -> bool:
        """Remove a budget rule"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("DELETE FROM budget_rules WHERE id = ?", (rule_id,))
            
            if rule_id in self.rules:
                del self.rules[rule_id]
            
            logger.info(f"Removed budget rule: {rule_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to remove budget rule {rule_id}: {e}")
            return False
    
    def check_budget_compliance(
        self,
        cost_breakdown: CostBreakdown,
        session_id: Optional[str] = None
    ) -> Tuple[bool, List[BudgetViolation], List[ApprovalRequest]]:
        """Check if operation complies with budget rules"""
        
        violations = []
        approval_requests = []
        allowed = True
        
        # Get current spending for relevant periods
        current_spending = self._get_current_spending()
        
        # Sort rules by priority (higher first)
        sorted_rules = sorted(
            [r for r in self.rules.values() if r.enabled],
            key=lambda x: x.priority,
            reverse=True
        )
        
        for rule in sorted_rules:
            # Check if rule applies to this operation
            if not self._rule_applies_to_operation(rule, cost_breakdown):
                continue
            
            # Get current spend for this rule's period and constraints
            period_spend = self._get_period_spending(rule, current_spending)
            projected_spend = period_spend + cost_breakdown.total_cost
            
            # Check if this operation would exceed the limit
            if projected_spend > rule.limit:
                severity = self._determine_severity(projected_spend, rule.limit, rule.alert_thresholds)
                
                violation = BudgetViolation(
                    id=f"violation_{int(datetime.now().timestamp())}_{rule.id}",
                    rule_id=rule.id,
                    rule_name=rule.name,
                    current_spend=period_spend,
                    limit=rule.limit,
                    percentage=float(projected_spend / rule.limit),
                    period=rule.period,
                    severity=severity,
                    action_taken=rule.enforcement_action,
                    cost_breakdown=cost_breakdown,
                    timestamp=datetime.now()
                )
                
                violations.append(violation)
                
                # Take enforcement action
                if rule.enforcement_action == EnforcementAction.BLOCK:
                    allowed = False
                    logger.warning(f"Operation blocked by budget rule: {rule.name}")
                
                elif rule.enforcement_action == EnforcementAction.APPROVE_REQUIRED:
                    allowed = False
                    approval_req = ApprovalRequest(
                        id=f"approval_{int(datetime.now().timestamp())}_{rule.id}",
                        operation_id=cost_breakdown.operation_id,
                        proposed_cost=cost_breakdown.total_cost,
                        current_spend=period_spend,
                        limit=rule.limit,
                        rule_id=rule.id
                    )
                    approval_requests.append(approval_req)
                    
                elif rule.enforcement_action == EnforcementAction.THROTTLE:
                    # Implement throttling logic here
                    logger.warning(f"Operation throttled by budget rule: {rule.name}")
                
                # Store violation
                self._store_violation(violation)
                
                # Trigger alerts
                self._trigger_alerts(violation)
        
        return allowed, violations, approval_requests
    
    def create_default_rules(self, daily_limit: Decimal, monthly_limit: Decimal) -> None:
        """Create a set of default budget rules"""
        
        # Daily spending limit
        daily_rule = BudgetRule(
            id="default_daily",
            name="Daily Spending Limit",
            description="Overall daily spending limit for all operations",
            period=BudgetPeriod.DAILY,
            limit=daily_limit,
            priority=100,
            enforcement_action=EnforcementAction.ALERT,
            alert_thresholds=[0.7, 0.85, 0.95]
        )
        
        # Monthly spending limit with hard stop
        monthly_rule = BudgetRule(
            id="default_monthly",
            name="Monthly Budget Hard Limit",
            description="Monthly budget with hard enforcement",
            period=BudgetPeriod.MONTHLY,
            limit=monthly_limit,
            priority=200,
            enforcement_action=EnforcementAction.APPROVE_REQUIRED,
            alert_thresholds=[0.5, 0.75, 0.9]
        )
        
        # High-cost operation protection
        expensive_op_rule = BudgetRule(
            id="expensive_operation",
            name="Expensive Operation Control",
            description="Control for individual expensive operations",
            period=BudgetPeriod.DAILY,
            limit=daily_limit * Decimal('0.1'),  # 10% of daily limit per operation
            priority=150,
            enforcement_action=EnforcementAction.APPROVE_REQUIRED
        )
        
        # Processing-heavy category limit
        processing_rule = BudgetRule(
            id="processing_category",
            name="Processing Operations Daily Limit",
            description="Daily limit for processing operations",
            period=BudgetPeriod.DAILY,
            limit=daily_limit * Decimal('0.7'),  # 70% of daily limit
            categories=[CostCategory.PROCESSING, CostCategory.ANALYSIS],
            priority=75,
            enforcement_action=EnforcementAction.THROTTLE
        )
        
        rules = [daily_rule, monthly_rule, expensive_op_rule, processing_rule]
        
        for rule in rules:
            self.add_budget_rule(rule)
        
        logger.info("Created default budget rules")
    
    def get_spending_summary(
        self,
        period: BudgetPeriod,
        category: Optional[CostCategory] = None
    ) -> Dict[str, Any]:
        """Get spending summary for a period"""
        
        # Get relevant cost history
        now = datetime.now()
        if period == BudgetPeriod.DAILY:
            start_date = now.replace(hour=0, minute=0, second=0, microsecond=0)
        elif period == BudgetPeriod.WEEKLY:
            start_date = now - timedelta(days=now.weekday())
            start_date = start_date.replace(hour=0, minute=0, second=0, microsecond=0)
        elif period == BudgetPeriod.MONTHLY:
            start_date = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        else:
            start_date = now - timedelta(days=30)
        
        relevant_costs = [
            c for c in self.cost_calculator.cost_history
            if c.timestamp >= start_date
        ]
        
        if category:
            relevant_costs = [c for c in relevant_costs if c.category == category]
        
        # Calculate summary
        total_cost = sum(c.total_cost for c in relevant_costs)
        total_operations = len(relevant_costs)
        
        # Group by model
        model_breakdown = {}
        for cost in relevant_costs:
            if cost.model not in model_breakdown:
                model_breakdown[cost.model] = Decimal('0')
            model_breakdown[cost.model] += cost.total_cost
        
        # Get applicable budget rules
        applicable_rules = []
        for rule in self.rules.values():
            if rule.enabled and rule.period == period:
                if not category or not rule.categories or category in rule.categories:
                    applicable_rules.append({
                        'id': rule.id,
                        'name': rule.name,
                        'limit': float(rule.limit),
                        'current_spend': float(total_cost),
                        'remaining': float(rule.limit - total_cost),
                        'percentage': float(total_cost / rule.limit * 100) if rule.limit > 0 else 0
                    })
        
        return {
            'period': period.value,
            'category': category.value if category else 'all',
            'start_date': start_date.isoformat(),
            'end_date': now.isoformat(),
            'total_cost': float(total_cost),
            'total_operations': total_operations,
            'avg_cost_per_operation': float(total_cost / total_operations) if total_operations > 0 else 0,
            'model_breakdown': {
                model: float(cost) for model, cost in model_breakdown.items()
            },
            'applicable_rules': applicable_rules
        }
    
    def get_budget_status(self) -> Dict[str, Any]:
        """Get overall budget status"""
        
        current_spending = self._get_current_spending()
        
        rule_status = []
        for rule in self.rules.values():
            if not rule.enabled:
                continue
            
            period_spend = self._get_period_spending(rule, current_spending)
            percentage = float(period_spend / rule.limit * 100) if rule.limit > 0 else 0
            
            # Determine status
            if percentage >= 95:
                status = "critical"
            elif percentage >= 75:
                status = "warning"
            elif percentage >= 50:
                status = "caution"
            else:
                status = "ok"
            
            rule_status.append({
                'rule_id': rule.id,
                'rule_name': rule.name,
                'period': rule.period.value,
                'limit': float(rule.limit),
                'current_spend': float(period_spend),
                'remaining': float(rule.limit - period_spend),
                'percentage': percentage,
                'status': status,
                'enforcement_action': rule.enforcement_action.value
            })
        
        # Get recent violations
        recent_violations = [
            {
                'id': v.id,
                'rule_name': v.rule_name,
                'severity': v.severity.value,
                'percentage': v.percentage,
                'timestamp': v.timestamp.isoformat(),
                'resolved': v.resolved
            }
            for v in sorted(self.violations.values(), key=lambda x: x.timestamp, reverse=True)[:10]
        ]
        
        # Get pending approvals
        pending_approvals = [
            {
                'id': req.id,
                'operation_id': req.operation_id,
                'proposed_cost': float(req.proposed_cost),
                'rule_id': req.rule_id,
                'requested_at': req.requested_at.isoformat()
            }
            for req in self.approval_requests.values()
            if req.status == 'pending'
        ]
        
        return {
            'timestamp': datetime.now().isoformat(),
            'rules': rule_status,
            'recent_violations': recent_violations,
            'pending_approvals': pending_approvals,
            'total_rules': len(self.rules),
            'active_rules': len([r for r in self.rules.values() if r.enabled])
        }
    
    def _load_rules(self):
        """Load budget rules from database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("SELECT * FROM budget_rules WHERE enabled = 1")
                
                for row in cursor.fetchall():
                    rule = BudgetRule(
                        id=row[0],
                        name=row[1],
                        description=row[2],
                        period=BudgetPeriod(row[3]),
                        limit=Decimal(str(row[4])),
                        categories=[CostCategory(c) for c in json.loads(row[5])] if row[5] else None,
                        models=json.loads(row[6]) if row[6] else None,
                        priority=row[7],
                        enforcement_action=EnforcementAction(row[8]),
                        alert_thresholds=json.loads(row[9]) if row[9] else [0.5, 0.75, 0.9],
                        enabled=bool(row[10]),
                        created_at=datetime.fromisoformat(row[11]),
                        updated_at=datetime.fromisoformat(row[12])
                    )
                    self.rules[rule.id] = rule
            
            logger.info(f"Loaded {len(self.rules)} budget rules")
            
        except Exception as e:
            logger.error(f"Failed to load budget rules: {e}")
    
    def _get_current_spending(self) -> Dict[str, Decimal]:
        """Get current spending across different periods"""
        now = datetime.now()
        spending = {}
        
        # Daily spending
        today = now.replace(hour=0, minute=0, second=0, microsecond=0)
        daily_costs = [
            c for c in self.cost_calculator.cost_history
            if c.timestamp >= today
        ]
        spending['daily'] = sum(c.total_cost for c in daily_costs)
        
        # Weekly spending
        week_start = now - timedelta(days=now.weekday())
        week_start = week_start.replace(hour=0, minute=0, second=0, microsecond=0)
        weekly_costs = [
            c for c in self.cost_calculator.cost_history
            if c.timestamp >= week_start
        ]
        spending['weekly'] = sum(c.total_cost for c in weekly_costs)
        
        # Monthly spending
        month_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        monthly_costs = [
            c for c in self.cost_calculator.cost_history
            if c.timestamp >= month_start
        ]
        spending['monthly'] = sum(c.total_cost for c in monthly_costs)
        
        return spending
    
    def _rule_applies_to_operation(self, rule: BudgetRule, cost_breakdown: CostBreakdown) -> bool:
        """Check if a rule applies to an operation"""
        # Check category filter
        if rule.categories and cost_breakdown.category not in rule.categories:
            return False
        
        # Check model filter
        if rule.models and cost_breakdown.model not in rule.models:
            return False
        
        return True
    
    def _get_period_spending(self, rule: BudgetRule, current_spending: Dict[str, Decimal]) -> Decimal:
        """Get spending for a rule's period"""
        period_map = {
            BudgetPeriod.DAILY: 'daily',
            BudgetPeriod.WEEKLY: 'weekly',
            BudgetPeriod.MONTHLY: 'monthly'
        }
        
        period_key = period_map.get(rule.period, 'daily')
        return current_spending.get(period_key, Decimal('0'))
    
    def _determine_severity(self, current_spend: Decimal, limit: Decimal, thresholds: List[float]) -> AlertSeverity:
        """Determine alert severity based on spending percentage"""
        percentage = float(current_spend / limit)
        
        if percentage >= 1.0:
            return AlertSeverity.EMERGENCY
        elif percentage >= max(thresholds, default=0.9):
            return AlertSeverity.CRITICAL
        elif percentage >= (max(thresholds[:-1], default=0.75) if len(thresholds) > 1 else 0.75):
            return AlertSeverity.WARNING
        else:
            return AlertSeverity.INFO
    
    def _store_violation(self, violation: BudgetViolation):
        """Store violation in database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO budget_violations 
                    (id, rule_id, rule_name, current_spend, limit_amount, percentage, 
                     period, severity, action_taken, cost_breakdown, timestamp)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    violation.id, violation.rule_id, violation.rule_name,
                    float(violation.current_spend), float(violation.limit), violation.percentage,
                    violation.period.value, violation.severity.value, violation.action_taken.value,
                    json.dumps(asdict(violation.cost_breakdown), default=str),
                    violation.timestamp.isoformat()
                ))
            
            self.violations[violation.id] = violation
            
        except Exception as e:
            logger.error(f"Failed to store violation {violation.id}: {e}")
    
    def _trigger_alerts(self, violation: BudgetViolation):
        """Trigger alerts for budget violations"""
        alert_data = {
            'type': 'budget_violation',
            'violation_id': violation.id,
            'rule_name': violation.rule_name,
            'severity': violation.severity.value,
            'current_spend': float(violation.current_spend),
            'limit': float(violation.limit),
            'percentage': violation.percentage,
            'timestamp': violation.timestamp.isoformat()
        }
        
        # Call registered alert callbacks
        for callback in self.alert_callbacks:
            try:
                callback(alert_data)
            except Exception as e:
                logger.error(f"Alert callback failed: {e}")
        
        # Log the violation
        logger.warning(f"Budget violation: {violation.rule_name} - {violation.percentage:.1f}% of limit")


# Global budget enforcer instance
_budget_enforcer: Optional[BudgetEnforcer] = None


def get_budget_enforcer(
    cost_calculator: Optional[CostCalculator] = None,
    **kwargs
) -> BudgetEnforcer:
    """Get the global budget enforcer instance"""
    global _budget_enforcer
    if _budget_enforcer is None:
        from .cost_calculator import get_cost_calculator
        calc = cost_calculator or get_cost_calculator()
        _budget_enforcer = BudgetEnforcer(calc, **kwargs)
    return _budget_enforcer