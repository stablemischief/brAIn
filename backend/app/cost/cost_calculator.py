"""
Cost calculation engine with real-time tracking, budget management, and optimization
Integrates with token counting utilities for accurate LLM cost calculation
"""

import logging
import json
from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import asyncio
from decimal import Decimal, ROUND_UP

from .token_counter import TokenCounter, TokenCount, ModelProvider, get_token_counter

logger = logging.getLogger(__name__)


class BudgetPeriod(Enum):
    """Budget period types"""
    HOURLY = "hourly"
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    YEARLY = "yearly"


class CostCategory(Enum):
    """Cost categorization for analysis"""
    PROCESSING = "processing"
    EXTRACTION = "extraction"
    SUMMARIZATION = "summarization"
    CLASSIFICATION = "classification"
    EMBEDDING = "embedding"
    CHAT = "chat"
    ANALYSIS = "analysis"


@dataclass
class CostBreakdown:
    """Detailed cost breakdown for an operation"""
    operation_id: str
    model: str
    provider: ModelProvider
    category: CostCategory
    input_tokens: int
    output_tokens: int
    total_tokens: int
    input_cost: Decimal
    output_cost: Decimal
    total_cost: Decimal
    timestamp: datetime
    metadata: Optional[Dict] = None


@dataclass
class BudgetAlert:
    """Budget alert configuration and tracking"""
    id: str
    name: str
    threshold_percentage: float  # 0.0 to 1.0
    period: BudgetPeriod
    budget_limit: Decimal
    current_spend: Decimal
    triggered: bool = False
    last_triggered: Optional[datetime] = None
    alert_count: int = 0


@dataclass
class BudgetConfig:
    """Budget configuration for cost control"""
    daily_limit: Optional[Decimal] = None
    weekly_limit: Optional[Decimal] = None
    monthly_limit: Optional[Decimal] = None
    per_operation_limit: Optional[Decimal] = None
    alerts: List[BudgetAlert] = field(default_factory=list)
    hard_stop_enabled: bool = False
    grace_percentage: float = 0.1  # 10% grace before hard stop


@dataclass
class CostForecast:
    """Cost forecasting data"""
    period: BudgetPeriod
    projected_cost: Decimal
    confidence: float  # 0.0 to 1.0
    trend: str  # "increasing", "decreasing", "stable"
    factors: List[str]  # Contributing factors
    generated_at: datetime


class CostCalculator:
    """
    Advanced cost calculation engine with budget management and optimization
    """
    
    def __init__(self, budget_config: Optional[BudgetConfig] = None):
        self.token_counter = get_token_counter()
        self.budget_config = budget_config or BudgetConfig()
        self.cost_history: List[CostBreakdown] = []
        self.session_costs: Dict[str, Decimal] = {}  # session_id -> total_cost
        self.daily_costs: Dict[str, Decimal] = {}  # YYYY-MM-DD -> total_cost
        self._cost_cache = {}  # Cache for expensive calculations
        
    def calculate_operation_cost(
        self,
        token_count: TokenCount,
        category: CostCategory = CostCategory.PROCESSING,
        operation_id: Optional[str] = None
    ) -> CostBreakdown:
        """Calculate cost for a specific operation"""
        
        # Validate no negative tokens
        if token_count.input_tokens < 0 or token_count.output_tokens < 0:
            raise ValueError(f"Token counts cannot be negative: input={token_count.input_tokens}, output={token_count.output_tokens}")
        
        model_config = self.token_counter.get_model_info(token_count.model)
        if not model_config:
            logger.warning(f"No pricing info for model {token_count.model}")
            return CostBreakdown(
                operation_id=operation_id or f"op_{int(datetime.now().timestamp())}",
                model=token_count.model,
                provider=token_count.provider,
                category=category,
                input_tokens=token_count.input_tokens,
                output_tokens=token_count.output_tokens,
                total_tokens=token_count.total_tokens,
                input_cost=Decimal('0'),
                output_cost=Decimal('0'),
                total_cost=Decimal('0'),
                timestamp=datetime.now()
            )
        
        # Calculate costs with high precision
        input_cost = Decimal(str(token_count.input_tokens / 1000)) * Decimal(str(model_config.input_cost_per_1k))
        output_cost = Decimal(str(token_count.output_tokens / 1000)) * Decimal(str(model_config.output_cost_per_1k))
        total_cost = input_cost + output_cost
        
        # Apply any category-specific multipliers
        multiplier = self._get_category_multiplier(category, token_count.provider)
        total_cost *= Decimal(str(multiplier))
        input_cost *= Decimal(str(multiplier))
        output_cost *= Decimal(str(multiplier))
        
        breakdown = CostBreakdown(
            operation_id=operation_id or f"op_{int(datetime.now().timestamp())}",
            model=token_count.model,
            provider=token_count.provider,
            category=category,
            input_tokens=token_count.input_tokens,
            output_tokens=token_count.output_tokens,
            total_tokens=token_count.total_tokens,
            input_cost=input_cost.quantize(Decimal('0.000001'), rounding=ROUND_UP),
            output_cost=output_cost.quantize(Decimal('0.000001'), rounding=ROUND_UP),
            total_cost=total_cost.quantize(Decimal('0.000001'), rounding=ROUND_UP),
            timestamp=datetime.now(),
            metadata=token_count.metadata
        )
        
        # Track in history
        self.cost_history.append(breakdown)
        
        # Update running totals
        today = datetime.now().strftime('%Y-%m-%d')
        self.daily_costs[today] = self.daily_costs.get(today, Decimal('0')) + total_cost
        
        return breakdown
    
    def calculate_cost(
        self,
        token_count: Union[int, TokenCount],
        category: CostCategory = CostCategory.PROCESSING,
        model: str = "gpt-4o-mini"
    ) -> CostBreakdown:
        """Calculate cost for tokens - simplified version for tests"""
        
        # If it's just a number, create a TokenCount
        if isinstance(token_count, int):
            tc = TokenCount(
                input_tokens=token_count // 2,
                output_tokens=token_count // 2,
                total_tokens=token_count,
                model=model,
                provider=ModelProvider.OPENAI
            )
        else:
            tc = token_count
            
        return self.calculate_operation_cost(tc, category)
    
    def check_budget(self, amount: Decimal) -> bool:
        """Check if we have budget available for the given amount"""
        if not self.budget_config:
            return True  # No budget limits
            
        # Check daily budget
        if self.budget_config.daily_limit:
            today = datetime.now().strftime('%Y-%m-%d')
            current_daily = self.daily_costs.get(today, Decimal('0'))
            if current_daily + amount > self.budget_config.daily_limit:
                return False
                
        # Check per-operation limit
        if self.budget_config.per_operation_limit:
            if amount > self.budget_config.per_operation_limit:
                return False
                
        return True
    
    def calculate_batch_cost(
        self,
        token_counts: List[TokenCount],
        category: CostCategory = CostCategory.PROCESSING,
        batch_id: Optional[str] = None
    ) -> List[CostBreakdown]:
        """Calculate costs for a batch of operations efficiently"""
        
        batch_id = batch_id or f"batch_{int(datetime.now().timestamp())}"
        breakdowns = []
        
        for i, token_count in enumerate(token_counts):
            operation_id = f"{batch_id}_op_{i}"
            breakdown = self.calculate_operation_cost(token_count, category, operation_id)
            breakdowns.append(breakdown)
        
        return breakdowns
    
    def check_budget_limits(
        self,
        proposed_cost: Decimal,
        session_id: Optional[str] = None
    ) -> Tuple[bool, List[str]]:
        """Check if proposed cost would exceed budget limits"""
        
        allowed = True
        warnings = []
        
        # Check per-operation limit
        if self.budget_config.per_operation_limit:
            if proposed_cost > self.budget_config.per_operation_limit:
                allowed = False
                warnings.append(f"Operation cost ${proposed_cost} exceeds per-operation limit ${self.budget_config.per_operation_limit}")
        
        # Check daily limit
        if self.budget_config.daily_limit:
            today = datetime.now().strftime('%Y-%m-%d')
            current_daily = self.daily_costs.get(today, Decimal('0'))
            projected_daily = current_daily + proposed_cost
            
            if projected_daily > self.budget_config.daily_limit:
                if self.budget_config.hard_stop_enabled:
                    grace_limit = self.budget_config.daily_limit * (1 + Decimal(str(self.budget_config.grace_percentage)))
                    if projected_daily > grace_limit:
                        allowed = False
                        warnings.append(f"Daily cost limit exceeded: ${projected_daily} > ${grace_limit} (with grace)")
                    else:
                        warnings.append(f"Daily cost limit exceeded but within grace: ${projected_daily}")
                else:
                    warnings.append(f"Daily cost limit exceeded: ${projected_daily} > ${self.budget_config.daily_limit}")
        
        # Check session limit if tracking sessions
        if session_id and self.budget_config.per_operation_limit:
            current_session = self.session_costs.get(session_id, Decimal('0'))
            # You might want to implement session-specific limits here
        
        # Check alert thresholds
        self._check_budget_alerts(proposed_cost)
        
        return allowed, warnings
    
    def estimate_document_processing_cost(
        self,
        document_content: str,
        processing_types: List[str],
        models: List[str]
    ) -> Dict[str, Dict[str, CostBreakdown]]:
        """Estimate costs for processing a document with multiple models and operations"""
        
        estimates = {}
        
        for model in models:
            estimates[model] = {}
            
            for processing_type in processing_types:
                # Get token estimate
                token_count = self.token_counter.estimate_processing_tokens(
                    document_content, processing_type, model
                )
                
                # Convert to cost category
                category = self._processing_type_to_category(processing_type)
                
                # Calculate cost
                breakdown = self.calculate_operation_cost(
                    token_count, 
                    category,
                    f"estimate_{model}_{processing_type}"
                )
                
                estimates[model][processing_type] = breakdown
        
        return estimates
    
    def optimize_model_selection(
        self,
        document_content: str,
        processing_type: str,
        quality_threshold: float = 0.8,
        available_models: Optional[List[str]] = None
    ) -> Tuple[str, CostBreakdown, str]:
        """Optimize model selection based on cost and quality requirements"""
        
        if not available_models:
            available_models = ["gpt-4o-mini", "gpt-4o", "claude-3-haiku", "claude-3-sonnet"]
        
        model_evaluations = []
        
        for model in available_models:
            # Estimate cost
            token_count = self.token_counter.estimate_processing_tokens(
                document_content, processing_type, model
            )
            category = self._processing_type_to_category(processing_type)
            breakdown = self.calculate_operation_cost(token_count, category)
            
            # Estimate quality (simplified heuristic)
            quality_score = self._estimate_quality_score(model, processing_type)
            
            # Calculate cost-effectiveness
            if quality_score >= quality_threshold:
                cost_effectiveness = quality_score / float(breakdown.total_cost)
            else:
                cost_effectiveness = 0  # Below threshold
            
            model_evaluations.append({
                'model': model,
                'cost': breakdown.total_cost,
                'quality': quality_score,
                'cost_effectiveness': cost_effectiveness,
                'breakdown': breakdown
            })
        
        # Sort by cost-effectiveness (higher is better)
        model_evaluations.sort(key=lambda x: x['cost_effectiveness'], reverse=True)
        
        if not model_evaluations or model_evaluations[0]['cost_effectiveness'] == 0:
            # Fallback to cheapest model that meets threshold
            suitable_models = [m for m in model_evaluations if m['quality'] >= quality_threshold]
            if suitable_models:
                best = min(suitable_models, key=lambda x: x['cost'])
                return best['model'], best['breakdown'], "cheapest_suitable"
            else:
                # No models meet threshold, return cheapest overall
                best = min(model_evaluations, key=lambda x: x['cost'])
                return best['model'], best['breakdown'], "cheapest_fallback"
        
        best = model_evaluations[0]
        return best['model'], best['breakdown'], "cost_effective"
    
    def generate_cost_forecast(
        self,
        period: BudgetPeriod,
        confidence_threshold: float = 0.7
    ) -> Optional[CostForecast]:
        """Generate cost forecast based on historical data"""
        
        if len(self.cost_history) < 10:  # Need minimum history
            return None
        
        # Get relevant historical data
        now = datetime.now()
        if period == BudgetPeriod.DAILY:
            lookback_days = 7
            projection_hours = 24
        elif period == BudgetPeriod.WEEKLY:
            lookback_days = 28
            projection_hours = 168  # 7 days
        elif period == BudgetPeriod.MONTHLY:
            lookback_days = 90
            projection_hours = 720  # 30 days
        else:
            return None
        
        # Filter historical data
        cutoff = now - timedelta(days=lookback_days)
        recent_costs = [c for c in self.cost_history if c.timestamp >= cutoff]
        
        if len(recent_costs) < 5:
            return None
        
        # Simple linear regression for trend
        total_cost = sum(float(c.total_cost) for c in recent_costs)
        avg_hourly_cost = total_cost / (lookback_days * 24)
        projected_cost = Decimal(str(avg_hourly_cost * projection_hours))
        
        # Determine trend
        first_half = recent_costs[:len(recent_costs)//2]
        second_half = recent_costs[len(recent_costs)//2:]
        
        first_avg = sum(float(c.total_cost) for c in first_half) / len(first_half)
        second_avg = sum(float(c.total_cost) for c in second_half) / len(second_half)
        
        if second_avg > first_avg * 1.1:
            trend = "increasing"
        elif second_avg < first_avg * 0.9:
            trend = "decreasing"
        else:
            trend = "stable"
        
        # Calculate confidence (simplified)
        variance = sum((float(c.total_cost) - (total_cost / len(recent_costs))) ** 2 for c in recent_costs) / len(recent_costs)
        confidence = max(0.0, min(1.0, 1.0 - (variance / (total_cost / len(recent_costs)))))
        
        if confidence < confidence_threshold:
            return None
        
        # Identify contributing factors
        factors = []
        model_distribution = {}
        for cost in recent_costs:
            model_distribution[cost.model] = model_distribution.get(cost.model, 0) + 1
        
        top_model = max(model_distribution, key=model_distribution.get)
        factors.append(f"Primary model: {top_model}")
        
        if trend == "increasing":
            factors.append("Usage trending upward")
        elif trend == "decreasing":
            factors.append("Usage trending downward")
        
        return CostForecast(
            period=period,
            projected_cost=projected_cost.quantize(Decimal('0.01'), rounding=ROUND_UP),
            confidence=confidence,
            trend=trend,
            factors=factors,
            generated_at=now
        )
    
    def get_cost_analytics(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Dict:
        """Get comprehensive cost analytics"""
        
        if not start_date:
            start_date = datetime.now() - timedelta(days=30)
        if not end_date:
            end_date = datetime.now()
        
        # Filter relevant costs
        relevant_costs = [
            c for c in self.cost_history
            if start_date <= c.timestamp <= end_date
        ]
        
        if not relevant_costs:
            return {"error": "No cost data in specified period"}
        
        # Calculate totals
        total_cost = sum(c.total_cost for c in relevant_costs)
        total_tokens = sum(c.total_tokens for c in relevant_costs)
        
        # Group by model
        model_breakdown = {}
        for cost in relevant_costs:
            if cost.model not in model_breakdown:
                model_breakdown[cost.model] = {
                    'cost': Decimal('0'),
                    'tokens': 0,
                    'operations': 0
                }
            model_breakdown[cost.model]['cost'] += cost.total_cost
            model_breakdown[cost.model]['tokens'] += cost.total_tokens
            model_breakdown[cost.model]['operations'] += 1
        
        # Group by category
        category_breakdown = {}
        for cost in relevant_costs:
            cat = cost.category.value
            if cat not in category_breakdown:
                category_breakdown[cat] = {
                    'cost': Decimal('0'),
                    'operations': 0
                }
            category_breakdown[cat]['cost'] += cost.total_cost
            category_breakdown[cat]['operations'] += 1
        
        # Daily costs
        daily_breakdown = {}
        for cost in relevant_costs:
            day = cost.timestamp.strftime('%Y-%m-%d')
            if day not in daily_breakdown:
                daily_breakdown[day] = Decimal('0')
            daily_breakdown[day] += cost.total_cost
        
        return {
            'period': {
                'start': start_date.isoformat(),
                'end': end_date.isoformat()
            },
            'totals': {
                'cost': float(total_cost),
                'tokens': total_tokens,
                'operations': len(relevant_costs),
                'avg_cost_per_operation': float(total_cost / len(relevant_costs)),
                'avg_cost_per_1k_tokens': float(total_cost / (total_tokens / 1000)) if total_tokens > 0 else 0
            },
            'models': {
                model: {
                    'cost': float(data['cost']),
                    'tokens': data['tokens'],
                    'operations': data['operations'],
                    'percentage': float(data['cost'] / total_cost * 100)
                }
                for model, data in model_breakdown.items()
            },
            'categories': {
                cat: {
                    'cost': float(data['cost']),
                    'operations': data['operations'],
                    'percentage': float(data['cost'] / total_cost * 100)
                }
                for cat, data in category_breakdown.items()
            },
            'daily_costs': {
                day: float(cost) for day, cost in daily_breakdown.items()
            }
        }
    
    def _get_category_multiplier(self, category: CostCategory, provider: ModelProvider) -> float:
        """Get cost multiplier for specific categories and providers"""
        # Some providers charge differently for different use cases
        multipliers = {
            (CostCategory.EMBEDDING, ModelProvider.OPENAI): 1.0,
            (CostCategory.CHAT, ModelProvider.ANTHROPIC): 1.0,
            # Add more specific multipliers as needed
        }
        
        return multipliers.get((category, provider), 1.0)
    
    def _processing_type_to_category(self, processing_type: str) -> CostCategory:
        """Convert processing type string to cost category"""
        mapping = {
            'extraction': CostCategory.EXTRACTION,
            'summarization': CostCategory.SUMMARIZATION,
            'classification': CostCategory.CLASSIFICATION,
            'embedding': CostCategory.EMBEDDING,
            'chat': CostCategory.CHAT,
            'analysis': CostCategory.ANALYSIS,
        }
        
        return mapping.get(processing_type, CostCategory.PROCESSING)
    
    def _estimate_quality_score(self, model: str, processing_type: str) -> float:
        """Estimate quality score for model-task combination (simplified heuristic)"""
        # This is a simplified scoring system - in reality you'd want
        # benchmarks, user feedback, or A/B test results
        
        quality_matrix = {
            # Model -> {processing_type -> score}
            'gpt-4': {
                'extraction': 0.95,
                'summarization': 0.95,
                'classification': 0.90,
                'analysis': 0.95,
                'chat': 0.95
            },
            'gpt-4o': {
                'extraction': 0.93,
                'summarization': 0.93,
                'classification': 0.88,
                'analysis': 0.93,
                'chat': 0.93
            },
            'gpt-4o-mini': {
                'extraction': 0.85,
                'summarization': 0.85,
                'classification': 0.85,
                'analysis': 0.80,
                'chat': 0.85
            },
            'claude-3-opus': {
                'extraction': 0.90,
                'summarization': 0.95,
                'classification': 0.85,
                'analysis': 0.90,
                'chat': 0.95
            },
            'claude-3-sonnet': {
                'extraction': 0.85,
                'summarization': 0.90,
                'classification': 0.80,
                'analysis': 0.85,
                'chat': 0.90
            },
            'claude-3-haiku': {
                'extraction': 0.75,
                'summarization': 0.80,
                'classification': 0.75,
                'analysis': 0.75,
                'chat': 0.80
            }
        }
        
        model_scores = quality_matrix.get(model, {})
        return model_scores.get(processing_type, 0.7)  # Default score
    
    def _check_budget_alerts(self, proposed_cost: Decimal) -> None:
        """Check and trigger budget alerts"""
        today = datetime.now().strftime('%Y-%m-%d')
        current_daily = self.daily_costs.get(today, Decimal('0'))
        
        for alert in self.budget_config.alerts:
            if alert.period == BudgetPeriod.DAILY and self.budget_config.daily_limit:
                threshold = self.budget_config.daily_limit * Decimal(str(alert.threshold_percentage))
                if current_daily + proposed_cost >= threshold and not alert.triggered:
                    alert.triggered = True
                    alert.last_triggered = datetime.now()
                    alert.alert_count += 1
                    logger.warning(f"Budget alert triggered: {alert.name} - Daily spend approaching ${threshold}")


# Global cost calculator instance
_cost_calculator: Optional[CostCalculator] = None


def get_cost_calculator(budget_config: Optional[BudgetConfig] = None) -> CostCalculator:
    """Get the global cost calculator instance"""
    global _cost_calculator
    if _cost_calculator is None:
        _cost_calculator = CostCalculator(budget_config)
    return _cost_calculator


# Convenience functions
def calculate_cost(
    text: str, 
    model: str = "gpt-4o-mini",
    category: CostCategory = CostCategory.PROCESSING
) -> CostBreakdown:
    """Calculate cost for processing text"""
    token_count = get_token_counter().count_tokens_text(text, model)
    return get_cost_calculator().calculate_operation_cost(token_count, category)


def estimate_batch_cost(
    texts: List[str],
    model: str = "gpt-4o-mini",
    category: CostCategory = CostCategory.PROCESSING
) -> Tuple[List[CostBreakdown], Decimal]:
    """Estimate cost for batch processing"""
    token_counts = get_token_counter().batch_count_tokens(texts, model)
    breakdowns = get_cost_calculator().calculate_batch_cost(token_counts, category)
    total_cost = sum(b.total_cost for b in breakdowns)
    return breakdowns, total_cost