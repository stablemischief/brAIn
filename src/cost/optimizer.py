"""
Cost optimization system with forecasting algorithms and intelligent optimization strategies
Provides recommendations for model selection, batch processing, and cost reduction
"""

import logging
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import json
from decimal import Decimal
import statistics
from collections import defaultdict, deque

from .cost_calculator import (
    CostCalculator, CostBreakdown, CostCategory, CostForecast, BudgetPeriod
)
from .token_counter import TokenCounter, ModelProvider

logger = logging.getLogger(__name__)


class OptimizationStrategy(Enum):
    """Cost optimization strategies"""
    MINIMIZE_COST = "minimize_cost"
    MAXIMIZE_QUALITY = "maximize_quality"
    BALANCED = "balanced"
    SPEED_OPTIMIZED = "speed_optimized"
    BATCH_OPTIMIZED = "batch_optimized"


class ForecastingMethod(Enum):
    """Forecasting methods"""
    LINEAR_REGRESSION = "linear_regression"
    EXPONENTIAL_SMOOTHING = "exponential_smoothing"
    MOVING_AVERAGE = "moving_average"
    SEASONAL_DECOMPOSITION = "seasonal_decomposition"


@dataclass
class OptimizationRecommendation:
    """Optimization recommendation with rationale"""
    strategy: OptimizationStrategy
    recommended_model: str
    estimated_cost: Decimal
    estimated_quality: float
    savings_percentage: float
    rationale: str
    confidence: float
    alternative_options: List[Dict[str, Any]]
    generated_at: datetime


@dataclass
class BatchOptimizationResult:
    """Results from batch processing optimization"""
    original_cost: Decimal
    optimized_cost: Decimal
    savings: Decimal
    savings_percentage: float
    batch_size: int
    processing_time_estimate: float
    model_assignments: Dict[str, str]  # operation_id -> model
    quality_impact: float
    recommendation: str


@dataclass
class CostTrend:
    """Cost trend analysis"""
    period: str
    trend_direction: str  # "increasing", "decreasing", "stable"
    trend_strength: float  # 0-1, how strong the trend is
    average_daily_cost: Decimal
    peak_cost_day: str
    lowest_cost_day: str
    volatility: float
    seasonal_factors: Dict[str, float]


class CostOptimizer:
    """
    Advanced cost optimization system with forecasting and intelligent recommendations
    """
    
    def __init__(self, cost_calculator: CostCalculator):
        self.cost_calculator = cost_calculator
        self.token_counter = cost_calculator.token_counter
        
        # Optimization cache
        self._optimization_cache = {}
        self._model_performance_history = defaultdict(list)
        self._seasonal_patterns = {}
        
        # Configuration
        self.quality_thresholds = {
            OptimizationStrategy.MINIMIZE_COST: 0.6,
            OptimizationStrategy.BALANCED: 0.8,
            OptimizationStrategy.MAXIMIZE_QUALITY: 0.95,
            OptimizationStrategy.SPEED_OPTIMIZED: 0.7,
            OptimizationStrategy.BATCH_OPTIMIZED: 0.75
        }
        
        # Model performance metrics (would be learned from actual usage)
        self._model_performance = {
            'gpt-4o-mini': {'quality': 0.85, 'speed': 0.9, 'cost_efficiency': 0.95},
            'gpt-4o': {'quality': 0.93, 'speed': 0.8, 'cost_efficiency': 0.85},
            'gpt-4': {'quality': 0.95, 'speed': 0.6, 'cost_efficiency': 0.7},
            'claude-3-haiku': {'quality': 0.80, 'speed': 0.95, 'cost_efficiency': 0.9},
            'claude-3-sonnet': {'quality': 0.90, 'speed': 0.85, 'cost_efficiency': 0.8},
            'claude-3-opus': {'quality': 0.95, 'speed': 0.7, 'cost_efficiency': 0.6},
        }
    
    def optimize_operation(
        self,
        content: str,
        operation_type: str,
        strategy: OptimizationStrategy = OptimizationStrategy.BALANCED,
        available_models: Optional[List[str]] = None,
        quality_threshold: Optional[float] = None,
        max_cost: Optional[Decimal] = None
    ) -> OptimizationRecommendation:
        """Optimize a single operation based on strategy"""
        
        if not available_models:
            available_models = list(self._model_performance.keys())
        
        if quality_threshold is None:
            quality_threshold = self.quality_thresholds[strategy]
        
        # Analyze content complexity
        content_complexity = self._analyze_content_complexity(content)
        
        # Evaluate each model option
        model_evaluations = []
        
        for model in available_models:
            # Estimate cost
            token_count = self.token_counter.estimate_processing_tokens(
                content, operation_type, model
            )
            cost_breakdown = self.cost_calculator.calculate_operation_cost(
                token_count,
                self._operation_type_to_category(operation_type)
            )
            
            # Get model performance metrics
            performance = self._model_performance.get(model, {
                'quality': 0.7, 'speed': 0.7, 'cost_efficiency': 0.7
            })
            
            # Adjust quality based on content complexity
            adjusted_quality = self._adjust_quality_for_complexity(
                performance['quality'], content_complexity, model
            )
            
            # Skip if below quality threshold
            if adjusted_quality < quality_threshold:
                continue
            
            # Skip if exceeds max cost
            if max_cost and cost_breakdown.total_cost > max_cost:
                continue
            
            # Calculate optimization score based on strategy
            opt_score = self._calculate_optimization_score(
                strategy, cost_breakdown.total_cost, adjusted_quality, 
                performance['speed'], content_complexity
            )
            
            model_evaluations.append({
                'model': model,
                'cost': cost_breakdown.total_cost,
                'quality': adjusted_quality,
                'speed': performance['speed'],
                'opt_score': opt_score,
                'breakdown': cost_breakdown
            })
        
        if not model_evaluations:
            raise ValueError("No models meet the specified criteria")
        
        # Sort by optimization score
        model_evaluations.sort(key=lambda x: x['opt_score'], reverse=True)
        
        best_option = model_evaluations[0]
        alternatives = model_evaluations[1:3]  # Top 2 alternatives
        
        # Calculate savings vs most expensive option
        max_cost_option = max(model_evaluations, key=lambda x: x['cost'])
        savings_pct = float((max_cost_option['cost'] - best_option['cost']) / max_cost_option['cost'] * 100)
        
        # Generate rationale
        rationale = self._generate_optimization_rationale(
            strategy, best_option, content_complexity, len(model_evaluations)
        )
        
        # Calculate confidence
        confidence = self._calculate_recommendation_confidence(
            model_evaluations, content_complexity
        )
        
        return OptimizationRecommendation(
            strategy=strategy,
            recommended_model=best_option['model'],
            estimated_cost=best_option['cost'],
            estimated_quality=best_option['quality'],
            savings_percentage=savings_pct,
            rationale=rationale,
            confidence=confidence,
            alternative_options=[
                {
                    'model': alt['model'],
                    'cost': float(alt['cost']),
                    'quality': alt['quality'],
                    'savings_vs_recommended': float((alt['cost'] - best_option['cost']) / best_option['cost'] * 100)
                }
                for alt in alternatives
            ],
            generated_at=datetime.now()
        )
    
    def optimize_batch_processing(
        self,
        operations: List[Dict[str, Any]],  # [{'content': str, 'type': str, 'priority': int}]
        strategy: OptimizationStrategy = OptimizationStrategy.BATCH_OPTIMIZED,
        max_batch_cost: Optional[Decimal] = None
    ) -> BatchOptimizationResult:
        """Optimize batch processing for cost efficiency"""
        
        # Calculate original costs (using default model selection)
        original_breakdowns = []
        for op in operations:
            token_count = self.token_counter.estimate_processing_tokens(
                op['content'], op['type'], 'gpt-4o'  # Default baseline
            )
            breakdown = self.cost_calculator.calculate_operation_cost(token_count)
            original_breakdowns.append(breakdown)
        
        original_total_cost = sum(b.total_cost for b in original_breakdowns)
        
        # Optimize each operation individually
        optimized_assignments = {}
        optimized_costs = []
        quality_scores = []
        
        for i, op in enumerate(operations):
            try:
                recommendation = self.optimize_operation(
                    op['content'], op['type'], strategy,
                    quality_threshold=0.75  # Slightly lower for batch processing
                )
                optimized_assignments[f"op_{i}"] = recommendation.recommended_model
                optimized_costs.append(recommendation.estimated_cost)
                quality_scores.append(recommendation.estimated_quality)
                
            except ValueError:
                # Fallback to cheapest available option
                optimized_assignments[f"op_{i}"] = 'gpt-4o-mini'
                token_count = self.token_counter.estimate_processing_tokens(
                    op['content'], op['type'], 'gpt-4o-mini'
                )
                breakdown = self.cost_calculator.calculate_operation_cost(token_count)
                optimized_costs.append(breakdown.total_cost)
                quality_scores.append(0.7)  # Estimated
        
        optimized_total_cost = sum(optimized_costs)
        savings = original_total_cost - optimized_total_cost
        savings_pct = float(savings / original_total_cost * 100) if original_total_cost > 0 else 0
        
        # Estimate processing time (simplified)
        avg_processing_time = self._estimate_batch_processing_time(
            operations, optimized_assignments
        )
        
        # Calculate average quality impact
        avg_quality = statistics.mean(quality_scores) if quality_scores else 0.7
        
        # Generate recommendation text
        recommendation_text = self._generate_batch_recommendation(
            len(operations), savings_pct, avg_quality, optimized_assignments
        )
        
        return BatchOptimizationResult(
            original_cost=original_total_cost,
            optimized_cost=optimized_total_cost,
            savings=savings,
            savings_percentage=savings_pct,
            batch_size=len(operations),
            processing_time_estimate=avg_processing_time,
            model_assignments=optimized_assignments,
            quality_impact=avg_quality,
            recommendation=recommendation_text
        )
    
    def forecast_costs(
        self,
        forecast_period: BudgetPeriod,
        method: ForecastingMethod = ForecastingMethod.EXPONENTIAL_SMOOTHING,
        confidence_level: float = 0.8
    ) -> Optional[CostForecast]:
        """Advanced cost forecasting using multiple methods"""
        
        if len(self.cost_calculator.cost_history) < 20:
            logger.warning("Insufficient data for reliable forecasting")
            return None
        
        # Prepare time series data
        daily_costs = self._prepare_time_series_data()
        
        if len(daily_costs) < 7:
            return None
        
        # Apply forecasting method
        if method == ForecastingMethod.LINEAR_REGRESSION:
            forecast = self._linear_regression_forecast(daily_costs, forecast_period)
        elif method == ForecastingMethod.EXPONENTIAL_SMOOTHING:
            forecast = self._exponential_smoothing_forecast(daily_costs, forecast_period)
        elif method == ForecastingMethod.MOVING_AVERAGE:
            forecast = self._moving_average_forecast(daily_costs, forecast_period)
        else:
            # Default to exponential smoothing
            forecast = self._exponential_smoothing_forecast(daily_costs, forecast_period)
        
        if not forecast:
            return None
        
        # Calculate confidence based on historical accuracy
        confidence = self._calculate_forecast_confidence(
            daily_costs, method, forecast['projected_cost']
        )
        
        if confidence < confidence_level:
            logger.info(f"Forecast confidence ({confidence:.2f}) below threshold ({confidence_level})")
            return None
        
        return CostForecast(
            period=forecast_period,
            projected_cost=Decimal(str(forecast['projected_cost'])).quantize(Decimal('0.01')),
            confidence=confidence,
            trend=forecast['trend'],
            factors=forecast['factors'],
            generated_at=datetime.now()
        )
    
    def analyze_cost_trends(
        self,
        analysis_period_days: int = 30
    ) -> CostTrend:
        """Analyze historical cost trends and patterns"""
        
        # Get daily cost data
        daily_costs = self._prepare_time_series_data(days=analysis_period_days)
        
        if len(daily_costs) < 7:
            raise ValueError("Insufficient data for trend analysis")
        
        dates = list(daily_costs.keys())
        costs = list(daily_costs.values())
        
        # Calculate trend direction and strength
        if len(costs) > 1:
            # Simple linear regression for trend
            x = np.arange(len(costs))
            slope = np.polyfit(x, costs, 1)[0]
            
            if abs(slope) < np.std(costs) * 0.1:
                trend_direction = "stable"
                trend_strength = 0.0
            elif slope > 0:
                trend_direction = "increasing"
                trend_strength = min(1.0, abs(slope) / np.std(costs))
            else:
                trend_direction = "decreasing"
                trend_strength = min(1.0, abs(slope) / np.std(costs))
        else:
            trend_direction = "stable"
            trend_strength = 0.0
        
        # Find peak and low cost days
        max_cost_day = max(daily_costs, key=daily_costs.get)
        min_cost_day = min(daily_costs, key=daily_costs.get)
        
        # Calculate volatility (coefficient of variation)
        mean_cost = statistics.mean(costs)
        std_cost = statistics.stdev(costs) if len(costs) > 1 else 0
        volatility = std_cost / mean_cost if mean_cost > 0 else 0
        
        # Simple seasonal analysis (day of week patterns)
        seasonal_factors = self._analyze_seasonal_patterns(daily_costs)
        
        return CostTrend(
            period=f"{analysis_period_days} days",
            trend_direction=trend_direction,
            trend_strength=trend_strength,
            average_daily_cost=Decimal(str(mean_cost)).quantize(Decimal('0.01')),
            peak_cost_day=max_cost_day,
            lowest_cost_day=min_cost_day,
            volatility=volatility,
            seasonal_factors=seasonal_factors
        )
    
    def analyze_costs(self, costs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze a list of costs and provide insights"""
        if not costs:
            return {
                'total_cost': 0.0,
                'average_cost': 0.0,
                'model_distribution': {},
                'cost_by_model': {},
                'recommendations': []
            }
        
        total_cost = sum(c.get('cost', 0) for c in costs)
        avg_cost = total_cost / len(costs) if costs else 0
        
        # Analyze model distribution
        model_dist = defaultdict(int)
        model_costs = defaultdict(float)
        
        for cost_item in costs:
            model = cost_item.get('model', 'unknown')
            model_dist[model] += 1
            model_costs[model] += cost_item.get('cost', 0)
        
        # Generate basic recommendations
        recommendations = []
        if total_cost > 1.0:
            recommendations.append("Consider batch processing to reduce costs")
        
        most_expensive_model = max(model_costs.items(), key=lambda x: x[1])[0] if model_costs else None
        if most_expensive_model and model_costs[most_expensive_model] > total_cost * 0.5:
            recommendations.append(f"Model {most_expensive_model} accounts for >50% of costs - consider alternatives")
        
        return {
            'total_cost': total_cost,
            'average_cost': avg_cost,
            'model_distribution': dict(model_dist),
            'cost_by_model': dict(model_costs),
            'recommendations': recommendations
        }
    
    def get_recommendations(self, usage_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get optimization recommendations based on usage data"""
        recommendations = []
        
        daily_cost = usage_data.get('daily_cost', 0)
        primary_model = usage_data.get('primary_model', '')
        avg_tokens = usage_data.get('average_tokens', 0)
        
        # Cost-based recommendations
        if daily_cost > 100:
            recommendations.append({
                'type': 'budget_alert',
                'priority': 'high',
                'description': 'Daily costs exceed $100 - implement budget controls',
                'action': 'Set up cost alerts and budget limits'
            })
        elif daily_cost > 50:
            recommendations.append({
                'type': 'cost_optimization',
                'priority': 'medium',
                'description': 'Consider cost optimization strategies',
                'action': 'Review model selection and batch processing options'
            })
        
        # Model-specific recommendations
        if primary_model == 'gpt-4' and avg_tokens < 1000:
            recommendations.append({
                'type': 'model_optimization',
                'priority': 'high',
                'description': 'Using GPT-4 for short prompts is not cost-effective',
                'action': 'Switch to gpt-4o-mini or gpt-3.5-turbo for shorter content'
            })
        
        # Token optimization
        if avg_tokens > 2000:
            recommendations.append({
                'type': 'token_optimization',
                'priority': 'medium',
                'description': 'High average token usage detected',
                'action': 'Consider summarization or chunking strategies'
            })
        
        # Batch processing recommendation
        if daily_cost > 10 and not usage_data.get('batch_processing_enabled', False):
            recommendations.append({
                'type': 'batch_processing',
                'priority': 'medium',
                'description': 'Enable batch processing for cost savings',
                'action': 'Group similar operations for batch processing',
                'potential_savings': '15-25%'
            })
        
        return recommendations
    
    def get_optimization_insights(self) -> Dict[str, Any]:
        """Get comprehensive optimization insights and recommendations"""
        
        # Analyze model usage patterns
        model_usage = defaultdict(int)
        model_costs = defaultdict(Decimal)
        
        for cost in self.cost_calculator.cost_history:
            model_usage[cost.model] += 1
            model_costs[cost.model] += cost.total_cost
        
        # Calculate cost per operation by model
        model_efficiency = {}
        for model in model_usage:
            avg_cost = model_costs[model] / model_usage[model] if model_usage[model] > 0 else Decimal('0')
            model_efficiency[model] = {
                'usage_count': model_usage[model],
                'total_cost': float(model_costs[model]),
                'avg_cost_per_operation': float(avg_cost),
                'efficiency_score': self._calculate_model_efficiency_score(model, avg_cost)
            }
        
        # Generate recommendations
        recommendations = []
        
        # Model optimization recommendations
        most_used_model = max(model_usage, key=model_usage.get) if model_usage else None
        if most_used_model and model_efficiency[most_used_model]['efficiency_score'] < 0.7:
            recommendations.append({
                'type': 'model_optimization',
                'priority': 'high',
                'description': f"Consider switching from {most_used_model} to more cost-effective alternatives",
                'potential_savings': self._estimate_model_switch_savings(most_used_model)
            })
        
        # Batch processing recommendations
        recent_ops = len([c for c in self.cost_calculator.cost_history 
                         if c.timestamp >= datetime.now() - timedelta(days=1)])
        if recent_ops > 10:
            recommendations.append({
                'type': 'batch_processing',
                'priority': 'medium',
                'description': "High operation volume detected - consider batch processing optimizations",
                'potential_savings': "15-25%"
            })
        
        # Budget optimization recommendations
        try:
            trend = self.analyze_cost_trends(7)  # Last 7 days
            if trend.trend_direction == "increasing" and trend.trend_strength > 0.5:
                recommendations.append({
                    'type': 'budget_control',
                    'priority': 'high',
                    'description': f"Costs trending upward strongly - implement stricter budget controls",
                    'trend_strength': trend.trend_strength
                })
        except ValueError:
            pass  # Not enough data
        
        return {
            'timestamp': datetime.now().isoformat(),
            'model_efficiency': model_efficiency,
            'recommendations': recommendations,
            'total_operations_analyzed': len(self.cost_calculator.cost_history),
            'analysis_period': '30 days'
        }
    
    def _analyze_content_complexity(self, content: str) -> float:
        """Analyze content complexity for model selection"""
        # Simple heuristics for content complexity
        factors = []
        
        # Length factor
        length_factor = min(1.0, len(content) / 10000)  # Normalize to 10k chars
        factors.append(length_factor * 0.3)
        
        # Technical vocabulary (simplified detection)
        technical_terms = ['API', 'function', 'class', 'method', 'algorithm', 'database', 'query']
        tech_count = sum(1 for term in technical_terms if term.lower() in content.lower())
        tech_factor = min(1.0, tech_count / len(technical_terms))
        factors.append(tech_factor * 0.3)
        
        # Sentence complexity (average sentence length)
        sentences = content.split('.')
        avg_sentence_length = np.mean([len(s.split()) for s in sentences if s.strip()])
        sentence_factor = min(1.0, avg_sentence_length / 25)  # Normalize to 25 words
        factors.append(sentence_factor * 0.2)
        
        # Special characters and formatting
        special_char_ratio = sum(1 for c in content if not c.isalnum() and not c.isspace()) / len(content)
        format_factor = min(1.0, special_char_ratio * 10)
        factors.append(format_factor * 0.2)
        
        return sum(factors)
    
    def _adjust_quality_for_complexity(self, base_quality: float, complexity: float, model: str) -> float:
        """Adjust quality estimates based on content complexity"""
        # More capable models handle complexity better
        complexity_factors = {
            'gpt-4': 0.1,      # Best at handling complexity
            'claude-3-opus': 0.15,
            'gpt-4o': 0.2,
            'claude-3-sonnet': 0.25,
            'gpt-4o-mini': 0.35,
            'claude-3-haiku': 0.4,
        }
        
        complexity_penalty = complexity_factors.get(model, 0.3) * complexity
        return max(0.1, base_quality - complexity_penalty)
    
    def _calculate_optimization_score(
        self,
        strategy: OptimizationStrategy,
        cost: Decimal,
        quality: float,
        speed: float,
        complexity: float
    ) -> float:
        """Calculate optimization score based on strategy"""
        
        # Normalize cost (assuming $0.10 as high cost reference)
        normalized_cost = max(0.0, 1.0 - float(cost) / 0.10)
        
        if strategy == OptimizationStrategy.MINIMIZE_COST:
            return normalized_cost * 0.8 + quality * 0.2
        
        elif strategy == OptimizationStrategy.MAXIMIZE_QUALITY:
            return quality * 0.8 + normalized_cost * 0.2
        
        elif strategy == OptimizationStrategy.BALANCED:
            return (normalized_cost * 0.4 + quality * 0.4 + speed * 0.2)
        
        elif strategy == OptimizationStrategy.SPEED_OPTIMIZED:
            return speed * 0.6 + normalized_cost * 0.3 + quality * 0.1
        
        elif strategy == OptimizationStrategy.BATCH_OPTIMIZED:
            # Prefer consistent performance for batching
            consistency_bonus = 1.0 - complexity * 0.3  # Bonus for handling simple, consistent content
            return (normalized_cost * 0.5 + quality * 0.3 + consistency_bonus * 0.2)
        
        return normalized_cost * 0.5 + quality * 0.5  # Default balanced
    
    def _operation_type_to_category(self, operation_type: str) -> CostCategory:
        """Convert operation type to cost category"""
        mapping = {
            'extraction': CostCategory.EXTRACTION,
            'summarization': CostCategory.SUMMARIZATION,
            'classification': CostCategory.CLASSIFICATION,
            'embedding': CostCategory.EMBEDDING,
            'chat': CostCategory.CHAT,
            'analysis': CostCategory.ANALYSIS,
        }
        
        return mapping.get(operation_type, CostCategory.PROCESSING)
    
    def _prepare_time_series_data(self, days: int = 30) -> Dict[str, float]:
        """Prepare daily cost time series data"""
        cutoff = datetime.now() - timedelta(days=days)
        relevant_costs = [
            c for c in self.cost_calculator.cost_history
            if c.timestamp >= cutoff
        ]
        
        # Group by day
        daily_costs = defaultdict(float)
        for cost in relevant_costs:
            day_key = cost.timestamp.strftime('%Y-%m-%d')
            daily_costs[day_key] += float(cost.total_cost)
        
        return dict(daily_costs)
    
    def _exponential_smoothing_forecast(
        self,
        daily_costs: Dict[str, float],
        period: BudgetPeriod,
        alpha: float = 0.3
    ) -> Optional[Dict[str, Any]]:
        """Exponential smoothing forecast"""
        costs = list(daily_costs.values())
        
        if len(costs) < 3:
            return None
        
        # Simple exponential smoothing
        smoothed = [costs[0]]
        for i in range(1, len(costs)):
            smoothed.append(alpha * costs[i] + (1 - alpha) * smoothed[i-1])
        
        # Forecast next period
        if period == BudgetPeriod.DAILY:
            forecast_days = 1
        elif period == BudgetPeriod.WEEKLY:
            forecast_days = 7
        else:
            forecast_days = 30
        
        projected_daily = smoothed[-1]
        projected_total = projected_daily * forecast_days
        
        # Determine trend
        recent_avg = np.mean(costs[-7:]) if len(costs) >= 7 else np.mean(costs)
        older_avg = np.mean(costs[:-7]) if len(costs) >= 14 else recent_avg
        
        if recent_avg > older_avg * 1.1:
            trend = "increasing"
        elif recent_avg < older_avg * 0.9:
            trend = "decreasing"
        else:
            trend = "stable"
        
        return {
            'projected_cost': projected_total,
            'trend': trend,
            'factors': ['Exponential smoothing based on recent cost patterns']
        }
    
    def _calculate_forecast_confidence(
        self,
        historical_data: Dict[str, float],
        method: ForecastingMethod,
        forecast_value: float
    ) -> float:
        """Calculate forecast confidence based on historical accuracy"""
        costs = list(historical_data.values())
        
        if len(costs) < 10:
            return 0.5  # Low confidence with limited data
        
        # Calculate historical variance
        variance = np.var(costs)
        mean_cost = np.mean(costs)
        coefficient_of_variation = np.sqrt(variance) / mean_cost if mean_cost > 0 else 1
        
        # Base confidence inversely related to volatility
        base_confidence = max(0.3, 1.0 - coefficient_of_variation)
        
        # Adjust based on trend consistency
        recent_trend = np.mean(costs[-5:]) / np.mean(costs[:-5]) if len(costs) >= 10 else 1.0
        trend_consistency = 1.0 - abs(recent_trend - 1.0)  # Closer to 1.0 = more stable
        
        # Method-specific adjustments
        method_confidence = {
            ForecastingMethod.EXPONENTIAL_SMOOTHING: 0.85,
            ForecastingMethod.LINEAR_REGRESSION: 0.75,
            ForecastingMethod.MOVING_AVERAGE: 0.7
        }
        
        method_factor = method_confidence.get(method, 0.7)
        
        return min(0.95, base_confidence * trend_consistency * method_factor)
    
    def _generate_optimization_rationale(
        self,
        strategy: OptimizationStrategy,
        best_option: Dict[str, Any],
        complexity: float,
        num_options: int
    ) -> str:
        """Generate human-readable optimization rationale"""
        
        model = best_option['model']
        cost = float(best_option['cost'])
        quality = best_option['quality']
        
        rationale_parts = []
        
        # Strategy explanation
        if strategy == OptimizationStrategy.MINIMIZE_COST:
            rationale_parts.append(f"Selected {model} as the most cost-effective option at ${cost:.4f}")
        elif strategy == OptimizationStrategy.MAXIMIZE_QUALITY:
            rationale_parts.append(f"Selected {model} for highest quality score ({quality:.2f})")
        elif strategy == OptimizationStrategy.BALANCED:
            rationale_parts.append(f"Selected {model} for optimal balance of cost (${cost:.4f}) and quality ({quality:.2f})")
        
        # Complexity consideration
        if complexity > 0.7:
            rationale_parts.append("High content complexity detected, favoring more capable models")
        elif complexity < 0.3:
            rationale_parts.append("Low complexity content allows for cost optimization")
        
        # Options considered
        rationale_parts.append(f"Evaluated {num_options} model options")
        
        return ". ".join(rationale_parts) + "."
    
    def _calculate_recommendation_confidence(
        self,
        evaluations: List[Dict[str, Any]],
        complexity: float
    ) -> float:
        """Calculate confidence in optimization recommendation"""
        
        if len(evaluations) < 2:
            return 0.5  # Low confidence with few options
        
        # Score spread (how much better is the best vs others)
        scores = [e['opt_score'] for e in evaluations]
        score_range = max(scores) - min(scores)
        
        # Higher spread = higher confidence in recommendation
        spread_confidence = min(1.0, score_range / 0.5)
        
        # Complexity factor (harder to be confident with complex content)
        complexity_confidence = 1.0 - complexity * 0.3
        
        # Number of options factor
        options_confidence = min(1.0, len(evaluations) / 5.0)
        
        return (spread_confidence * 0.5 + complexity_confidence * 0.3 + options_confidence * 0.2)
    
    def _estimate_batch_processing_time(
        self,
        operations: List[Dict[str, Any]],
        assignments: Dict[str, str]
    ) -> float:
        """Estimate batch processing time in minutes"""
        
        # Simple time estimates per model (tokens per minute)
        model_speeds = {
            'gpt-4o-mini': 50000,
            'gpt-4o': 30000,
            'gpt-4': 20000,
            'claude-3-haiku': 45000,
            'claude-3-sonnet': 25000,
            'claude-3-opus': 15000,
        }
        
        total_time = 0
        for i, op in enumerate(operations):
            op_id = f"op_{i}"
            model = assignments.get(op_id, 'gpt-4o-mini')
            
            # Estimate tokens
            estimated_tokens = len(op['content']) * 0.75  # Rough approximation
            
            # Estimate time
            speed = model_speeds.get(model, 30000)
            time_minutes = estimated_tokens / speed
            total_time += time_minutes
        
        return total_time
    
    def _generate_batch_recommendation(
        self,
        batch_size: int,
        savings_pct: float,
        avg_quality: float,
        assignments: Dict[str, str]
    ) -> str:
        """Generate batch optimization recommendation text"""
        
        model_counts = defaultdict(int)
        for model in assignments.values():
            model_counts[model] += 1
        
        primary_model = max(model_counts, key=model_counts.get)
        
        recommendation = f"Batch of {batch_size} operations optimized for {savings_pct:.1f}% cost savings. "
        recommendation += f"Primary model: {primary_model} ({model_counts[primary_model]} operations). "
        recommendation += f"Average quality maintained at {avg_quality:.2f}. "
        
        if len(model_counts) > 1:
            recommendation += f"Mixed model assignment across {len(model_counts)} different models for optimal efficiency."
        else:
            recommendation += "Consistent model assignment for predictable performance."
        
        return recommendation
    
    def _analyze_seasonal_patterns(self, daily_costs: Dict[str, float]) -> Dict[str, float]:
        """Analyze day-of-week seasonal patterns"""
        
        day_costs = defaultdict(list)
        
        for date_str, cost in daily_costs.items():
            try:
                date = datetime.strptime(date_str, '%Y-%m-%d')
                day_name = date.strftime('%A')
                day_costs[day_name].append(cost)
            except ValueError:
                continue
        
        # Calculate average cost per day of week
        seasonal_factors = {}
        overall_avg = np.mean(list(daily_costs.values())) if daily_costs else 0
        
        for day, costs in day_costs.items():
            if costs and overall_avg > 0:
                day_avg = np.mean(costs)
                seasonal_factors[day] = day_avg / overall_avg
            else:
                seasonal_factors[day] = 1.0
        
        return seasonal_factors
    
    def _calculate_model_efficiency_score(self, model: str, avg_cost: Decimal) -> float:
        """Calculate efficiency score for a model"""
        performance = self._model_performance.get(model, {'cost_efficiency': 0.7})
        
        # Combine performance metrics with actual cost data
        efficiency_score = performance['cost_efficiency']
        
        # Adjust based on actual vs expected cost
        expected_costs = {
            'gpt-4o-mini': 0.001,
            'gpt-4o': 0.01,
            'gpt-4': 0.03,
            'claude-3-haiku': 0.0005,
            'claude-3-sonnet': 0.005,
            'claude-3-opus': 0.025,
        }
        
        expected = expected_costs.get(model, 0.01)
        if avg_cost > 0:
            cost_factor = min(2.0, expected / float(avg_cost))
            efficiency_score *= cost_factor
        
        return min(1.0, efficiency_score)
    
    def _estimate_model_switch_savings(self, current_model: str) -> str:
        """Estimate potential savings from switching models"""
        
        current_performance = self._model_performance.get(current_model, {})
        current_efficiency = current_performance.get('cost_efficiency', 0.7)
        
        # Find most efficient alternatives
        alternatives = [
            (model, perf['cost_efficiency'])
            for model, perf in self._model_performance.items()
            if model != current_model and perf['cost_efficiency'] > current_efficiency
        ]
        
        if alternatives:
            best_alternative = max(alternatives, key=lambda x: x[1])
            improvement = (best_alternative[1] - current_efficiency) * 100
            return f"{improvement:.0f}% by switching to {best_alternative[0]}"
        
        return "5-15% with optimized model selection"
    
    # Additional forecasting methods
    def _linear_regression_forecast(self, daily_costs: Dict[str, float], period: BudgetPeriod) -> Optional[Dict[str, Any]]:
        """Linear regression forecast"""
        costs = list(daily_costs.values())
        x = np.arange(len(costs))
        
        if len(costs) < 3:
            return None
        
        # Fit linear regression
        slope, intercept = np.polyfit(x, costs, 1)
        
        # Forecast
        if period == BudgetPeriod.DAILY:
            forecast_days = 1
        elif period == BudgetPeriod.WEEKLY:
            forecast_days = 7
        else:
            forecast_days = 30
        
        next_x = len(costs)
        projected_daily = slope * next_x + intercept
        projected_total = max(0, projected_daily * forecast_days)  # Ensure non-negative
        
        # Determine trend
        if abs(slope) < np.std(costs) * 0.1:
            trend = "stable"
        elif slope > 0:
            trend = "increasing"
        else:
            trend = "decreasing"
        
        return {
            'projected_cost': projected_total,
            'trend': trend,
            'factors': [f'Linear trend analysis (slope: {slope:.6f})']
        }
    
    def _moving_average_forecast(self, daily_costs: Dict[str, float], period: BudgetPeriod, window: int = 7) -> Optional[Dict[str, Any]]:
        """Moving average forecast"""
        costs = list(daily_costs.values())
        
        if len(costs) < window:
            window = max(3, len(costs) // 2)
        
        # Calculate moving average
        recent_avg = np.mean(costs[-window:])
        
        # Forecast
        if period == BudgetPeriod.DAILY:
            forecast_days = 1
        elif period == BudgetPeriod.WEEKLY:
            forecast_days = 7
        else:
            forecast_days = 30
        
        projected_total = recent_avg * forecast_days
        
        # Simple trend detection
        if len(costs) >= window * 2:
            older_avg = np.mean(costs[-window*2:-window])
            if recent_avg > older_avg * 1.1:
                trend = "increasing"
            elif recent_avg < older_avg * 0.9:
                trend = "decreasing"
            else:
                trend = "stable"
        else:
            trend = "stable"
        
        return {
            'projected_cost': projected_total,
            'trend': trend,
            'factors': [f'{window}-day moving average projection']
        }


# Global optimizer instance
_cost_optimizer: Optional[CostOptimizer] = None


def get_cost_optimizer(cost_calculator: Optional[CostCalculator] = None) -> CostOptimizer:
    """Get the global cost optimizer instance"""
    global _cost_optimizer
    if _cost_optimizer is None:
        from .cost_calculator import get_cost_calculator
        calc = cost_calculator or get_cost_calculator()
        _cost_optimizer = CostOptimizer(calc)
    return _cost_optimizer