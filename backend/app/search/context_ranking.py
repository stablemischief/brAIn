"""
Context-aware ranking system for search results.

This module provides sophisticated ranking algorithms that consider user context,
search history, document relevance, and behavioral patterns to improve search
result quality and personalization.
"""

import math
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import asyncio
from uuid import UUID

from .hybrid_search import SearchResult, QueryAnalysis


class RankingFactor(Enum):
    """Different factors that influence result ranking"""
    RELEVANCE_SCORE = "relevance_score"
    RECENCY = "recency"
    USER_PREFERENCE = "user_preference"
    CLICK_THROUGH_RATE = "click_through_rate"
    DOCUMENT_AUTHORITY = "document_authority"
    CONTENT_QUALITY = "content_quality"
    USER_CONTEXT = "user_context"
    SEMANTIC_SIMILARITY = "semantic_similarity"


@dataclass
class UserContext:
    """User context information for personalized ranking"""
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    search_history: List[str] = field(default_factory=list)
    clicked_documents: List[str] = field(default_factory=list)
    user_preferences: Dict[str, float] = field(default_factory=dict)
    current_topic_focus: Optional[str] = None
    expertise_level: str = "intermediate"  # beginner, intermediate, expert
    language_preference: str = "en"
    time_zone: Optional[str] = None
    device_type: str = "desktop"  # mobile, tablet, desktop


@dataclass
class RankingConfig:
    """Configuration for the ranking system"""
    # Factor weights
    relevance_weight: float = 0.4
    recency_weight: float = 0.1
    user_preference_weight: float = 0.2
    ctr_weight: float = 0.15
    authority_weight: float = 0.1
    quality_weight: float = 0.05
    
    # Decay parameters
    time_decay_half_life: timedelta = timedelta(days=30)
    click_decay_half_life: timedelta = timedelta(days=7)
    
    # Personalization parameters
    personalization_strength: float = 0.3
    min_history_for_personalization: int = 5
    
    # Quality thresholds
    min_content_quality_score: float = 0.3
    boost_high_quality_threshold: float = 0.8


@dataclass
class DocumentMetrics:
    """Metrics and metadata for a document used in ranking"""
    document_id: str
    click_through_rate: float = 0.0
    total_clicks: int = 0
    total_impressions: int = 0
    average_dwell_time: float = 0.0
    authority_score: float = 0.5
    content_quality_score: float = 0.5
    creation_date: Optional[datetime] = None
    last_updated: Optional[datetime] = None
    author_authority: float = 0.5
    document_type: str = "article"
    topic_categories: List[str] = field(default_factory=list)


class ContextAwareRanker:
    """
    Advanced ranking system that considers multiple factors including user context,
    document metrics, and search patterns to provide personalized results.
    """
    
    def __init__(self, config: Optional[RankingConfig] = None):
        self.config = config or RankingConfig()
        self._document_metrics_cache: Dict[str, DocumentMetrics] = {}
        self._user_profile_cache: Dict[str, UserContext] = {}
    
    async def rank_results(
        self,
        results: List[SearchResult],
        query: str,
        query_analysis: QueryAnalysis,
        user_context: Optional[UserContext] = None,
        ranking_factors: Optional[List[RankingFactor]] = None
    ) -> List[SearchResult]:
        """
        Rank search results using context-aware algorithms.
        
        Args:
            results: Initial search results to rank
            query: Original search query
            query_analysis: Analysis of the query
            user_context: User context for personalization
            ranking_factors: Specific factors to emphasize
            
        Returns:
            Reranked search results
        """
        if not results:
            return results
        
        # Default ranking factors
        if ranking_factors is None:
            ranking_factors = [
                RankingFactor.RELEVANCE_SCORE,
                RankingFactor.RECENCY,
                RankingFactor.USER_PREFERENCE,
                RankingFactor.CLICK_THROUGH_RATE,
                RankingFactor.CONTENT_QUALITY
            ]
        
        # Calculate ranking scores for each result
        scored_results = []
        for result in results:
            ranking_score = await self._calculate_ranking_score(
                result, query, query_analysis, user_context, ranking_factors
            )
            
            # Create updated result with ranking score
            updated_result = SearchResult(
                document_id=result.document_id,
                title=result.title,
                content=result.content,
                similarity_score=result.similarity_score,
                keyword_score=result.keyword_score,
                combined_score=ranking_score,  # Use ranking score as combined score
                metadata=result.metadata,
                highlights=result.highlights,
                search_strategy=result.search_strategy
            )
            
            scored_results.append(updated_result)
        
        # Sort by ranking score (descending)
        scored_results.sort(key=lambda x: x.combined_score, reverse=True)
        
        return scored_results
    
    async def _calculate_ranking_score(
        self,
        result: SearchResult,
        query: str,
        query_analysis: QueryAnalysis,
        user_context: Optional[UserContext],
        ranking_factors: List[RankingFactor]
    ) -> float:
        """Calculate comprehensive ranking score for a search result"""
        
        # Get document metrics
        doc_metrics = await self._get_document_metrics(result.document_id)
        
        # Initialize score components
        score_components = {}
        
        # Base relevance score (from search engine)
        if RankingFactor.RELEVANCE_SCORE in ranking_factors:
            score_components[RankingFactor.RELEVANCE_SCORE] = result.combined_score
        
        # Recency factor
        if RankingFactor.RECENCY in ranking_factors:
            score_components[RankingFactor.RECENCY] = self._calculate_recency_score(doc_metrics)
        
        # User preference factor
        if RankingFactor.USER_PREFERENCE in ranking_factors and user_context:
            score_components[RankingFactor.USER_PREFERENCE] = await self._calculate_user_preference_score(
                result, query, user_context
            )
        
        # Click-through rate factor
        if RankingFactor.CLICK_THROUGH_RATE in ranking_factors:
            score_components[RankingFactor.CLICK_THROUGH_RATE] = self._calculate_ctr_score(doc_metrics)
        
        # Document authority factor
        if RankingFactor.DOCUMENT_AUTHORITY in ranking_factors:
            score_components[RankingFactor.DOCUMENT_AUTHORITY] = doc_metrics.authority_score
        
        # Content quality factor
        if RankingFactor.CONTENT_QUALITY in ranking_factors:
            score_components[RankingFactor.CONTENT_QUALITY] = self._calculate_quality_score(doc_metrics)
        
        # User context factor
        if RankingFactor.USER_CONTEXT in ranking_factors and user_context:
            score_components[RankingFactor.USER_CONTEXT] = await self._calculate_context_score(
                result, query_analysis, user_context
            )
        
        # Semantic similarity factor
        if RankingFactor.SEMANTIC_SIMILARITY in ranking_factors:
            score_components[RankingFactor.SEMANTIC_SIMILARITY] = result.similarity_score or 0.0
        
        # Combine scores with weights
        final_score = await self._combine_scores(score_components, user_context)
        
        return max(0.0, min(1.0, final_score))  # Clamp between 0 and 1
    
    def _calculate_recency_score(self, doc_metrics: DocumentMetrics) -> float:
        """Calculate recency score with time decay"""
        if not doc_metrics.last_updated and not doc_metrics.creation_date:
            return 0.5  # Neutral score for unknown dates
        
        reference_date = doc_metrics.last_updated or doc_metrics.creation_date
        if not reference_date:
            return 0.5
        
        # Calculate days since last update
        days_old = (datetime.utcnow() - reference_date).days
        
        # Apply exponential decay
        half_life_days = self.config.time_decay_half_life.days
        decay_factor = 2 ** (-days_old / half_life_days)
        
        return min(1.0, decay_factor)
    
    async def _calculate_user_preference_score(
        self,
        result: SearchResult,
        query: str,
        user_context: UserContext
    ) -> float:
        """Calculate score based on user preferences and history"""
        if not user_context or len(user_context.search_history) < self.config.min_history_for_personalization:
            return 0.5  # Neutral score for new users
        
        preference_score = 0.0
        
        # Check if document was previously clicked by user
        if result.document_id in user_context.clicked_documents:
            preference_score += 0.3
        
        # Check topic preferences
        doc_topics = result.metadata.get("topics", [])
        if doc_topics:
            topic_matches = sum(
                user_context.user_preferences.get(topic, 0.0) 
                for topic in doc_topics
            ) / len(doc_topics)
            preference_score += topic_matches * 0.4
        
        # Check expertise level match
        doc_level = result.metadata.get("difficulty_level", "intermediate")
        if doc_level == user_context.expertise_level:
            preference_score += 0.2
        elif abs(self._get_level_numeric(doc_level) - self._get_level_numeric(user_context.expertise_level)) <= 1:
            preference_score += 0.1
        
        # Current topic focus bonus
        if user_context.current_topic_focus and user_context.current_topic_focus in doc_topics:
            preference_score += 0.1
        
        return min(1.0, preference_score)
    
    def _calculate_ctr_score(self, doc_metrics: DocumentMetrics) -> float:
        """Calculate click-through rate score"""
        if doc_metrics.total_impressions == 0:
            return 0.5  # Neutral score for new documents
        
        # Basic CTR
        base_ctr = doc_metrics.click_through_rate
        
        # Adjust for sample size (Wilson score interval approach)
        n = doc_metrics.total_impressions
        if n < 10:  # Small sample size
            confidence_adjustment = 0.1  # Conservative adjustment
            adjusted_ctr = (base_ctr + confidence_adjustment) / (1 + confidence_adjustment)
        else:
            adjusted_ctr = base_ctr
        
        # Apply time decay for clicks
        if doc_metrics.total_clicks > 0:
            # Assume recent clicks are more valuable (simplified approach)
            time_decay = 0.9  # Could be calculated based on click timestamps
            adjusted_ctr *= time_decay
        
        return min(1.0, adjusted_ctr * 2)  # Scale up CTR for ranking
    
    def _calculate_quality_score(self, doc_metrics: DocumentMetrics) -> float:
        """Calculate content quality score"""
        quality_score = doc_metrics.content_quality_score
        
        # Apply quality threshold
        if quality_score < self.config.min_content_quality_score:
            quality_score *= 0.5  # Penalize low-quality content
        
        # Boost high-quality content
        if quality_score >= self.config.boost_high_quality_threshold:
            quality_score = min(1.0, quality_score * 1.2)
        
        # Consider author authority
        author_boost = doc_metrics.author_authority * 0.2
        quality_score = min(1.0, quality_score + author_boost)
        
        return quality_score
    
    async def _calculate_context_score(
        self,
        result: SearchResult,
        query_analysis: QueryAnalysis,
        user_context: UserContext
    ) -> float:
        """Calculate score based on current user context"""
        context_score = 0.0
        
        # Device type optimization
        doc_format = result.metadata.get("format", "text")
        if user_context.device_type == "mobile":
            if doc_format in ["mobile_optimized", "summary"]:
                context_score += 0.2
            elif doc_format in ["long_form", "technical"]:
                context_score -= 0.1
        
        # Time-based context
        current_hour = datetime.utcnow().hour
        if user_context.time_zone:
            # Adjust for user timezone (simplified)
            local_hour = (current_hour - 8) % 24  # Assume UTC-8 for simplicity
        else:
            local_hour = current_hour
        
        # Morning: prefer news, updates
        # Afternoon: prefer in-depth articles
        # Evening: prefer lighter content
        doc_type = result.metadata.get("content_type", "article")
        if 6 <= local_hour <= 10 and doc_type == "news":
            context_score += 0.1
        elif 14 <= local_hour <= 18 and doc_type in ["tutorial", "guide"]:
            context_score += 0.1
        elif 19 <= local_hour <= 23 and doc_type in ["summary", "overview"]:
            context_score += 0.1
        
        # Query intent matching
        if query_analysis.intent_type == "informational" and doc_type in ["article", "guide"]:
            context_score += 0.15
        elif query_analysis.intent_type == "navigational" and doc_type == "landing_page":
            context_score += 0.15
        elif query_analysis.intent_type == "transactional" and doc_type in ["product", "service"]:
            context_score += 0.15
        
        return min(1.0, max(0.0, context_score))
    
    async def _combine_scores(
        self,
        score_components: Dict[RankingFactor, float],
        user_context: Optional[UserContext]
    ) -> float:
        """Combine individual ranking scores into final score"""
        
        # Base weighted combination
        weighted_score = 0.0
        total_weight = 0.0
        
        for factor, score in score_components.items():
            weight = self._get_factor_weight(factor)
            weighted_score += score * weight
            total_weight += weight
        
        # Normalize by total weight
        if total_weight > 0:
            base_score = weighted_score / total_weight
        else:
            base_score = 0.5
        
        # Apply personalization boost if applicable
        if user_context and len(user_context.search_history) >= self.config.min_history_for_personalization:
            personalization_boost = score_components.get(RankingFactor.USER_PREFERENCE, 0.5)
            personalized_score = (
                base_score * (1 - self.config.personalization_strength) +
                personalization_boost * self.config.personalization_strength
            )
            return personalized_score
        
        return base_score
    
    def _get_factor_weight(self, factor: RankingFactor) -> float:
        """Get weight for a ranking factor"""
        weight_map = {
            RankingFactor.RELEVANCE_SCORE: self.config.relevance_weight,
            RankingFactor.RECENCY: self.config.recency_weight,
            RankingFactor.USER_PREFERENCE: self.config.user_preference_weight,
            RankingFactor.CLICK_THROUGH_RATE: self.config.ctr_weight,
            RankingFactor.DOCUMENT_AUTHORITY: self.config.authority_weight,
            RankingFactor.CONTENT_QUALITY: self.config.quality_weight,
            RankingFactor.USER_CONTEXT: 0.1,  # Additional context weight
            RankingFactor.SEMANTIC_SIMILARITY: 0.2  # Semantic weight
        }
        return weight_map.get(factor, 0.1)
    
    def _get_level_numeric(self, level: str) -> int:
        """Convert expertise level to numeric value"""
        level_map = {"beginner": 1, "intermediate": 2, "expert": 3}
        return level_map.get(level.lower(), 2)
    
    async def _get_document_metrics(self, document_id: str) -> DocumentMetrics:
        """Get or create document metrics"""
        if document_id not in self._document_metrics_cache:
            # In a real implementation, this would fetch from database
            # For now, return default metrics
            self._document_metrics_cache[document_id] = DocumentMetrics(
                document_id=document_id,
                creation_date=datetime.utcnow() - timedelta(days=30)
            )
        
        return self._document_metrics_cache[document_id]
    
    async def update_document_metrics(
        self,
        document_id: str,
        click_occurred: bool = False,
        dwell_time: Optional[float] = None,
        impression_occurred: bool = True
    ):
        """Update document metrics based on user interaction"""
        metrics = await self._get_document_metrics(document_id)
        
        if impression_occurred:
            metrics.total_impressions += 1
        
        if click_occurred:
            metrics.total_clicks += 1
            metrics.click_through_rate = metrics.total_clicks / max(1, metrics.total_impressions)
        
        if dwell_time is not None:
            # Update average dwell time with exponential moving average
            if metrics.average_dwell_time == 0:
                metrics.average_dwell_time = dwell_time
            else:
                alpha = 0.1  # Smoothing factor
                metrics.average_dwell_time = (
                    alpha * dwell_time + (1 - alpha) * metrics.average_dwell_time
                )
    
    async def update_user_context(
        self,
        user_context: UserContext,
        query: str,
        clicked_document_id: Optional[str] = None,
        topic_preferences: Optional[Dict[str, float]] = None
    ):
        """Update user context based on search behavior"""
        # Add to search history
        if query not in user_context.search_history:
            user_context.search_history.append(query)
            # Keep only recent history
            if len(user_context.search_history) > 100:
                user_context.search_history = user_context.search_history[-100:]
        
        # Add clicked document
        if clicked_document_id and clicked_document_id not in user_context.clicked_documents:
            user_context.clicked_documents.append(clicked_document_id)
            # Keep only recent clicks
            if len(user_context.clicked_documents) > 200:
                user_context.clicked_documents = user_context.clicked_documents[-200:]
        
        # Update topic preferences
        if topic_preferences:
            for topic, weight in topic_preferences.items():
                current_weight = user_context.user_preferences.get(topic, 0.0)
                # Exponential moving average update
                user_context.user_preferences[topic] = 0.1 * weight + 0.9 * current_weight
    
    def get_ranking_explanation(
        self,
        result: SearchResult,
        score_components: Dict[RankingFactor, float]
    ) -> Dict[str, Any]:
        """Generate explanation for ranking decision"""
        return {
            "final_score": result.combined_score,
            "components": {
                factor.value: score 
                for factor, score in score_components.items()
            },
            "primary_factors": [
                factor.value for factor, score in score_components.items()
                if score > 0.7
            ],
            "ranking_rationale": self._generate_ranking_rationale(score_components)
        }
    
    def _generate_ranking_rationale(
        self,
        score_components: Dict[RankingFactor, float]
    ) -> str:
        """Generate human-readable explanation for ranking"""
        explanations = []
        
        # Find top contributing factors
        sorted_factors = sorted(
            score_components.items(),
            key=lambda x: x[1],
            reverse=True
        )[:3]
        
        for factor, score in sorted_factors:
            if score > 0.7:
                if factor == RankingFactor.RELEVANCE_SCORE:
                    explanations.append("high content relevance to query")
                elif factor == RankingFactor.USER_PREFERENCE:
                    explanations.append("matches your interests and search history")
                elif factor == RankingFactor.RECENCY:
                    explanations.append("recently updated content")
                elif factor == RankingFactor.CLICK_THROUGH_RATE:
                    explanations.append("popular with other users")
                elif factor == RankingFactor.CONTENT_QUALITY:
                    explanations.append("high-quality content from authoritative source")
        
        if not explanations:
            return "balanced ranking across multiple factors"
        
        return "Ranked highly due to: " + ", ".join(explanations)