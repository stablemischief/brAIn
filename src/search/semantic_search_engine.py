"""
Main Semantic Search Engine orchestrator.

This module provides the primary interface for the semantic search system,
orchestrating hybrid search, context-aware ranking, search history tracking,
document suggestions, and analytics.
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
from uuid import uuid4
import logging

from .hybrid_search import (
    HybridSearchEngine, 
    SearchStrategy, 
    SearchResult, 
    QueryAnalysis,
    HybridSearchConfig,
    QueryProcessor
)
from .context_ranking import (
    ContextAwareRanker,
    UserContext,
    RankingConfig,
    RankingFactor
)
from .search_history import (
    SearchHistoryManager,
    SearchQuery,
    SearchInteraction,
    InteractionType,
    SearchIntent
)
from .document_suggestions import (
    DocumentSuggestionEngine,
    DocumentSuggestion,
    SuggestionConfig,
    SuggestionType
)
from .search_analytics import (
    SearchAnalyticsEngine,
    AnalyticsReport,
    MetricType
)


class SearchMode(Enum):
    """Search modes for different use cases"""
    STANDARD = "standard"
    RESEARCH = "research"
    QUICK_ANSWER = "quick_answer"
    EXPLORATION = "exploration"
    FOCUSED = "focused"


@dataclass 
class SemanticSearchConfig:
    """Configuration for the semantic search engine"""
    # Component configurations
    hybrid_search_config: Optional[HybridSearchConfig] = None
    ranking_config: Optional[RankingConfig] = None
    suggestion_config: Optional[SuggestionConfig] = None
    
    # Search behavior
    default_search_mode: SearchMode = SearchMode.STANDARD
    enable_personalization: bool = True
    enable_search_history: bool = True
    enable_suggestions: bool = True
    enable_analytics: bool = True
    
    # Performance settings
    max_results_per_query: int = 50
    result_cache_ttl: timedelta = timedelta(minutes=30)
    enable_result_caching: bool = True
    
    # Quality thresholds
    min_result_confidence: float = 0.1
    suggestion_trigger_threshold: int = 3  # Min results to show suggestions
    
    # Database connection
    database_url: str = ""


@dataclass
class SearchRequest:
    """Complete search request with context"""
    query: str
    user_context: Optional[UserContext] = None
    search_mode: Optional[SearchMode] = None
    strategy: Optional[SearchStrategy] = None
    filters: Optional[Dict[str, Any]] = None
    limit: int = 20
    include_suggestions: bool = True
    include_analytics: bool = True
    
    # Request metadata
    request_id: str = field(default_factory=lambda: str(uuid4()))
    timestamp: datetime = field(default_factory=datetime.utcnow)
    session_id: Optional[str] = None


@dataclass
class SearchResponse:
    """Complete search response with all components"""
    request_id: str
    query: str
    
    # Core search results
    results: List[SearchResult]
    total_results: int
    
    # Enhanced features
    suggestions: List[DocumentSuggestion] = field(default_factory=list)
    query_analysis: Optional[QueryAnalysis] = None
    
    # Performance metrics
    processing_time: float = 0.0
    strategy_used: Optional[SearchStrategy] = None
    
    # Context and insights
    search_insights: Dict[str, Any] = field(default_factory=dict)
    ranking_explanation: Dict[str, Any] = field(default_factory=dict)
    
    # Metadata
    timestamp: datetime = field(default_factory=datetime.utcnow)
    cache_hit: bool = False
    
    # Debugging and analytics
    debug_info: Dict[str, Any] = field(default_factory=dict)


class SemanticSearchEngine:
    """
    Main semantic search engine that orchestrates all search components
    to provide intelligent, context-aware, and personalized search experiences.
    """
    
    def __init__(self, config: SemanticSearchConfig):
        self.config = config
        
        # Initialize components
        self.hybrid_search = HybridSearchEngine(
            config.hybrid_search_config or HybridSearchConfig()
        )
        self.context_ranker = ContextAwareRanker(
            config.ranking_config or RankingConfig()
        )
        
        # Optional components
        self.history_manager: Optional[SearchHistoryManager] = None
        self.suggestion_engine: Optional[DocumentSuggestionEngine] = None
        self.analytics_engine: Optional[SearchAnalyticsEngine] = None
        
        # Result caching
        self._result_cache: Dict[str, Tuple[SearchResponse, datetime]] = {}
        
        # Performance tracking
        self._performance_stats = {
            'total_queries': 0,
            'cache_hits': 0,
            'average_response_time': 0.0
        }
        
        # Logger
        self.logger = logging.getLogger(__name__)
    
    async def initialize(self) -> bool:
        """Initialize the semantic search engine and all components"""
        try:
            # Initialize hybrid search
            success = await self.hybrid_search.initialize()
            if not success:
                self.logger.error("Failed to initialize hybrid search engine")
                return False
            
            # Initialize optional components
            if self.config.enable_search_history and self.config.database_url:
                self.history_manager = SearchHistoryManager(self.config.database_url)
                success = await self.history_manager.initialize()
                if not success:
                    self.logger.warning("Failed to initialize search history manager")
            
            if self.config.enable_suggestions:
                self.suggestion_engine = DocumentSuggestionEngine(
                    self.config.suggestion_config or SuggestionConfig()
                )
            
            if self.config.enable_analytics and self.config.database_url:
                self.analytics_engine = SearchAnalyticsEngine(self.config.database_url)
                success = await self.analytics_engine.initialize()
                if not success:
                    self.logger.warning("Failed to initialize analytics engine")
            
            # Start background tasks
            asyncio.create_task(self._cleanup_cache_periodically())
            
            self.logger.info("Semantic search engine initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize semantic search engine: {e}")
            return False
    
    async def search(self, request: SearchRequest) -> SearchResponse:
        """
        Perform comprehensive semantic search with all enhancements.
        
        Args:
            request: Complete search request with context
            
        Returns:
            Complete search response with results and enhancements
        """
        start_time = datetime.utcnow()
        
        try:
            # Check cache first
            if self.config.enable_result_caching:
                cached_response = await self._check_cache(request)
                if cached_response:
                    self._performance_stats['cache_hits'] += 1
                    return cached_response
            
            # Create response object
            response = SearchResponse(
                request_id=request.request_id,
                query=request.query
            )
            
            # 1. Analyze the query
            query_analysis = await self._analyze_query(request)
            response.query_analysis = query_analysis
            
            # 2. Determine optimal search strategy
            search_strategy = await self._determine_strategy(request, query_analysis)
            response.strategy_used = search_strategy
            
            # 3. Execute hybrid search
            search_results = await self.hybrid_search.search(
                query=request.query,
                strategy=search_strategy,
                filters=request.filters,
                limit=request.limit * 2,  # Get extra for ranking
                user_context=request.user_context
            )
            
            # 4. Apply context-aware ranking
            if self.config.enable_personalization and request.user_context:
                search_results = await self.context_ranker.rank_results(
                    results=search_results,
                    query=request.query,
                    query_analysis=query_analysis,
                    user_context=request.user_context
                )
            
            # 5. Filter and limit results
            filtered_results = await self._filter_results(search_results, request)
            response.results = filtered_results[:request.limit]
            response.total_results = len(search_results)
            
            # 6. Generate document suggestions
            if (request.include_suggestions and 
                self.suggestion_engine and 
                len(response.results) >= self.config.suggestion_trigger_threshold):
                
                suggestions = await self._generate_suggestions(request, response.results)
                response.suggestions = suggestions
            
            # 7. Generate search insights
            response.search_insights = await self._generate_search_insights(
                request, response, query_analysis
            )
            
            # 8. Record search event
            await self._record_search_event(request, response, search_strategy)
            
            # 9. Cache results
            if self.config.enable_result_caching:
                await self._cache_results(request, response)
            
            # Calculate processing time
            processing_time = (datetime.utcnow() - start_time).total_seconds()
            response.processing_time = processing_time
            
            # Update performance stats
            self._update_performance_stats(processing_time)
            
            self.logger.info(
                f"Search completed: query='{request.query}', "
                f"results={len(response.results)}, "
                f"time={processing_time:.3f}s"
            )
            
            return response
            
        except Exception as e:
            self.logger.error(f"Search failed: {e}")
            
            # Return error response
            error_response = SearchResponse(
                request_id=request.request_id,
                query=request.query,
                results=[],
                total_results=0,
                processing_time=(datetime.utcnow() - start_time).total_seconds(),
                debug_info={"error": str(e)}
            )
            
            return error_response
    
    async def record_interaction(
        self,
        query_id: str,
        document_id: str,
        interaction_type: InteractionType,
        user_context: Optional[UserContext] = None,
        dwell_time: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Record user interaction with search results"""
        try:
            interaction = SearchInteraction(
                document_id=document_id,
                interaction_type=interaction_type,
                dwell_time=dwell_time,
                metadata=metadata or {}
            )
            
            # Record in history manager
            if self.history_manager:
                await self.history_manager.record_interaction(query_id, interaction)
            
            # Record in analytics
            if self.analytics_engine:
                user_id = user_context.user_id if user_context else None
                await self.analytics_engine.record_user_interaction(
                    query_id, interaction, user_id
                )
            
            # Update ranking system with feedback
            if self.context_ranker and user_context:
                await self.context_ranker.update_document_metrics(
                    document_id=document_id,
                    click_occurred=(interaction_type == InteractionType.CLICK),
                    dwell_time=dwell_time,
                    impression_occurred=True
                )
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to record interaction: {e}")
            return False
    
    async def get_search_suggestions(
        self,
        partial_query: str,
        user_context: Optional[UserContext] = None,
        limit: int = 5
    ) -> List[str]:
        """Get query completion suggestions"""
        if not self.history_manager:
            return []
        
        try:
            user_id = user_context.user_id if user_context else None
            suggestions = await self.history_manager.get_query_suggestions(
                partial_query, user_id, limit
            )
            return suggestions
            
        except Exception as e:
            self.logger.error(f"Failed to get search suggestions: {e}")
            return []
    
    async def get_user_search_analytics(
        self,
        user_id: str,
        days_back: int = 30
    ) -> Dict[str, Any]:
        """Get analytics for a specific user"""
        if not self.history_manager:
            return {}
        
        try:
            analytics = await self.history_manager.analyze_search_patterns(
                user_id, days_back
            )
            return analytics
            
        except Exception as e:
            self.logger.error(f"Failed to get user analytics: {e}")
            return {}
    
    async def generate_analytics_report(
        self,
        start_date: datetime,
        end_date: datetime,
        report_type: str = "comprehensive"
    ) -> Optional[AnalyticsReport]:
        """Generate comprehensive analytics report"""
        if not self.analytics_engine:
            return None
        
        try:
            report = await self.analytics_engine.generate_analytics_report(
                start_date, end_date, report_type
            )
            return report
            
        except Exception as e:
            self.logger.error(f"Failed to generate analytics report: {e}")
            return None
    
    async def _analyze_query(self, request: SearchRequest) -> QueryAnalysis:
        """Analyze the search query for optimal processing"""
        processor = QueryProcessor()
        
        # Basic analysis
        analysis = await processor.analyze_query(
            request.query, 
            request.user_context
        )
        
        # Enhance with search history if available
        if self.history_manager and request.user_context:
            similar_queries = await self.history_manager.get_similar_queries(
                request.query, 
                request.user_context.user_id, 
                limit=5
            )
            
            # Use similar queries to refine analysis
            if similar_queries:
                analysis.similar_queries = [q.query_text for q in similar_queries]
        
        return analysis
    
    async def _determine_strategy(
        self, 
        request: SearchRequest, 
        query_analysis: QueryAnalysis
    ) -> SearchStrategy:
        """Determine optimal search strategy based on request and analysis"""
        
        # Use explicitly requested strategy
        if request.strategy:
            return request.strategy
        
        # Mode-based strategy selection
        if request.search_mode == SearchMode.QUICK_ANSWER:
            return SearchStrategy.KEYWORD_ONLY
        elif request.search_mode == SearchMode.RESEARCH:
            return SearchStrategy.HYBRID_BALANCED
        elif request.search_mode == SearchMode.EXPLORATION:
            return SearchStrategy.VECTOR_ONLY
        
        # Analysis-based strategy selection
        if query_analysis.intent_type == "navigational":
            return SearchStrategy.KEYWORD_ONLY
        elif query_analysis.intent_type == "informational":
            return SearchStrategy.HYBRID_BALANCED
        elif query_analysis.contains_entities and len(query_analysis.keywords) <= 3:
            return SearchStrategy.VECTOR_ONLY
        
        # Default to adaptive
        return SearchStrategy.ADAPTIVE
    
    async def _filter_results(
        self, 
        results: List[SearchResult], 
        request: SearchRequest
    ) -> List[SearchResult]:
        """Filter search results based on quality and relevance"""
        filtered = []
        
        for result in results:
            # Apply minimum confidence threshold
            if result.combined_score < self.config.min_result_confidence:
                continue
            
            # Apply user-specific filters
            if request.filters:
                if not self._apply_filters(result, request.filters):
                    continue
            
            filtered.append(result)
        
        return filtered
    
    def _apply_filters(
        self, 
        result: SearchResult, 
        filters: Dict[str, Any]
    ) -> bool:
        """Apply filters to a search result"""
        # Example filter implementations
        if "document_type" in filters:
            allowed_types = filters["document_type"]
            doc_type = result.metadata.get("document_type")
            if doc_type and doc_type not in allowed_types:
                return False
        
        if "date_range" in filters:
            date_range = filters["date_range"]
            doc_date = result.metadata.get("created_date")
            if doc_date:
                if doc_date < date_range.get("start") or doc_date > date_range.get("end"):
                    return False
        
        return True
    
    async def _generate_suggestions(
        self, 
        request: SearchRequest, 
        results: List[SearchResult]
    ) -> List[DocumentSuggestion]:
        """Generate document suggestions based on search results"""
        if not self.suggestion_engine or not results:
            return []
        
        try:
            # Use the top result as the base for suggestions
            primary_result = results[0]
            
            # Get search history for context
            search_history = []
            if self.history_manager and request.user_context:
                search_history = await self.history_manager.get_user_search_history(
                    request.user_context.user_id, limit=10
                )
            
            # Generate suggestions
            suggestions = await self.suggestion_engine.get_suggestions(
                current_document_id=primary_result.document_id,
                user_context=request.user_context,
                search_history=search_history,
                exclude_documents={r.document_id for r in results}
            )
            
            return suggestions
            
        except Exception as e:
            self.logger.error(f"Failed to generate suggestions: {e}")
            return []
    
    async def _generate_search_insights(
        self,
        request: SearchRequest,
        response: SearchResponse,
        query_analysis: QueryAnalysis
    ) -> Dict[str, Any]:
        """Generate insights about the search results and process"""
        insights = {
            "query_complexity": len(query_analysis.keywords) + len(query_analysis.entities),
            "result_diversity": len(set(r.metadata.get("document_type", "unknown") for r in response.results)),
            "confidence_distribution": {
                "high": len([r for r in response.results if r.combined_score > 0.7]),
                "medium": len([r for r in response.results if 0.4 <= r.combined_score <= 0.7]),
                "low": len([r for r in response.results if r.combined_score < 0.4])
            }
        }
        
        # Add personalization insights
        if request.user_context and self.config.enable_personalization:
            insights["personalized"] = True
            insights["user_expertise_level"] = request.user_context.expertise_level
            insights["search_history_influence"] = len(request.user_context.search_history) > 5
        
        return insights
    
    async def _record_search_event(
        self,
        request: SearchRequest,
        response: SearchResponse,
        strategy: SearchStrategy
    ):
        """Record search event in history and analytics"""
        try:
            # Create search query record
            search_query = SearchQuery(
                user_id=request.user_context.user_id if request.user_context else None,
                session_id=request.session_id,
                query_text=request.query,
                query_analysis=response.query_analysis,
                search_strategy_used=strategy.value,
                total_results=response.total_results,
                processing_time=response.processing_time,
                user_context=request.user_context
            )
            
            # Record in history manager
            if self.history_manager:
                await self.history_manager.record_search_query(
                    search_query, request.session_id
                )
            
            # Record in analytics engine
            if self.analytics_engine:
                await self.analytics_engine.record_search_event(
                    search_query, response.results, response.processing_time, strategy
                )
                
        except Exception as e:
            self.logger.error(f"Failed to record search event: {e}")
    
    async def _check_cache(self, request: SearchRequest) -> Optional[SearchResponse]:
        """Check if search results are cached"""
        cache_key = self._generate_cache_key(request)
        
        if cache_key in self._result_cache:
            cached_response, cached_time = self._result_cache[cache_key]
            
            # Check if cache is still valid
            if datetime.utcnow() - cached_time < self.config.result_cache_ttl:
                # Mark as cache hit
                cached_response.cache_hit = True
                cached_response.timestamp = datetime.utcnow()
                return cached_response
            else:
                # Remove expired cache
                del self._result_cache[cache_key]
        
        return None
    
    async def _cache_results(self, request: SearchRequest, response: SearchResponse):
        """Cache search results"""
        cache_key = self._generate_cache_key(request)
        self._result_cache[cache_key] = (response, datetime.utcnow())
    
    def _generate_cache_key(self, request: SearchRequest) -> str:
        """Generate cache key for search request"""
        # Simple cache key - in production, would be more sophisticated
        key_components = [
            request.query.lower().strip(),
            str(request.strategy.value if request.strategy else "auto"),
            str(request.limit),
            str(sorted(request.filters.items()) if request.filters else "")
        ]
        
        # Add user context hash if personalization is enabled
        if self.config.enable_personalization and request.user_context:
            user_key = f"{request.user_context.user_id}_{request.user_context.expertise_level}"
            key_components.append(user_key)
        
        return "|".join(key_components)
    
    def _update_performance_stats(self, processing_time: float):
        """Update performance statistics"""
        self._performance_stats['total_queries'] += 1
        
        # Update average response time with exponential moving average
        alpha = 0.1
        current_avg = self._performance_stats['average_response_time']
        self._performance_stats['average_response_time'] = (
            alpha * processing_time + (1 - alpha) * current_avg
        )
    
    async def _cleanup_cache_periodically(self):
        """Background task to clean up expired cache entries"""
        while True:
            try:
                now = datetime.utcnow()
                expired_keys = []
                
                for cache_key, (_, cached_time) in self._result_cache.items():
                    if now - cached_time > self.config.result_cache_ttl:
                        expired_keys.append(cache_key)
                
                # Remove expired entries
                for key in expired_keys:
                    del self._result_cache[key]
                
                if expired_keys:
                    self.logger.debug(f"Cleaned up {len(expired_keys)} expired cache entries")
                
                # Sleep for 5 minutes before next cleanup
                await asyncio.sleep(300)
                
            except Exception as e:
                self.logger.error(f"Error in cache cleanup: {e}")
                await asyncio.sleep(300)
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get current performance statistics"""
        stats = self._performance_stats.copy()
        
        # Add cache statistics
        stats['cache_size'] = len(self._result_cache)
        stats['cache_hit_rate'] = (
            stats['cache_hits'] / max(1, stats['total_queries'])
        )
        
        return stats
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on all components"""
        health = {
            "status": "healthy",
            "components": {},
            "timestamp": datetime.utcnow().isoformat()
        }
        
        # Check hybrid search
        try:
            # Simple test query
            test_results = await self.hybrid_search.search("test", limit=1)
            health["components"]["hybrid_search"] = "healthy"
        except Exception as e:
            health["components"]["hybrid_search"] = f"unhealthy: {e}"
            health["status"] = "degraded"
        
        # Check optional components
        if self.history_manager:
            try:
                # Test database connection
                await self.history_manager.get_query_suggestions("test", limit=1)
                health["components"]["search_history"] = "healthy"
            except Exception as e:
                health["components"]["search_history"] = f"unhealthy: {e}"
                health["status"] = "degraded"
        
        if self.analytics_engine:
            health["components"]["analytics"] = "healthy"  # Simplified check
        
        return health
    
    async def close(self):
        """Cleanup and close all connections"""
        try:
            if self.history_manager:
                await self.history_manager.close()
            
            if self.analytics_engine:
                await self.analytics_engine.close()
            
            if self.hybrid_search:
                await self.hybrid_search.close()
            
            self.logger.info("Semantic search engine closed successfully")
            
        except Exception as e:
            self.logger.error(f"Error closing semantic search engine: {e}")