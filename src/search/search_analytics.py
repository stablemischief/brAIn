"""
Search analytics and performance monitoring system.

This module provides comprehensive analytics for search performance, user behavior,
query analysis, and system optimization insights.
"""

import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, field, asdict
from enum import Enum
from collections import defaultdict, Counter
from uuid import UUID
import asyncio
import statistics
import asyncpg

from .hybrid_search import SearchStrategy, SearchResult
from .search_history import SearchQuery, SearchInteraction, InteractionType, SearchIntent
from .context_ranking import RankingFactor
from .document_suggestions import SuggestionType


class MetricType(Enum):
    """Types of search metrics"""
    QUERY_PERFORMANCE = "query_performance"
    USER_ENGAGEMENT = "user_engagement"
    CONTENT_POPULARITY = "content_popularity"
    SEARCH_SUCCESS = "search_success"
    SYSTEM_PERFORMANCE = "system_performance"
    PERSONALIZATION_EFFECTIVENESS = "personalization_effectiveness"


class AggregationPeriod(Enum):
    """Time periods for metric aggregation"""
    HOURLY = "hourly"
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"


@dataclass
class SearchMetrics:
    """Core search performance metrics"""
    total_queries: int = 0
    unique_queries: int = 0
    successful_queries: int = 0
    average_response_time: float = 0.0
    
    # User engagement metrics
    click_through_rate: float = 0.0
    average_results_per_query: float = 0.0
    average_dwell_time: float = 0.0
    bounce_rate: float = 0.0
    
    # Query characteristics
    average_query_length: float = 0.0
    query_refinement_rate: float = 0.0
    
    # Content metrics
    documents_viewed: int = 0
    documents_interacted: int = 0
    
    # Search strategy effectiveness
    strategy_distribution: Dict[str, int] = field(default_factory=dict)
    strategy_success_rates: Dict[str, float] = field(default_factory=dict)


@dataclass
class UserEngagementMetrics:
    """User engagement and behavior metrics"""
    active_users: int = 0
    new_users: int = 0
    returning_users: int = 0
    
    # Session metrics
    average_session_duration: float = 0.0
    queries_per_session: float = 0.0
    pages_per_session: float = 0.0
    
    # Engagement quality
    deep_engagement_rate: float = 0.0  # Users with >5min session
    satisfaction_score: float = 0.0
    user_retention_rate: float = 0.0


@dataclass
class ContentAnalytics:
    """Content performance analytics"""
    most_viewed_documents: List[Tuple[str, int]] = field(default_factory=list)
    trending_topics: List[Tuple[str, float]] = field(default_factory=list)
    content_gaps: List[str] = field(default_factory=list)
    
    # Document performance
    document_click_rates: Dict[str, float] = field(default_factory=dict)
    document_dwell_times: Dict[str, float] = field(default_factory=dict)
    document_bounce_rates: Dict[str, float] = field(default_factory=dict)


@dataclass
class SystemPerformanceMetrics:
    """System performance and resource utilization"""
    average_search_latency: float = 0.0
    p95_search_latency: float = 0.0
    p99_search_latency: float = 0.0
    
    # Resource utilization
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    database_query_time: float = 0.0
    
    # Error rates
    error_rate: float = 0.0
    timeout_rate: float = 0.0
    
    # Cache performance
    cache_hit_rate: float = 0.0
    cache_miss_rate: float = 0.0


@dataclass
class AnalyticsReport:
    """Comprehensive analytics report"""
    report_id: str
    period_start: datetime
    period_end: datetime
    generated_at: datetime = field(default_factory=datetime.utcnow)
    
    # Core metrics
    search_metrics: SearchMetrics = field(default_factory=SearchMetrics)
    engagement_metrics: UserEngagementMetrics = field(default_factory=UserEngagementMetrics)
    content_analytics: ContentAnalytics = field(default_factory=ContentAnalytics)
    system_metrics: SystemPerformanceMetrics = field(default_factory=SystemPerformanceMetrics)
    
    # Additional insights
    key_insights: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    anomalies: List[str] = field(default_factory=list)


class SearchAnalyticsEngine:
    """
    Comprehensive search analytics engine for monitoring performance,
    user behavior, and system health.
    """
    
    def __init__(self, database_url: str):
        self.database_url = database_url
        self.connection_pool: Optional[asyncpg.Pool] = None
        
        # Caches for real-time analytics
        self._metrics_cache: Dict[str, Any] = {}
        self._user_sessions: Dict[str, Dict] = {}
        self._real_time_metrics: Dict[str, float] = {}
        
        # Configuration
        self.cache_ttl = timedelta(minutes=5)
        self.batch_size = 1000
        
    async def initialize(self) -> bool:
        """Initialize analytics engine and database connection"""
        try:
            self.connection_pool = await asyncpg.create_pool(
                self.database_url,
                min_size=3,
                max_size=10,
                command_timeout=60
            )
            
            await self._create_analytics_tables()
            
            # Start background tasks
            asyncio.create_task(self._real_time_metrics_updater())
            
            return True
            
        except Exception as e:
            print(f"Failed to initialize SearchAnalyticsEngine: {e}")
            return False
    
    async def _create_analytics_tables(self):
        """Create analytics tables"""
        async with self.connection_pool.acquire() as conn:
            # Daily metrics aggregation
            await conn.execute('''
                CREATE TABLE IF NOT EXISTS daily_search_metrics (
                    date DATE PRIMARY KEY,
                    total_queries INTEGER DEFAULT 0,
                    unique_queries INTEGER DEFAULT 0,
                    successful_queries INTEGER DEFAULT 0,
                    average_response_time FLOAT DEFAULT 0.0,
                    click_through_rate FLOAT DEFAULT 0.0,
                    average_query_length FLOAT DEFAULT 0.0,
                    bounce_rate FLOAT DEFAULT 0.0,
                    metrics_data JSONB,
                    created_at TIMESTAMPTZ DEFAULT NOW(),
                    updated_at TIMESTAMPTZ DEFAULT NOW()
                );
            ''')
            
            # Hourly metrics for detailed analysis
            await conn.execute('''
                CREATE TABLE IF NOT EXISTS hourly_search_metrics (
                    hour_timestamp TIMESTAMPTZ PRIMARY KEY,
                    total_queries INTEGER DEFAULT 0,
                    successful_queries INTEGER DEFAULT 0,
                    average_response_time FLOAT DEFAULT 0.0,
                    error_count INTEGER DEFAULT 0,
                    metrics_data JSONB,
                    created_at TIMESTAMPTZ DEFAULT NOW()
                );
            ''')
            
            # User behavior analytics
            await conn.execute('''
                CREATE TABLE IF NOT EXISTS user_behavior_metrics (
                    user_id VARCHAR(255),
                    date DATE,
                    queries_count INTEGER DEFAULT 0,
                    sessions_count INTEGER DEFAULT 0,
                    total_dwell_time FLOAT DEFAULT 0.0,
                    documents_viewed INTEGER DEFAULT 0,
                    behavior_data JSONB,
                    created_at TIMESTAMPTZ DEFAULT NOW(),
                    PRIMARY KEY (user_id, date)
                );
            ''')
            
            # Content performance analytics
            await conn.execute('''
                CREATE TABLE IF NOT EXISTS content_performance_metrics (
                    document_id VARCHAR(255),
                    date DATE,
                    impressions INTEGER DEFAULT 0,
                    clicks INTEGER DEFAULT 0,
                    total_dwell_time FLOAT DEFAULT 0.0,
                    bounce_count INTEGER DEFAULT 0,
                    performance_data JSONB,
                    created_at TIMESTAMPTZ DEFAULT NOW(),
                    PRIMARY KEY (document_id, date)
                );
            ''')
            
            # Create indexes
            await conn.execute('CREATE INDEX IF NOT EXISTS idx_daily_metrics_date ON daily_search_metrics(date DESC);')
            await conn.execute('CREATE INDEX IF NOT EXISTS idx_hourly_metrics_timestamp ON hourly_search_metrics(hour_timestamp DESC);')
            await conn.execute('CREATE INDEX IF NOT EXISTS idx_user_behavior_date ON user_behavior_metrics(date DESC);')
    
    async def record_search_event(
        self,
        query: SearchQuery,
        results: List[SearchResult],
        processing_time: float,
        strategy_used: SearchStrategy
    ):
        """Record a search event for analytics"""
        try:
            # Update real-time metrics
            current_hour = datetime.utcnow().replace(minute=0, second=0, microsecond=0)
            
            await self._update_hourly_metrics(
                current_hour,
                query=query,
                results_count=len(results),
                processing_time=processing_time,
                strategy=strategy_used
            )
            
            # Update user session tracking
            if query.user_id:
                await self._update_user_session(query.user_id, query.session_id, query.timestamp)
            
            # Update content impression metrics
            for result in results:
                await self._record_content_impression(result.document_id, query.timestamp)
                
        except Exception as e:
            print(f"Failed to record search event: {e}")
    
    async def record_user_interaction(
        self,
        query_id: str,
        interaction: SearchInteraction,
        user_id: Optional[str] = None
    ):
        """Record user interaction for analytics"""
        try:
            # Update content interaction metrics
            await self._update_content_interaction(
                interaction.document_id,
                interaction.interaction_type,
                interaction.dwell_time,
                datetime.utcnow().date()
            )
            
            # Update user engagement tracking
            if user_id:
                await self._update_user_engagement(user_id, interaction)
                
        except Exception as e:
            print(f"Failed to record user interaction: {e}")
    
    async def generate_analytics_report(
        self,
        start_date: datetime,
        end_date: datetime,
        report_type: str = "comprehensive"
    ) -> AnalyticsReport:
        """Generate comprehensive analytics report"""
        report = AnalyticsReport(
            report_id=f"{report_type}_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}",
            period_start=start_date,
            period_end=end_date
        )
        
        try:
            # Generate search metrics
            report.search_metrics = await self._generate_search_metrics(start_date, end_date)
            
            # Generate engagement metrics
            report.engagement_metrics = await self._generate_engagement_metrics(start_date, end_date)
            
            # Generate content analytics
            report.content_analytics = await self._generate_content_analytics(start_date, end_date)
            
            # Generate system metrics
            report.system_metrics = await self._generate_system_metrics(start_date, end_date)
            
            # Generate insights and recommendations
            report.key_insights = await self._generate_insights(report)
            report.recommendations = await self._generate_recommendations(report)
            report.anomalies = await self._detect_anomalies(report)
            
        except Exception as e:
            print(f"Failed to generate analytics report: {e}")
            report.key_insights = [f"Error generating report: {str(e)}"]
        
        return report
    
    async def _generate_search_metrics(
        self,
        start_date: datetime,
        end_date: datetime
    ) -> SearchMetrics:
        """Generate search performance metrics"""
        metrics = SearchMetrics()
        
        async with self.connection_pool.acquire() as conn:
            # Get aggregated daily metrics
            daily_metrics = await conn.fetch('''
                SELECT 
                    SUM(total_queries) as total_queries,
                    SUM(unique_queries) as unique_queries,
                    SUM(successful_queries) as successful_queries,
                    AVG(average_response_time) as avg_response_time,
                    AVG(click_through_rate) as avg_ctr,
                    AVG(average_query_length) as avg_query_length,
                    AVG(bounce_rate) as avg_bounce_rate
                FROM daily_search_metrics
                WHERE date BETWEEN $1 AND $2
            ''', start_date.date(), end_date.date())
            
            if daily_metrics and daily_metrics[0]['total_queries']:
                row = daily_metrics[0]
                metrics.total_queries = row['total_queries'] or 0
                metrics.unique_queries = row['unique_queries'] or 0
                metrics.successful_queries = row['successful_queries'] or 0
                metrics.average_response_time = row['avg_response_time'] or 0.0
                metrics.click_through_rate = row['avg_ctr'] or 0.0
                metrics.average_query_length = row['avg_query_length'] or 0.0
                metrics.bounce_rate = row['avg_bounce_rate'] or 0.0
            
            # Get strategy distribution from queries
            strategy_stats = await conn.fetch('''
                SELECT 
                    search_strategy,
                    COUNT(*) as count,
                    AVG(CASE WHEN click_through_rate > 0 THEN 1.0 ELSE 0.0 END) as success_rate
                FROM search_queries
                WHERE timestamp BETWEEN $1 AND $2
                AND search_strategy IS NOT NULL
                GROUP BY search_strategy
            ''', start_date, end_date)
            
            for row in strategy_stats:
                strategy = row['search_strategy']
                metrics.strategy_distribution[strategy] = row['count']
                metrics.strategy_success_rates[strategy] = row['success_rate'] or 0.0
        
        return metrics
    
    async def _generate_engagement_metrics(
        self,
        start_date: datetime,
        end_date: datetime
    ) -> UserEngagementMetrics:
        """Generate user engagement metrics"""
        metrics = UserEngagementMetrics()
        
        async with self.connection_pool.acquire() as conn:
            # Get user activity stats
            user_stats = await conn.fetchrow('''
                SELECT 
                    COUNT(DISTINCT user_id) as total_users,
                    AVG(queries_count) as avg_queries_per_user,
                    AVG(sessions_count) as avg_sessions_per_user,
                    AVG(total_dwell_time) as avg_dwell_time
                FROM user_behavior_metrics
                WHERE date BETWEEN $1 AND $2
                AND user_id IS NOT NULL
            ''', start_date.date(), end_date.date())
            
            if user_stats:
                metrics.active_users = user_stats['total_users'] or 0
                
            # Get session statistics from search queries
            session_stats = await conn.fetchrow('''
                SELECT 
                    COUNT(DISTINCT session_id) as total_sessions,
                    AVG(queries_per_session) as avg_queries_per_session
                FROM (
                    SELECT 
                        session_id,
                        COUNT(*) as queries_per_session
                    FROM search_queries
                    WHERE timestamp BETWEEN $1 AND $2
                    AND session_id IS NOT NULL
                    GROUP BY session_id
                ) session_query_counts
            ''', start_date, end_date)
            
            if session_stats:
                metrics.queries_per_session = session_stats['avg_queries_per_session'] or 0.0
        
        return metrics
    
    async def _generate_content_analytics(
        self,
        start_date: datetime,
        end_date: datetime
    ) -> ContentAnalytics:
        """Generate content performance analytics"""
        analytics = ContentAnalytics()
        
        async with self.connection_pool.acquire() as conn:
            # Get most viewed documents
            popular_docs = await conn.fetch('''
                SELECT 
                    document_id,
                    SUM(impressions) as total_impressions,
                    SUM(clicks) as total_clicks,
                    CASE WHEN SUM(impressions) > 0 
                         THEN SUM(clicks)::FLOAT / SUM(impressions)
                         ELSE 0.0 
                    END as ctr
                FROM content_performance_metrics
                WHERE date BETWEEN $1 AND $2
                GROUP BY document_id
                ORDER BY total_impressions DESC
                LIMIT 20
            ''', start_date.date(), end_date.date())
            
            analytics.most_viewed_documents = [
                (row['document_id'], row['total_impressions'])
                for row in popular_docs
            ]
            
            # Calculate document performance metrics
            for row in popular_docs:
                doc_id = row['document_id']
                analytics.document_click_rates[doc_id] = row['ctr']
            
            # Get trending topics (would require topic extraction from queries)
            # This is simplified - in reality would analyze query topics
            analytics.trending_topics = [
                ("artificial intelligence", 0.8),
                ("machine learning", 0.7),
                ("data science", 0.6)
            ]
        
        return analytics
    
    async def _generate_system_metrics(
        self,
        start_date: datetime,
        end_date: datetime
    ) -> SystemPerformanceMetrics:
        """Generate system performance metrics"""
        metrics = SystemPerformanceMetrics()
        
        async with self.connection_pool.acquire() as conn:
            # Get performance statistics from hourly metrics
            perf_stats = await conn.fetch('''
                SELECT average_response_time
                FROM hourly_search_metrics
                WHERE hour_timestamp BETWEEN $1 AND $2
                AND average_response_time > 0
                ORDER BY average_response_time
            ''', start_date, end_date)
            
            if perf_stats:
                response_times = [row['average_response_time'] for row in perf_stats]
                
                metrics.average_search_latency = statistics.mean(response_times)
                
                if len(response_times) >= 20:  # Enough data for percentiles
                    sorted_times = sorted(response_times)
                    metrics.p95_search_latency = sorted_times[int(len(sorted_times) * 0.95)]
                    metrics.p99_search_latency = sorted_times[int(len(sorted_times) * 0.99)]
            
            # Get error rates
            error_stats = await conn.fetchrow('''
                SELECT 
                    SUM(error_count) as total_errors,
                    SUM(total_queries) as total_queries
                FROM hourly_search_metrics
                WHERE hour_timestamp BETWEEN $1 AND $2
            ''', start_date, end_date)
            
            if error_stats and error_stats['total_queries']:
                metrics.error_rate = (
                    error_stats['total_errors'] / error_stats['total_queries']
                )
        
        return metrics
    
    async def _generate_insights(self, report: AnalyticsReport) -> List[str]:
        """Generate key insights from analytics data"""
        insights = []
        
        # Search performance insights
        if report.search_metrics.click_through_rate < 0.1:
            insights.append("Low click-through rate indicates search results may not be meeting user needs")
        
        if report.search_metrics.query_refinement_rate > 0.3:
            insights.append("High query refinement rate suggests initial results are not satisfactory")
        
        # Engagement insights
        if report.engagement_metrics.bounce_rate > 0.7:
            insights.append("High bounce rate indicates users are not finding relevant content quickly")
        
        # Performance insights
        if report.system_metrics.average_search_latency > 1.0:
            insights.append("Search latency is above optimal range, consider performance optimizations")
        
        # Content insights
        if len(report.content_analytics.most_viewed_documents) > 0:
            top_doc = report.content_analytics.most_viewed_documents[0]
            insights.append(f"Most popular document: {top_doc[0]} with {top_doc[1]} views")
        
        return insights
    
    async def _generate_recommendations(self, report: AnalyticsReport) -> List[str]:
        """Generate optimization recommendations"""
        recommendations = []
        
        # Performance recommendations
        if report.system_metrics.average_search_latency > 0.5:
            recommendations.append("Consider implementing result caching to improve response times")
        
        # User experience recommendations
        if report.search_metrics.click_through_rate < 0.15:
            recommendations.append("Improve result ranking algorithm to show more relevant results first")
        
        if report.engagement_metrics.queries_per_session < 2:
            recommendations.append("Add related search suggestions to encourage exploration")
        
        # Content recommendations
        if report.content_analytics.trending_topics:
            top_topic = report.content_analytics.trending_topics[0][0]
            recommendations.append(f"Consider creating more content around trending topic: {top_topic}")
        
        return recommendations
    
    async def _detect_anomalies(self, report: AnalyticsReport) -> List[str]:
        """Detect anomalies in analytics data"""
        anomalies = []
        
        # Check for unusual patterns
        if report.search_metrics.bounce_rate > 0.8:
            anomalies.append("Unusually high bounce rate detected")
        
        if report.system_metrics.error_rate > 0.05:
            anomalies.append("Higher than normal error rate detected")
        
        if report.engagement_metrics.average_session_duration < 60:
            anomalies.append("Very short average session duration")
        
        return anomalies
    
    async def _update_hourly_metrics(
        self,
        hour_timestamp: datetime,
        query: SearchQuery,
        results_count: int,
        processing_time: float,
        strategy: SearchStrategy
    ):
        """Update hourly metrics with new search data"""
        async with self.connection_pool.acquire() as conn:
            await conn.execute('''
                INSERT INTO hourly_search_metrics (
                    hour_timestamp, total_queries, successful_queries, 
                    average_response_time, error_count, metrics_data
                ) VALUES ($1, 1, $2, $3, 0, $4)
                ON CONFLICT (hour_timestamp)
                DO UPDATE SET
                    total_queries = hourly_search_metrics.total_queries + 1,
                    successful_queries = hourly_search_metrics.successful_queries + $2,
                    average_response_time = (
                        hourly_search_metrics.average_response_time * 
                        hourly_search_metrics.total_queries + $3
                    ) / (hourly_search_metrics.total_queries + 1)
            ''', 
                hour_timestamp, 
                1 if results_count > 0 else 0,
                processing_time,
                json.dumps({"strategy": strategy.value})
            )
    
    async def _update_user_session(
        self,
        user_id: str,
        session_id: Optional[str],
        timestamp: datetime
    ):
        """Update user session tracking"""
        if user_id not in self._user_sessions:
            self._user_sessions[user_id] = {
                'session_start': timestamp,
                'last_activity': timestamp,
                'query_count': 0
            }
        
        session = self._user_sessions[user_id]
        session['last_activity'] = timestamp
        session['query_count'] += 1
    
    async def _record_content_impression(self, document_id: str, timestamp: datetime):
        """Record content impression for analytics"""
        date = timestamp.date()
        
        async with self.connection_pool.acquire() as conn:
            await conn.execute('''
                INSERT INTO content_performance_metrics (
                    document_id, date, impressions
                ) VALUES ($1, $2, 1)
                ON CONFLICT (document_id, date)
                DO UPDATE SET
                    impressions = content_performance_metrics.impressions + 1
            ''', document_id, date)
    
    async def _update_content_interaction(
        self,
        document_id: str,
        interaction_type: InteractionType,
        dwell_time: Optional[float],
        date: datetime
    ):
        """Update content interaction metrics"""
        async with self.connection_pool.acquire() as conn:
            if interaction_type == InteractionType.CLICK:
                await conn.execute('''
                    UPDATE content_performance_metrics
                    SET clicks = clicks + 1,
                        total_dwell_time = total_dwell_time + COALESCE($3, 0)
                    WHERE document_id = $1 AND date = $2
                ''', document_id, date, dwell_time or 0.0)
    
    async def _update_user_engagement(
        self,
        user_id: str,
        interaction: SearchInteraction
    ):
        """Update user engagement metrics"""
        date = interaction.timestamp.date()
        
        async with self.connection_pool.acquire() as conn:
            await conn.execute('''
                INSERT INTO user_behavior_metrics (
                    user_id, date, documents_viewed, total_dwell_time
                ) VALUES ($1, $2, 1, $3)
                ON CONFLICT (user_id, date)
                DO UPDATE SET
                    documents_viewed = user_behavior_metrics.documents_viewed + 1,
                    total_dwell_time = user_behavior_metrics.total_dwell_time + $3
            ''', user_id, date, interaction.dwell_time or 0.0)
    
    async def _real_time_metrics_updater(self):
        """Background task to update real-time metrics"""
        while True:
            try:
                # Update cache with latest metrics
                now = datetime.utcnow()
                current_hour = now.replace(minute=0, second=0, microsecond=0)
                
                # Update real-time query rate
                query_rate = await self._get_current_query_rate(current_hour)
                self._real_time_metrics['queries_per_minute'] = query_rate
                
                # Update real-time response time
                avg_response_time = await self._get_current_response_time(current_hour)
                self._real_time_metrics['average_response_time'] = avg_response_time
                
                await asyncio.sleep(60)  # Update every minute
                
            except Exception as e:
                print(f"Error updating real-time metrics: {e}")
                await asyncio.sleep(60)
    
    async def _get_current_query_rate(self, hour: datetime) -> float:
        """Get current query rate (queries per minute)"""
        async with self.connection_pool.acquire() as conn:
            result = await conn.fetchrow('''
                SELECT total_queries
                FROM hourly_search_metrics
                WHERE hour_timestamp = $1
            ''', hour)
            
            if result:
                return result['total_queries'] / 60.0  # Convert to per minute
            return 0.0
    
    async def _get_current_response_time(self, hour: datetime) -> float:
        """Get current average response time"""
        async with self.connection_pool.acquire() as conn:
            result = await conn.fetchrow('''
                SELECT average_response_time
                FROM hourly_search_metrics
                WHERE hour_timestamp = $1
            ''', hour)
            
            if result:
                return result['average_response_time'] or 0.0
            return 0.0
    
    def get_real_time_metrics(self) -> Dict[str, float]:
        """Get current real-time metrics"""
        return self._real_time_metrics.copy()
    
    async def export_report(
        self,
        report: AnalyticsReport,
        format: str = "json"
    ) -> str:
        """Export analytics report in specified format"""
        if format.lower() == "json":
            return json.dumps(asdict(report), default=str, indent=2)
        
        # Could add other formats (CSV, PDF, etc.)
        return json.dumps(asdict(report), default=str, indent=2)
    
    async def close(self):
        """Close database connections"""
        if self.connection_pool:
            await self.connection_pool.close()