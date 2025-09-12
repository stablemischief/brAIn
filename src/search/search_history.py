"""
Search history tracking and management system.

This module provides comprehensive search history tracking, including query storage,
user behavior analysis, search pattern recognition, and personalization features.
"""

import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, field, asdict
from enum import Enum
from uuid import UUID, uuid4
import asyncio
from collections import defaultdict, Counter
import asyncpg

from .hybrid_search import QueryAnalysis, SearchResult
from .context_ranking import UserContext


class InteractionType(Enum):
    """Types of user interactions with search results"""
    CLICK = "click"
    HOVER = "hover"
    BOOKMARK = "bookmark"
    SHARE = "share"
    COPY = "copy"
    DOWNLOAD = "download"
    PRINT = "print"


class SearchIntent(Enum):
    """Inferred search intent categories"""
    INFORMATIONAL = "informational"
    NAVIGATIONAL = "navigational"
    TRANSACTIONAL = "transactional"
    RESEARCH = "research"
    COMPARISON = "comparison"
    TROUBLESHOOTING = "troubleshooting"


@dataclass
class SearchInteraction:
    """Individual interaction with a search result"""
    interaction_id: str = field(default_factory=lambda: str(uuid4()))
    document_id: str = ""
    interaction_type: InteractionType = InteractionType.CLICK
    timestamp: datetime = field(default_factory=datetime.utcnow)
    dwell_time: Optional[float] = None  # Time spent on document
    scroll_depth: Optional[float] = None  # Percentage of document scrolled
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SearchQuery:
    """Complete search query record with context and results"""
    query_id: str = field(default_factory=lambda: str(uuid4()))
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    query_text: str = ""
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    # Query analysis
    query_analysis: Optional[QueryAnalysis] = None
    inferred_intent: Optional[SearchIntent] = None
    
    # Search execution details
    search_strategy_used: Optional[str] = None
    total_results: int = 0
    processing_time: float = 0.0
    
    # User interactions
    interactions: List[SearchInteraction] = field(default_factory=list)
    
    # Context
    user_context: Optional[UserContext] = None
    device_info: Dict[str, str] = field(default_factory=dict)
    location_info: Dict[str, str] = field(default_factory=dict)
    
    # Refinements and follow-ups
    refinement_of: Optional[str] = None  # Query ID if this is a refinement
    follow_up_queries: List[str] = field(default_factory=list)
    
    # Performance metrics
    click_through_rate: float = 0.0
    satisfaction_score: Optional[float] = None  # Inferred satisfaction
    
    def add_interaction(self, interaction: SearchInteraction):
        """Add user interaction to this search query"""
        self.interactions.append(interaction)
        
        # Update click-through rate
        clicks = sum(1 for i in self.interactions if i.interaction_type == InteractionType.CLICK)
        self.click_through_rate = clicks / max(1, self.total_results) if self.total_results > 0 else 0.0


@dataclass
class SearchSession:
    """Search session containing multiple related queries"""
    session_id: str = field(default_factory=lambda: str(uuid4()))
    user_id: Optional[str] = None
    start_time: datetime = field(default_factory=datetime.utcnow)
    end_time: Optional[datetime] = None
    
    queries: List[SearchQuery] = field(default_factory=list)
    session_topic: Optional[str] = None
    session_intent: Optional[SearchIntent] = None
    
    # Session metrics
    total_queries: int = 0
    successful_queries: int = 0
    average_query_time: float = 0.0
    session_duration: float = 0.0


@dataclass
class UserSearchProfile:
    """Comprehensive user search behavior profile"""
    user_id: str
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_updated: datetime = field(default_factory=datetime.utcnow)
    
    # Query patterns
    total_queries: int = 0
    unique_queries: int = 0
    average_query_length: float = 0.0
    common_keywords: List[Tuple[str, int]] = field(default_factory=list)
    
    # Intent patterns
    intent_distribution: Dict[SearchIntent, int] = field(default_factory=dict)
    topic_interests: Dict[str, float] = field(default_factory=dict)
    
    # Interaction patterns
    click_patterns: Dict[str, float] = field(default_factory=dict)
    dwell_time_patterns: Dict[str, float] = field(default_factory=dict)
    
    # Temporal patterns
    active_hours: List[int] = field(default_factory=list)
    active_days: List[str] = field(default_factory=list)
    search_frequency: float = 0.0  # Queries per day
    
    # Preferences
    preferred_content_types: List[str] = field(default_factory=list)
    expertise_indicators: Dict[str, str] = field(default_factory=dict)
    
    # Performance metrics
    satisfaction_trend: List[float] = field(default_factory=list)
    engagement_score: float = 0.5


class SearchHistoryManager:
    """
    Comprehensive search history management with advanced analytics
    and personalization features.
    """
    
    def __init__(self, database_url: str):
        self.database_url = database_url
        self.connection_pool: Optional[asyncpg.Pool] = None
        
        # In-memory caches
        self._session_cache: Dict[str, SearchSession] = {}
        self._user_profile_cache: Dict[str, UserSearchProfile] = {}
        self._recent_queries_cache: Dict[str, List[SearchQuery]] = {}
        
    async def initialize(self) -> bool:
        """Initialize database connection and create tables"""
        try:
            self.connection_pool = await asyncpg.create_pool(
                self.database_url,
                min_size=5,
                max_size=20,
                command_timeout=30
            )
            
            await self._create_tables()
            return True
            
        except Exception as e:
            print(f"Failed to initialize SearchHistoryManager: {e}")
            return False
    
    async def _create_tables(self):
        """Create database tables for search history"""
        async with self.connection_pool.acquire() as conn:
            # Search queries table
            await conn.execute('''
                CREATE TABLE IF NOT EXISTS search_queries (
                    query_id UUID PRIMARY KEY,
                    user_id VARCHAR(255),
                    session_id UUID,
                    query_text TEXT NOT NULL,
                    timestamp TIMESTAMPTZ DEFAULT NOW(),
                    query_analysis JSONB,
                    inferred_intent VARCHAR(50),
                    search_strategy VARCHAR(100),
                    total_results INTEGER DEFAULT 0,
                    processing_time FLOAT DEFAULT 0.0,
                    click_through_rate FLOAT DEFAULT 0.0,
                    satisfaction_score FLOAT,
                    refinement_of UUID,
                    user_context JSONB,
                    device_info JSONB,
                    location_info JSONB,
                    created_at TIMESTAMPTZ DEFAULT NOW(),
                    updated_at TIMESTAMPTZ DEFAULT NOW()
                );
            ''')
            
            # Search interactions table
            await conn.execute('''
                CREATE TABLE IF NOT EXISTS search_interactions (
                    interaction_id UUID PRIMARY KEY,
                    query_id UUID NOT NULL REFERENCES search_queries(query_id),
                    document_id VARCHAR(255) NOT NULL,
                    interaction_type VARCHAR(50) NOT NULL,
                    timestamp TIMESTAMPTZ DEFAULT NOW(),
                    dwell_time FLOAT,
                    scroll_depth FLOAT,
                    metadata JSONB,
                    created_at TIMESTAMPTZ DEFAULT NOW()
                );
            ''')
            
            # Search sessions table
            await conn.execute('''
                CREATE TABLE IF NOT EXISTS search_sessions (
                    session_id UUID PRIMARY KEY,
                    user_id VARCHAR(255),
                    start_time TIMESTAMPTZ DEFAULT NOW(),
                    end_time TIMESTAMPTZ,
                    session_topic VARCHAR(255),
                    session_intent VARCHAR(50),
                    total_queries INTEGER DEFAULT 0,
                    successful_queries INTEGER DEFAULT 0,
                    average_query_time FLOAT DEFAULT 0.0,
                    session_duration FLOAT DEFAULT 0.0,
                    created_at TIMESTAMPTZ DEFAULT NOW(),
                    updated_at TIMESTAMPTZ DEFAULT NOW()
                );
            ''')
            
            # User search profiles table
            await conn.execute('''
                CREATE TABLE IF NOT EXISTS user_search_profiles (
                    user_id VARCHAR(255) PRIMARY KEY,
                    profile_data JSONB NOT NULL,
                    created_at TIMESTAMPTZ DEFAULT NOW(),
                    updated_at TIMESTAMPTZ DEFAULT NOW()
                );
            ''')
            
            # Create indexes for performance
            await conn.execute('CREATE INDEX IF NOT EXISTS idx_search_queries_user_timestamp ON search_queries(user_id, timestamp DESC);')
            await conn.execute('CREATE INDEX IF NOT EXISTS idx_search_queries_session ON search_queries(session_id);')
            await conn.execute('CREATE INDEX IF NOT EXISTS idx_search_interactions_query ON search_interactions(query_id);')
            await conn.execute('CREATE INDEX IF NOT EXISTS idx_search_interactions_document ON search_interactions(document_id);')
    
    async def record_search_query(
        self,
        query: SearchQuery,
        session_id: Optional[str] = None
    ) -> bool:
        """Record a search query with full context"""
        try:
            # Ensure session exists
            if session_id and session_id not in self._session_cache:
                await self._create_session(session_id, query.user_id)
            
            # Store in database
            async with self.connection_pool.acquire() as conn:
                await conn.execute('''
                    INSERT INTO search_queries (
                        query_id, user_id, session_id, query_text, timestamp,
                        query_analysis, inferred_intent, search_strategy,
                        total_results, processing_time, click_through_rate,
                        satisfaction_score, refinement_of, user_context,
                        device_info, location_info
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16)
                ''', 
                    query.query_id, query.user_id, session_id, query.query_text,
                    query.timestamp, 
                    json.dumps(asdict(query.query_analysis)) if query.query_analysis else None,
                    query.inferred_intent.value if query.inferred_intent else None,
                    query.search_strategy_used, query.total_results,
                    query.processing_time, query.click_through_rate,
                    query.satisfaction_score, query.refinement_of,
                    json.dumps(asdict(query.user_context)) if query.user_context else None,
                    json.dumps(query.device_info), json.dumps(query.location_info)
                )
            
            # Update caches
            if query.user_id:
                if query.user_id not in self._recent_queries_cache:
                    self._recent_queries_cache[query.user_id] = []
                self._recent_queries_cache[query.user_id].append(query)
                
                # Keep only recent queries in cache
                if len(self._recent_queries_cache[query.user_id]) > 100:
                    self._recent_queries_cache[query.user_id] = self._recent_queries_cache[query.user_id][-100:]
            
            # Update session
            if session_id and session_id in self._session_cache:
                self._session_cache[session_id].queries.append(query)
                self._session_cache[session_id].total_queries += 1
            
            return True
            
        except Exception as e:
            print(f"Failed to record search query: {e}")
            return False
    
    async def record_interaction(
        self,
        query_id: str,
        interaction: SearchInteraction
    ) -> bool:
        """Record user interaction with search result"""
        try:
            async with self.connection_pool.acquire() as conn:
                await conn.execute('''
                    INSERT INTO search_interactions (
                        interaction_id, query_id, document_id, interaction_type,
                        timestamp, dwell_time, scroll_depth, metadata
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                ''',
                    interaction.interaction_id, query_id, interaction.document_id,
                    interaction.interaction_type.value, interaction.timestamp,
                    interaction.dwell_time, interaction.scroll_depth,
                    json.dumps(interaction.metadata)
                )
            
            # Update query click-through rate
            await self._update_query_metrics(query_id)
            
            return True
            
        except Exception as e:
            print(f"Failed to record interaction: {e}")
            return False
    
    async def get_user_search_history(
        self,
        user_id: str,
        limit: int = 50,
        days_back: int = 30
    ) -> List[SearchQuery]:
        """Get user's search history with interactions"""
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=days_back)
            
            async with self.connection_pool.acquire() as conn:
                # Get queries
                query_rows = await conn.fetch('''
                    SELECT * FROM search_queries
                    WHERE user_id = $1 AND timestamp >= $2
                    ORDER BY timestamp DESC
                    LIMIT $3
                ''', user_id, cutoff_date, limit)
                
                queries = []
                for row in query_rows:
                    # Get interactions for this query
                    interaction_rows = await conn.fetch('''
                        SELECT * FROM search_interactions
                        WHERE query_id = $1
                        ORDER BY timestamp
                    ''', row['query_id'])
                    
                    interactions = [
                        SearchInteraction(
                            interaction_id=str(irow['interaction_id']),
                            document_id=irow['document_id'],
                            interaction_type=InteractionType(irow['interaction_type']),
                            timestamp=irow['timestamp'],
                            dwell_time=irow['dwell_time'],
                            scroll_depth=irow['scroll_depth'],
                            metadata=json.loads(irow['metadata']) if irow['metadata'] else {}
                        )
                        for irow in interaction_rows
                    ]
                    
                    # Reconstruct SearchQuery
                    query = SearchQuery(
                        query_id=str(row['query_id']),
                        user_id=row['user_id'],
                        session_id=str(row['session_id']) if row['session_id'] else None,
                        query_text=row['query_text'],
                        timestamp=row['timestamp'],
                        inferred_intent=SearchIntent(row['inferred_intent']) if row['inferred_intent'] else None,
                        search_strategy_used=row['search_strategy'],
                        total_results=row['total_results'],
                        processing_time=row['processing_time'],
                        interactions=interactions,
                        click_through_rate=row['click_through_rate'],
                        satisfaction_score=row['satisfaction_score'],
                        refinement_of=str(row['refinement_of']) if row['refinement_of'] else None,
                        device_info=json.loads(row['device_info']) if row['device_info'] else {},
                        location_info=json.loads(row['location_info']) if row['location_info'] else {}
                    )
                    
                    queries.append(query)
                
                return queries
                
        except Exception as e:
            print(f"Failed to get user search history: {e}")
            return []
    
    async def get_similar_queries(
        self,
        query_text: str,
        user_id: Optional[str] = None,
        limit: int = 10
    ) -> List[SearchQuery]:
        """Find similar queries using text similarity"""
        try:
            # Simple similarity using PostgreSQL full-text search
            async with self.connection_pool.acquire() as conn:
                base_query = '''
                    SELECT *, similarity(query_text, $1) as sim_score
                    FROM search_queries
                    WHERE similarity(query_text, $1) > 0.3
                '''
                
                params = [query_text]
                
                if user_id:
                    base_query += ' AND user_id = $2'
                    params.append(user_id)
                
                base_query += ' ORDER BY sim_score DESC LIMIT $' + str(len(params) + 1)
                params.append(limit)
                
                rows = await conn.fetch(base_query, *params)
                
                return [
                    SearchQuery(
                        query_id=str(row['query_id']),
                        user_id=row['user_id'],
                        query_text=row['query_text'],
                        timestamp=row['timestamp']
                    )
                    for row in rows
                ]
                
        except Exception as e:
            print(f"Failed to find similar queries: {e}")
            return []
    
    async def analyze_search_patterns(
        self,
        user_id: str,
        days_back: int = 90
    ) -> Dict[str, Any]:
        """Analyze user's search patterns and behavior"""
        queries = await self.get_user_search_history(user_id, limit=1000, days_back=days_back)
        
        if not queries:
            return {"error": "No search history found"}
        
        # Temporal patterns
        hour_distribution = defaultdict(int)
        day_distribution = defaultdict(int)
        
        # Query patterns
        query_lengths = []
        keywords = []
        intent_distribution = defaultdict(int)
        
        # Interaction patterns
        click_counts = []
        dwell_times = []
        successful_queries = 0
        
        for query in queries:
            # Temporal analysis
            hour_distribution[query.timestamp.hour] += 1
            day_distribution[query.timestamp.strftime("%A")] += 1
            
            # Query analysis
            query_lengths.append(len(query.query_text.split()))
            keywords.extend(query.query_text.lower().split())
            
            if query.inferred_intent:
                intent_distribution[query.inferred_intent] += 1
            
            # Interaction analysis
            click_count = len([i for i in query.interactions if i.interaction_type == InteractionType.CLICK])
            click_counts.append(click_count)
            
            if click_count > 0:
                successful_queries += 1
            
            # Dwell time analysis
            for interaction in query.interactions:
                if interaction.dwell_time:
                    dwell_times.append(interaction.dwell_time)
        
        # Calculate statistics
        keyword_counter = Counter(keywords)
        common_keywords = keyword_counter.most_common(20)
        
        avg_query_length = sum(query_lengths) / len(query_lengths) if query_lengths else 0
        avg_clicks_per_query = sum(click_counts) / len(click_counts) if click_counts else 0
        avg_dwell_time = sum(dwell_times) / len(dwell_times) if dwell_times else 0
        success_rate = successful_queries / len(queries) if queries else 0
        
        return {
            "summary": {
                "total_queries": len(queries),
                "unique_queries": len(set(q.query_text for q in queries)),
                "success_rate": success_rate,
                "average_query_length": avg_query_length,
                "average_clicks_per_query": avg_clicks_per_query,
                "average_dwell_time": avg_dwell_time
            },
            "temporal_patterns": {
                "most_active_hours": sorted(hour_distribution.items(), key=lambda x: x[1], reverse=True)[:5],
                "most_active_days": sorted(day_distribution.items(), key=lambda x: x[1], reverse=True)[:3],
                "search_frequency": len(queries) / days_back  # Queries per day
            },
            "content_patterns": {
                "common_keywords": common_keywords[:10],
                "intent_distribution": dict(intent_distribution),
                "query_length_distribution": {
                    "short": len([l for l in query_lengths if l <= 3]),
                    "medium": len([l for l in query_lengths if 4 <= l <= 7]),
                    "long": len([l for l in query_lengths if l > 7])
                }
            },
            "engagement_patterns": {
                "click_distribution": {
                    "no_clicks": click_counts.count(0),
                    "one_click": click_counts.count(1),
                    "multiple_clicks": len([c for c in click_counts if c > 1])
                },
                "dwell_time_ranges": {
                    "quick": len([t for t in dwell_times if t < 30]),
                    "moderate": len([t for t in dwell_times if 30 <= t < 180]),
                    "long": len([t for t in dwell_times if t >= 180])
                }
            }
        }
    
    async def get_query_suggestions(
        self,
        partial_query: str,
        user_id: Optional[str] = None,
        limit: int = 5
    ) -> List[str]:
        """Get query suggestions based on search history"""
        try:
            async with self.connection_pool.acquire() as conn:
                base_query = '''
                    SELECT query_text, COUNT(*) as frequency
                    FROM search_queries
                    WHERE query_text ILIKE $1
                '''
                
                params = [f'{partial_query}%']
                
                if user_id:
                    # Prioritize user's own queries
                    base_query = '''
                        SELECT query_text, 
                               COUNT(*) + CASE WHEN user_id = $2 THEN 10 ELSE 0 END as frequency
                        FROM search_queries
                        WHERE query_text ILIKE $1
                    '''
                    params.append(user_id)
                
                base_query += '''
                    GROUP BY query_text
                    ORDER BY frequency DESC
                    LIMIT $''' + str(len(params) + 1)
                
                params.append(limit)
                
                rows = await conn.fetch(base_query, *params)
                
                return [row['query_text'] for row in rows]
                
        except Exception as e:
            print(f"Failed to get query suggestions: {e}")
            return []
    
    async def _create_session(self, session_id: str, user_id: Optional[str]) -> SearchSession:
        """Create a new search session"""
        session = SearchSession(session_id=session_id, user_id=user_id)
        self._session_cache[session_id] = session
        
        # Store in database
        try:
            async with self.connection_pool.acquire() as conn:
                await conn.execute('''
                    INSERT INTO search_sessions (session_id, user_id, start_time)
                    VALUES ($1, $2, $3)
                ''', session_id, user_id, session.start_time)
        except Exception as e:
            print(f"Failed to create session in database: {e}")
        
        return session
    
    async def _update_query_metrics(self, query_id: str):
        """Update query metrics based on interactions"""
        try:
            async with self.connection_pool.acquire() as conn:
                # Count clicks and calculate CTR
                result = await conn.fetchrow('''
                    SELECT 
                        COUNT(*) FILTER (WHERE interaction_type = 'click') as clicks,
                        q.total_results
                    FROM search_interactions i
                    JOIN search_queries q ON i.query_id = q.query_id
                    WHERE i.query_id = $1
                    GROUP BY q.total_results
                ''', query_id)
                
                if result:
                    clicks = result['clicks']
                    total_results = result['total_results']
                    ctr = clicks / max(1, total_results)
                    
                    await conn.execute('''
                        UPDATE search_queries
                        SET click_through_rate = $2, updated_at = NOW()
                        WHERE query_id = $1
                    ''', query_id, ctr)
                    
        except Exception as e:
            print(f"Failed to update query metrics: {e}")
    
    async def close(self):
        """Close database connections"""
        if self.connection_pool:
            await self.connection_pool.close()