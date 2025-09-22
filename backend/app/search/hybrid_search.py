"""
Hybrid search implementation combining vector similarity and keyword search.
Provides sophisticated multi-modal search capabilities for document retrieval.
"""

import logging
import asyncio
from typing import Dict, List, Set, Optional, Tuple, Union, Any
from dataclasses import dataclass
from enum import Enum
import math
import re

import asyncpg
import numpy as np
from pydantic import BaseModel, Field
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)


class SearchStrategy(str, Enum):
    """Search strategy types"""
    VECTOR_ONLY = "vector_only"
    KEYWORD_ONLY = "keyword_only"
    HYBRID_BALANCED = "hybrid_balanced"
    VECTOR_FIRST = "vector_first"
    KEYWORD_FIRST = "keyword_first"
    ADAPTIVE = "adaptive"


class RankingMethod(str, Enum):
    """Ranking method types"""
    COSINE_SIMILARITY = "cosine_similarity"
    BM25 = "bm25"
    TF_IDF = "tf_idf"
    COMBINED_SCORE = "combined_score"
    RRF = "reciprocal_rank_fusion"  # Reciprocal Rank Fusion


@dataclass
class SearchResult:
    """Individual search result"""
    document_id: str
    title: str
    content: str
    score: float
    vector_score: Optional[float] = None
    keyword_score: Optional[float] = None
    combined_score: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None
    highlights: Optional[List[str]] = None
    ranking_factors: Optional[Dict[str, float]] = None


class HybridSearchConfig(BaseModel):
    """Configuration for hybrid search"""
    
    # Search strategy
    default_strategy: SearchStrategy = SearchStrategy.HYBRID_BALANCED
    enable_adaptive_strategy: bool = True
    
    # Vector search settings
    vector_similarity_threshold: float = 0.5
    max_vector_results: int = 100
    vector_weight: float = 0.6
    
    # Keyword search settings
    keyword_weight: float = 0.4
    max_keyword_results: int = 100
    min_keyword_score: float = 0.1
    
    # BM25 parameters
    bm25_k1: float = 1.5
    bm25_b: float = 0.75
    
    # Result fusion
    fusion_method: RankingMethod = RankingMethod.RRF
    max_combined_results: int = 50
    
    # Performance settings
    enable_caching: bool = True
    cache_ttl_seconds: int = 300
    enable_query_expansion: bool = True
    
    # Text processing
    enable_stemming: bool = True
    enable_stopword_removal: bool = True
    min_query_length: int = 2
    max_query_length: int = 1000


class QueryAnalysis(BaseModel):
    """Analysis of search query characteristics"""
    query: str
    cleaned_query: str
    tokens: List[str]
    entity_count: int
    is_semantic_query: bool
    is_keyword_query: bool
    query_length: int
    language: str = "en"
    suggested_strategy: SearchStrategy
    confidence: float


class HybridSearchEngine:
    """
    Advanced hybrid search engine combining vector similarity and keyword search.
    
    Features:
    - Multiple search strategies (vector, keyword, hybrid)
    - Adaptive strategy selection based on query analysis
    - BM25 and TF-IDF keyword scoring
    - Reciprocal Rank Fusion for result combination
    - Query expansion and preprocessing
    - Performance optimization with caching
    """
    
    def __init__(self, config: HybridSearchConfig = None, db_pool: asyncpg.Pool = None):
        self.config = config or HybridSearchConfig()
        self.db_pool = db_pool
        
        # Initialize components
        self._tfidf_vectorizer = None
        self._query_cache: Dict[str, List[SearchResult]] = {}
        self._stats = {
            "total_searches": 0,
            "vector_searches": 0,
            "keyword_searches": 0,
            "hybrid_searches": 0,
            "cache_hits": 0,
            "avg_response_time": 0.0
        }
        
        # Stopwords for query processing
        self._stopwords = {
            'a', 'an', 'and', 'are', 'as', 'at', 'be', 'been', 'by', 'for', 
            'from', 'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that',
            'the', 'to', 'was', 'will', 'with', 'the', 'this', 'but', 'they',
            'have', 'had', 'what', 'said', 'each', 'which', 'she', 'do',
            'how', 'their', 'if', 'up', 'out', 'many', 'then', 'them'
        }
        
    async def search(
        self,
        query: str,
        strategy: Optional[SearchStrategy] = None,
        filters: Optional[Dict[str, Any]] = None,
        limit: int = 20,
        user_context: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """
        Perform hybrid search with multiple strategies.
        
        Args:
            query: Search query string
            strategy: Search strategy to use (None for auto-detection)
            filters: Additional filters (document_type, date_range, etc.)
            limit: Maximum number of results
            user_context: User context for personalization
            
        Returns:
            List of search results sorted by relevance
        """
        if not query or len(query.strip()) < self.config.min_query_length:
            return []
        
        # Update stats
        self._stats["total_searches"] += 1
        
        # Check cache first
        cache_key = self._generate_cache_key(query, strategy, filters, limit)
        if self.config.enable_caching and cache_key in self._query_cache:
            self._stats["cache_hits"] += 1
            return self._query_cache[cache_key]
        
        try:
            # Analyze query characteristics
            query_analysis = await self._analyze_query(query, user_context)
            
            # Select search strategy
            final_strategy = strategy or self._select_strategy(query_analysis, user_context)
            
            # Execute search based on strategy
            results = await self._execute_search(
                query, query_analysis, final_strategy, filters, limit, user_context
            )
            
            # Cache results
            if self.config.enable_caching:
                self._query_cache[cache_key] = results
            
            return results
        
        except Exception as e:
            logger.error(f"Error in hybrid search: {e}")
            return []
    
    async def _analyze_query(
        self, 
        query: str, 
        user_context: Optional[Dict[str, Any]] = None
    ) -> QueryAnalysis:
        """Analyze query characteristics to determine optimal search strategy"""
        
        # Clean and tokenize query
        cleaned_query = self._clean_query(query)
        tokens = self._tokenize_query(cleaned_query)
        
        # Analyze query characteristics
        entity_count = len([t for t in tokens if t.istitle()])
        query_length = len(tokens)
        
        # Determine if query is more semantic or keyword-oriented
        is_semantic_query = self._is_semantic_query(tokens)
        is_keyword_query = self._is_keyword_query(tokens)
        
        # Suggest optimal strategy
        suggested_strategy, confidence = self._suggest_strategy(
            tokens, is_semantic_query, is_keyword_query, query_length, user_context
        )
        
        return QueryAnalysis(
            query=query,
            cleaned_query=cleaned_query,
            tokens=tokens,
            entity_count=entity_count,
            is_semantic_query=is_semantic_query,
            is_keyword_query=is_keyword_query,
            query_length=query_length,
            suggested_strategy=suggested_strategy,
            confidence=confidence
        )
    
    def _clean_query(self, query: str) -> str:
        """Clean and normalize query string"""
        # Remove extra whitespace and special characters
        cleaned = re.sub(r'[^\w\s\-]', ' ', query.lower())
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        
        # Truncate if too long
        if len(cleaned) > self.config.max_query_length:
            cleaned = cleaned[:self.config.max_query_length]
        
        return cleaned
    
    def _tokenize_query(self, query: str) -> List[str]:
        """Tokenize query into meaningful terms"""
        tokens = query.split()
        
        # Remove stopwords if enabled
        if self.config.enable_stopword_removal:
            tokens = [token for token in tokens if token not in self._stopwords]
        
        # Basic stemming if enabled
        if self.config.enable_stemming:
            tokens = [self._simple_stem(token) for token in tokens]
        
        return [token for token in tokens if len(token) >= 2]
    
    def _simple_stem(self, word: str) -> str:
        """Simple stemming algorithm"""
        # Basic suffix removal
        suffixes = ['ing', 'ed', 'er', 'est', 'ly', 's']
        for suffix in suffixes:
            if word.endswith(suffix) and len(word) > len(suffix) + 2:
                return word[:-len(suffix)]
        return word
    
    def _is_semantic_query(self, tokens: List[str]) -> bool:
        """Determine if query is semantic in nature"""
        semantic_indicators = [
            'how', 'what', 'why', 'when', 'where', 'explain', 'describe',
            'compare', 'difference', 'relationship', 'similar', 'like',
            'about', 'regarding', 'concerning', 'related', 'means'
        ]
        
        semantic_count = sum(1 for token in tokens if token in semantic_indicators)
        return semantic_count > 0 or len(tokens) > 5
    
    def _is_keyword_query(self, tokens: List[str]) -> bool:
        """Determine if query is keyword-focused"""
        # Keyword queries typically have:
        # - Few tokens (1-3)
        # - Proper nouns or technical terms
        # - No question words
        
        if len(tokens) <= 3:
            return True
        
        proper_noun_count = sum(1 for token in tokens if token.istitle())
        return proper_noun_count > len(tokens) * 0.5
    
    def _suggest_strategy(
        self, 
        tokens: List[str],
        is_semantic: bool,
        is_keyword: bool,
        query_length: int,
        user_context: Optional[Dict[str, Any]]
    ) -> Tuple[SearchStrategy, float]:
        """Suggest optimal search strategy based on query analysis"""
        
        confidence = 0.5
        
        # Strong semantic indicators
        if is_semantic and query_length > 5:
            return SearchStrategy.VECTOR_FIRST, 0.8
        
        # Strong keyword indicators
        if is_keyword and query_length <= 3:
            return SearchStrategy.KEYWORD_FIRST, 0.8
        
        # Mixed queries
        if is_semantic and is_keyword:
            return SearchStrategy.HYBRID_BALANCED, 0.7
        
        # Consider user context
        if user_context:
            # If user typically does semantic searches
            if user_context.get('prefers_semantic', False):
                return SearchStrategy.VECTOR_FIRST, 0.6
            
            # If user typically does keyword searches
            if user_context.get('prefers_keyword', False):
                return SearchStrategy.KEYWORD_FIRST, 0.6
        
        # Default to balanced hybrid
        return SearchStrategy.HYBRID_BALANCED, confidence
    
    def _select_strategy(
        self, 
        query_analysis: QueryAnalysis,
        user_context: Optional[Dict[str, Any]]
    ) -> SearchStrategy:
        """Select final search strategy"""
        
        if self.config.enable_adaptive_strategy:
            return query_analysis.suggested_strategy
        else:
            return self.config.default_strategy
    
    async def _execute_search(
        self,
        query: str,
        query_analysis: QueryAnalysis,
        strategy: SearchStrategy,
        filters: Optional[Dict[str, Any]],
        limit: int,
        user_context: Optional[Dict[str, Any]]
    ) -> List[SearchResult]:
        """Execute search based on selected strategy"""
        
        if strategy == SearchStrategy.VECTOR_ONLY:
            self._stats["vector_searches"] += 1
            return await self._vector_search(query_analysis, filters, limit)
        
        elif strategy == SearchStrategy.KEYWORD_ONLY:
            self._stats["keyword_searches"] += 1
            return await self._keyword_search(query_analysis, filters, limit)
        
        elif strategy in [SearchStrategy.HYBRID_BALANCED, SearchStrategy.VECTOR_FIRST, SearchStrategy.KEYWORD_FIRST]:
            self._stats["hybrid_searches"] += 1
            return await self._hybrid_search(query_analysis, strategy, filters, limit)
        
        elif strategy == SearchStrategy.ADAPTIVE:
            # Adaptive strategy tries multiple approaches and combines results
            return await self._adaptive_search(query_analysis, filters, limit, user_context)
        
        else:
            # Fallback to hybrid balanced
            return await self._hybrid_search(query_analysis, SearchStrategy.HYBRID_BALANCED, filters, limit)
    
    async def _vector_search(
        self,
        query_analysis: QueryAnalysis,
        filters: Optional[Dict[str, Any]],
        limit: int
    ) -> List[SearchResult]:
        """Perform vector similarity search"""
        
        if not self.db_pool:
            return []
        
        try:
            # Generate query embedding (placeholder - would use actual embedding service)
            query_embedding = await self._generate_query_embedding(query_analysis.cleaned_query)
            if not query_embedding:
                return []
            
            # Build filter conditions
            filter_conditions, filter_params = self._build_filter_conditions(filters)
            
            # Construct vector search query
            query_sql = f"""
                SELECT 
                    d.id,
                    d.title,
                    d.content,
                    d.metadata,
                    1 - (d.embedding <=> $1) as similarity_score
                FROM documents d
                WHERE d.embedding IS NOT NULL
                    AND 1 - (d.embedding <=> $1) >= $2
                    {filter_conditions}
                ORDER BY d.embedding <=> $1
                LIMIT $3
            """
            
            params = [query_embedding, self.config.vector_similarity_threshold] + filter_params + [limit]
            
            async with self.db_pool.acquire() as conn:
                rows = await conn.fetch(query_sql, *params)
                
                results = []
                for row in rows:
                    result = SearchResult(
                        document_id=str(row['id']),
                        title=row['title'] or '',
                        content=row['content'] or '',
                        score=float(row['similarity_score']),
                        vector_score=float(row['similarity_score']),
                        metadata=row.get('metadata'),
                        ranking_factors={'vector_similarity': float(row['similarity_score'])}
                    )
                    results.append(result)
                
                return results
        
        except Exception as e:
            logger.error(f"Error in vector search: {e}")
            return []
    
    async def _keyword_search(
        self,
        query_analysis: QueryAnalysis,
        filters: Optional[Dict[str, Any]],
        limit: int
    ) -> List[SearchResult]:
        """Perform full-text keyword search using PostgreSQL"""
        
        if not self.db_pool:
            return []
        
        try:
            # Build search query for PostgreSQL full-text search
            search_query = ' & '.join(query_analysis.tokens)
            
            # Build filter conditions
            filter_conditions, filter_params = self._build_filter_conditions(filters)
            
            # Construct full-text search query
            query_sql = f"""
                SELECT 
                    d.id,
                    d.title,
                    d.content,
                    d.metadata,
                    ts_rank(d.search_vector, plainto_tsquery($1)) as rank_score,
                    ts_headline(d.content, plainto_tsquery($1), 'MaxWords=50, MinWords=25') as highlight
                FROM documents d
                WHERE d.search_vector @@ plainto_tsquery($1)
                    {filter_conditions}
                ORDER BY ts_rank(d.search_vector, plainto_tsquery($1)) DESC
                LIMIT $2
            """
            
            params = [search_query] + filter_params + [limit]
            
            async with self.db_pool.acquire() as conn:
                rows = await conn.fetch(query_sql, *params)
                
                results = []
                for row in rows:
                    # Calculate BM25 score (simplified)
                    bm25_score = self._calculate_bm25_score(
                        query_analysis.tokens, 
                        row['content'] or '', 
                        float(row['rank_score'])
                    )
                    
                    result = SearchResult(
                        document_id=str(row['id']),
                        title=row['title'] or '',
                        content=row['content'] or '',
                        score=bm25_score,
                        keyword_score=bm25_score,
                        metadata=row.get('metadata'),
                        highlights=[row['highlight']] if row.get('highlight') else None,
                        ranking_factors={'bm25_score': bm25_score, 'rank_score': float(row['rank_score'])}
                    )
                    results.append(result)
                
                return results
        
        except Exception as e:
            logger.error(f"Error in keyword search: {e}")
            return []
    
    async def _hybrid_search(
        self,
        query_analysis: QueryAnalysis,
        strategy: SearchStrategy,
        filters: Optional[Dict[str, Any]],
        limit: int
    ) -> List[SearchResult]:
        """Perform hybrid search combining vector and keyword results"""
        
        try:
            # Determine result limits for each search type
            if strategy == SearchStrategy.VECTOR_FIRST:
                vector_limit = min(self.config.max_vector_results, limit * 2)
                keyword_limit = min(self.config.max_keyword_results, limit)
            elif strategy == SearchStrategy.KEYWORD_FIRST:
                vector_limit = min(self.config.max_vector_results, limit)
                keyword_limit = min(self.config.max_keyword_results, limit * 2)
            else:  # HYBRID_BALANCED
                vector_limit = min(self.config.max_vector_results, limit)
                keyword_limit = min(self.config.max_keyword_results, limit)
            
            # Execute both searches concurrently
            vector_task = asyncio.create_task(
                self._vector_search(query_analysis, filters, vector_limit)
            )
            keyword_task = asyncio.create_task(
                self._keyword_search(query_analysis, filters, keyword_limit)
            )
            
            vector_results, keyword_results = await asyncio.gather(
                vector_task, keyword_task, return_exceptions=True
            )
            
            # Handle exceptions
            if isinstance(vector_results, Exception):
                logger.error(f"Vector search failed: {vector_results}")
                vector_results = []
            
            if isinstance(keyword_results, Exception):
                logger.error(f"Keyword search failed: {keyword_results}")
                keyword_results = []
            
            # Combine and rank results
            combined_results = await self._combine_search_results(
                vector_results, keyword_results, strategy, limit
            )
            
            return combined_results
        
        except Exception as e:
            logger.error(f"Error in hybrid search: {e}")
            return []
    
    async def _adaptive_search(
        self,
        query_analysis: QueryAnalysis,
        filters: Optional[Dict[str, Any]],
        limit: int,
        user_context: Optional[Dict[str, Any]]
    ) -> List[SearchResult]:
        """Adaptive search that tries multiple strategies and selects the best"""
        
        # Try multiple strategies
        strategies_to_try = [
            SearchStrategy.HYBRID_BALANCED,
            SearchStrategy.VECTOR_FIRST if query_analysis.is_semantic_query else SearchStrategy.KEYWORD_FIRST
        ]
        
        all_results = []
        
        for strategy in strategies_to_try:
            try:
                strategy_results = await self._hybrid_search(
                    query_analysis, strategy, filters, limit
                )
                
                # Add strategy information to results
                for result in strategy_results:
                    if not result.metadata:
                        result.metadata = {}
                    result.metadata['strategy'] = strategy.value
                
                all_results.extend(strategy_results)
                
            except Exception as e:
                logger.error(f"Error in adaptive search strategy {strategy}: {e}")
        
        # Remove duplicates and re-rank
        unique_results = self._deduplicate_results(all_results)
        
        # Sort by combined score
        unique_results.sort(key=lambda x: x.score, reverse=True)
        
        return unique_results[:limit]
    
    async def _combine_search_results(
        self,
        vector_results: List[SearchResult],
        keyword_results: List[SearchResult],
        strategy: SearchStrategy,
        limit: int
    ) -> List[SearchResult]:
        """Combine vector and keyword search results using various fusion methods"""
        
        if self.config.fusion_method == RankingMethod.RRF:
            return self._reciprocal_rank_fusion(vector_results, keyword_results, limit)
        
        elif self.config.fusion_method == RankingMethod.COMBINED_SCORE:
            return self._weighted_score_combination(vector_results, keyword_results, strategy, limit)
        
        else:
            # Fallback to weighted combination
            return self._weighted_score_combination(vector_results, keyword_results, strategy, limit)
    
    def _reciprocal_rank_fusion(
        self,
        vector_results: List[SearchResult],
        keyword_results: List[SearchResult],
        limit: int,
        k: int = 60
    ) -> List[SearchResult]:
        """
        Combine results using Reciprocal Rank Fusion (RRF).
        RRF is a method that combines multiple ranked lists without requiring
        the underlying scoring functions to be related.
        """
        
        # Create a dictionary to store combined scores
        doc_scores = {}
        doc_results = {}
        
        # Process vector results
        for rank, result in enumerate(vector_results, 1):
            doc_id = result.document_id
            rrf_score = 1 / (k + rank)
            
            doc_scores[doc_id] = doc_scores.get(doc_id, 0) + rrf_score * self.config.vector_weight
            doc_results[doc_id] = result
            
            # Update result with RRF info
            result.vector_score = result.score
            result.combined_score = doc_scores[doc_id]
            
            if not result.ranking_factors:
                result.ranking_factors = {}
            result.ranking_factors['vector_rrf'] = rrf_score * self.config.vector_weight
        
        # Process keyword results
        for rank, result in enumerate(keyword_results, 1):
            doc_id = result.document_id
            rrf_score = 1 / (k + rank)
            
            doc_scores[doc_id] = doc_scores.get(doc_id, 0) + rrf_score * self.config.keyword_weight
            
            if doc_id in doc_results:
                # Update existing result
                doc_results[doc_id].keyword_score = result.score
                doc_results[doc_id].combined_score = doc_scores[doc_id]
                
                if result.highlights:
                    if not doc_results[doc_id].highlights:
                        doc_results[doc_id].highlights = []
                    doc_results[doc_id].highlights.extend(result.highlights)
                
                doc_results[doc_id].ranking_factors['keyword_rrf'] = rrf_score * self.config.keyword_weight
            else:
                # New result from keyword search only
                result.keyword_score = result.score
                result.combined_score = doc_scores[doc_id]
                
                if not result.ranking_factors:
                    result.ranking_factors = {}
                result.ranking_factors['keyword_rrf'] = rrf_score * self.config.keyword_weight
                
                doc_results[doc_id] = result
        
        # Update final scores and sort
        final_results = list(doc_results.values())
        for result in final_results:
            result.score = result.combined_score or 0
        
        final_results.sort(key=lambda x: x.score, reverse=True)
        return final_results[:limit]
    
    def _weighted_score_combination(
        self,
        vector_results: List[SearchResult],
        keyword_results: List[SearchResult],
        strategy: SearchStrategy,
        limit: int
    ) -> List[SearchResult]:
        """Combine results using weighted score combination"""
        
        # Normalize scores to 0-1 range
        vector_results = self._normalize_scores(vector_results)
        keyword_results = self._normalize_scores(keyword_results)
        
        # Adjust weights based on strategy
        if strategy == SearchStrategy.VECTOR_FIRST:
            vector_weight = 0.7
            keyword_weight = 0.3
        elif strategy == SearchStrategy.KEYWORD_FIRST:
            vector_weight = 0.3
            keyword_weight = 0.7
        else:  # HYBRID_BALANCED
            vector_weight = self.config.vector_weight
            keyword_weight = self.config.keyword_weight
        
        # Combine results
        doc_scores = {}
        doc_results = {}
        
        # Process vector results
        for result in vector_results:
            doc_id = result.document_id
            weighted_score = result.score * vector_weight
            
            doc_scores[doc_id] = doc_scores.get(doc_id, 0) + weighted_score
            doc_results[doc_id] = result
            
            result.vector_score = result.score
        
        # Process keyword results
        for result in keyword_results:
            doc_id = result.document_id
            weighted_score = result.score * keyword_weight
            
            doc_scores[doc_id] = doc_scores.get(doc_id, 0) + weighted_score
            
            if doc_id in doc_results:
                # Update existing result
                doc_results[doc_id].keyword_score = result.score
                if result.highlights:
                    if not doc_results[doc_id].highlights:
                        doc_results[doc_id].highlights = []
                    doc_results[doc_id].highlights.extend(result.highlights)
            else:
                # New result from keyword search only
                result.keyword_score = result.score
                doc_results[doc_id] = result
        
        # Update final scores
        final_results = list(doc_results.values())
        for result in final_results:
            result.combined_score = doc_scores[result.document_id]
            result.score = result.combined_score
            
            # Add ranking factors
            if not result.ranking_factors:
                result.ranking_factors = {}
            result.ranking_factors.update({
                'vector_weight': vector_weight,
                'keyword_weight': keyword_weight,
                'final_score': result.score
            })
        
        # Sort and return
        final_results.sort(key=lambda x: x.score, reverse=True)
        return final_results[:limit]
    
    def _normalize_scores(self, results: List[SearchResult]) -> List[SearchResult]:
        """Normalize scores to 0-1 range"""
        if not results:
            return results
        
        scores = [result.score for result in results]
        min_score = min(scores)
        max_score = max(scores)
        
        if max_score > min_score:
            for result in results:
                result.score = (result.score - min_score) / (max_score - min_score)
        
        return results
    
    def _deduplicate_results(self, results: List[SearchResult]) -> List[SearchResult]:
        """Remove duplicate results based on document ID"""
        seen = set()
        unique_results = []
        
        for result in results:
            if result.document_id not in seen:
                seen.add(result.document_id)
                unique_results.append(result)
        
        return unique_results
    
    def _calculate_bm25_score(
        self, 
        query_tokens: List[str], 
        document: str, 
        base_score: float
    ) -> float:
        """Calculate BM25 score for document"""
        
        if not query_tokens or not document:
            return base_score
        
        # Simplified BM25 calculation
        # In production, this would use proper document frequency statistics
        
        doc_tokens = document.lower().split()
        doc_length = len(doc_tokens)
        
        if doc_length == 0:
            return base_score
        
        # Assumed average document length (would be calculated from corpus)
        avgdl = 100
        
        score = 0.0
        for token in query_tokens:
            # Term frequency in document
            tf = doc_tokens.count(token.lower())
            if tf == 0:
                continue
            
            # Simplified IDF (would use proper collection statistics)
            idf = math.log(1000 / max(1, tf))  # Assume collection size of 1000
            
            # BM25 formula
            numerator = tf * (self.config.bm25_k1 + 1)
            denominator = tf + self.config.bm25_k1 * (1 - self.config.bm25_b + self.config.bm25_b * doc_length / avgdl)
            
            score += idf * numerator / denominator
        
        return max(score, base_score)
    
    def _build_filter_conditions(self, filters: Optional[Dict[str, Any]]) -> Tuple[str, List[Any]]:
        """Build SQL filter conditions and parameters"""
        if not filters:
            return "", []
        
        conditions = []
        params = []
        param_count = 3  # Start after existing params
        
        for key, value in filters.items():
            if key == "document_type":
                conditions.append(f"AND d.document_type = ${param_count}")
                params.append(value)
                param_count += 1
            elif key == "date_range":
                if isinstance(value, dict) and "start" in value and "end" in value:
                    conditions.append(f"AND d.created_at BETWEEN ${param_count} AND ${param_count + 1}")
                    params.extend([value["start"], value["end"]])
                    param_count += 2
            elif key == "metadata":
                if isinstance(value, dict):
                    for meta_key, meta_value in value.items():
                        conditions.append(f"AND d.metadata->>'{meta_key}' = ${param_count}")
                        params.append(str(meta_value))
                        param_count += 1
        
        return " ".join(conditions), params
    
    def _generate_cache_key(
        self, 
        query: str, 
        strategy: Optional[SearchStrategy], 
        filters: Optional[Dict[str, Any]], 
        limit: int
    ) -> str:
        """Generate cache key for query"""
        import hashlib
        
        key_parts = [
            query.lower().strip(),
            str(strategy),
            str(sorted(filters.items())) if filters else "",
            str(limit)
        ]
        
        key_string = "|".join(key_parts)
        return hashlib.md5(key_string.encode()).hexdigest()
    
    async def _generate_query_embedding(self, query: str) -> Optional[List[float]]:
        """Generate embedding for query (placeholder implementation)"""
        # This would integrate with actual embedding service (OpenAI, etc.)
        # For now, return None to skip vector search
        return None
    
    def get_search_stats(self) -> Dict[str, Any]:
        """Get search engine statistics"""
        return self._stats.copy()
    
    def clear_cache(self):
        """Clear query cache"""
        self._query_cache.clear()
    
    def update_config(self, new_config: HybridSearchConfig):
        """Update search configuration"""
        self.config = new_config


# Global search engine instance
_search_engine: Optional[HybridSearchEngine] = None


def get_hybrid_search_engine() -> Optional[HybridSearchEngine]:
    """Get global hybrid search engine instance"""
    return _search_engine


def initialize_hybrid_search_engine(config: HybridSearchConfig, db_pool: asyncpg.Pool) -> HybridSearchEngine:
    """Initialize global hybrid search engine"""
    global _search_engine
    _search_engine = HybridSearchEngine(config, db_pool)
    return _search_engine