"""
Test suite for knowledge graph builder functionality.
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4, UUID
from datetime import datetime, timezone

from app.knowledge_graph.builder import (
    KnowledgeGraphBuilder,
    GraphBuildConfig,
    GraphBuildResult,
)
from app.knowledge_graph.entities import (
    EntityExtractor,
    ExtractedEntity,
    EntityType,
    EntityMention,
    ExtractionMethod,
)
from src.knowledge_graph.relationships import (
    RelationshipDetector,
    DetectedRelationship,
    RelationshipType,
)
from src.knowledge_graph.storage import (
    KnowledgeGraphStorage,
    GraphStorageConfig,
    GraphNode,
    GraphEdge,
)


@pytest.fixture
def mock_storage_config():
    """Mock storage configuration for testing"""
    return GraphStorageConfig(
        database_url="postgresql://test:test@localhost/test", max_connections=5
    )


@pytest.fixture
def test_config(mock_storage_config):
    """Test configuration for graph builder"""
    return GraphBuildConfig(
        storage=mock_storage_config,
        batch_size=10,
        max_concurrent_tasks=2,
        enable_realtime_updates=False,
    )


@pytest.fixture
def sample_entities():
    """Sample extracted entities for testing"""
    return [
        ExtractedEntity(
            name="Artificial Intelligence",
            entity_type=EntityType.CONCEPT,
            mentions=[
                EntityMention(
                    text="AI",
                    start_pos=0,
                    end_pos=2,
                    entity_type=EntityType.CONCEPT,
                    confidence=0.9,
                    context="AI is transforming industries",
                    method=ExtractionMethod.SPACY_NER,
                )
            ],
            confidence=0.9,
            aliases=["AI", "Machine Intelligence"],
        ),
        ExtractedEntity(
            name="Machine Learning",
            entity_type=EntityType.CONCEPT,
            mentions=[
                EntityMention(
                    text="Machine Learning",
                    start_pos=20,
                    end_pos=36,
                    entity_type=EntityType.CONCEPT,
                    confidence=0.85,
                    context="Machine Learning algorithms",
                    method=ExtractionMethod.SPACY_NER,
                )
            ],
            confidence=0.85,
            aliases=["ML"],
        ),
    ]


@pytest.fixture
def sample_relationships():
    """Sample detected relationships for testing"""
    return [
        DetectedRelationship(
            source_entity="Machine Learning",
            target_entity="Artificial Intelligence",
            relationship_type=RelationshipType.PART_OF,
            relationship_name="subset_of",
            strength=0.8,
            confidence=0.9,
            description="Machine Learning is part of Artificial Intelligence",
            evidence=[],
        )
    ]


@pytest.fixture
def sample_graph_nodes():
    """Sample graph nodes for testing"""
    return [
        GraphNode(
            id=uuid4(),
            name="Artificial Intelligence",
            entity_type=EntityType.CONCEPT,
            confidence=0.9,
            properties={"aliases": ["AI"]},
        ),
        GraphNode(
            id=uuid4(),
            name="Machine Learning",
            entity_type=EntityType.CONCEPT,
            confidence=0.85,
            properties={"aliases": ["ML"]},
        ),
    ]


class TestKnowledgeGraphBuilder:
    """Test cases for KnowledgeGraphBuilder"""

    @pytest.mark.asyncio
    async def test_initialization(self, test_config):
        """Test builder initialization"""
        builder = KnowledgeGraphBuilder(test_config)

        # Mock storage initialization
        with patch.object(KnowledgeGraphStorage, "initialize", return_value=True):
            success = await builder.initialize()
            assert success is True
            assert builder.storage is not None
            assert builder.query_engine is not None

    @pytest.mark.asyncio
    async def test_initialization_failure(self, test_config):
        """Test builder initialization failure"""
        builder = KnowledgeGraphBuilder(test_config)

        # Mock storage initialization failure
        with patch.object(KnowledgeGraphStorage, "initialize", return_value=False):
            success = await builder.initialize()
            assert success is False

    @pytest.mark.asyncio
    async def test_process_document_success(
        self, test_config, sample_entities, sample_relationships
    ):
        """Test successful document processing"""
        builder = KnowledgeGraphBuilder(test_config)

        # Mock dependencies
        builder.entity_extractor = AsyncMock()
        builder.entity_extractor.extract_entities.return_value = sample_entities

        builder.relationship_detector = AsyncMock()
        builder.relationship_detector.detect_relationships.return_value = (
            sample_relationships
        )

        builder.storage = AsyncMock()
        builder.storage.create_node.side_effect = [
            GraphNode(
                id=uuid4(), name="AI", entity_type=EntityType.CONCEPT, confidence=0.9
            ),
            GraphNode(
                id=uuid4(), name="ML", entity_type=EntityType.CONCEPT, confidence=0.85
            ),
        ]
        builder.storage.create_edge.return_value = GraphEdge(
            id=uuid4(),
            source_node_id=uuid4(),
            target_node_id=uuid4(),
            relationship_type=RelationshipType.PART_OF,
            strength=0.8,
        )
        builder.storage.get_node_by_name.return_value = None
        builder.storage.get_graph_statistics.return_value = {"nodes": 2, "edges": 1}

        # Test processing
        document_id = uuid4()
        result = await builder.process_document(
            text="AI and Machine Learning are related", document_id=document_id
        )

        assert result.success is True
        assert result.document_id == document_id
        assert result.entities_created == 2
        assert result.relationships_created == 1
        assert len(result.errors) == 0

    @pytest.mark.asyncio
    async def test_process_document_no_entities(self, test_config):
        """Test document processing with no entities extracted"""
        builder = KnowledgeGraphBuilder(test_config)

        # Mock no entities extracted
        builder.entity_extractor = AsyncMock()
        builder.entity_extractor.extract_entities.return_value = []

        document_id = uuid4()
        result = await builder.process_document(
            text="This text has no extractable entities", document_id=document_id
        )

        assert result.success is True
        assert result.entities_created == 0
        assert result.relationships_created == 0
        assert "No entities extracted" in result.warnings[0]

    @pytest.mark.asyncio
    async def test_merge_similar_entities(self, test_config):
        """Test entity similarity merging"""
        builder = KnowledgeGraphBuilder(test_config)
        builder.config.merge_similar_entities = True
        builder.config.entity_similarity_threshold = 0.8

        # Create similar entities
        entities = [
            ExtractedEntity(
                name="AI", entity_type=EntityType.CONCEPT, confidence=0.9, mentions=[]
            ),
            ExtractedEntity(
                name="Artificial Intelligence",
                entity_type=EntityType.CONCEPT,
                confidence=0.85,
                mentions=[],
                aliases=["AI"],
            ),
            ExtractedEntity(
                name="Machine Learning",
                entity_type=EntityType.CONCEPT,
                confidence=0.8,
                mentions=[],
            ),
        ]

        merged = await builder._merge_similar_entities(entities)

        # Should merge the first two entities
        assert len(merged) == 2

        # Check merged entity contains both names
        merged_ai = next(e for e in merged if "Artificial" in e.name or "AI" in e.name)
        assert "AI" in merged_ai.aliases or merged_ai.name == "AI"

    @pytest.mark.asyncio
    async def test_process_text_batch(self, test_config, sample_entities):
        """Test batch text processing"""
        builder = KnowledgeGraphBuilder(test_config)

        # Mock process_document to return success
        async def mock_process_document(text, doc_id, metadata=None, created_by=None):
            return GraphBuildResult(
                success=True,
                document_id=doc_id,
                entities_created=1,
                relationships_created=1,
            )

        builder.process_document = mock_process_document

        # Test batch processing
        texts = [
            ("Document 1 text", uuid4()),
            ("Document 2 text", uuid4()),
            ("Document 3 text", uuid4()),
        ]

        results = await builder.process_text_batch(texts)

        assert len(results) == 3
        assert all(r.success for r in results)
        assert sum(r.entities_created for r in results) == 3
        assert sum(r.relationships_created for r in results) == 3

    def test_calculate_entity_similarity(self, test_config):
        """Test entity similarity calculation"""
        builder = KnowledgeGraphBuilder(test_config)

        entity1 = ExtractedEntity(
            name="AI", entity_type=EntityType.CONCEPT, confidence=0.9, mentions=[]
        )

        entity2 = ExtractedEntity(
            name="Artificial Intelligence",
            entity_type=EntityType.CONCEPT,
            confidence=0.85,
            mentions=[],
            aliases=["AI"],
        )

        # Should have high similarity due to alias match
        similarity = builder._calculate_entity_similarity(entity1, entity2)
        assert similarity >= 0.8

        # Test exact name match
        entity3 = ExtractedEntity(
            name="AI", entity_type=EntityType.CONCEPT, confidence=0.8, mentions=[]
        )

        similarity_exact = builder._calculate_entity_similarity(entity1, entity3)
        assert similarity_exact == 1.0

        # Test different types (should be 0)
        entity4 = ExtractedEntity(
            name="AI", entity_type=EntityType.ORGANIZATION, confidence=0.8, mentions=[]
        )

        similarity_diff_type = builder._calculate_entity_similarity(entity1, entity4)
        assert similarity_diff_type == 0.0

    def test_merge_entities(self, test_config):
        """Test entity merging logic"""
        builder = KnowledgeGraphBuilder(test_config)

        entities = [
            ExtractedEntity(
                name="AI",
                entity_type=EntityType.CONCEPT,
                confidence=0.9,
                mentions=[],
                aliases=["Artificial Intelligence"],
                properties={"prop1": "value1"},
            ),
            ExtractedEntity(
                name="Artificial Intelligence",
                entity_type=EntityType.CONCEPT,
                confidence=0.85,
                mentions=[],
                aliases=["AI"],
                properties={"prop2": "value2"},
            ),
        ]

        merged = builder._merge_entities(entities)

        # Should use highest confidence entity as base
        assert merged.name == "AI"  # Higher confidence
        assert merged.confidence == (0.9 + 0.85) / 2  # Average confidence

        # Should combine aliases
        assert "Artificial Intelligence" in merged.aliases

        # Should combine properties
        assert "prop1" in merged.properties
        assert "prop2" in merged.properties

    @pytest.mark.asyncio
    async def test_get_graph_summary(self, test_config):
        """Test graph summary generation"""
        builder = KnowledgeGraphBuilder(test_config)

        # Mock storage
        mock_storage = AsyncMock()
        mock_storage.get_graph_statistics.return_value = {
            "total_nodes": 100,
            "total_edges": 150,
            "entity_types": 5,
        }
        builder.storage = mock_storage

        summary = await builder.get_graph_summary()

        assert "basic_statistics" in summary
        assert "last_updated" in summary
        assert "cache_status" in summary
        assert summary["basic_statistics"]["total_nodes"] == 100

    @pytest.mark.asyncio
    async def test_get_graph_summary_no_storage(self, test_config):
        """Test graph summary when storage not initialized"""
        builder = KnowledgeGraphBuilder(test_config)

        summary = await builder.get_graph_summary()

        assert "error" in summary
        assert summary["error"] == "Storage not initialized"

    def test_clear_caches(self, test_config):
        """Test cache clearing"""
        builder = KnowledgeGraphBuilder(test_config)

        # Add some test data to caches
        builder._entity_cache["test"] = uuid4()
        builder._similarity_cache["test"] = [(uuid4(), 0.8)]

        assert len(builder._entity_cache) > 0
        assert len(builder._similarity_cache) > 0

        builder.clear_caches()

        assert len(builder._entity_cache) == 0
        assert len(builder._similarity_cache) == 0

    @pytest.mark.asyncio
    async def test_rebuild_graph_from_documents(self, test_config):
        """Test graph rebuilding from documents"""
        builder = KnowledgeGraphBuilder(test_config)

        # Mock process_text_batch
        async def mock_process_batch(texts, metadata=None, created_by=None):
            return [
                GraphBuildResult(
                    success=True,
                    document_id=doc_id,
                    entities_created=2,
                    relationships_created=1,
                    processing_time_seconds=1.0,
                )
                for _, doc_id in texts
            ]

        builder.process_text_batch = mock_process_batch

        # Test rebuild
        documents = [("Document 1", uuid4()), ("Document 2", uuid4())]

        results = await builder.rebuild_graph_from_documents(documents)

        assert len(results) == 2
        assert all(r.success for r in results)
        assert sum(r.entities_created for r in results) == 4
        assert sum(r.relationships_created for r in results) == 2


class TestGraphBuildConfig:
    """Test cases for GraphBuildConfig"""

    def test_default_config(self, mock_storage_config):
        """Test default configuration values"""
        config = GraphBuildConfig(storage=mock_storage_config)

        assert config.batch_size == 50
        assert config.max_concurrent_tasks == 5
        assert config.enable_realtime_updates is True
        assert config.merge_similar_entities is True
        assert config.entity_similarity_threshold == 0.85

    def test_custom_config(self, mock_storage_config):
        """Test custom configuration values"""
        config = GraphBuildConfig(
            storage=mock_storage_config,
            batch_size=25,
            max_concurrent_tasks=3,
            enable_realtime_updates=False,
            entity_similarity_threshold=0.9,
        )

        assert config.batch_size == 25
        assert config.max_concurrent_tasks == 3
        assert config.enable_realtime_updates is False
        assert config.entity_similarity_threshold == 0.9


class TestGraphBuildResult:
    """Test cases for GraphBuildResult"""

    def test_result_creation(self):
        """Test result object creation"""
        document_id = uuid4()

        result = GraphBuildResult(
            success=True,
            document_id=document_id,
            entities_created=5,
            relationships_created=3,
            processing_time_seconds=2.5,
        )

        assert result.success is True
        assert result.document_id == document_id
        assert result.entities_created == 5
        assert result.relationships_created == 3
        assert result.processing_time_seconds == 2.5
        assert len(result.errors) == 0
        assert len(result.warnings) == 0

    def test_result_with_errors(self):
        """Test result object with errors and warnings"""
        result = GraphBuildResult(
            success=False,
            errors=["Entity extraction failed", "Database error"],
            warnings=["Low confidence entities detected"],
        )

        assert result.success is False
        assert len(result.errors) == 2
        assert len(result.warnings) == 1
        assert "Entity extraction failed" in result.errors
