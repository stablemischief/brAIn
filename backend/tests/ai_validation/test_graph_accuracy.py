"""
AI Validation Tests for Knowledge Graph Accuracy
Tests the knowledge graph system for entity extraction and relationship accuracy.
"""

import pytest
import json
import numpy as np
from pathlib import Path
from unittest.mock import MagicMock, patch
from typing import List, Dict, Tuple

# Import knowledge graph modules
from src.knowledge_graph.builder import KnowledgeGraphBuilder
from src.knowledge_graph.entities import EntityExtractor
from src.knowledge_graph.relationships import RelationshipDetector
from src.knowledge_graph.queries import GraphQueryEngine


@pytest.mark.ai
class TestKnowledgeGraphAccuracy:
    """Test suite for knowledge graph accuracy validation."""

    @pytest.fixture
    def kg_test_data(self):
        """Load knowledge graph test data from fixtures."""
        fixture_path = (
            Path(__file__).parent.parent
            / "fixtures"
            / "ai_test_data"
            / "knowledge_graph_test_data.json"
        )
        with open(fixture_path, "r") as f:
            return json.load(f)

    @pytest.fixture
    def entity_extractor(self):
        """Create EntityExtractor instance for testing."""
        return EntityExtractor(confidence_threshold=0.8, model="en_core_web_sm")

    @pytest.fixture
    def relationship_detector(self):
        """Create RelationshipDetector instance for testing."""
        return RelationshipDetector(confidence_threshold=0.8)

    @pytest.fixture
    def graph_builder(self, entity_extractor, relationship_detector):
        """Create KnowledgeGraphBuilder instance for testing."""
        return KnowledgeGraphBuilder(
            entity_extractor=entity_extractor,
            relationship_detector=relationship_detector,
        )

    def test_entity_extraction_accuracy(self, entity_extractor, kg_test_data):
        """Test entity extraction accuracy against expected results."""
        test_documents = kg_test_data["test_documents"]

        for doc in test_documents:
            # Extract entities from document content
            extracted_entities = entity_extractor.extract_entities(doc["content"])
            expected_entities = doc["expected_entities"]

            # Calculate accuracy metrics
            precision, recall, f1_score = self._calculate_entity_metrics(
                extracted_entities, expected_entities
            )

            # Assert minimum accuracy thresholds
            assert (
                precision >= 0.80
            ), f"Entity precision {precision:.3f} below threshold for doc {doc['id']}"
            assert (
                recall >= 0.75
            ), f"Entity recall {recall:.3f} below threshold for doc {doc['id']}"
            assert (
                f1_score >= 0.77
            ), f"Entity F1 score {f1_score:.3f} below threshold for doc {doc['id']}"

    def test_relationship_extraction_accuracy(
        self, relationship_detector, kg_test_data
    ):
        """Test relationship extraction accuracy."""
        test_documents = kg_test_data["test_documents"]

        for doc in test_documents:
            # First extract entities (needed for relationship detection)
            entities = [e["text"] for e in doc["expected_entities"]]

            # Extract relationships
            extracted_relationships = relationship_detector.extract_relationships(
                doc["content"], entities
            )
            expected_relationships = doc["expected_relationships"]

            # Calculate relationship accuracy
            precision, recall, f1_score = self._calculate_relationship_metrics(
                extracted_relationships, expected_relationships
            )

            # Assert minimum accuracy thresholds
            assert (
                precision >= 0.75
            ), f"Relationship precision {precision:.3f} below threshold for doc {doc['id']}"
            assert (
                recall >= 0.70
            ), f"Relationship recall {recall:.3f} below threshold for doc {doc['id']}"
            assert (
                f1_score >= 0.72
            ), f"Relationship F1 score {f1_score:.3f} below threshold for doc {doc['id']}"

    def test_graph_construction_accuracy(self, graph_builder, kg_test_data):
        """Test complete graph construction accuracy."""
        test_documents = kg_test_data["test_documents"]
        validation_metrics = kg_test_data["graph_metrics_validation"]

        # Build graph from test documents
        graph = graph_builder.build_graph([doc["content"] for doc in test_documents])

        # Validate graph metrics
        node_count = graph.get_node_count()
        relationship_count = graph.get_relationship_count()

        assert (
            node_count >= validation_metrics["expected_node_count"] * 0.8
        ), f"Node count {node_count} significantly below expected {validation_metrics['expected_node_count']}"

        assert (
            relationship_count
            >= validation_metrics["expected_relationship_count"] * 0.7
        ), f"Relationship count {relationship_count} below expected {validation_metrics['expected_relationship_count']}"

        # Test node type distribution
        node_types = graph.get_node_type_distribution()
        assert (
            node_types.get("ORGANIZATION", 0)
            >= validation_metrics["expected_organization_nodes"]
        )
        assert (
            node_types.get("PERSON", 0)
            >= validation_metrics["expected_person_nodes"] * 0.8
        )

    def test_graph_query_accuracy(self, graph_builder, kg_test_data):
        """Test graph query accuracy for relationship questions."""
        test_documents = kg_test_data["test_documents"]
        accuracy_tests = kg_test_data["relationship_accuracy_tests"]

        # Build graph
        graph = graph_builder.build_graph([doc["content"] for doc in test_documents])
        query_engine = GraphQueryEngine(graph)

        for test in accuracy_tests:
            # Execute query
            results = query_engine.query(test["query"])

            # Calculate accuracy
            accuracy = self._calculate_query_accuracy(results, test["expected_results"])

            assert (
                accuracy >= test["minimum_accuracy"]
            ), f"Query accuracy {accuracy:.3f} below threshold {test['minimum_accuracy']} for query: {test['query']}"

    def test_entity_confidence_scoring(self, entity_extractor, kg_test_data):
        """Test entity confidence scoring accuracy."""
        test_documents = kg_test_data["test_documents"]

        for doc in test_documents:
            extracted_entities = entity_extractor.extract_entities(doc["content"])

            for extracted in extracted_entities:
                # Find corresponding expected entity
                expected = self._find_matching_entity(
                    extracted, doc["expected_entities"]
                )

                if expected:
                    # Confidence should be within reasonable range of expected
                    confidence_diff = abs(
                        extracted["confidence"] - expected["confidence"]
                    )
                    assert (
                        confidence_diff <= 0.15
                    ), f"Confidence score difference {confidence_diff:.3f} too large for entity {extracted['text']}"

    def test_relationship_confidence_scoring(self, relationship_detector, kg_test_data):
        """Test relationship confidence scoring accuracy."""
        test_documents = kg_test_data["test_documents"]

        for doc in test_documents:
            entities = [e["text"] for e in doc["expected_entities"]]
            extracted_relationships = relationship_detector.extract_relationships(
                doc["content"], entities
            )

            for extracted in extracted_relationships:
                expected = self._find_matching_relationship(
                    extracted, doc["expected_relationships"]
                )

                if expected:
                    confidence_diff = abs(
                        extracted["confidence"] - expected["confidence"]
                    )
                    assert (
                        confidence_diff <= 0.20
                    ), f"Relationship confidence difference {confidence_diff:.3f} too large"

    def test_graph_connectivity_metrics(self, graph_builder, kg_test_data):
        """Test graph connectivity and clustering metrics."""
        test_documents = kg_test_data["test_documents"]
        validation_metrics = kg_test_data["graph_metrics_validation"]

        graph = graph_builder.build_graph([doc["content"] for doc in test_documents])

        # Calculate connectivity score
        connectivity_score = graph.calculate_connectivity_score()
        assert (
            connectivity_score >= validation_metrics["minimum_connectivity_score"]
        ), f"Connectivity score {connectivity_score:.3f} below minimum {validation_metrics['minimum_connectivity_score']}"

        # Calculate clustering coefficient
        clustering_coefficient = graph.calculate_clustering_coefficient()
        assert (
            clustering_coefficient
            >= validation_metrics["expected_clustering_coefficient"] * 0.8
        ), f"Clustering coefficient {clustering_coefficient:.3f} significantly below expected"

    @pytest.mark.performance
    def test_graph_building_performance(self, graph_builder):
        """Test knowledge graph building performance."""
        import time

        # Test documents of varying sizes
        test_documents = [
            "Apple Inc. was founded by Steve Jobs.",  # Small
            "Apple Inc. was founded by Steve Jobs, Steve Wozniak, and Ronald Wayne in 1976. "
            * 10,  # Medium
            "Apple Inc. was founded by Steve Jobs, Steve Wozniak, and Ronald Wayne in 1976. "
            * 100,  # Large
        ]

        for i, doc in enumerate(test_documents):
            start_time = time.time()
            graph = graph_builder.build_graph([doc])
            end_time = time.time()

            # Performance thresholds based on document size
            max_times = [0.5, 2.0, 10.0]  # seconds
            assert (
                end_time - start_time < max_times[i]
            ), f"Graph building took {end_time - start_time:.2f}s, expected < {max_times[i]}s"

    def test_entity_disambiguation(self, entity_extractor):
        """Test entity disambiguation accuracy."""
        # Test cases with ambiguous entities
        test_cases = [
            {
                "text": "Apple released a new iPhone model and Apple pie is popular in America.",
                "expected_disambiguation": {
                    "Apple": ["Apple Inc.", "apple (fruit)"],
                    "contexts": ["technology", "food"],
                },
            }
        ]

        for case in test_cases:
            entities = entity_extractor.extract_entities_with_disambiguation(
                case["text"]
            )

            # Verify disambiguation was performed
            apple_entities = [e for e in entities if "apple" in e["text"].lower()]
            assert (
                len(apple_entities) >= 2
            ), "Should distinguish between different 'Apple' entities"

    def test_temporal_relationship_extraction(self, relationship_detector):
        """Test extraction of temporal relationships."""
        text = "Steve Jobs founded Apple in 1976. Tim Cook became CEO in 2011."

        relationships = relationship_detector.extract_temporal_relationships(text)

        # Should extract temporal context for events
        founding_rels = [r for r in relationships if "1976" in str(r)]
        ceo_rels = [r for r in relationships if "2011" in str(r)]

        assert len(founding_rels) > 0, "Should extract founding date relationship"
        assert len(ceo_rels) > 0, "Should extract CEO appointment date relationship"

    def test_graph_update_consistency(self, graph_builder):
        """Test graph update consistency and incremental building."""
        # Build initial graph
        initial_docs = ["Apple Inc. was founded by Steve Jobs."]
        graph = graph_builder.build_graph(initial_docs)

        initial_node_count = graph.get_node_count()

        # Add new document
        new_docs = ["Tim Cook is the CEO of Apple Inc."]
        updated_graph = graph_builder.update_graph(graph, new_docs)

        # Verify incremental update
        new_node_count = updated_graph.get_node_count()
        assert (
            new_node_count > initial_node_count
        ), "Graph should have new nodes after update"

        # Verify consistency
        apple_node = updated_graph.get_node("Apple Inc.")
        assert apple_node is not None, "Apple Inc. node should exist after update"

    # Helper methods for metric calculations

    def _calculate_entity_metrics(
        self, extracted: List[Dict], expected: List[Dict]
    ) -> Tuple[float, float, float]:
        """Calculate precision, recall, and F1 score for entity extraction."""
        extracted_entities = {e["text"].lower(): e["type"] for e in extracted}
        expected_entities = {e["text"].lower(): e["type"] for e in expected}

        true_positives = 0
        for entity, entity_type in extracted_entities.items():
            if entity in expected_entities and expected_entities[entity] == entity_type:
                true_positives += 1

        precision = (
            true_positives / len(extracted_entities) if extracted_entities else 0
        )
        recall = true_positives / len(expected_entities) if expected_entities else 0
        f1_score = (
            2 * (precision * recall) / (precision + recall)
            if (precision + recall) > 0
            else 0
        )

        return precision, recall, f1_score

    def _calculate_relationship_metrics(
        self, extracted: List[Dict], expected: List[Dict]
    ) -> Tuple[float, float, float]:
        """Calculate precision, recall, and F1 score for relationship extraction."""
        extracted_rels = {
            (r["source"], r["target"], r["relationship"]) for r in extracted
        }
        expected_rels = {
            (r["source"], r["target"], r["relationship"]) for r in expected
        }

        true_positives = len(extracted_rels & expected_rels)

        precision = true_positives / len(extracted_rels) if extracted_rels else 0
        recall = true_positives / len(expected_rels) if expected_rels else 0
        f1_score = (
            2 * (precision * recall) / (precision + recall)
            if (precision + recall) > 0
            else 0
        )

        return precision, recall, f1_score

    def _calculate_query_accuracy(
        self, results: List[str], expected: List[str]
    ) -> float:
        """Calculate query result accuracy."""
        if not expected:
            return 1.0 if not results else 0.0

        results_set = {r.lower() for r in results}
        expected_set = {e.lower() for e in expected}

        intersection = results_set & expected_set
        return len(intersection) / len(expected_set)

    def _find_matching_entity(
        self, extracted: Dict, expected_entities: List[Dict]
    ) -> Dict:
        """Find matching expected entity for extracted entity."""
        for expected in expected_entities:
            if (
                expected["text"].lower() == extracted["text"].lower()
                and expected["type"] == extracted["type"]
            ):
                return expected
        return None

    def _find_matching_relationship(
        self, extracted: Dict, expected_relationships: List[Dict]
    ) -> Dict:
        """Find matching expected relationship for extracted relationship."""
        for expected in expected_relationships:
            if (
                expected["source"] == extracted["source"]
                and expected["target"] == extracted["target"]
                and expected["relationship"] == extracted["relationship"]
            ):
                return expected
        return None
