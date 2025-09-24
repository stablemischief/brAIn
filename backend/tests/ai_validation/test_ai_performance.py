"""
AI Validation Tests for AI Performance Benchmarking
Tests performance characteristics of AI operations under various load conditions.
"""

import pytest
import time
import asyncio
import statistics
from concurrent.futures import ThreadPoolExecutor, as_completed
from unittest.mock import MagicMock, patch

# Import AI performance modules
from app.core.text_processor import EnhancedTextProcessor
from app.cost.cost_calculator import CostCalculator
from app.knowledge_graph.builder import KnowledgeGraphBuilder
from app.search.hybrid_search import HybridSearchEngine


@pytest.mark.performance
@pytest.mark.ai
class TestAIPerformanceBenchmarks:
    """Test suite for AI performance benchmarking."""

    @pytest.fixture
    def text_processor(self):
        """Create EnhancedTextProcessor for performance testing."""
        return EnhancedTextProcessor(
            api_key="test-key", batch_size=10, max_concurrent_requests=5
        )

    @pytest.fixture
    def cost_calculator(self):
        """Create CostCalculator for performance testing."""
        return CostCalculator()

    @pytest.fixture
    def kg_builder(self):
        """Create KnowledgeGraphBuilder for performance testing."""
        return KnowledgeGraphBuilder()

    @pytest.fixture
    def search_engine(self):
        """Create HybridSearchEngine for performance testing."""
        return HybridSearchEngine(vector_weight=0.7, keyword_weight=0.3)

    @pytest.fixture
    def performance_test_data(self):
        """Generate test data of various sizes for performance testing."""
        return {
            "small_text": "This is a small text sample for testing.",
            "medium_text": "This is a medium-sized text sample. " * 50,
            "large_text": "This is a large text sample for performance testing. " * 500,
            "xlarge_text": "This is an extra large text sample. " * 2000,
            "small_documents": [f"Document {i} content" for i in range(10)],
            "medium_documents": [
                f"Document {i} with more content. " * 20 for i in range(100)
            ],
            "large_documents": [
                f"Document {i} with extensive content. " * 100 for i in range(500)
            ],
        }

    def test_text_processing_latency(self, text_processor, performance_test_data):
        """Test text processing latency across different input sizes."""
        test_cases = [
            ("small_text", 0.1),
            ("medium_text", 0.5),
            ("large_text", 2.0),
            ("xlarge_text", 5.0),
        ]

        for text_key, max_latency in test_cases:
            text = performance_test_data[text_key]

            start_time = time.time()
            with patch.object(
                text_processor, "generate_embedding", return_value=[0.1] * 1536
            ):
                result = text_processor.process_text_sync(text)
            end_time = time.time()

            latency = end_time - start_time

            assert (
                latency < max_latency
            ), f"Text processing latency {latency:.2f}s exceeds threshold {max_latency}s for {text_key}"
            assert result is not None, f"Processing failed for {text_key}"

    def test_batch_processing_throughput(self, text_processor, performance_test_data):
        """Test batch processing throughput."""
        batch_sizes = [5, 10, 25, 50]
        expected_throughput = [10, 15, 20, 25]  # documents per second

        for batch_size, min_throughput in zip(batch_sizes, expected_throughput):
            documents = performance_test_data["small_documents"][:batch_size]

            start_time = time.time()
            with patch.object(
                text_processor, "generate_embedding", return_value=[0.1] * 1536
            ):
                results = text_processor.batch_process_texts_sync(documents)
            end_time = time.time()

            total_time = end_time - start_time
            actual_throughput = len(documents) / total_time

            assert (
                actual_throughput >= min_throughput
            ), f"Batch throughput {actual_throughput:.2f} docs/sec below threshold {min_throughput} for batch size {batch_size}"

    def test_concurrent_processing_performance(
        self, text_processor, performance_test_data
    ):
        """Test performance under concurrent load."""
        num_concurrent = [1, 5, 10, 20]
        max_latency_increase = [1.0, 1.5, 2.0, 3.0]  # multipliers

        baseline_time = None
        documents = performance_test_data["medium_documents"][:20]

        for concurrent, max_increase in zip(num_concurrent, max_latency_increase):
            times = []

            # Run concurrent processing test
            with ThreadPoolExecutor(max_workers=concurrent) as executor:
                futures = []
                start_time = time.time()

                for i in range(concurrent):
                    doc_subset = documents[i : i + 2]  # Process 2 docs per thread
                    future = executor.submit(
                        self._process_documents_mock, text_processor, doc_subset
                    )
                    futures.append(future)

                for future in as_completed(futures):
                    future.result()

                end_time = time.time()

            total_time = end_time - start_time

            if baseline_time is None:
                baseline_time = total_time
            else:
                latency_increase = total_time / baseline_time
                assert (
                    latency_increase <= max_increase
                ), f"Concurrent processing latency increased {latency_increase:.2f}x, max allowed {max_increase}x"

    def test_memory_usage_under_load(self, text_processor, performance_test_data):
        """Test memory usage during intensive processing."""
        import psutil
        import os

        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        # Process large amount of data
        large_documents = performance_test_data["large_documents"]

        with patch.object(
            text_processor, "generate_embedding", return_value=[0.1] * 1536
        ):
            for batch in self._chunk_list(large_documents, 10):
                text_processor.batch_process_texts_sync(batch)

        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory

        # Memory increase should be reasonable (less than 500MB for this test)
        assert (
            memory_increase < 500
        ), f"Memory usage increased by {memory_increase:.1f}MB, exceeds 500MB threshold"

    def test_cost_calculation_performance(self, cost_calculator):
        """Test cost calculation performance under high load."""
        num_calculations = 10000

        start_time = time.time()
        for i in range(num_calculations):
            cost_calculator.calculate_operation_cost(
                model="gpt-4",
                provider="openai",
                input_tokens=1000 + i % 1000,
                output_tokens=500 + i % 500,
                category="processing",
            )
        end_time = time.time()

        total_time = end_time - start_time
        calculations_per_second = num_calculations / total_time

        # Should handle at least 1000 calculations per second
        assert (
            calculations_per_second >= 1000
        ), f"Cost calculations: {calculations_per_second:.0f}/sec, expected >= 1000/sec"

    def test_knowledge_graph_building_performance(
        self, kg_builder, performance_test_data
    ):
        """Test knowledge graph building performance."""
        document_sets = [
            ("small", performance_test_data["small_documents"], 2.0),
            ("medium", performance_test_data["medium_documents"][:50], 10.0),
            ("large", performance_test_data["large_documents"][:20], 30.0),
        ]

        for set_name, documents, max_time in document_sets:
            start_time = time.time()

            with patch.object(kg_builder, "extract_entities") as mock_extract:
                mock_extract.return_value = [
                    {"text": "Entity", "type": "PERSON", "confidence": 0.9}
                ]
                with patch.object(kg_builder, "extract_relationships") as mock_rel:
                    mock_rel.return_value = [
                        {"source": "A", "target": "B", "relation": "KNOWS"}
                    ]
                    graph = kg_builder.build_graph(documents)

            end_time = time.time()
            build_time = end_time - start_time

            assert (
                build_time < max_time
            ), f"Knowledge graph building took {build_time:.2f}s, expected < {max_time}s for {set_name} set"

    def test_search_performance_scaling(self, search_engine, performance_test_data):
        """Test search performance scaling with document count."""
        document_counts = [100, 500, 1000, 2000]
        max_search_times = [0.1, 0.3, 0.8, 2.0]  # seconds

        query = "machine learning algorithms"

        for doc_count, max_time in zip(document_counts, max_search_times):
            # Create test document set
            documents = [
                {
                    "id": f"doc_{i}",
                    "content": f"Content about machine learning and AI algorithms {i}",
                }
                for i in range(doc_count)
            ]

            start_time = time.time()
            with patch.object(
                search_engine, "calculate_vector_similarity", return_value=0.8
            ):
                results = search_engine.search(query, documents, limit=10)
            end_time = time.time()

            search_time = end_time - start_time

            assert (
                search_time < max_time
            ), f"Search time {search_time:.2f}s exceeds {max_time}s for {doc_count} documents"

    @pytest.mark.asyncio
    async def test_async_processing_performance(self, text_processor):
        """Test asynchronous processing performance."""
        num_tasks = 50
        texts = [
            f"Test document {i} with content for processing." for i in range(num_tasks)
        ]

        start_time = time.time()

        # Create async tasks
        tasks = []
        with patch.object(
            text_processor, "generate_embedding", return_value=[0.1] * 1536
        ):
            for text in texts:
                task = asyncio.create_task(text_processor.process_text_async(text))
                tasks.append(task)

            # Wait for all tasks to complete
            results = await asyncio.gather(*tasks)

        end_time = time.time()

        total_time = end_time - start_time
        tasks_per_second = num_tasks / total_time

        # Async processing should be faster than synchronous
        assert (
            tasks_per_second >= 10
        ), f"Async processing: {tasks_per_second:.1f} tasks/sec, expected >= 10 tasks/sec"
        assert len(results) == num_tasks, "All async tasks should complete"

    def test_error_handling_performance(self, text_processor):
        """Test performance impact of error handling."""
        error_rates = [0.0, 0.1, 0.3, 0.5]  # 0%, 10%, 30%, 50% error rates
        baseline_time = None

        for error_rate in error_rates:
            texts = [f"Test text {i}" for i in range(100)]

            start_time = time.time()

            with patch.object(text_processor, "generate_embedding") as mock_embed:
                # Configure mock to fail based on error rate
                def side_effect(text):
                    if hash(text) % 10 < error_rate * 10:
                        raise Exception("Simulated API error")
                    return [0.1] * 1536

                mock_embed.side_effect = side_effect

                results = []
                for text in texts:
                    try:
                        result = text_processor.process_text_sync(text)
                        results.append(result)
                    except Exception:
                        pass  # Expected errors

            end_time = time.time()
            processing_time = end_time - start_time

            if baseline_time is None:
                baseline_time = processing_time
            else:
                # Error handling shouldn't dramatically slow processing
                time_increase = processing_time / baseline_time
                assert (
                    time_increase <= 2.0
                ), f"Error handling increased processing time by {time_increase:.2f}x"

    def test_cache_performance_impact(self, search_engine):
        """Test performance impact of caching mechanisms."""
        query = "test query"
        documents = [{"id": f"doc_{i}", "content": f"Content {i}"} for i in range(1000)]

        # First search (cache miss)
        start_time = time.time()
        with patch.object(
            search_engine, "calculate_vector_similarity", return_value=0.8
        ):
            results1 = search_engine.search(query, documents, limit=10)
        end_time = time.time()
        cold_time = end_time - start_time

        # Second search (cache hit)
        start_time = time.time()
        with patch.object(
            search_engine, "calculate_vector_similarity", return_value=0.8
        ):
            results2 = search_engine.search(query, documents, limit=10)
        end_time = time.time()
        warm_time = end_time - start_time

        # Cache should improve performance
        speedup = cold_time / warm_time if warm_time > 0 else 1
        assert (
            speedup >= 1.5
        ), f"Cache speedup {speedup:.2f}x below expected 1.5x minimum"

    def test_resource_cleanup_performance(self, text_processor):
        """Test resource cleanup doesn't impact performance."""
        # Process many documents and measure cleanup time
        documents = [f"Document {i} content" for i in range(100)]

        processing_times = []
        cleanup_times = []

        for i in range(10):  # Multiple iterations
            start_time = time.time()

            with patch.object(
                text_processor, "generate_embedding", return_value=[0.1] * 1536
            ):
                results = text_processor.batch_process_texts_sync(documents)

            mid_time = time.time()

            # Force cleanup
            text_processor.cleanup_resources()

            end_time = time.time()

            processing_times.append(mid_time - start_time)
            cleanup_times.append(end_time - mid_time)

        avg_processing_time = statistics.mean(processing_times)
        avg_cleanup_time = statistics.mean(cleanup_times)

        # Cleanup should be fast relative to processing
        cleanup_ratio = avg_cleanup_time / avg_processing_time
        assert (
            cleanup_ratio <= 0.1
        ), f"Cleanup time ratio {cleanup_ratio:.3f} exceeds 10% of processing time"

    # Helper methods

    def _process_documents_mock(self, processor, documents):
        """Mock document processing for concurrent testing."""
        with patch.object(processor, "generate_embedding", return_value=[0.1] * 1536):
            return [processor.process_text_sync(doc) for doc in documents]

    def _chunk_list(self, lst, chunk_size):
        """Split list into chunks of specified size."""
        for i in range(0, len(lst), chunk_size):
            yield lst[i : i + chunk_size]

    def _simulate_api_latency(self, base_latency=0.1, variance=0.05):
        """Simulate API call latency for realistic testing."""
        import random

        latency = base_latency + random.uniform(-variance, variance)
        time.sleep(max(0, latency))

    def _measure_memory_usage(self):
        """Measure current memory usage."""
        import psutil
        import os

        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024  # MB
