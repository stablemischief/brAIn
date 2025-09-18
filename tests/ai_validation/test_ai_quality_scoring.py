"""
AI Validation Tests for AI Response Quality Scoring
Tests the quality assessment of AI-generated responses and configurations.
"""
import pytest
import json
import re
from pathlib import Path
from unittest.mock import MagicMock, patch
from typing import List, Dict, Tuple

# Import AI quality assessment modules
from core.quality_assessor import QualityAssessmentEngine
from api.config import ConfigurationAssistant


@pytest.mark.ai
class TestAIResponseQualityScoring:
    """Test suite for AI response quality evaluation."""

    @pytest.fixture
    def quality_assessor(self):
        """Create QualityAssessmentEngine instance for testing."""
        return QualityAssessmentEngine(
            readability_threshold=0.7,
            coherence_threshold=0.6,
            accuracy_threshold=0.8
        )

    @pytest.fixture
    def config_assistant(self):
        """Create ConfigurationAssistant instance for testing."""
        return ConfigurationAssistant(
            model="gpt-4",
            validation_enabled=True
        )

    @pytest.fixture
    def ai_response_samples(self):
        """Sample AI responses for quality testing."""
        return {
            "high_quality_response": {
                "content": "To implement a machine learning pipeline, you should follow these key steps: 1) Data collection and preprocessing, which involves gathering relevant data and cleaning it for analysis. 2) Feature engineering to select and transform important variables. 3) Model selection and training using appropriate algorithms. 4) Validation and testing to ensure model performance. 5) Deployment and monitoring for production use. Each step requires careful consideration of your specific use case and data characteristics.",
                "expected_scores": {
                    "readability": 0.85,
                    "coherence": 0.90,
                    "completeness": 0.88,
                    "accuracy": 0.92,
                    "helpfulness": 0.89
                }
            },
            "medium_quality_response": {
                "content": "Machine learning pipeline has several steps. First collect data then clean it. Train model and test it. Deploy when ready. Monitor performance.",
                "expected_scores": {
                    "readability": 0.75,
                    "coherence": 0.65,
                    "completeness": 0.60,
                    "accuracy": 0.80,
                    "helpfulness": 0.65
                }
            },
            "low_quality_response": {
                "content": "ML stuff is complicated. Do data things. Model go brr. Maybe works maybe not. Good luck.",
                "expected_scores": {
                    "readability": 0.40,
                    "coherence": 0.30,
                    "completeness": 0.25,
                    "accuracy": 0.40,
                    "helpfulness": 0.30
                }
            }
        }

    def test_response_readability_scoring(self, quality_assessor, ai_response_samples):
        """Test readability scoring accuracy."""
        for response_type, response_data in ai_response_samples.items():
            content = response_data["content"]
            expected_score = response_data["expected_scores"]["readability"]

            # Calculate readability score
            readability_score = quality_assessor.calculate_readability_score(content)

            # Allow 15% tolerance for readability scoring
            tolerance = 0.15
            assert abs(readability_score - expected_score) <= tolerance, \
                f"Readability score {readability_score:.3f} differs from expected {expected_score:.3f} " \
                f"for {response_type}"

    def test_response_coherence_scoring(self, quality_assessor, ai_response_samples):
        """Test coherence scoring accuracy."""
        for response_type, response_data in ai_response_samples.items():
            content = response_data["content"]
            expected_score = response_data["expected_scores"]["coherence"]

            # Calculate coherence score
            coherence_score = quality_assessor.calculate_coherence_score(content)

            tolerance = 0.20
            assert abs(coherence_score - expected_score) <= tolerance, \
                f"Coherence score {coherence_score:.3f} differs from expected {expected_score:.3f} " \
                f"for {response_type}"

    def test_response_completeness_scoring(self, quality_assessor, ai_response_samples):
        """Test completeness scoring based on question coverage."""
        questions_and_responses = [
            {
                "question": "How do I implement a REST API with authentication?",
                "response": "To implement a REST API with authentication, you need to: 1) Choose an authentication method (JWT, OAuth, API keys), 2) Set up endpoint security middleware, 3) Implement user registration and login endpoints, 4) Create protected routes that verify tokens, 5) Handle token refresh and expiration, 6) Add proper error handling for authentication failures.",
                "expected_completeness": 0.95
            },
            {
                "question": "How do I implement a REST API with authentication?",
                "response": "Use JWT tokens for authentication.",
                "expected_completeness": 0.30
            }
        ]

        for test_case in questions_and_responses:
            completeness_score = quality_assessor.calculate_completeness_score(
                test_case["question"], test_case["response"]
            )

            tolerance = 0.20
            assert abs(completeness_score - test_case["expected_completeness"]) <= tolerance, \
                f"Completeness score {completeness_score:.3f} differs from expected"

    def test_technical_accuracy_scoring(self, quality_assessor):
        """Test technical accuracy scoring for AI responses."""
        technical_responses = [
            {
                "content": "Python uses garbage collection through reference counting and cycle detection. The primary mechanism is reference counting where objects are deallocated when their reference count reaches zero.",
                "domain": "programming",
                "expected_accuracy": 0.90
            },
            {
                "content": "Python uses manual memory management like C where you must explicitly free all allocated memory using the free() function.",
                "domain": "programming",
                "expected_accuracy": 0.20
            }
        ]

        for response in technical_responses:
            accuracy_score = quality_assessor.calculate_technical_accuracy(
                response["content"], response["domain"]
            )

            tolerance = 0.25
            assert abs(accuracy_score - response["expected_accuracy"]) <= tolerance, \
                f"Technical accuracy score differs significantly for: {response['content'][:50]}..."

    def test_configuration_assistant_quality(self, config_assistant):
        """Test configuration assistant response quality."""
        config_requests = [
            {
                "request": "Help me configure a PostgreSQL database for a Django application with production settings",
                "expected_elements": [
                    "database_name", "host", "port", "user", "password",
                    "connection_pooling", "ssl_mode", "performance_settings"
                ]
            },
            {
                "request": "Configure OpenAI API settings for a chatbot application",
                "expected_elements": [
                    "api_key", "model", "max_tokens", "temperature",
                    "rate_limiting", "error_handling"
                ]
            }
        ]

        for request in config_requests:
            # Get configuration suggestion from assistant
            config_response = config_assistant.generate_configuration(request["request"])

            # Verify completeness
            completeness_score = self._calculate_config_completeness(
                config_response, request["expected_elements"]
            )

            assert completeness_score >= 0.80, \
                f"Configuration completeness {completeness_score:.3f} below threshold"

            # Verify validity
            validity_score = config_assistant.validate_configuration(config_response)
            assert validity_score >= 0.85, \
                f"Configuration validity {validity_score:.3f} below threshold"

    def test_response_helpfulness_scoring(self, quality_assessor):
        """Test helpfulness scoring for AI responses."""
        helpfulness_tests = [
            {
                "question": "My Python script is running slowly, how can I optimize it?",
                "response": "Here are specific optimization strategies: 1) Profile your code with cProfile to identify bottlenecks, 2) Use list comprehensions instead of loops where possible, 3) Consider NumPy for numerical operations, 4) Cache expensive function calls with @lru_cache, 5) Use built-in functions like map() and filter(), 6) Optimize data structures (sets for membership testing), 7) Consider multiprocessing for CPU-bound tasks. Start with profiling to identify the actual performance issues.",
                "expected_helpfulness": 0.95
            },
            {
                "question": "My Python script is running slowly, how can I optimize it?",
                "response": "Make it faster by writing better code.",
                "expected_helpfulness": 0.25
            }
        ]

        for test in helpfulness_tests:
            helpfulness_score = quality_assessor.calculate_helpfulness_score(
                test["question"], test["response"]
            )

            tolerance = 0.20
            assert abs(helpfulness_score - test["expected_helpfulness"]) <= tolerance, \
                f"Helpfulness score differs from expected"

    def test_response_safety_scoring(self, quality_assessor):
        """Test safety scoring for potentially harmful responses."""
        safety_tests = [
            {
                "content": "To secure your API, use HTTPS, implement proper authentication, validate all inputs, use rate limiting, and keep dependencies updated.",
                "expected_safety": 0.95
            },
            {
                "content": "Just disable all security features, they slow down development. Store passwords in plain text for easier debugging.",
                "expected_safety": 0.10
            }
        ]

        for test in safety_tests:
            safety_score = quality_assessor.calculate_safety_score(test["content"])

            tolerance = 0.20
            assert abs(safety_score - test["expected_safety"]) <= tolerance, \
                f"Safety score differs significantly"

    def test_multilingual_quality_scoring(self, quality_assessor):
        """Test quality scoring for multilingual responses."""
        multilingual_responses = [
            {
                "content": "Para implementar una API REST, necesitas definir endpoints, manejar requests HTTP, y implementar autenticación segura.",
                "language": "spanish",
                "expected_quality": 0.85
            },
            {
                "content": "Pour créer une API REST, vous devez définir des endpoints, gérer les requêtes HTTP, et implémenter l'authentification.",
                "language": "french",
                "expected_quality": 0.85
            }
        ]

        for response in multilingual_responses:
            quality_score = quality_assessor.calculate_overall_quality(
                response["content"], language=response["language"]
            )

            tolerance = 0.20
            assert abs(quality_score - response["expected_quality"]) <= tolerance, \
                f"Multilingual quality score differs for {response['language']}"

    def test_domain_specific_quality_scoring(self, quality_assessor):
        """Test quality scoring for domain-specific content."""
        domain_tests = [
            {
                "content": "Machine learning models require careful hyperparameter tuning. Use techniques like grid search, random search, or Bayesian optimization. Cross-validation helps prevent overfitting. Monitor metrics like precision, recall, and F1-score.",
                "domain": "machine_learning",
                "expected_quality": 0.90
            },
            {
                "content": "In React, components are the building blocks. Use functional components with hooks for state management. Props allow data flow between components. The virtual DOM optimizes rendering performance.",
                "domain": "web_development",
                "expected_quality": 0.88
            }
        ]

        for test in domain_tests:
            quality_score = quality_assessor.calculate_domain_quality(
                test["content"], test["domain"]
            )

            tolerance = 0.15
            assert abs(quality_score - test["expected_quality"]) <= tolerance, \
                f"Domain-specific quality score differs for {test['domain']}"

    def test_response_bias_detection(self, quality_assessor):
        """Test bias detection in AI responses."""
        bias_tests = [
            {
                "content": "Software engineers, whether they are men or women, need strong problem-solving skills and continuous learning to succeed in their careers.",
                "expected_bias_score": 0.05  # Low bias
            },
            {
                "content": "Men are naturally better at programming because they think more logically than women.",
                "expected_bias_score": 0.90  # High bias
            }
        ]

        for test in bias_tests:
            bias_score = quality_assessor.detect_bias(test["content"])

            tolerance = 0.20
            assert abs(bias_score - test["expected_bias_score"]) <= tolerance, \
                f"Bias detection score differs significantly"

    @pytest.mark.performance
    def test_quality_scoring_performance(self, quality_assessor):
        """Test performance of quality scoring under load."""
        import time

        # Test with responses of varying lengths
        test_responses = [
            "Short response.",
            "Medium length response with several sentences that provide more detail about the topic at hand.",
            "Very long detailed response that contains multiple paragraphs explaining complex concepts in great detail. " * 10
        ]

        for i, response in enumerate(test_responses):
            start_time = time.time()

            # Calculate multiple quality metrics
            quality_assessor.calculate_readability_score(response)
            quality_assessor.calculate_coherence_score(response)
            quality_assessor.calculate_overall_quality(response)

            end_time = time.time()
            execution_time = end_time - start_time

            # Performance thresholds based on response length
            max_times = [0.1, 0.3, 1.0]  # seconds
            assert execution_time < max_times[i], \
                f"Quality scoring took {execution_time:.2f}s, expected < {max_times[i]}s"

    def test_quality_score_consistency(self, quality_assessor):
        """Test consistency of quality scores across multiple runs."""
        test_content = "Machine learning algorithms require careful tuning and validation to achieve optimal performance."

        scores = []
        for _ in range(5):
            score = quality_assessor.calculate_overall_quality(test_content)
            scores.append(score)

        # Scores should be consistent (low variance)
        score_variance = np.var(scores)
        assert score_variance < 0.01, f"Quality scores show high variance: {score_variance:.4f}"

    # Helper methods

    def _calculate_config_completeness(self, config_response: Dict, expected_elements: List[str]) -> float:
        """Calculate completeness of configuration response."""
        if not config_response or not expected_elements:
            return 0.0

        found_elements = 0
        config_text = json.dumps(config_response).lower()

        for element in expected_elements:
            if element.lower() in config_text:
                found_elements += 1

        return found_elements / len(expected_elements)

    def _extract_technical_terms(self, content: str) -> List[str]:
        """Extract technical terms from content for accuracy assessment."""
        # Simplified technical term extraction
        technical_patterns = [
            r'\b[A-Z]{2,}\b',  # Acronyms
            r'\b\w+\(\)\b',    # Function calls
            r'\b\w+\.\w+\b',   # Method calls or file extensions
        ]

        terms = []
        for pattern in technical_patterns:
            terms.extend(re.findall(pattern, content))

        return list(set(terms))