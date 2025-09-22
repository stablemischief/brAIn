"""
Pytest configuration and shared fixtures for the test suite.
"""
import os
import sys
import asyncio
from pathlib import Path
from typing import Generator, AsyncGenerator, Any
import pytest
import tempfile
import shutil
from unittest.mock import MagicMock, AsyncMock

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Configure pytest plugins
pytest_plugins = ["pytest_asyncio"]


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for test files."""
    temp_path = tempfile.mkdtemp()
    yield Path(temp_path)
    shutil.rmtree(temp_path)


@pytest.fixture
def mock_openai_client():
    """Mock OpenAI client for testing."""
    client = MagicMock()

    # Mock embeddings response
    embedding_response = MagicMock()
    embedding_response.data = [MagicMock(embedding=[0.1] * 1536)]
    client.embeddings.create.return_value = embedding_response

    # Mock completion response
    completion_response = MagicMock()
    completion_response.choices = [
        MagicMock(message=MagicMock(content="Test response"))
    ]
    client.chat.completions.create.return_value = completion_response

    return client


@pytest.fixture
def mock_supabase_client():
    """Mock Supabase client for testing."""
    client = MagicMock()

    # Mock table operations
    table_mock = MagicMock()
    table_mock.select.return_value.execute.return_value.data = []
    table_mock.insert.return_value.execute.return_value.data = [{"id": "test-id"}]
    table_mock.update.return_value.eq.return_value.execute.return_value.data = [{"id": "test-id"}]
    table_mock.delete.return_value.eq.return_value.execute.return_value.data = []

    client.table.return_value = table_mock
    client.from_.return_value = table_mock

    # Mock RPC operations
    client.rpc.return_value.execute.return_value.data = []

    # Mock auth
    client.auth.sign_in_with_password.return_value.user = MagicMock(id="test-user-id")

    return client


@pytest.fixture
def sample_document():
    """Sample document for testing."""
    return {
        "id": "doc-123",
        "google_drive_id": "drive-456",
        "title": "Test Document",
        "content": "This is test content for our document.",
        "folder_id": "folder-789",
        "metadata": {
            "author": "Test Author",
            "created_at": "2024-01-01T00:00:00Z"
        },
        "embedding": [0.1] * 1536,
        "token_count": 10,
        "processing_cost": 0.001
    }


@pytest.fixture
def sample_config():
    """Sample configuration for testing."""
    return {
        "openai_api_key": "test-key",
        "supabase_url": "https://test.supabase.co",
        "supabase_key": "test-supabase-key",
        "google_drive_folder_id": "test-folder-id",
        "database_url": "postgresql://test:test@localhost/test",
        "max_tokens": 4000,
        "embedding_model": "text-embedding-ada-002",
        "chat_model": "gpt-4",
        "cost_limit_daily": 50.0,
        "cost_limit_monthly": 1000.0,
        "batch_size": 10,
        "max_workers": 4
    }


@pytest.fixture
async def async_mock_db_connection():
    """Async mock database connection."""
    conn = AsyncMock()
    conn.execute.return_value = None
    conn.fetch.return_value = []
    conn.fetchrow.return_value = {"id": "test-id"}
    conn.fetchval.return_value = 1
    return conn


@pytest.fixture
def mock_langfuse_client():
    """Mock Langfuse client for testing."""
    client = MagicMock()

    # Mock trace operations
    trace_mock = MagicMock()
    trace_mock.id = "trace-123"
    trace_mock.span.return_value = MagicMock()
    trace_mock.generation.return_value = MagicMock()

    client.trace.return_value = trace_mock

    return client


@pytest.fixture
def mock_redis_client():
    """Mock Redis client for caching tests."""
    client = MagicMock()
    client.get.return_value = None
    client.set.return_value = True
    client.delete.return_value = 1
    client.exists.return_value = 0
    client.expire.return_value = True
    return client


@pytest.fixture
def test_api_client():
    """Create test client for FastAPI app."""
    from httpx import AsyncClient
    from main import app

    return AsyncClient(app=app, base_url="http://test")


@pytest.fixture
def auth_headers():
    """Authentication headers for API tests."""
    return {"Authorization": "Bearer test-token"}


@pytest.fixture
def sample_processing_result():
    """Sample processing result for testing."""
    return {
        "success": True,
        "document_id": "doc-123",
        "content": "Processed content",
        "metadata": {
            "extraction_method": "ai",
            "quality_score": 0.95,
            "token_count": 100
        },
        "embedding": [0.1] * 1536,
        "cost": 0.002,
        "processing_time": 1.5
    }


# Environment setup for tests
@pytest.fixture(scope="session", autouse=True)
def setup_test_env():
    """Set up test environment variables."""
    os.environ["ENVIRONMENT"] = "test"
    os.environ["OPENAI_API_KEY"] = "test-key"
    os.environ["SUPABASE_URL"] = "https://test.supabase.co"
    os.environ["SUPABASE_KEY"] = "test-key"
    os.environ["DATABASE_URL"] = "postgresql://test:test@localhost/test"
    os.environ["LOG_LEVEL"] = "ERROR"  # Reduce log noise in tests
    yield
    # Cleanup if needed


# Markers for test categorization
def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line("markers", "unit: mark test as a unit test")
    config.addinivalue_line("markers", "integration: mark test as an integration test")
    config.addinivalue_line("markers", "e2e: mark test as an end-to-end test")
    config.addinivalue_line("markers", "performance: mark test as a performance test")
    config.addinivalue_line("markers", "slow: mark test as slow running")
    config.addinivalue_line("markers", "ai: mark test as requiring AI services")