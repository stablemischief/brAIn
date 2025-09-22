"""
Unit tests for the DatabaseHandler module.
"""
import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from datetime import datetime
import uuid

from core.database_handler import (
    DatabaseHandler,
    Document,
    DatabaseError,
    DuplicateError,
    ConnectionPool
)


@pytest.mark.unit
class TestDatabaseHandler:
    """Test suite for DatabaseHandler class."""

    @pytest.fixture
    def db_handler(self, mock_supabase_client):
        """Create DatabaseHandler instance with mocked dependencies."""
        handler = DatabaseHandler(
            supabase_client=mock_supabase_client,
            database_url="postgresql://test:test@localhost/test"
        )
        return handler

    @pytest.fixture
    def sample_document_dict(self):
        """Sample document dictionary for testing."""
        return {
            "id": str(uuid.uuid4()),
            "google_drive_id": "drive-123",
            "title": "Test Document",
            "content": "Test content",
            "folder_id": "folder-456",
            "embedding": [0.1] * 1536,
            "metadata": {"author": "Test"},
            "token_count": 10,
            "processing_cost": 0.001,
            "created_at": datetime.utcnow().isoformat(),
            "updated_at": datetime.utcnow().isoformat()
        }

    def test_handler_initialization(self, db_handler):
        """Test DatabaseHandler initialization."""
        assert db_handler is not None
        assert db_handler.supabase_client is not None

    @pytest.mark.asyncio
    async def test_create_document(self, db_handler, sample_document_dict):
        """Test document creation."""
        db_handler.supabase_client.table().insert().execute.return_value.data = [
            sample_document_dict
        ]

        document = await db_handler.create_document(sample_document_dict)

        assert isinstance(document, Document)
        assert document.id == sample_document_dict["id"]
        assert document.title == "Test Document"
        db_handler.supabase_client.table().insert.assert_called_once()

    @pytest.mark.asyncio
    async def test_create_document_duplicate(self, db_handler, sample_document_dict):
        """Test duplicate document detection."""
        # Simulate duplicate error
        db_handler.supabase_client.table().insert().execute.side_effect = Exception(
            "duplicate key value violates unique constraint"
        )

        with pytest.raises(DuplicateError):
            await db_handler.create_document(sample_document_dict)

    @pytest.mark.asyncio
    async def test_get_document_by_id(self, db_handler, sample_document_dict):
        """Test retrieving document by ID."""
        db_handler.supabase_client.table().select().eq().execute.return_value.data = [
            sample_document_dict
        ]

        document = await db_handler.get_document(sample_document_dict["id"])

        assert document is not None
        assert document.id == sample_document_dict["id"]
        assert document.title == "Test Document"

    @pytest.mark.asyncio
    async def test_get_document_not_found(self, db_handler):
        """Test retrieving non-existent document."""
        db_handler.supabase_client.table().select().eq().execute.return_value.data = []

        document = await db_handler.get_document("nonexistent-id")
        assert document is None

    @pytest.mark.asyncio
    async def test_update_document(self, db_handler, sample_document_dict):
        """Test document update."""
        updated_data = {**sample_document_dict, "title": "Updated Title"}
        db_handler.supabase_client.table().update().eq().execute.return_value.data = [
            updated_data
        ]

        updated = await db_handler.update_document(
            sample_document_dict["id"],
            {"title": "Updated Title"}
        )

        assert updated is True
        db_handler.supabase_client.table().update.assert_called_once()

    @pytest.mark.asyncio
    async def test_delete_document(self, db_handler):
        """Test document deletion."""
        db_handler.supabase_client.table().delete().eq().execute.return_value.data = []

        deleted = await db_handler.delete_document("doc-123")

        assert deleted is True
        db_handler.supabase_client.table().delete.assert_called_once()

    @pytest.mark.asyncio
    async def test_search_documents(self, db_handler, sample_document_dict):
        """Test document search."""
        db_handler.supabase_client.rpc().execute.return_value.data = [
            {**sample_document_dict, "similarity": 0.95}
        ]

        results = await db_handler.search_documents(
            query="test query",
            limit=10
        )

        assert len(results) == 1
        assert results[0]["similarity"] == 0.95
        db_handler.supabase_client.rpc.assert_called_once()

    @pytest.mark.asyncio
    async def test_search_documents_with_filters(self, db_handler):
        """Test document search with filters."""
        db_handler.supabase_client.rpc().execute.return_value.data = []

        results = await db_handler.search_documents(
            query="test query",
            folder_id="folder-123",
            date_from=datetime(2024, 1, 1),
            limit=5
        )

        assert results == []
        # Verify RPC was called with correct parameters

    @pytest.mark.asyncio
    async def test_batch_create_documents(self, db_handler):
        """Test batch document creation."""
        docs = [
            {"title": "Doc 1", "content": "Content 1"},
            {"title": "Doc 2", "content": "Content 2"}
        ]

        db_handler.supabase_client.table().insert().execute.return_value.data = docs

        created = await db_handler.batch_create_documents(docs)

        assert len(created) == 2
        db_handler.supabase_client.table().insert.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_documents_by_folder(self, db_handler, sample_document_dict):
        """Test retrieving documents by folder."""
        db_handler.supabase_client.table().select().eq().execute.return_value.data = [
            sample_document_dict
        ]

        documents = await db_handler.get_documents_by_folder("folder-456")

        assert len(documents) == 1
        assert documents[0].folder_id == "folder-456"

    @pytest.mark.asyncio
    async def test_count_documents(self, db_handler):
        """Test document counting."""
        db_handler.supabase_client.table().select().execute.return_value.data = [
            {"count": 42}
        ]

        count = await db_handler.count_documents()

        assert count == 42

    @pytest.mark.asyncio
    async def test_get_recent_documents(self, db_handler, sample_document_dict):
        """Test retrieving recent documents."""
        db_handler.supabase_client.table().select().order().limit().execute.return_value.data = [
            sample_document_dict
        ]

        recent = await db_handler.get_recent_documents(limit=10)

        assert len(recent) == 1
        assert recent[0].id == sample_document_dict["id"]

    @pytest.mark.asyncio
    async def test_check_duplicate_by_hash(self, db_handler):
        """Test duplicate checking by content hash."""
        content_hash = "abc123def456"

        # No duplicate
        db_handler.supabase_client.table().select().eq().execute.return_value.data = []
        is_duplicate = await db_handler.check_duplicate_by_hash(content_hash)
        assert is_duplicate is False

        # Has duplicate
        db_handler.supabase_client.table().select().eq().execute.return_value.data = [
            {"id": "existing-doc"}
        ]
        is_duplicate = await db_handler.check_duplicate_by_hash(content_hash)
        assert is_duplicate is True

    @pytest.mark.asyncio
    async def test_update_document_embedding(self, db_handler):
        """Test updating document embedding."""
        new_embedding = [0.2] * 1536

        db_handler.supabase_client.table().update().eq().execute.return_value.data = [
            {"id": "doc-123"}
        ]

        updated = await db_handler.update_document_embedding(
            "doc-123",
            new_embedding
        )

        assert updated is True

    @pytest.mark.asyncio
    async def test_database_connection_error(self, db_handler):
        """Test database connection error handling."""
        db_handler.supabase_client.table().select().execute.side_effect = Exception(
            "Connection refused"
        )

        with pytest.raises(DatabaseError):
            await db_handler.get_document("doc-123")

    @pytest.mark.asyncio
    async def test_transaction_rollback(self, db_handler):
        """Test transaction rollback on error."""
        # Simulate error during transaction
        db_handler.supabase_client.table().insert().execute.side_effect = [
            MagicMock(data=[{"id": "doc-1"}]),  # First insert succeeds
            Exception("Constraint violation")    # Second insert fails
        ]

        docs = [
            {"title": "Doc 1"},
            {"title": "Doc 2"}
        ]

        with pytest.raises(DatabaseError):
            await db_handler.batch_create_documents(docs, transaction=True)

        # Verify rollback was attempted

    def test_connection_pool_management(self, db_handler):
        """Test connection pool management."""
        pool = db_handler.get_connection_pool()

        assert isinstance(pool, ConnectionPool)
        assert pool.min_size > 0
        assert pool.max_size >= pool.min_size

    @pytest.mark.asyncio
    async def test_document_statistics(self, db_handler):
        """Test retrieving document statistics."""
        stats = {
            "total_documents": 100,
            "total_tokens": 50000,
            "total_cost": 5.50,
            "average_quality_score": 0.92
        }

        db_handler.supabase_client.rpc().execute.return_value.data = [stats]

        result = await db_handler.get_document_statistics()

        assert result["total_documents"] == 100
        assert result["total_cost"] == 5.50

    @pytest.mark.asyncio
    async def test_cleanup_old_documents(self, db_handler):
        """Test cleaning up old documents."""
        db_handler.supabase_client.table().delete().lt().execute.return_value.data = []

        deleted_count = await db_handler.cleanup_old_documents(days=30)

        assert isinstance(deleted_count, int)
        db_handler.supabase_client.table().delete.assert_called_once()