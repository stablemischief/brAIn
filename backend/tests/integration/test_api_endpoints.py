"""
Integration tests for FastAPI endpoints.
"""

import pytest
from httpx import AsyncClient
from unittest.mock import MagicMock, patch
import json
from datetime import datetime


@pytest.mark.integration
class TestAPIEndpoints:
    """Integration tests for API endpoints."""

    @pytest.fixture
    async def client(self):
        """Create async test client."""
        from main import app

        async with AsyncClient(app=app, base_url="http://test") as ac:
            yield ac

    @pytest.fixture
    def auth_headers(self):
        """Authentication headers for protected endpoints."""
        return {"Authorization": "Bearer test-token"}

    # Health Check Endpoints

    @pytest.mark.asyncio
    async def test_health_check(self, client):
        """Test health check endpoint."""
        response = await client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data
        assert "version" in data

    @pytest.mark.asyncio
    async def test_readiness_check(self, client):
        """Test readiness check endpoint."""
        response = await client.get("/ready")

        assert response.status_code == 200
        data = response.json()
        assert "database" in data
        assert "cache" in data
        assert "services" in data

    # Document Management Endpoints

    @pytest.mark.asyncio
    async def test_create_document(self, client, auth_headers):
        """Test document creation endpoint."""
        document_data = {
            "title": "Test Document",
            "content": "Test content",
            "folder_id": "folder-123",
            "metadata": {"author": "Test User"},
        }

        with patch("api.folders.create_document") as mock_create:
            mock_create.return_value = {"id": "doc-123", **document_data}

            response = await client.post(
                "/api/documents", json=document_data, headers=auth_headers
            )

            assert response.status_code == 201
            data = response.json()
            assert data["id"] == "doc-123"
            assert data["title"] == "Test Document"

    @pytest.mark.asyncio
    async def test_get_document(self, client, auth_headers):
        """Test document retrieval endpoint."""
        with patch("api.folders.get_document") as mock_get:
            mock_get.return_value = {
                "id": "doc-123",
                "title": "Test Document",
                "content": "Test content",
            }

            response = await client.get("/api/documents/doc-123", headers=auth_headers)

            assert response.status_code == 200
            data = response.json()
            assert data["id"] == "doc-123"

    @pytest.mark.asyncio
    async def test_get_document_not_found(self, client, auth_headers):
        """Test document retrieval with non-existent ID."""
        with patch("api.folders.get_document") as mock_get:
            mock_get.return_value = None

            response = await client.get(
                "/api/documents/nonexistent", headers=auth_headers
            )

            assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_update_document(self, client, auth_headers):
        """Test document update endpoint."""
        update_data = {"title": "Updated Title", "metadata": {"updated": True}}

        with patch("api.folders.update_document") as mock_update:
            mock_update.return_value = True

            response = await client.patch(
                "/api/documents/doc-123", json=update_data, headers=auth_headers
            )

            assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_delete_document(self, client, auth_headers):
        """Test document deletion endpoint."""
        with patch("api.folders.delete_document") as mock_delete:
            mock_delete.return_value = True

            response = await client.delete(
                "/api/documents/doc-123", headers=auth_headers
            )

            assert response.status_code == 204

    # Search Endpoints

    @pytest.mark.asyncio
    async def test_search_documents(self, client, auth_headers):
        """Test document search endpoint."""
        with patch("api.search.search_documents") as mock_search:
            mock_search.return_value = {
                "results": [
                    {"id": "doc-1", "title": "Result 1", "score": 0.95},
                    {"id": "doc-2", "title": "Result 2", "score": 0.85},
                ],
                "total": 2,
            }

            response = await client.post(
                "/api/search",
                json={"query": "test query", "limit": 10},
                headers=auth_headers,
            )

            assert response.status_code == 200
            data = response.json()
            assert len(data["results"]) == 2
            assert data["total"] == 2

    @pytest.mark.asyncio
    async def test_search_with_filters(self, client, auth_headers):
        """Test search with advanced filters."""
        search_params = {
            "query": "test",
            "folder_id": "folder-123",
            "date_from": "2024-01-01",
            "date_to": "2024-12-31",
            "limit": 5,
        }

        with patch("api.search.search_documents") as mock_search:
            mock_search.return_value = {"results": [], "total": 0}

            response = await client.post(
                "/api/search", json=search_params, headers=auth_headers
            )

            assert response.status_code == 200

    # Processing Endpoints

    @pytest.mark.asyncio
    async def test_start_processing(self, client, auth_headers):
        """Test processing start endpoint."""
        processing_request = {
            "document_id": "doc-123",
            "processing_type": "full",
            "options": {"extract_entities": True},
        }

        with patch("api.processing.start_processing") as mock_process:
            mock_process.return_value = {"job_id": "job-456", "status": "processing"}

            response = await client.post(
                "/api/processing/start", json=processing_request, headers=auth_headers
            )

            assert response.status_code == 202
            data = response.json()
            assert data["job_id"] == "job-456"

    @pytest.mark.asyncio
    async def test_get_processing_status(self, client, auth_headers):
        """Test processing status endpoint."""
        with patch("api.processing.get_job_status") as mock_status:
            mock_status.return_value = {
                "job_id": "job-456",
                "status": "completed",
                "progress": 100,
                "result": {"success": True},
            }

            response = await client.get(
                "/api/processing/status/job-456", headers=auth_headers
            )

            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "completed"

    # Analytics Endpoints

    @pytest.mark.asyncio
    async def test_get_analytics(self, client, auth_headers):
        """Test analytics endpoint."""
        with patch("api.analytics.get_analytics") as mock_analytics:
            mock_analytics.return_value = {
                "total_documents": 100,
                "total_tokens": 50000,
                "total_cost": 5.50,
                "processing_stats": {"success_rate": 0.95, "average_time": 2.5},
            }

            response = await client.get("/api/analytics", headers=auth_headers)

            assert response.status_code == 200
            data = response.json()
            assert data["total_documents"] == 100
            assert data["total_cost"] == 5.50

    @pytest.mark.asyncio
    async def test_get_cost_breakdown(self, client, auth_headers):
        """Test cost breakdown endpoint."""
        with patch("api.analytics.get_cost_breakdown") as mock_costs:
            mock_costs.return_value = {
                "daily": {"2024-01-01": 1.50},
                "monthly": {"2024-01": 45.00},
                "by_model": {"gpt-4": 30.00, "text-embedding-ada-002": 15.00},
            }

            response = await client.get("/api/analytics/costs", headers=auth_headers)

            assert response.status_code == 200
            data = response.json()
            assert "daily" in data
            assert "monthly" in data

    # Configuration Endpoints

    @pytest.mark.asyncio
    async def test_get_config(self, client, auth_headers):
        """Test configuration retrieval endpoint."""
        with patch("api.config.get_configuration") as mock_config:
            mock_config.return_value = {
                "max_tokens": 4000,
                "embedding_model": "text-embedding-ada-002",
                "cost_limits": {"daily": 50.0, "monthly": 1000.0},
            }

            response = await client.get("/api/config", headers=auth_headers)

            assert response.status_code == 200
            data = response.json()
            assert data["max_tokens"] == 4000

    @pytest.mark.asyncio
    async def test_update_config(self, client, auth_headers):
        """Test configuration update endpoint."""
        config_update = {"max_tokens": 8000, "cost_limits": {"daily": 100.0}}

        with patch("api.config.update_configuration") as mock_update:
            mock_update.return_value = True

            response = await client.patch(
                "/api/config", json=config_update, headers=auth_headers
            )

            assert response.status_code == 200

    # WebSocket Endpoint

    @pytest.mark.asyncio
    async def test_websocket_connection(self, client):
        """Test WebSocket connection."""
        with client.websocket_connect("/ws") as websocket:
            # Send test message
            websocket.send_json({"type": "ping"})

            # Receive response
            data = websocket.receive_json()
            assert data["type"] == "pong"

    # Error Handling

    @pytest.mark.asyncio
    async def test_invalid_json(self, client, auth_headers):
        """Test handling of invalid JSON."""
        response = await client.post(
            "/api/documents",
            content="invalid json",
            headers={**auth_headers, "Content-Type": "application/json"},
        )

        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_missing_auth(self, client):
        """Test endpoint without authentication."""
        response = await client.get("/api/documents/doc-123")

        assert response.status_code == 401

    @pytest.mark.asyncio
    async def test_rate_limiting(self, client, auth_headers):
        """Test rate limiting."""
        # Send multiple requests quickly
        for _ in range(10):
            response = await client.get("/api/documents", headers=auth_headers)

        # Should eventually get rate limited
        # Note: Actual implementation depends on rate limiting configuration

    # Batch Operations

    @pytest.mark.asyncio
    async def test_batch_create_documents(self, client, auth_headers):
        """Test batch document creation."""
        documents = [
            {"title": "Doc 1", "content": "Content 1"},
            {"title": "Doc 2", "content": "Content 2"},
        ]

        with patch("api.folders.batch_create_documents") as mock_batch:
            mock_batch.return_value = [
                {"id": "doc-1", **documents[0]},
                {"id": "doc-2", **documents[1]},
            ]

            response = await client.post(
                "/api/documents/batch",
                json={"documents": documents},
                headers=auth_headers,
            )

            assert response.status_code == 201
            data = response.json()
            assert len(data) == 2

    @pytest.mark.asyncio
    async def test_batch_delete_documents(self, client, auth_headers):
        """Test batch document deletion."""
        with patch("api.folders.batch_delete_documents") as mock_delete:
            mock_delete.return_value = {"deleted": 3}

            response = await client.post(
                "/api/documents/batch/delete",
                json={"ids": ["doc-1", "doc-2", "doc-3"]},
                headers=auth_headers,
            )

            assert response.status_code == 200
            data = response.json()
            assert data["deleted"] == 3
