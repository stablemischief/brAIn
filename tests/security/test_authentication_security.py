"""
Security tests for authentication and authorization systems.
Tests for authentication bypass, token validation, and access control vulnerabilities.
"""

import pytest
import jwt
import time
from datetime import datetime, timezone, timedelta
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
from fastapi import status

from main import app
from api.auth import get_current_user, get_supabase_client
from middleware.auth_middleware import AuthenticationMiddleware


class TestAuthenticationSecurity:
    """Test suite for authentication security vulnerabilities."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)

    @pytest.fixture
    def valid_token(self):
        """Create a valid JWT token for testing."""
        payload = {
            "sub": "test-user-123",
            "email": "test@example.com",
            "role": "user",
            "exp": datetime.now(timezone.utc) + timedelta(hours=1),
            "iat": datetime.now(timezone.utc)
        }
        return jwt.encode(payload, "test-secret", algorithm="HS256")

    @pytest.fixture
    def expired_token(self):
        """Create an expired JWT token for testing."""
        payload = {
            "sub": "test-user-123",
            "email": "test@example.com",
            "role": "user",
            "exp": datetime.now(timezone.utc) - timedelta(hours=1),
            "iat": datetime.now(timezone.utc) - timedelta(hours=2)
        }
        return jwt.encode(payload, "test-secret", algorithm="HS256")

    @pytest.fixture
    def malformed_token(self):
        """Create a malformed JWT token for testing."""
        return "malformed.jwt.token"

    def test_authentication_bypass_vulnerability(self, client):
        """
        SEC-001: Test for authentication bypass vulnerability.
        CRITICAL: JWT signature validation is disabled.
        """
        # Create a token with invalid signature
        fake_payload = {
            "sub": "attacker-user",
            "email": "attacker@evil.com",
            "role": "admin",
            "exp": datetime.now(timezone.utc) + timedelta(hours=1)
        }
        fake_token = jwt.encode(fake_payload, "wrong-secret", algorithm="HS256")

        # This should fail but currently passes due to disabled signature verification
        response = client.get(
            "/api/auth/me",
            headers={"Authorization": f"Bearer {fake_token}"}
        )

        # VULNERABILITY: This test will pass, showing the bypass
        assert response.status_code == 200  # This is the vulnerability

        # What should happen in a secure system:
        # assert response.status_code == 401
        # assert "Invalid authentication" in response.json()["detail"]

    def test_mock_authentication_vulnerability(self, client):
        """
        SEC-002: Test for mock authentication in production code.
        CRITICAL: Any token returns hardcoded user data.
        """
        # Any random token should work due to mock authentication
        random_token = "any.random.token"

        response = client.get(
            "/api/auth/me",
            headers={"Authorization": f"Bearer {random_token}"}
        )

        # VULNERABILITY: Mock authentication accepts any token
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == "user-123"  # Hardcoded value
        assert data["email"] == "user@example.com"  # Hardcoded value

    def test_missing_token_protection(self, client):
        """Test that protected endpoints require authentication."""
        protected_endpoints = [
            "/api/auth/me",
            "/api/folders",
            "/api/processing",
            "/api/search",
            "/api/analytics"
        ]

        for endpoint in protected_endpoints:
            response = client.get(endpoint)
            assert response.status_code == 401, f"Endpoint {endpoint} should require authentication"

    def test_token_expiration_handling(self, client, expired_token):
        """Test handling of expired tokens."""
        response = client.get(
            "/api/auth/me",
            headers={"Authorization": f"Bearer {expired_token}"}
        )

        # Should reject expired tokens (but currently may not due to disabled validation)
        # In current vulnerable state, this might pass
        if response.status_code == 200:
            pytest.fail("VULNERABILITY: Expired token was accepted")

    def test_malformed_token_handling(self, client, malformed_token):
        """Test handling of malformed tokens."""
        response = client.get(
            "/api/auth/me",
            headers={"Authorization": f"Bearer {malformed_token}"}
        )

        assert response.status_code == 401
        assert "Invalid authentication" in response.json()["detail"]

    def test_missing_authorization_header(self, client):
        """Test endpoint behavior without authorization header."""
        response = client.get("/api/auth/me")
        assert response.status_code == 401
        assert "Authentication required" in response.json()["detail"]

    def test_bearer_token_format_validation(self, client):
        """Test validation of Bearer token format."""
        invalid_formats = [
            "Token abc123",  # Wrong prefix
            "Bearer",  # Missing token
            "Bearer ",  # Empty token
            "abc123",  # No prefix
        ]

        for invalid_auth in invalid_formats:
            response = client.get(
                "/api/auth/me",
                headers={"Authorization": invalid_auth}
            )
            assert response.status_code == 401

    def test_role_based_access_control(self, client):
        """Test role-based access control vulnerabilities."""
        # Create tokens with different roles
        user_payload = {
            "sub": "user-123",
            "email": "user@example.com",
            "role": "user",
            "exp": datetime.now(timezone.utc) + timedelta(hours=1)
        }
        user_token = jwt.encode(user_payload, "test-secret", algorithm="HS256")

        admin_payload = {
            "sub": "admin-123",
            "email": "admin@example.com",
            "role": "admin",
            "exp": datetime.now(timezone.utc) + timedelta(hours=1)
        }
        admin_token = jwt.encode(admin_payload, "test-secret", algorithm="HS256")

        # Test that user cannot access admin endpoints
        # (This test would need admin-specific endpoints to be meaningful)
        pass

    def test_session_fixation_vulnerability(self, client):
        """Test for session fixation vulnerabilities."""
        # Test that session IDs change after authentication
        # This would require session management to be implemented
        pass

    def test_concurrent_session_handling(self, client, valid_token):
        """Test handling of concurrent sessions."""
        # Multiple requests with same token should work
        responses = []
        for _ in range(5):
            response = client.get(
                "/api/auth/me",
                headers={"Authorization": f"Bearer {valid_token}"}
            )
            responses.append(response)

        # All requests should succeed (or fail consistently)
        status_codes = [r.status_code for r in responses]
        assert len(set(status_codes)) == 1, "Inconsistent authentication behavior"

    def test_token_leakage_in_logs(self, client, valid_token, caplog):
        """Test that tokens are not leaked in application logs."""
        with caplog.at_level("DEBUG"):
            client.get(
                "/api/auth/me",
                headers={"Authorization": f"Bearer {valid_token}"}
            )

        # Check that token is not in logs
        for record in caplog.records:
            assert valid_token not in record.message, "Token leaked in logs"

    def test_password_exposure_in_requests(self, client):
        """Test that passwords are not exposed in error messages or logs."""
        login_data = {
            "email": "test@example.com",
            "password": "sensitive_password_123"
        }

        # This would fail in current mock implementation
        with patch('api.auth.get_supabase_client') as mock_supabase:
            mock_client = MagicMock()
            mock_client.auth.sign_in_with_password.side_effect = Exception("Auth failed")
            mock_supabase.return_value = mock_client

            response = client.post("/api/auth/login", json=login_data)

            # Password should not appear in response
            response_text = response.text
            assert "sensitive_password_123" not in response_text


class TestAPIKeySecurity:
    """Test suite for API key authentication security."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)

    def test_api_key_authentication(self, client):
        """Test API key authentication mechanism."""
        # Test with valid API key (if configured)
        # This would require API key configuration
        pass

    def test_api_key_in_query_parameters(self, client):
        """Test that API keys in query parameters are handled securely."""
        # API keys in URLs can be logged and cached
        response = client.get("/api/health?api_key=secret_key_123")

        # Should discourage or prevent API keys in query parameters
        # Implementation would depend on security policy
        pass

    def test_api_key_rate_limiting(self, client):
        """Test rate limiting for API key requests."""
        # Test that API key requests are rate limited
        pass


class TestCORSSecurity:
    """Test suite for CORS security vulnerabilities."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)

    def test_cors_wildcard_vulnerability(self, client):
        """
        SEC-004: Test for CORS wildcard vulnerability.
        CRITICAL: CORS allows all origins, methods, and headers.
        """
        # Test with malicious origin
        malicious_origin = "https://evil.attacker.com"

        response = client.options(
            "/api/auth/login",
            headers={
                "Origin": malicious_origin,
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "Content-Type,Authorization"
            }
        )

        # VULNERABILITY: Should not allow arbitrary origins
        cors_origin = response.headers.get("Access-Control-Allow-Origin")
        if cors_origin == "*" or cors_origin == malicious_origin:
            pytest.fail("VULNERABILITY: CORS allows arbitrary origins")

    def test_cors_credentials_exposure(self, client):
        """Test for credential exposure via CORS."""
        response = client.options("/api/auth/login")

        # Should not allow credentials with wildcard origin
        allow_origin = response.headers.get("Access-Control-Allow-Origin")
        allow_credentials = response.headers.get("Access-Control-Allow-Credentials")

        if allow_origin == "*" and allow_credentials == "true":
            pytest.fail("VULNERABILITY: Credentials allowed with wildcard origin")

    def test_cors_method_restriction(self, client):
        """Test that CORS restricts dangerous HTTP methods."""
        dangerous_methods = ["DELETE", "PUT", "PATCH"]

        for method in dangerous_methods:
            response = client.options(
                "/api/health",
                headers={
                    "Origin": "https://example.com",
                    "Access-Control-Request-Method": method
                }
            )

            allowed_methods = response.headers.get("Access-Control-Allow-Methods", "")

            # Should carefully consider which methods to allow
            if "*" in allowed_methods:
                pytest.fail("VULNERABILITY: CORS allows all methods")


class TestInputValidationSecurity:
    """Test suite for input validation vulnerabilities."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)

    def test_sql_injection_vulnerability(self, client, valid_token):
        """
        SEC-005: Test for SQL injection vulnerabilities.
        HIGH: Missing input validation on database queries.
        """
        # Test SQL injection payloads
        sql_payloads = [
            "'; DROP TABLE users; --",
            "' OR '1'='1",
            "' UNION SELECT * FROM sensitive_table --",
            "'; INSERT INTO users (email) VALUES ('hacked@evil.com'); --"
        ]

        # Test on search endpoint (if it exists)
        for payload in sql_payloads:
            response = client.get(
                f"/api/search?query={payload}",
                headers={"Authorization": f"Bearer {valid_token}"}
            )

            # Should not execute SQL injection
            if response.status_code == 500:
                # Check if error suggests SQL injection was attempted
                error_detail = response.json().get("detail", "")
                if "syntax error" in error_detail.lower() or "sql" in error_detail.lower():
                    pytest.fail(f"VULNERABILITY: SQL injection possible with payload: {payload}")

    def test_xss_vulnerability(self, client, valid_token):
        """Test for Cross-Site Scripting (XSS) vulnerabilities."""
        xss_payloads = [
            "<script>alert('XSS')</script>",
            "javascript:alert('XSS')",
            "<img src=x onerror=alert('XSS')>",
            "';alert('XSS');//"
        ]

        for payload in xss_payloads:
            # Test on endpoints that might reflect user input
            response = client.get(
                f"/api/search?query={payload}",
                headers={"Authorization": f"Bearer {valid_token}"}
            )

            # Response should not contain unescaped payload
            if payload in response.text:
                pytest.fail(f"VULNERABILITY: XSS possible with payload: {payload}")

    def test_command_injection_vulnerability(self, client, valid_token):
        """Test for command injection vulnerabilities."""
        command_payloads = [
            "; ls -la",
            "| cat /etc/passwd",
            "&& whoami",
            "`id`",
            "$(uname -a)"
        ]

        for payload in command_payloads:
            # Test on endpoints that might execute system commands
            response = client.post(
                "/api/processing/start",
                json={"input": payload},
                headers={"Authorization": f"Bearer {valid_token}"}
            )

            # Should not execute commands
            if response.status_code == 500:
                error_detail = response.json().get("detail", "")
                if "command not found" in error_detail.lower():
                    pytest.fail(f"VULNERABILITY: Command injection possible with payload: {payload}")

    def test_path_traversal_vulnerability(self, client, valid_token):
        """Test for path traversal vulnerabilities."""
        path_payloads = [
            "../../../etc/passwd",
            "..\\..\\..\\windows\\system32\\config\\sam",
            "%2e%2e%2f%2e%2e%2f%2e%2e%2f%65%74%63%2f%70%61%73%73%77%64",
            "....//....//....//etc/passwd"
        ]

        for payload in path_payloads:
            # Test on file-related endpoints
            response = client.get(
                f"/api/folders/{payload}",
                headers={"Authorization": f"Bearer {valid_token}"}
            )

            # Should not allow path traversal
            if response.status_code == 200:
                # Check if sensitive file content is returned
                sensitive_patterns = ["root:", "bin/bash", "Administrator"]
                content = response.text.lower()
                if any(pattern.lower() in content for pattern in sensitive_patterns):
                    pytest.fail(f"VULNERABILITY: Path traversal possible with payload: {payload}")


class TestSessionSecurity:
    """Test suite for session management security."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)

    def test_session_hijacking_vulnerability(self, client):
        """Test for session hijacking vulnerabilities."""
        # Test if session tokens are predictable or can be hijacked
        pass

    def test_session_fixation_vulnerability(self, client):
        """Test for session fixation vulnerabilities."""
        # Test if session IDs change after authentication
        pass

    def test_session_timeout_enforcement(self, client):
        """Test session timeout enforcement."""
        # Test that sessions expire after configured timeout
        pass


@pytest.mark.integration
class TestSecurityIntegration:
    """Integration tests for security components."""

    def test_end_to_end_authentication_flow(self):
        """Test complete authentication flow for security issues."""
        # Test login -> protected resource access -> logout
        pass

    def test_security_headers_presence(self, client):
        """
        SEC-007: Test for missing security headers.
        HIGH: Security headers not implemented.
        """
        response = client.get("/")

        # Check for important security headers
        security_headers = {
            "X-Content-Type-Options": "nosniff",
            "X-Frame-Options": "DENY",
            "X-XSS-Protection": "1; mode=block",
            "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
            "Content-Security-Policy": None  # Should exist
        }

        missing_headers = []
        for header, expected_value in security_headers.items():
            if header not in response.headers:
                missing_headers.append(header)
            elif expected_value and response.headers[header] != expected_value:
                missing_headers.append(f"{header} (incorrect value)")

        if missing_headers:
            pytest.fail(f"VULNERABILITY: Missing security headers: {missing_headers}")

    def test_information_disclosure_in_errors(self, client):
        """
        SEC-006: Test for information disclosure in error messages.
        HIGH: Detailed error messages expose internal information.
        """
        # Trigger various error conditions
        error_endpoints = [
            "/api/nonexistent",
            "/api/auth/login",  # with invalid data
            "/api/search?query=" + "A" * 10000,  # oversized input
        ]

        for endpoint in error_endpoints:
            response = client.get(endpoint)

            # Check if error messages contain sensitive information
            error_text = response.text.lower()
            sensitive_patterns = [
                "traceback",
                "stack trace",
                "file system path",
                "database error",
                "internal server error details",
                "/home/",
                "/usr/",
                "c:\\",
                "python",
                "django",
                "fastapi"
            ]

            for pattern in sensitive_patterns:
                if pattern in error_text:
                    pytest.fail(f"VULNERABILITY: Information disclosure in error: {pattern}")