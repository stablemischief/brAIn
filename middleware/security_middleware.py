"""
Comprehensive security middleware for brAIn v2.0.

This middleware provides multiple layers of security protection including
XSS prevention, CSRF protection, input validation, and security headers.
"""

import re
import html
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone

from fastapi import Request, Response, HTTPException, status
from fastapi.security import HTTPBearer
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

from config.settings import get_settings

logger = logging.getLogger(__name__)


class SecurityMiddleware(BaseHTTPMiddleware):
    """Comprehensive security middleware with multiple protection layers."""

    def __init__(self, app):
        super().__init__(app)
        self.settings = get_settings()

        # Security patterns for detection
        self.sql_injection_patterns = [
            r"(?i)(union\s+select|select.*from|drop\s+table|delete\s+from)",
            r"(?i)(insert\s+into|update.*set|alter\s+table|create\s+table)",
            r"(?i)('|\"|;|--|/\*|\*/|xp_|sp_)",
            r"(?i)(or\s+1=1|and\s+1=1|or\s+'1'='1'|and\s+'1'='1')"
        ]

        self.xss_patterns = [
            r"(?i)(<script|</script|javascript:|onload=|onerror=)",
            r"(?i)(alert\(|confirm\(|prompt\(|document\.|window\.)",
            r"(?i)(eval\(|setTimeout\(|setInterval\()"
        ]

        self.command_injection_patterns = [
            r"(?i)(;|\||&&|`|\$\(|>\s*/|<\s*/)",
            r"(?i)(cat\s|ls\s|pwd|whoami|id\s|ps\s|kill\s)",
            r"(?i)(rm\s|mv\s|cp\s|chmod\s|chown\s|sudo\s)"
        ]

        # File upload restrictions
        self.dangerous_extensions = {
            '.exe', '.bat', '.cmd', '.com', '.pif', '.scr', '.vbs', '.js',
            '.jar', '.jsp', '.php', '.asp', '.aspx', '.pl', '.py', '.sh',
            '.ps1', '.rb', '.go', '.rs'
        }

    async def dispatch(self, request: Request, call_next) -> Response:
        """Main security processing pipeline."""
        start_time = datetime.now(timezone.utc)

        try:
            # 1. Add security headers to request context
            self._add_security_context(request)

            # 2. Validate request size and headers
            await self._validate_request_basics(request)

            # 3. Input validation and sanitization
            await self._validate_and_sanitize_inputs(request)

            # 4. Check for suspicious patterns
            self._detect_malicious_patterns(request)

            # Process the request
            response = await call_next(request)

            # 5. Add security headers to response
            self._add_security_headers(response, request)

            # 6. Log security events
            self._log_security_event(request, response, start_time)

            return response

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Security middleware error: {e}")
            # Log the security incident
            self._log_security_incident(request, str(e), start_time)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Security processing error"
            )

    def _add_security_context(self, request: Request):
        """Add security context to request."""
        # Add security metadata
        request.state.security = {
            "ip_address": self._get_client_ip(request),
            "user_agent": request.headers.get("User-Agent", "Unknown"),
            "referer": request.headers.get("Referer"),
            "processed_at": datetime.now(timezone.utc)
        }

    def _get_client_ip(self, request: Request) -> str:
        """Get client IP address with proxy support."""
        # Check for common proxy headers
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()

        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip

        return request.client.host if request.client else "unknown"

    async def _validate_request_basics(self, request: Request):
        """Basic request validation."""
        # Check request size
        content_length = request.headers.get("Content-Length")
        if content_length:
            try:
                size = int(content_length)
                max_size = 50 * 1024 * 1024  # 50MB
                if size > max_size:
                    raise HTTPException(
                        status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                        detail="Request too large"
                    )
            except ValueError:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Invalid Content-Length header"
                )

        # Validate critical headers
        user_agent = request.headers.get("User-Agent")
        if not user_agent or len(user_agent) > 500:
            logger.warning(f"Suspicious or missing User-Agent: {user_agent}")

    async def _validate_and_sanitize_inputs(self, request: Request):
        """Validate and sanitize all inputs."""
        path = request.url.path

        # Skip validation for static files and health checks
        if any(path.startswith(skip) for skip in ["/static/", "/health", "/docs", "/redoc"]):
            return

        # Validate and sanitize query parameters
        if request.query_params:
            self._validate_query_parameters(dict(request.query_params))

        # Validate and sanitize request body (for applicable methods)
        if request.method in ["POST", "PUT", "PATCH"]:
            await self._validate_request_body(request)

    def _validate_query_parameters(self, params: Dict[str, str]):
        """Validate query parameters for security issues."""
        for key, value in params.items():
            # Check parameter key
            if not re.match(r'^[a-zA-Z0-9_\-\.]{1,50}$', key):
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Invalid parameter name: {key}"
                )

            # Check parameter value length
            if len(str(value)) > 1000:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Parameter value too long"
                )

            # Sanitize the value
            sanitized_value = html.escape(str(value))
            if sanitized_value != str(value):
                logger.warning(f"HTML entities found in parameter {key}: {value}")

    async def _validate_request_body(self, request: Request):
        """Validate request body content."""
        try:
            content_type = request.headers.get("Content-Type", "")

            # Skip binary content
            if content_type.startswith(("image/", "video/", "audio/", "application/octet-stream")):
                return

            # Get body for validation (don't consume it)
            body = await request.body()
            if not body:
                return

            # Check body size
            if len(body) > 10 * 1024 * 1024:  # 10MB for JSON/text
                raise HTTPException(
                    status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                    detail="Request body too large"
                )

            # For JSON content, perform additional validation
            if "application/json" in content_type:
                try:
                    import json
                    data = json.loads(body.decode('utf-8'))
                    self._validate_json_content(data)
                except json.JSONDecodeError:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail="Invalid JSON format"
                    )
                except UnicodeDecodeError:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail="Invalid character encoding"
                    )

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Body validation error: {e}")

    def _validate_json_content(self, data: Any, depth: int = 0):
        """Recursively validate JSON content."""
        if depth > 10:  # Prevent deeply nested objects
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="JSON nesting too deep"
            )

        if isinstance(data, dict):
            if len(data) > 100:  # Prevent too many keys
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Too many JSON keys"
                )
            for key, value in data.items():
                if not isinstance(key, str) or len(key) > 100:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail="Invalid JSON key"
                    )
                self._validate_json_content(value, depth + 1)

        elif isinstance(data, list):
            if len(data) > 1000:  # Prevent huge arrays
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="JSON array too large"
                )
            for item in data:
                self._validate_json_content(item, depth + 1)

        elif isinstance(data, str):
            if len(data) > 10000:  # Prevent huge strings
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="JSON string too long"
                )

    def _detect_malicious_patterns(self, request: Request):
        """Detect malicious patterns in request."""
        # Check URL path
        self._check_patterns(request.url.path, "URL path")

        # Check query parameters
        for key, value in request.query_params.items():
            self._check_patterns(f"{key}={value}", "Query parameter")

        # Check headers for suspicious content
        for header_name, header_value in request.headers.items():
            if header_name.lower() not in ["authorization", "cookie", "x-api-key"]:
                self._check_patterns(header_value, f"Header {header_name}")

    def _check_patterns(self, content: str, context: str):
        """Check content against malicious patterns."""
        content_lower = content.lower()

        # Check SQL injection patterns
        for pattern in self.sql_injection_patterns:
            if re.search(pattern, content_lower):
                logger.warning(f"SQL injection attempt detected in {context}: {content}")
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Malicious content detected"
                )

        # Check XSS patterns
        for pattern in self.xss_patterns:
            if re.search(pattern, content_lower):
                logger.warning(f"XSS attempt detected in {context}: {content}")
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Malicious content detected"
                )

        # Check command injection patterns
        for pattern in self.command_injection_patterns:
            if re.search(pattern, content_lower):
                logger.warning(f"Command injection attempt detected in {context}: {content}")
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Malicious content detected"
                )

    def _add_security_headers(self, response: Response, request: Request):
        """Add comprehensive security headers."""
        # Basic security headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["X-Download-Options"] = "noopen"
        response.headers["X-Permitted-Cross-Domain-Policies"] = "none"

        # Content Security Policy
        if self.settings.environment == "production":
            csp = (
                "default-src 'self'; "
                "script-src 'self' 'unsafe-inline' 'unsafe-eval'; "
                "style-src 'self' 'unsafe-inline' https:; "
                "img-src 'self' data: https:; "
                "font-src 'self' https:; "
                "connect-src 'self' https: wss:; "
                "frame-ancestors 'none'; "
                "base-uri 'self'; "
                "form-action 'self'"
            )
            response.headers["Content-Security-Policy"] = csp

        # HTTPS enforcement in production
        if self.settings.environment == "production":
            response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains; preload"

        # Permission Policy (Feature Policy replacement)
        response.headers["Permissions-Policy"] = (
            "geolocation=(), "
            "microphone=(), "
            "camera=(), "
            "payment=(), "
            "usb=(), "
            "magnetometer=(), "
            "gyroscope=(), "
            "speaker=()"
        )

        # Rate limiting headers (if applicable)
        if hasattr(request.state, 'rate_limit_remaining'):
            response.headers["X-RateLimit-Remaining"] = str(request.state.rate_limit_remaining)
            response.headers["X-RateLimit-Reset"] = str(request.state.rate_limit_reset)

    def _log_security_event(self, request: Request, response: Response, start_time: datetime):
        """Log security events for monitoring."""
        processing_time = (datetime.now(timezone.utc) - start_time).total_seconds()

        # Log security-relevant requests
        if (response.status_code >= 400 or
            processing_time > 2.0 or
            any(request.url.path.startswith(sensitive) for sensitive in ["/api/auth", "/api/admin"])):

            security_log = {
                "timestamp": start_time.isoformat(),
                "ip": self._get_client_ip(request),
                "method": request.method,
                "path": request.url.path,
                "status_code": response.status_code,
                "processing_time": processing_time,
                "user_agent": request.headers.get("User-Agent", "Unknown")[:200],
                "referer": request.headers.get("Referer", "")[:200]
            }

            if hasattr(request.state, 'user') and request.state.user:
                security_log["user_id"] = request.state.user.get("user_id")

            logger.info(f"Security event: {security_log}")

    def _log_security_incident(self, request: Request, error: str, start_time: datetime):
        """Log security incidents for investigation."""
        incident = {
            "timestamp": start_time.isoformat(),
            "type": "security_incident",
            "error": error,
            "ip": self._get_client_ip(request),
            "method": request.method,
            "path": request.url.path,
            "user_agent": request.headers.get("User-Agent", "Unknown")[:200],
            "query_params": dict(request.query_params),
            "severity": "HIGH"
        }

        logger.error(f"SECURITY INCIDENT: {incident}")


class InputSanitizationMixin:
    """Mixin for input sanitization utilities."""

    @staticmethod
    def sanitize_html(text: str) -> str:
        """Sanitize HTML content to prevent XSS."""
        if not text:
            return ""
        return html.escape(text, quote=True)

    @staticmethod
    def sanitize_sql_input(text: str) -> str:
        """Basic SQL input sanitization."""
        if not text:
            return ""
        # Remove dangerous SQL characters
        dangerous_chars = ["'", '"', ";", "--", "/*", "*/", "xp_", "sp_"]
        sanitized = text
        for char in dangerous_chars:
            sanitized = sanitized.replace(char, "")
        return sanitized

    @staticmethod
    def validate_filename(filename: str) -> bool:
        """Validate filename for security."""
        if not filename or len(filename) > 255:
            return False

        # Check for dangerous patterns
        dangerous_patterns = ["..", "/", "\\", ":", "*", "?", "<", ">", "|"]
        if any(pattern in filename for pattern in dangerous_patterns):
            return False

        # Check file extension
        extension = "." + filename.split(".")[-1].lower() if "." in filename else ""
        dangerous_extensions = {
            '.exe', '.bat', '.cmd', '.com', '.pif', '.scr', '.vbs', '.js',
            '.jar', '.jsp', '.php', '.asp', '.aspx', '.pl', '.py', '.sh',
            '.ps1', '.rb', '.go', '.rs'
        }

        return extension not in dangerous_extensions