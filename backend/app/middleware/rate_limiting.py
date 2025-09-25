"""
Rate limiting middleware for brAIn v2.0
Implements request rate limiting and throttling
"""

import time
import logging
from typing import Dict, Optional, Tuple
from collections import defaultdict, deque
from datetime import datetime, timedelta

from fastapi import Request, Response, HTTPException, status
from starlette.middleware.base import BaseHTTPMiddleware

from app.config.settings import get_settings

logger = logging.getLogger(__name__)


class RateLimitStore:
    """
    In-memory store for rate limiting data.
    In production, this should be replaced with Redis or similar.
    """

    def __init__(self):
        # Store request timestamps per client
        self.requests: Dict[str, deque] = defaultdict(lambda: deque())
        # Store blocked clients
        self.blocked: Dict[str, float] = {}
        # Store rate limit rules per endpoint
        self.rules: Dict[str, Dict[str, int]] = {}

    def add_request(self, client_id: str, timestamp: float):
        """Add a request timestamp for a client."""
        self.requests[client_id].append(timestamp)

    def get_request_count(self, client_id: str, window_seconds: int) -> int:
        """Get number of requests within the time window."""
        current_time = time.time()
        cutoff_time = current_time - window_seconds

        # Remove old requests
        while self.requests[client_id] and self.requests[client_id][0] < cutoff_time:
            self.requests[client_id].popleft()

        return len(self.requests[client_id])

    def is_blocked(self, client_id: str) -> bool:
        """Check if a client is currently blocked."""
        if client_id in self.blocked:
            if time.time() < self.blocked[client_id]:
                return True
            else:
                # Block has expired
                del self.blocked[client_id]
        return False

    def block_client(self, client_id: str, duration_seconds: int):
        """Block a client for a specific duration."""
        self.blocked[client_id] = time.time() + duration_seconds
        logger.warning(f"Client {client_id} blocked for {duration_seconds} seconds")

    def cleanup_old_data(self, max_age_seconds: int = 3600):
        """Clean up old request data to prevent memory leaks."""
        current_time = time.time()
        cutoff_time = current_time - max_age_seconds

        # Clean up old requests
        for client_id in list(self.requests.keys()):
            while (
                self.requests[client_id] and self.requests[client_id][0] < cutoff_time
            ):
                self.requests[client_id].popleft()

            # Remove empty queues
            if not self.requests[client_id]:
                del self.requests[client_id]

        # Clean up expired blocks
        expired_blocks = [
            client_id
            for client_id, expiry in self.blocked.items()
            if current_time >= expiry
        ]
        for client_id in expired_blocks:
            del self.blocked[client_id]


class RateLimitRule:
    """Rate limit rule configuration."""

    def __init__(
        self,
        requests_per_minute: int = 60,
        requests_per_hour: int = 1000,
        burst_requests: int = 10,
        burst_window_seconds: int = 10,
        block_duration_seconds: int = 300,
    ):
        self.requests_per_minute = requests_per_minute
        self.requests_per_hour = requests_per_hour
        self.burst_requests = burst_requests
        self.burst_window_seconds = burst_window_seconds
        self.block_duration_seconds = block_duration_seconds


class RateLimitingMiddleware(BaseHTTPMiddleware):
    """
    Rate limiting middleware with configurable rules per endpoint.
    """

    def __init__(self, app):
        super().__init__(app)
        self.settings = get_settings()
        self.store = RateLimitStore()
        self.enabled = self.settings.rate_limiting_enabled

        # Default rate limit rules
        self.default_rules = {
            "default": RateLimitRule(
                requests_per_minute=60,
                requests_per_hour=1000,
                burst_requests=10,
                burst_window_seconds=10,
            ),
            "auth": RateLimitRule(
                requests_per_minute=10,
                requests_per_hour=100,
                burst_requests=3,
                burst_window_seconds=60,
                block_duration_seconds=600,  # 10 minutes for auth endpoints
            ),
            "processing": RateLimitRule(
                requests_per_minute=30,
                requests_per_hour=500,
                burst_requests=5,
                burst_window_seconds=30,
            ),
            "search": RateLimitRule(
                requests_per_minute=120,
                requests_per_hour=2000,
                burst_requests=20,
                burst_window_seconds=10,
            ),
        }

        # Endpoint to rule mapping
        self.endpoint_rules = {
            "/api/auth/": "auth",
            "/api/processing/": "processing",
            "/api/search/": "search",
            "/api/analytics/": "search",  # Analytics uses same limits as search
        }

        # Last cleanup time
        self.last_cleanup = time.time()

    def _get_client_identifier(self, request: Request) -> str:
        """Get unique identifier for the client."""
        # Try to get user ID if authenticated
        if hasattr(request.state, "user") and request.state.user:
            return f"user_{request.state.user['user_id']}"

        # Fall back to IP address
        # In production behind a proxy, use X-Forwarded-For
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            # Take the first IP in the chain
            client_ip = forwarded_for.split(",")[0].strip()
        else:
            client_ip = request.client.host if request.client else "unknown"

        return f"ip_{client_ip}"

    def _get_rule_for_path(self, path: str) -> RateLimitRule:
        """Get rate limit rule for a specific path."""
        for prefix, rule_name in self.endpoint_rules.items():
            if path.startswith(prefix):
                return self.default_rules.get(rule_name, self.default_rules["default"])

        return self.default_rules["default"]

    def _check_rate_limit(
        self, client_id: str, rule: RateLimitRule
    ) -> Tuple[bool, Dict[str, str]]:
        """
        Check if request should be rate limited.
        Returns (is_allowed, headers).
        """
        current_time = time.time()

        # Check if client is blocked
        if self.store.is_blocked(client_id):
            return False, {
                "X-RateLimit-Blocked": "true",
                "X-RateLimit-Block-Expires": str(int(self.store.blocked[client_id])),
            }

        # Check burst rate (short window)
        burst_count = self.store.get_request_count(client_id, rule.burst_window_seconds)

        if burst_count >= rule.burst_requests:
            logger.warning(
                f"Burst rate limit exceeded for {client_id}: {burst_count}/{rule.burst_requests}"
            )
            self.store.block_client(client_id, rule.block_duration_seconds)
            return False, {
                "X-RateLimit-Burst-Exceeded": "true",
                "X-RateLimit-Burst-Limit": str(rule.burst_requests),
                "X-RateLimit-Burst-Window": str(rule.burst_window_seconds),
            }

        # Check minute rate limit
        minute_count = self.store.get_request_count(client_id, 60)
        if minute_count >= rule.requests_per_minute:
            logger.warning(
                f"Minute rate limit exceeded for {client_id}: {minute_count}/{rule.requests_per_minute}"
            )
            return False, {
                "X-RateLimit-Minute-Exceeded": "true",
                "X-RateLimit-Minute-Limit": str(rule.requests_per_minute),
                "X-RateLimit-Minute-Remaining": "0",
            }

        # Check hourly rate limit
        hour_count = self.store.get_request_count(client_id, 3600)
        if hour_count >= rule.requests_per_hour:
            logger.warning(
                f"Hour rate limit exceeded for {client_id}: {hour_count}/{rule.requests_per_hour}"
            )
            return False, {
                "X-RateLimit-Hour-Exceeded": "true",
                "X-RateLimit-Hour-Limit": str(rule.requests_per_hour),
                "X-RateLimit-Hour-Remaining": "0",
            }

        # Request is allowed
        headers = {
            "X-RateLimit-Minute-Limit": str(rule.requests_per_minute),
            "X-RateLimit-Minute-Remaining": str(
                rule.requests_per_minute - minute_count
            ),
            "X-RateLimit-Hour-Limit": str(rule.requests_per_hour),
            "X-RateLimit-Hour-Remaining": str(rule.requests_per_hour - hour_count),
            "X-RateLimit-Burst-Limit": str(rule.burst_requests),
            "X-RateLimit-Burst-Remaining": str(rule.burst_requests - burst_count),
        }

        return True, headers

    def _cleanup_if_needed(self):
        """Perform periodic cleanup of old data."""
        current_time = time.time()
        if current_time - self.last_cleanup > 300:  # Cleanup every 5 minutes
            self.store.cleanup_old_data()
            self.last_cleanup = current_time

    async def dispatch(self, request: Request, call_next) -> Response:
        """Process request through rate limiting middleware."""
        # Skip if rate limiting is disabled
        if not self.enabled:
            return await call_next(request)

        # Skip rate limiting for certain paths
        path = request.url.path
        if path in ["/", "/health", "/docs", "/redoc", "/openapi.json"]:
            return await call_next(request)

        # Periodic cleanup
        self._cleanup_if_needed()

        # Get client identifier and rule
        client_id = self._get_client_identifier(request)
        rule = self._get_rule_for_path(path)

        # Check rate limits
        is_allowed, headers = self._check_rate_limit(client_id, rule)

        if not is_allowed:
            # Rate limit exceeded
            logger.info(f"Rate limit exceeded for {client_id} on {path}")

            response_data = {
                "error": "Rate limit exceeded",
                "detail": "Too many requests. Please try again later.",
                "retry_after": headers.get("X-RateLimit-Block-Expires", "300"),
            }

            response = Response(
                content=str(response_data).replace("'", '"'),  # Simple JSON conversion
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                headers={
                    "content-type": "application/json",
                    "Retry-After": headers.get("X-RateLimit-Block-Expires", "300"),
                    **headers,
                },
            )

            return response

        # Record the request
        self.store.add_request(client_id, time.time())

        # Process the request
        response = await call_next(request)

        # Add rate limit headers to response
        for header_name, header_value in headers.items():
            response.headers[header_name] = header_value

        return response


class IPWhitelistMiddleware(BaseHTTPMiddleware):
    """
    Middleware to whitelist specific IP addresses or ranges.
    Useful for admin endpoints or internal APIs.
    """

    def __init__(self, app, whitelist: Optional[list] = None):
        super().__init__(app)
        self.whitelist = whitelist or []
        self.settings = get_settings()

        # Add configured admin IPs
        if self.settings.admin_ips:
            self.whitelist.extend(self.settings.admin_ips.split(","))

        # Always allow localhost for development
        if self.settings.environment == "development":
            self.whitelist.extend(["127.0.0.1", "::1", "localhost"])

    def _is_ip_whitelisted(self, ip: str) -> bool:
        """Check if an IP address is whitelisted."""
        if not self.whitelist:
            return True  # No whitelist means all IPs allowed

        # Simple IP matching (could be enhanced with CIDR support)
        return ip in self.whitelist

    def _get_client_ip(self, request: Request) -> str:
        """Get the client's IP address."""
        # Check X-Forwarded-For header (for proxy setups)
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()

        # Check X-Real-IP header
        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip.strip()

        # Fall back to direct connection IP
        return request.client.host if request.client else "unknown"

    async def dispatch(self, request: Request, call_next) -> Response:
        """Process request through IP whitelist middleware."""
        # Only apply to admin endpoints
        path = request.url.path
        if not path.startswith("/api/admin/"):
            return await call_next(request)

        client_ip = self._get_client_ip(request)

        if not self._is_ip_whitelisted(client_ip):
            logger.warning(f"IP address {client_ip} blocked for admin endpoint {path}")

            return Response(
                content='{"error": "Access denied", "detail": "IP address not whitelisted"}',
                status_code=status.HTTP_403_FORBIDDEN,
                headers={"content-type": "application/json"},
            )

        return await call_next(request)
