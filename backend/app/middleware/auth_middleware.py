"""
Authentication middleware for brAIn v2.0
Handles JWT token validation and user authentication with Supabase
"""

import logging
from typing import Optional, Callable
from datetime import datetime, timezone

from fastapi import Request, Response, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from starlette.middleware.base import BaseHTTPMiddleware
from supabase import create_client, Client
import jwt

from app.config.settings import get_settings

logger = logging.getLogger(__name__)

# Security scheme for JWT tokens
security = HTTPBearer(auto_error=False)


class AuthenticationMiddleware(BaseHTTPMiddleware):
    """
    Middleware for handling authentication and authorization.
    Validates JWT tokens and injects user context into requests.
    """

    def __init__(self, app):
        super().__init__(app)
        self.settings = get_settings()
        self.supabase_client = self._get_supabase_client()

        # Routes that don't require authentication
        self.public_routes = {
            "/",
            "/docs",
            "/redoc",
            "/openapi.json",
            "/health",
            "/api/health",
            "/api/health/",
            "/api/auth/login",
            "/api/auth/refresh",
        }

        # Route prefixes that don't require authentication
        self.public_prefixes = ["/static/", "/docs/", "/redoc/", "/api/health/"]

    def _get_supabase_client(self) -> Optional[Client]:
        """Get Supabase client instance."""
        try:
            if self.settings.supabase_url and self.settings.supabase_anon_key:
                return create_client(
                    self.settings.supabase_url, self.settings.supabase_anon_key
                )
        except Exception as e:
            logger.warning(f"Failed to initialize Supabase client: {e}")
        return None

    def _is_public_route(self, path: str) -> bool:
        """Check if a route is public (doesn't require authentication)."""
        # Check exact matches
        if path in self.public_routes:
            return True

        # Check prefix matches
        return any(path.startswith(prefix) for prefix in self.public_prefixes)

    def _extract_token_from_request(self, request: Request) -> Optional[str]:
        """Extract JWT token from request headers."""
        # Try Authorization header first
        auth_header = request.headers.get("Authorization")
        if auth_header and auth_header.startswith("Bearer "):
            return auth_header[7:]  # Remove "Bearer " prefix

        # Try cookies as fallback
        return request.cookies.get("access_token")

    async def _validate_token(self, token: str) -> Optional[dict]:
        """Validate JWT token with proper signature verification."""
        try:
            if not self.supabase_client:
                logger.warning("Supabase client not available for token validation")
                return None

            # Get JWT secret for verification
            jwt_secret = self.settings.jwt_secret or self.settings.supabase_jwt_secret
            if not jwt_secret:
                logger.error("JWT secret not configured - cannot validate tokens")
                return None

            try:
                # Decode token WITH signature verification (SECURITY FIX)
                decoded_token = jwt.decode(
                    token,
                    jwt_secret,
                    algorithms=["HS256"],  # Explicitly specify allowed algorithms
                    options={
                        "verify_signature": True,
                        "verify_exp": True,
                        "verify_iat": True,
                        "verify_aud": False,  # Can be enabled if audience is configured
                        "require": ["sub", "exp", "iat"],  # Require essential claims
                    },
                )

                # Extract user information
                user_id = decoded_token.get("sub")
                email = decoded_token.get("email")
                role = decoded_token.get("role", "authenticated")

                # Validate essential claims
                if not user_id:
                    logger.warning("JWT token missing required 'sub' claim")
                    return None

                # Additional security: Check token age (optional)
                iat = decoded_token.get("iat")
                if iat:
                    current_time = datetime.now(timezone.utc).timestamp()
                    # Reject tokens older than 24 hours even if not expired
                    if current_time - iat > 86400:  # 24 hours
                        logger.warning("JWT token is older than maximum allowed age")
                        return None

                return {
                    "user_id": user_id,
                    "email": email or f"user-{user_id}@system.local",
                    "role": role,
                    "authenticated": True,
                    "token_exp": decoded_token.get("exp"),
                    "token_iat": decoded_token.get("iat"),
                    "auth_method": "jwt",
                }

            except jwt.ExpiredSignatureError:
                logger.warning("JWT token has expired")
                return None
            except jwt.InvalidSignatureError:
                logger.warning("JWT token has invalid signature")
                return None
            except jwt.InvalidTokenError as e:
                logger.warning(f"Invalid JWT token: {e}")
                return None
            except jwt.MissingRequiredClaimError as e:
                logger.warning(f"JWT token missing required claim: {e}")
                return None

        except Exception as e:
            logger.error(f"Token validation error: {e}")
            return None

        return None

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process the request through authentication middleware."""
        path = request.url.path

        # Skip authentication for public routes
        if self._is_public_route(path):
            return await call_next(request)

        # Extract token from request
        token = self._extract_token_from_request(request)

        if not token:
            # No token provided for protected route
            logger.info(f"No authentication token provided for protected route: {path}")
            return Response(
                content='{"error": "Authentication required", "detail": "No authentication token provided"}',
                status_code=status.HTTP_401_UNAUTHORIZED,
                headers={"content-type": "application/json"},
            )

        # Validate token
        user_info = await self._validate_token(token)

        if not user_info:
            # Invalid token
            logger.info(f"Invalid authentication token for route: {path}")
            return Response(
                content='{"error": "Invalid authentication", "detail": "Authentication token is invalid or expired"}',
                status_code=status.HTTP_401_UNAUTHORIZED,
                headers={"content-type": "application/json"},
            )

        # Check token expiration
        if user_info.get("token_exp"):
            exp_timestamp = user_info["token_exp"]
            current_timestamp = datetime.now(timezone.utc).timestamp()

            if exp_timestamp < current_timestamp:
                logger.info(f"Expired authentication token for route: {path}")
                return Response(
                    content='{"error": "Token expired", "detail": "Authentication token has expired"}',
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    headers={"content-type": "application/json"},
                )

        # Add user context to request state
        request.state.user = user_info
        request.state.authenticated = True

        # Log successful authentication
        logger.debug(f"Authenticated user {user_info['email']} for route: {path}")

        # Continue to the next middleware/route handler
        response = await call_next(request)

        # Add authentication headers to response
        response.headers["X-User-ID"] = user_info["user_id"]
        response.headers["X-User-Email"] = user_info["email"]

        return response


def get_current_user_from_request(request: Request) -> Optional[dict]:
    """
    Extract current user information from request state.
    Used by dependency injection in route handlers.
    """
    return getattr(request.state, "user", None)


def require_authentication(request: Request) -> dict:
    """
    Dependency function that requires authentication.
    Raises HTTPException if user is not authenticated.
    """
    user = get_current_user_from_request(request)

    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return user


def require_role(required_role: str):
    """
    Dependency factory that requires a specific user role.
    Returns a dependency function that checks for the required role.
    """

    def role_dependency(request: Request) -> dict:
        user = require_authentication(request)

        user_role = user.get("role", "")

        # Admin role has access to everything
        if user_role == "admin":
            return user

        # Check specific role requirement
        if user_role != required_role:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Insufficient permissions. Required role: {required_role}",
            )

        return user

    return role_dependency


# Pre-defined role dependencies
require_admin = require_role("admin")
require_user = require_role("authenticated")


class APIKeyAuthenticationMiddleware(BaseHTTPMiddleware):
    """
    Alternative authentication middleware for API key based authentication.
    Can be used alongside or instead of JWT authentication.
    """

    def __init__(self, app):
        super().__init__(app)
        self.settings = get_settings()
        self.api_keys = self._load_api_keys()

    def _load_api_keys(self) -> dict:
        """Load valid API keys from configuration."""
        # In production, this would load from a secure store
        return (
            {self.settings.admin_api_key: {"role": "admin", "name": "Admin API Key"}}
            if self.settings.admin_api_key
            else {}
        )

    def _extract_api_key(self, request: Request) -> Optional[str]:
        """Extract API key from request headers."""
        # Try X-API-Key header
        api_key = request.headers.get("X-API-Key")
        if api_key:
            return api_key

        # Try query parameter as fallback
        return request.query_params.get("api_key")

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process the request through API key authentication middleware."""
        # Only process API routes
        if not request.url.path.startswith("/api/"):
            return await call_next(request)

        # Check if JWT authentication is already present
        if hasattr(request.state, "authenticated") and request.state.authenticated:
            return await call_next(request)

        # Extract API key
        api_key = self._extract_api_key(request)

        if api_key and api_key in self.api_keys:
            # Valid API key found
            key_info = self.api_keys[api_key]

            # Add API key context to request state
            request.state.api_authenticated = True
            request.state.api_key_info = key_info

            # Create user-like context for consistency
            request.state.user = {
                "user_id": f"api_key_{api_key[:8]}",
                "email": f"api_key@{key_info['name'].lower().replace(' ', '_')}",
                "role": key_info["role"],
                "authenticated": True,
                "auth_type": "api_key",
            }
            request.state.authenticated = True

            logger.debug(f"API key authentication successful for {key_info['name']}")

        return await call_next(request)
