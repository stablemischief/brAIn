"""
Authentication API endpoints
Handles user authentication, session management, and security
"""

from datetime import datetime, timezone, timedelta
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.ext.asyncio import AsyncSession
import jwt
from supabase import create_client, Client

from config.settings import get_settings
from models.api import (
    LoginRequest,
    LoginResponse,
    UserResponse,
    TokenResponse,
    LogoutResponse,
)
from database.connection import get_database_session

router = APIRouter()
security = HTTPBearer(auto_error=False)


def get_supabase_client() -> Client:
    """Get Supabase client instance."""
    settings = get_settings()
    return create_client(settings.supabase_url, settings.supabase_anon_key)


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: AsyncSession = Depends(get_database_session),
) -> dict:
    """Get current authenticated user from JWT token."""
    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Authentication required"
        )

    try:
        # Verify JWT token with Supabase
        supabase = get_supabase_client()

        # In a real implementation, you would validate the token
        # For now, we'll create a mock user
        user_data = {
            "id": "user-123",
            "email": "user@example.com",
            "role": "user",
            "created_at": datetime.now(timezone.utc),
        }

        return user_data
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
        )


@router.post("/login", response_model=LoginResponse)
async def login(request: LoginRequest):
    """Authenticate user and return access token."""
    supabase = get_supabase_client()

    try:
        # Authenticate with Supabase
        response = supabase.auth.sign_in_with_password(
            {"email": request.email, "password": request.password}
        )

        if response.user:
            return LoginResponse(
                access_token=response.session.access_token,
                refresh_token=response.session.refresh_token,
                token_type="bearer",
                expires_in=3600,
                user=UserResponse(
                    id=response.user.id, email=response.user.email, role="user"
                ),
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid email or password",
            )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Authentication failed"
        )


@router.post("/logout", response_model=LogoutResponse)
async def logout(current_user: dict = Depends(get_current_user)):
    """Logout current user and invalidate session."""
    try:
        supabase = get_supabase_client()
        supabase.auth.sign_out()

        return LogoutResponse(message="Logged out successfully", success=True)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Logout failed"
        )


@router.post("/refresh", response_model=TokenResponse)
async def refresh_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Refresh access token using refresh token."""
    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Refresh token required"
        )

    try:
        supabase = get_supabase_client()

        # Refresh session with Supabase
        response = supabase.auth.refresh_session()

        if response.session:
            return TokenResponse(
                access_token=response.session.access_token,
                refresh_token=response.session.refresh_token,
                token_type="bearer",
                expires_in=3600,
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid refresh token"
            )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Token refresh failed"
        )


@router.get("/me", response_model=UserResponse)
async def get_current_user_profile(current_user: dict = Depends(get_current_user)):
    """Get current user profile information."""
    return UserResponse(
        id=current_user["id"],
        email=current_user["email"],
        role=current_user.get("role", "user"),
        created_at=current_user.get("created_at"),
    )


@router.get("/session")
async def get_session_info(current_user: dict = Depends(get_current_user)):
    """Get current session information."""
    return {
        "user_id": current_user["id"],
        "email": current_user["email"],
        "role": current_user.get("role", "user"),
        "session_expires": datetime.now(timezone.utc) + timedelta(hours=1),
        "is_authenticated": True,
    }
