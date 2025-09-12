"""
Error handling utilities for brAIn v2.0
Provides comprehensive error handling, logging, and response formatting
"""

import logging
import traceback
from typing import Dict, Any, Optional, Union
from datetime import datetime, timezone

from fastapi import Request, HTTPException, status
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
from pydantic import ValidationError
import sqlalchemy.exc

from config.settings import get_settings

logger = logging.getLogger(__name__)


class ErrorCode:
    """Standardized error codes for the application."""
    
    # Generic errors
    INTERNAL_ERROR = "INTERNAL_ERROR"
    INVALID_REQUEST = "INVALID_REQUEST"
    RESOURCE_NOT_FOUND = "RESOURCE_NOT_FOUND"
    
    # Authentication errors
    AUTHENTICATION_REQUIRED = "AUTHENTICATION_REQUIRED"
    AUTHENTICATION_FAILED = "AUTHENTICATION_FAILED"
    INSUFFICIENT_PERMISSIONS = "INSUFFICIENT_PERMISSIONS"
    TOKEN_EXPIRED = "TOKEN_EXPIRED"
    
    # Validation errors
    VALIDATION_ERROR = "VALIDATION_ERROR"
    INVALID_DATA_FORMAT = "INVALID_DATA_FORMAT"
    MISSING_REQUIRED_FIELD = "MISSING_REQUIRED_FIELD"
    
    # Database errors
    DATABASE_ERROR = "DATABASE_ERROR"
    DUPLICATE_RESOURCE = "DUPLICATE_RESOURCE"
    FOREIGN_KEY_VIOLATION = "FOREIGN_KEY_VIOLATION"
    
    # Processing errors
    PROCESSING_FAILED = "PROCESSING_FAILED"
    PROCESSING_TIMEOUT = "PROCESSING_TIMEOUT"
    INSUFFICIENT_CREDITS = "INSUFFICIENT_CREDITS"
    
    # External service errors
    EXTERNAL_SERVICE_ERROR = "EXTERNAL_SERVICE_ERROR"
    GOOGLE_DRIVE_ERROR = "GOOGLE_DRIVE_ERROR"
    OPENAI_API_ERROR = "OPENAI_API_ERROR"
    SUPABASE_ERROR = "SUPABASE_ERROR"
    
    # Rate limiting
    RATE_LIMIT_EXCEEDED = "RATE_LIMIT_EXCEEDED"
    
    # File processing errors
    FILE_TOO_LARGE = "FILE_TOO_LARGE"
    UNSUPPORTED_FILE_TYPE = "UNSUPPORTED_FILE_TYPE"
    FILE_PROCESSING_ERROR = "FILE_PROCESSING_ERROR"


class AppException(Exception):
    """Base application exception with structured error information."""
    
    def __init__(
        self,
        message: str,
        error_code: str = ErrorCode.INTERNAL_ERROR,
        status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR,
        details: Optional[Dict[str, Any]] = None,
        user_message: Optional[str] = None
    ):
        self.message = message
        self.error_code = error_code
        self.status_code = status_code
        self.details = details or {}
        self.user_message = user_message or message
        super().__init__(message)


class ValidationException(AppException):
    """Exception for validation errors."""
    
    def __init__(
        self,
        message: str,
        field_errors: Optional[Dict[str, str]] = None,
        user_message: Optional[str] = None
    ):
        super().__init__(
            message=message,
            error_code=ErrorCode.VALIDATION_ERROR,
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            details={"field_errors": field_errors or {}},
            user_message=user_message
        )


class AuthenticationException(AppException):
    """Exception for authentication errors."""
    
    def __init__(
        self,
        message: str = "Authentication required",
        error_code: str = ErrorCode.AUTHENTICATION_REQUIRED
    ):
        super().__init__(
            message=message,
            error_code=error_code,
            status_code=status.HTTP_401_UNAUTHORIZED,
            user_message="Please log in to continue"
        )


class AuthorizationException(AppException):
    """Exception for authorization errors."""
    
    def __init__(self, message: str = "Insufficient permissions"):
        super().__init__(
            message=message,
            error_code=ErrorCode.INSUFFICIENT_PERMISSIONS,
            status_code=status.HTTP_403_FORBIDDEN,
            user_message="You don't have permission to perform this action"
        )


class ProcessingException(AppException):
    """Exception for processing errors."""
    
    def __init__(
        self,
        message: str,
        details: Optional[Dict[str, Any]] = None,
        user_message: Optional[str] = None
    ):
        super().__init__(
            message=message,
            error_code=ErrorCode.PROCESSING_FAILED,
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            details=details,
            user_message=user_message or "Processing failed. Please try again."
        )


class ExternalServiceException(AppException):
    """Exception for external service errors."""
    
    def __init__(
        self,
        service_name: str,
        message: str,
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(
            message=f"{service_name}: {message}",
            error_code=ErrorCode.EXTERNAL_SERVICE_ERROR,
            status_code=status.HTTP_502_BAD_GATEWAY,
            details={**(details or {}), "service": service_name},
            user_message=f"External service ({service_name}) is currently unavailable"
        )


def format_error_response(
    error_code: str,
    message: str,
    details: Optional[Dict[str, Any]] = None,
    user_message: Optional[str] = None,
    request_id: Optional[str] = None
) -> Dict[str, Any]:
    """Format a standardized error response."""
    response = {
        "error": {
            "code": error_code,
            "message": message,
            "user_message": user_message or message,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    }
    
    if details:
        response["error"]["details"] = details
    
    if request_id:
        response["error"]["request_id"] = request_id
    
    # Add debug information in development
    settings = get_settings()
    if settings.debug and details:
        response["error"]["debug"] = details
    
    return response


def log_error(
    error: Exception,
    request: Optional[Request] = None,
    extra_context: Optional[Dict[str, Any]] = None
) -> str:
    """Log an error with context information."""
    
    # Generate request ID for tracking
    request_id = f"req_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
    
    # Gather context information
    context = {
        "request_id": request_id,
        "error_type": type(error).__name__,
        "error_message": str(error),
    }
    
    if request:
        context.update({
            "method": request.method,
            "url": str(request.url),
            "user_agent": request.headers.get("User-Agent"),
            "ip_address": getattr(request.client, "host", None) if request.client else None,
        })
        
        # Add user context if available
        if hasattr(request.state, "user") and request.state.user:
            context["user_id"] = request.state.user.get("user_id")
            context["user_email"] = request.state.user.get("email")
    
    if extra_context:
        context.update(extra_context)
    
    # Log at appropriate level
    if isinstance(error, AppException):
        if error.status_code >= 500:
            logger.error(f"Application error: {error.message}", extra=context)
        else:
            logger.warning(f"Client error: {error.message}", extra=context)
    else:
        logger.error(f"Unhandled error: {str(error)}", extra=context, exc_info=True)
    
    return request_id


# Exception handlers for FastAPI


async def app_exception_handler(request: Request, exc: AppException) -> JSONResponse:
    """Handle custom application exceptions."""
    request_id = log_error(exc, request)
    
    response = format_error_response(
        error_code=exc.error_code,
        message=exc.message,
        details=exc.details,
        user_message=exc.user_message,
        request_id=request_id
    )
    
    return JSONResponse(
        status_code=exc.status_code,
        content=response
    )


async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    """Handle FastAPI HTTP exceptions."""
    request_id = log_error(exc, request)
    
    # Map HTTP status codes to error codes
    error_code_map = {
        400: ErrorCode.INVALID_REQUEST,
        401: ErrorCode.AUTHENTICATION_REQUIRED,
        403: ErrorCode.INSUFFICIENT_PERMISSIONS,
        404: ErrorCode.RESOURCE_NOT_FOUND,
        422: ErrorCode.VALIDATION_ERROR,
        429: ErrorCode.RATE_LIMIT_EXCEEDED,
        500: ErrorCode.INTERNAL_ERROR,
    }
    
    error_code = error_code_map.get(exc.status_code, ErrorCode.INTERNAL_ERROR)
    
    response = format_error_response(
        error_code=error_code,
        message=str(exc.detail),
        request_id=request_id
    )
    
    return JSONResponse(
        status_code=exc.status_code,
        content=response
    )


async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    """Handle Pydantic validation errors."""
    request_id = log_error(exc, request)
    
    # Extract field errors
    field_errors = {}
    for error in exc.errors():
        field_name = ".".join(str(loc) for loc in error["loc"])
        field_errors[field_name] = error["msg"]
    
    response = format_error_response(
        error_code=ErrorCode.VALIDATION_ERROR,
        message="Request validation failed",
        details={"field_errors": field_errors, "validation_errors": exc.errors()},
        user_message="Please check your input and try again",
        request_id=request_id
    )
    
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content=response
    )


async def database_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Handle database-related exceptions."""
    request_id = log_error(exc, request)
    
    # Determine specific database error type
    error_code = ErrorCode.DATABASE_ERROR
    user_message = "A database error occurred. Please try again."
    
    if isinstance(exc, sqlalchemy.exc.IntegrityError):
        if "unique constraint" in str(exc).lower():
            error_code = ErrorCode.DUPLICATE_RESOURCE
            user_message = "This resource already exists."
        elif "foreign key constraint" in str(exc).lower():
            error_code = ErrorCode.FOREIGN_KEY_VIOLATION
            user_message = "Related resource not found."
    
    response = format_error_response(
        error_code=error_code,
        message=str(exc),
        user_message=user_message,
        request_id=request_id
    )
    
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=response
    )


async def generic_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Handle any unhandled exceptions."""
    request_id = log_error(exc, request, {"traceback": traceback.format_exc()})
    
    settings = get_settings()
    
    # In production, don't expose internal error details
    if settings.environment == "production":
        message = "An internal error occurred"
        details = None
    else:
        message = str(exc)
        details = {"traceback": traceback.format_exc()}
    
    response = format_error_response(
        error_code=ErrorCode.INTERNAL_ERROR,
        message=message,
        details=details,
        user_message="An unexpected error occurred. Please try again later.",
        request_id=request_id
    )
    
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=response
    )


def setup_error_handlers(app):
    """Set up error handlers for the FastAPI application."""
    
    # Custom application exceptions
    app.add_exception_handler(AppException, app_exception_handler)
    app.add_exception_handler(ValidationException, app_exception_handler)
    app.add_exception_handler(AuthenticationException, app_exception_handler)
    app.add_exception_handler(AuthorizationException, app_exception_handler)
    app.add_exception_handler(ProcessingException, app_exception_handler)
    app.add_exception_handler(ExternalServiceException, app_exception_handler)
    
    # FastAPI and Starlette exceptions
    app.add_exception_handler(HTTPException, http_exception_handler)
    app.add_exception_handler(StarletteHTTPException, http_exception_handler)
    app.add_exception_handler(RequestValidationError, validation_exception_handler)
    
    # Database exceptions
    app.add_exception_handler(sqlalchemy.exc.SQLAlchemyError, database_exception_handler)
    
    # Catch-all for any unhandled exceptions
    app.add_exception_handler(Exception, generic_exception_handler)


# Utility functions for raising common errors

def raise_not_found(resource_name: str, resource_id: Union[str, int]):
    """Raise a standardized not found error."""
    raise AppException(
        message=f"{resource_name} with ID '{resource_id}' not found",
        error_code=ErrorCode.RESOURCE_NOT_FOUND,
        status_code=status.HTTP_404_NOT_FOUND,
        user_message=f"The requested {resource_name.lower()} was not found"
    )


def raise_validation_error(message: str, field_errors: Optional[Dict[str, str]] = None):
    """Raise a standardized validation error."""
    raise ValidationException(
        message=message,
        field_errors=field_errors,
        user_message="Please check your input and try again"
    )


def raise_processing_error(message: str, details: Optional[Dict[str, Any]] = None):
    """Raise a standardized processing error."""
    raise ProcessingException(
        message=message,
        details=details,
        user_message="Processing failed. Please try again later."
    )