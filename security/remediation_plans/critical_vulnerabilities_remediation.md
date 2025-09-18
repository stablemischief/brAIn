# Critical Vulnerabilities Remediation Plan

**Project:** brAIn v2.0 RAG Pipeline Management System
**Date:** September 18, 2025
**Priority:** CRITICAL - Immediate Action Required
**Target Completion:** Within 1 Week

## Overview

This document outlines the immediate remediation steps for critical security vulnerabilities discovered in the brAIn v2.0 system. These vulnerabilities present an unacceptable security risk and must be addressed before any production deployment.

## Critical Vulnerabilities Summary

| ID | Vulnerability | CVSS | Status | Assignee | Due Date |
|----|--------------|------|--------|----------|----------|
| SEC-001 | JWT Signature Validation Disabled | 9.8 | 🔴 Open | Dev Team | Day 1 |
| SEC-002 | Mock Authentication in Production | 9.5 | 🔴 Open | Dev Team | Day 1 |
| SEC-003 | Hardcoded Secrets Exposure | 9.1 | 🔴 Open | DevOps Team | Day 2 |
| SEC-004 | CORS Wildcard Configuration | 8.5 | 🔴 Open | Dev Team | Day 2 |
| SEC-005 | Missing Input Validation | 8.7 | 🔴 Open | Dev Team | Day 3 |

## Phase 1: Emergency Authentication Fixes (Day 1)

### SEC-001: Fix JWT Signature Validation

**Current Vulnerable Code:** `middleware/auth_middleware.py:99`
```python
# VULNERABLE
decoded_token = jwt.decode(token, options={"verify_signature": False})
```

**Remediation Steps:**

1. **Immediate Fix:**
```python
# SECURE - Enable signature verification
try:
    settings = get_settings()
    decoded_token = jwt.decode(
        token,
        settings.jwt_secret_key,
        algorithms=["HS256"],
        options={
            "verify_signature": True,
            "verify_exp": True,
            "verify_iat": True,
            "verify_aud": False  # Set based on requirements
        }
    )
except jwt.ExpiredSignatureError:
    logger.warning("JWT token has expired")
    return None
except jwt.InvalidTokenError as e:
    logger.warning(f"Invalid JWT token: {e}")
    return None
```

2. **Add Secret Key Management:**
```python
# config/settings.py
class Settings(BaseSettings):
    jwt_secret_key: str = Field(..., env="JWT_SECRET_KEY")
    jwt_algorithm: str = Field(default="HS256", env="JWT_ALGORITHM")
    jwt_expiration_hours: int = Field(default=24, env="JWT_EXPIRATION_HOURS")

    @validator('jwt_secret_key')
    def validate_jwt_secret(cls, v):
        if len(v) < 32:
            raise ValueError('JWT secret key must be at least 32 characters')
        return v
```

3. **Generate Strong Secret Key:**
```bash
# Generate secure secret key
python -c "import secrets; print(secrets.token_urlsafe(32))"
```

**Validation:**
- [ ] JWT tokens with invalid signatures are rejected
- [ ] Expired tokens are rejected
- [ ] Strong secret key (32+ characters) is configured
- [ ] Algorithm is explicitly specified

---

### SEC-002: Remove Mock Authentication

**Current Vulnerable Code:** `api/auth.py:50-58`

**Remediation Steps:**

1. **Replace Mock Authentication:**
```python
async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: AsyncSession = Depends(get_database_session)
) -> dict:
    """Get current authenticated user from JWT token."""
    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required"
        )

    try:
        # Proper JWT validation
        settings = get_settings()
        payload = jwt.decode(
            credentials.credentials,
            settings.jwt_secret_key,
            algorithms=[settings.jwt_algorithm]
        )

        user_id = payload.get("sub")
        if not user_id:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token: missing subject"
            )

        # Validate user exists in database
        user = await get_user_by_id(db, user_id)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User not found"
            )

        return {
            "id": user.id,
            "email": user.email,
            "role": user.role,
            "created_at": user.created_at
        }

    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has expired"
        )
    except jwt.InvalidTokenError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials"
        )
```

2. **Implement Proper Supabase Integration:**
```python
def verify_supabase_token(token: str) -> dict:
    """Verify JWT token with Supabase."""
    try:
        supabase = get_supabase_client()

        # Get JWT secret from Supabase
        jwt_secret = get_settings().supabase_jwt_secret

        # Verify token
        payload = jwt.decode(
            token,
            jwt_secret,
            algorithms=["HS256"],
            audience="authenticated"
        )

        return payload

    except jwt.InvalidTokenError as e:
        logger.warning(f"Invalid Supabase token: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication token"
        )
```

**Validation:**
- [ ] Mock authentication code completely removed
- [ ] Real JWT validation implemented
- [ ] Supabase integration working correctly
- [ ] User validation against database implemented

---

## Phase 2: Secret Management (Day 2)

### SEC-003: Implement Secure Secret Management

**Remediation Steps:**

1. **Create Environment Variables Template:**
```bash
# .env.example
JWT_SECRET_KEY=your-secure-jwt-secret-key-here-minimum-32-chars
SUPABASE_URL=your-supabase-url
SUPABASE_ANON_KEY=your-supabase-anon-key
SUPABASE_SERVICE_KEY=your-supabase-service-key
SUPABASE_JWT_SECRET=your-supabase-jwt-secret
DATABASE_URL=your-database-url
ENCRYPTION_KEY=your-encryption-key-for-sensitive-data
```

2. **Update Settings Configuration:**
```python
class Settings(BaseSettings):
    # Security
    jwt_secret_key: SecretStr = Field(..., env="JWT_SECRET_KEY")
    supabase_jwt_secret: SecretStr = Field(..., env="SUPABASE_JWT_SECRET")
    encryption_key: SecretStr = Field(..., env="ENCRYPTION_KEY")

    # Database
    database_url: SecretStr = Field(..., env="DATABASE_URL")

    # Supabase
    supabase_url: str = Field(..., env="SUPABASE_URL")
    supabase_anon_key: SecretStr = Field(..., env="SUPABASE_ANON_KEY")
    supabase_service_key: SecretStr = Field(..., env="SUPABASE_SERVICE_KEY")

    class Config:
        env_file = ".env"
        case_sensitive = True

    @validator('jwt_secret_key', 'encryption_key')
    def validate_secret_strength(cls, v):
        secret = v.get_secret_value()
        if len(secret) < 32:
            raise ValueError('Secret must be at least 32 characters')
        return v
```

3. **Add .env to .gitignore:**
```bash
# Add to .gitignore
.env
.env.local
.env.production
*.key
secrets/
```

4. **Create Secret Rotation Script:**
```python
# scripts/rotate_secrets.py
import secrets
import os
from datetime import datetime

def generate_secure_secret(length: int = 32) -> str:
    """Generate cryptographically secure secret."""
    return secrets.token_urlsafe(length)

def rotate_jwt_secret():
    """Rotate JWT secret key."""
    new_secret = generate_secure_secret(32)

    # Log rotation for audit
    print(f"JWT secret rotated at {datetime.utcnow()}")
    print(f"New secret (first 8 chars): {new_secret[:8]}...")

    return new_secret

if __name__ == "__main__":
    new_jwt_secret = rotate_jwt_secret()
    print(f"New JWT_SECRET_KEY={new_jwt_secret}")
```

**Validation:**
- [ ] All hardcoded secrets removed from source code
- [ ] Environment variables properly configured
- [ ] Secret validation implemented
- [ ] .env files added to .gitignore
- [ ] Secret rotation procedure documented

---

### SEC-004: Fix CORS Configuration

**Current Vulnerable Code:** `main.py:95-101`

**Remediation Steps:**

1. **Secure CORS Configuration:**
```python
# Secure CORS setup
def configure_cors(app: FastAPI, settings: Settings):
    """Configure CORS with security best practices."""

    # Define allowed origins based on environment
    if settings.environment == "development":
        allowed_origins = [
            "http://localhost:3000",
            "http://127.0.0.1:3000",
            "http://localhost:8080"
        ]
    elif settings.environment == "staging":
        allowed_origins = [
            "https://staging.brain.example.com",
            "https://staging-admin.brain.example.com"
        ]
    else:  # production
        allowed_origins = [
            "https://brain.example.com",
            "https://admin.brain.example.com"
        ]

    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,  # NO WILDCARDS
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE"],  # Specific methods only
        allow_headers=[
            "Authorization",
            "Content-Type",
            "X-Requested-With",
            "Accept",
            "Origin",
            "Cache-Control",
            "X-File-Name"
        ],
        expose_headers=["X-Total-Count", "X-User-ID"],
        max_age=600  # Cache preflight for 10 minutes
    )
```

2. **Add Origin Validation:**
```python
def validate_origin(origin: str, allowed_origins: List[str]) -> bool:
    """Validate origin against allowed list."""
    if not origin:
        return False

    # Exact match check
    if origin in allowed_origins:
        return True

    # Subdomain validation for trusted domains
    for allowed in allowed_origins:
        if allowed.startswith("*."):
            domain = allowed[2:]
            if origin.endswith(f".{domain}") or origin == domain:
                return True

    return False
```

**Validation:**
- [ ] No wildcard origins allowed
- [ ] Environment-specific origin lists configured
- [ ] Specific HTTP methods allowed only
- [ ] Specific headers allowed only
- [ ] Credentials properly restricted

---

## Phase 3: Input Validation Framework (Day 3)

### SEC-005: Implement Comprehensive Input Validation

**Remediation Steps:**

1. **Create Validation Models:**
```python
# validators/security_validators.py
from pydantic import BaseModel, validator, Field
from typing import Optional, List
import re

class SecureBaseModel(BaseModel):
    """Base model with security validations."""

    @validator('*', pre=True)
    def prevent_null_bytes(cls, v):
        """Prevent null byte injection."""
        if isinstance(v, str) and '\x00' in v:
            raise ValueError('Null bytes not allowed')
        return v

    @validator('*', pre=True)
    def prevent_oversized_strings(cls, v):
        """Prevent oversized string attacks."""
        if isinstance(v, str) and len(v) > 10000:  # Configurable limit
            raise ValueError('String too long')
        return v

class SecureSearchRequest(SecureBaseModel):
    """Secure search request validation."""
    query: str = Field(..., min_length=1, max_length=500)
    filters: Optional[dict] = Field(default=None)
    limit: int = Field(default=10, ge=1, le=100)
    offset: int = Field(default=0, ge=0)

    @validator('query')
    def validate_query(cls, v):
        # Prevent SQL injection patterns
        dangerous_patterns = [
            r"('|\"|;|--|\*|\/\*|\*\/)",
            r"\b(union|select|insert|update|delete|drop|exec|execute)\b",
            r"(script|javascript|vbscript|onload|onerror)"
        ]

        for pattern in dangerous_patterns:
            if re.search(pattern, v, re.IGNORECASE):
                raise ValueError('Query contains dangerous patterns')

        return v

    @validator('filters')
    def validate_filters(cls, v):
        if v is None:
            return v

        # Prevent NoSQL injection
        dangerous_keys = ['$where', '$regex', '$ne', '$gt', '$gte', '$lt', '$lte']

        def check_dict(d):
            if isinstance(d, dict):
                for key in d.keys():
                    if key in dangerous_keys:
                        raise ValueError(f'Dangerous filter key: {key}')
                    check_dict(d[key])
            elif isinstance(d, list):
                for item in d:
                    check_dict(item)

        check_dict(v)
        return v
```

2. **Implement Input Sanitization:**
```python
# utils/sanitization.py
import html
import re
from urllib.parse import quote
from typing import Any, Dict

class InputSanitizer:
    """Comprehensive input sanitization."""

    @staticmethod
    def sanitize_html(value: str) -> str:
        """Sanitize HTML content."""
        # HTML encode dangerous characters
        return html.escape(value, quote=True)

    @staticmethod
    def sanitize_sql(value: str) -> str:
        """Sanitize potential SQL injection."""
        # Remove or escape SQL meta-characters
        dangerous_chars = ["'", '"', ';', '--', '/*', '*/']
        sanitized = value
        for char in dangerous_chars:
            sanitized = sanitized.replace(char, '')
        return sanitized

    @staticmethod
    def sanitize_path(value: str) -> str:
        """Sanitize file paths."""
        # Prevent path traversal
        sanitized = value.replace('..', '').replace('\\', '').replace('/', '')
        return quote(sanitized)

    @staticmethod
    def sanitize_command(value: str) -> str:
        """Sanitize command injection attempts."""
        dangerous_chars = [';', '|', '&', '$', '`', '(', ')', '<', '>']
        sanitized = value
        for char in dangerous_chars:
            sanitized = sanitized.replace(char, '')
        return sanitized
```

3. **Update API Endpoints:**
```python
# api/search.py
from validators.security_validators import SecureSearchRequest
from utils.sanitization import InputSanitizer

@router.post("/search")
async def search_documents(
    request: SecureSearchRequest,  # Pydantic validation
    current_user: dict = Depends(get_current_user)
):
    """Search documents with security validation."""

    # Additional sanitization after Pydantic validation
    sanitized_query = InputSanitizer.sanitize_html(request.query)

    try:
        # Secure database query with parameterized statements
        results = await search_service.search(
            query=sanitized_query,
            filters=request.filters,
            limit=request.limit,
            offset=request.offset,
            user_id=current_user["id"]
        )

        return {
            "results": results,
            "total": len(results),
            "query": sanitized_query  # Return sanitized version
        }

    except Exception as e:
        logger.error(f"Search error for user {current_user['id']}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Search operation failed"  # Generic error message
        )
```

**Validation:**
- [ ] Pydantic models validate all inputs
- [ ] SQL injection prevention implemented
- [ ] XSS prevention implemented
- [ ] Path traversal prevention implemented
- [ ] Command injection prevention implemented
- [ ] Input sanitization applied consistently

---

## Testing and Validation

### Security Test Suite

1. **Run Security Tests:**
```bash
# Run comprehensive security test suite
pytest tests/security/ -v --tb=short

# Run specific vulnerability tests
pytest tests/security/test_authentication_security.py::TestAuthenticationSecurity::test_authentication_bypass_vulnerability -v
```

2. **Automated Security Scanning:**
```bash
# Install security tools
pip install bandit safety

# Run static analysis
bandit -r . -f json -o security_scan_results.json

# Check for known vulnerabilities
safety check --json --output vulnerability_report.json
```

3. **Manual Penetration Testing:**
```bash
# Test authentication bypass
curl -H "Authorization: Bearer fake.jwt.token" http://localhost:8000/api/auth/me

# Test SQL injection
curl "http://localhost:8000/api/search?query='; DROP TABLE users; --"

# Test XSS
curl "http://localhost:8000/api/search?query=<script>alert('XSS')</script>"
```

### Validation Checklist

**Phase 1 Validation:**
- [ ] JWT signature validation enabled and working
- [ ] Mock authentication completely removed
- [ ] Strong secret keys generated and configured
- [ ] CORS wildcards removed and specific origins configured

**Phase 2 Validation:**
- [ ] All hardcoded secrets removed from source code
- [ ] Environment variables properly configured
- [ ] Secret rotation procedures documented

**Phase 3 Validation:**
- [ ] Input validation framework implemented
- [ ] All API endpoints use secure validation models
- [ ] Sanitization applied to all user inputs
- [ ] Security test suite passes

### Post-Remediation Security Scan

After implementing all fixes, run comprehensive security assessment:

1. **Automated Vulnerability Scan**
2. **Manual Penetration Testing**
3. **Code Review for Security**
4. **Configuration Review**

### Emergency Rollback Plan

If any remediation causes system instability:

1. **Immediate Actions:**
   - Revert to previous stable code
   - Disable affected endpoints temporarily
   - Enable enhanced logging

2. **Assessment:**
   - Identify root cause of instability
   - Assess security vs availability trade-offs
   - Plan alternative remediation approach

3. **Communication:**
   - Notify security team immediately
   - Document rollback reason
   - Update remediation timeline

---

## Success Criteria

### Critical Vulnerabilities Resolved:
- [ ] Authentication bypass prevented (SEC-001, SEC-002)
- [ ] Secret exposure eliminated (SEC-003)
- [ ] CORS properly configured (SEC-004)
- [ ] Input validation comprehensive (SEC-005)

### Security Posture Improved:
- [ ] All critical vulnerabilities addressed
- [ ] Security test suite implemented and passing
- [ ] Monitoring and alerting for security events
- [ ] Documentation updated with security procedures

### Production Readiness:
- [ ] Security assessment shows acceptable risk level
- [ ] Penetration testing validates fixes
- [ ] Team trained on secure development practices
- [ ] Incident response procedures defined

**Target Completion Date:** September 25, 2025
**Next Security Review:** October 15, 2025

---

*This remediation plan is classified as **CONFIDENTIAL** and should only be shared with authorized security and development team members.*