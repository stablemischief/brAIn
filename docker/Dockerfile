# Multi-stage Dockerfile for brAIn v2.0
# Optimized for AI/ML dependencies with production-ready configuration

# ================== Stage 1: Python Builder ==================
FROM python:3.11-slim as python-builder

# Set environment variables for build
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    POETRY_VERSION=1.7.1 \
    POETRY_HOME="/opt/poetry" \
    POETRY_VIRTUALENVS_CREATE=false \
    POETRY_NO_INTERACTION=1

# Install system dependencies and build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    gcc \
    g++ \
    postgresql-client \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry for dependency management
RUN curl -sSL https://install.python-poetry.org | python3 -
ENV PATH="$POETRY_HOME/bin:$PATH"

# Set working directory
WORKDIR /app

# Copy dependency files
COPY backend/pyproject.toml backend/poetry.lock* ./

# Install Python dependencies
RUN poetry install --no-dev --no-root && \
    pip install --upgrade pip setuptools wheel

# ================== Stage 2: Node Builder ==================
FROM node:18-alpine as node-builder

# Set working directory
WORKDIR /app

# Copy package files
COPY frontend/package.json frontend/package-lock.json* ./

# Install dependencies
RUN npm ci --only=production

# Copy frontend source code
COPY frontend ./frontend

# Build frontend application
RUN cd frontend && npm run build

# ================== Stage 3: Production Runtime ==================
FROM python:3.11-slim as production

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    NODE_ENV=production \
    PORT=8000

# Create non-root user for security
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    supervisor \
    postgresql-client \
    curl \
    wget \
    ca-certificates \
    gnupg \
    lsb-release \
    && rm -rf /var/lib/apt/lists/*

# Install Node.js for runtime (needed for some AI tools)
RUN curl -fsSL https://deb.nodesource.com/setup_18.x | bash - && \
    apt-get install -y nodejs && \
    rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy Python dependencies from builder
COPY --from=python-builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=python-builder /usr/local/bin /usr/local/bin

# Copy Node modules from builder
COPY --from=node-builder /app/node_modules ./node_modules

# Copy built frontend from builder
COPY --from=node-builder /app/frontend/build ./frontend/build

# Copy application code
COPY backend ./backend
COPY docker/supervisord.conf /etc/supervisor/conf.d/supervisord.conf

# Create necessary directories
RUN mkdir -p /app/logs /app/data /app/tmp && \
    chown -R appuser:appuser /app

# Copy and set permissions for health check script
COPY backend/scripts/health-check.py ./backend/scripts/
RUN chmod +x ./backend/scripts/health-check.py

# Health check configuration
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python /app/backend/scripts/health-check.py || exit 1

# Expose ports
EXPOSE 8000 3000 8080

# Switch to non-root user
USER appuser

# Volume mounts for persistent data
VOLUME ["/app/logs", "/app/data", "/app/config"]

# Start supervisor to manage multiple processes
CMD ["/usr/bin/supervisord", "-c", "/etc/supervisor/conf.d/supervisord.conf"]

# ================== Stage 4: Development ==================
FROM production as development

# Switch back to root for development tools installation
USER root

# Install development dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    vim \
    htop \
    iputils-ping \
    telnet \
    && rm -rf /var/lib/apt/lists/*

# Install Python development dependencies
RUN pip install --no-cache-dir \
    pytest \
    pytest-asyncio \
    pytest-cov \
    black \
    ruff \
    mypy \
    ipython \
    ipdb

# Set development environment
ENV ENVIRONMENT=development \
    DEBUG=true

# Switch back to appuser
USER appuser

# Development command (run from /app directory, backend.main refers to backend/main.py)
CMD ["python", "-m", "uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]