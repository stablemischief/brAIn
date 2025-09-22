# Multi-stage Dockerfile for brAIn v2.0 Production
# Optimized for security, size, and performance

# ==================== Stage 1: Python Builder ====================
FROM python:3.11-slim as python-builder

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    build-essential \
    libpq-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /build

# Copy requirements first for better caching
COPY requirements.txt .
COPY requirements-prod.txt .

# Create virtual environment and install dependencies
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Upgrade pip and install production dependencies
RUN pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir -r requirements-prod.txt

# ==================== Stage 2: Frontend Builder ====================
FROM node:18-alpine as frontend-builder

# Set working directory
WORKDIR /frontend

# Copy package files
COPY frontend/package*.json ./

# Install dependencies
RUN npm ci --only=production

# Copy frontend source
COPY frontend/ ./

# Build production frontend
RUN npm run build

# ==================== Stage 3: Production Runtime ====================
FROM python:3.11-slim as runtime

# Install runtime dependencies only
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    curl \
    supervisor \
    nginx \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd -r brain && useradd -r -g brain brain

# Set working directory
WORKDIR /app

# Copy virtual environment from builder
COPY --from=python-builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy frontend build
COPY --from=frontend-builder /frontend/dist /app/frontend/dist

# Copy application code
COPY --chown=brain:brain . .

# Create necessary directories
RUN mkdir -p /app/logs /app/data /app/secrets && \
    chown -R brain:brain /app/logs /app/data /app/secrets

# Copy configuration files
COPY deployment/production/supervisord.conf /etc/supervisor/conf.d/supervisord.conf
COPY deployment/production/nginx.conf /etc/nginx/sites-available/default

# Security: Remove unnecessary files
RUN find /app -name "*.pyc" -delete && \
    find /app -name "__pycache__" -type d -delete && \
    rm -rf /app/.git /app/tests /app/docs

# Environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    ENVIRONMENT=production \
    PORT=8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:${PORT}/health || exit 1

# Switch to non-root user
USER brain

# Expose port
EXPOSE ${PORT}

# Start supervisor
CMD ["/usr/bin/supervisord", "-c", "/etc/supervisor/conf.d/supervisord.conf"]