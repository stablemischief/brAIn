#!/bin/bash

# brAIn v2.0 - Installation Verification Script
# This script validates your Docker installation and environment setup

set -e  # Exit on any error

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[CHECK]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[✓]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[⚠]${NC} $1"
}

print_error() {
    echo -e "${RED}[✗]${NC} $1"
}

echo "=========================================="
echo "   brAIn v2.0 - Installation Verifier"
echo "=========================================="
echo ""

# Track overall status
ERRORS=0
WARNINGS=0

# 1. Check Docker installation
print_status "Checking Docker installation..."
if command -v docker &> /dev/null; then
    DOCKER_VERSION=$(docker --version)
    print_success "Docker installed: $DOCKER_VERSION"
else
    print_error "Docker is not installed!"
    echo "  Install Docker from: https://docs.docker.com/get-docker/"
    ERRORS=$((ERRORS + 1))
fi

# 2. Check Docker Compose installation
print_status "Checking Docker Compose installation..."
if command -v docker-compose &> /dev/null; then
    COMPOSE_VERSION=$(docker-compose --version)
    print_success "Docker Compose installed: $COMPOSE_VERSION"
else
    print_error "Docker Compose is not installed!"
    echo "  Install Docker Compose from: https://docs.docker.com/compose/install/"
    ERRORS=$((ERRORS + 1))
fi

# 3. Check Docker daemon is running
print_status "Checking Docker daemon status..."
if docker info &> /dev/null; then
    print_success "Docker daemon is running"
else
    print_error "Docker daemon is not running!"
    echo "  Start Docker Desktop or run: sudo systemctl start docker"
    ERRORS=$((ERRORS + 1))
fi

# 4. Check .env file exists
print_status "Checking environment configuration..."
if [ -f ".env" ]; then
    print_success "Environment file (.env) found"

    # Check for critical environment variables
    if grep -q "OPENAI_API_KEY=" .env && grep -v "OPENAI_API_KEY=$" .env | grep -q "OPENAI_API_KEY="; then
        print_success "OpenAI API key configured"
    else
        print_warning "OpenAI API key not set in .env file"
        WARNINGS=$((WARNINGS + 1))
    fi
else
    if [ -f ".env.example" ]; then
        print_warning ".env file not found, but .env.example exists"
        echo "  Run: cp .env.example .env"
        echo "  Then edit .env to add your configuration"
        WARNINGS=$((WARNINGS + 1))
    else
        print_error "No .env or .env.example file found!"
        ERRORS=$((ERRORS + 1))
    fi
fi

# 5. Check required directories
print_status "Checking project structure..."
REQUIRED_DIRS=("backend" "frontend" "docker" "docs")
for dir in "${REQUIRED_DIRS[@]}"; do
    if [ -d "$dir" ]; then
        print_success "Directory '$dir' exists"
    else
        print_error "Directory '$dir' is missing!"
        ERRORS=$((ERRORS + 1))
    fi
done

# 6. Check critical files
print_status "Checking critical files..."
CRITICAL_FILES=("docker-compose.yml" "backend/main.py" "frontend/package.json")
for file in "${CRITICAL_FILES[@]}"; do
    if [ -f "$file" ]; then
        print_success "File '$file' exists"
    else
        print_error "File '$file' is missing!"
        ERRORS=$((ERRORS + 1))
    fi
done

# 7. Check Docker images (if built)
print_status "Checking Docker images..."
if docker images | grep -q "brain-brain-app"; then
    print_success "brAIn Docker image found"
else
    print_warning "brAIn Docker image not built yet"
    echo "  This is normal for first-time setup"
    echo "  Run: docker-compose build"
fi

# 8. Check if containers are running
print_status "Checking running containers..."
RUNNING_CONTAINERS=$(docker-compose ps --services 2>/dev/null | wc -l)
if [ "$RUNNING_CONTAINERS" -gt 0 ]; then
    print_success "$RUNNING_CONTAINERS service(s) running"
    docker-compose ps
else
    print_warning "No containers are currently running"
    echo "  This is normal if you haven't started the system yet"
    echo "  Run: docker-compose up -d"
fi

# 9. Check port availability
print_status "Checking port availability..."
PORTS=(3000 8000 5432 6379)
PORT_CONFLICTS=0
for port in "${PORTS[@]}"; do
    if lsof -Pi :$port -sTCP:LISTEN -t >/dev/null 2>&1; then
        # Check if it's our container using the port
        if docker ps --format '{{.Ports}}' | grep -q "$port"; then
            print_success "Port $port is used by brAIn container"
        else
            print_warning "Port $port is already in use by another process"
            PORT_CONFLICTS=$((PORT_CONFLICTS + 1))
        fi
    else
        print_success "Port $port is available"
    fi
done

if [ $PORT_CONFLICTS -gt 0 ]; then
    WARNINGS=$((WARNINGS + 1))
    echo "  You may need to stop conflicting services or change ports in docker-compose.yml"
fi

# 10. Quick health check (if running)
if [ "$RUNNING_CONTAINERS" -gt 0 ]; then
    print_status "Performing health check..."

    # Check backend health
    if curl -f http://localhost:8000/api/health &>/dev/null; then
        print_success "Backend API is responding"
    else
        print_warning "Backend API is not responding on port 8000"
        echo "  The service may still be starting up"
    fi

    # Check frontend
    if curl -f http://localhost:3000 &>/dev/null; then
        print_success "Frontend is responding"
    else
        print_warning "Frontend is not responding on port 3000"
        echo "  The service may still be starting up"
    fi
fi

# Summary
echo ""
echo "=========================================="
echo "           VERIFICATION SUMMARY"
echo "=========================================="

if [ $ERRORS -eq 0 ] && [ $WARNINGS -eq 0 ]; then
    echo -e "${GREEN}✅ All checks passed!${NC}"
    echo ""
    echo "Your brAIn installation is ready!"
    echo ""
    echo "Next steps:"
    echo "1. Configure your .env file (if not done)"
    echo "2. Run: docker-compose up --build"
    echo "3. Access: http://localhost:3000"
elif [ $ERRORS -eq 0 ]; then
    echo -e "${YELLOW}⚠️  Installation functional with $WARNINGS warning(s)${NC}"
    echo ""
    echo "Your installation should work but review the warnings above."
    echo ""
    echo "To start brAIn:"
    echo "1. Address any warnings if needed"
    echo "2. Run: docker-compose up --build"
    echo "3. Access: http://localhost:3000"
else
    echo -e "${RED}❌ Installation has $ERRORS error(s) and $WARNINGS warning(s)${NC}"
    echo ""
    echo "Please fix the errors above before proceeding."
    echo "Check the documentation at: docs/user_guide/quick-start.md"
fi

echo ""
echo "=========================================="
echo "For detailed setup instructions, see:"
echo "- Quick Start: docs/user_guide/quick-start.md"
echo "- Deployment: docs/user_guide/deployment-basic.md"
echo "- Troubleshooting: docs/troubleshooting/README.md"
echo "=========================================="