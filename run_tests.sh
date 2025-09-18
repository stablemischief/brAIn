#!/bin/bash

# Test runner script for brAIn Enhanced RAG Pipeline
# This script runs the test suite with proper coverage reporting

echo "🧪 Running brAIn Test Suite"
echo "==========================="

# Activate virtual environment
source venv/bin/activate

# Install missing dependencies if needed
echo "📦 Checking dependencies..."
pip install -q pytest pytest-cov pytest-asyncio pytest-mock httpx

# Create coverage config if not exists
if [ ! -f .coveragerc ]; then
    echo "[run]
source = .
omit =
    tests/*
    */test_*.py
    */__pycache__/*
    venv/*
    .venv/*
    frontend/*
    node_modules/*
    htmlcov/*
    .pytest_cache/*
    */migrations/*
    */config/*
    setup.py

[report]
exclude_lines =
    pragma: no cover
    def __repr__
    raise AssertionError
    raise NotImplementedError
    if __name__ == .__main__.:
    if TYPE_CHECKING:
    @abstract
    @abstractmethod
" > .coveragerc
fi

echo ""
echo "🔬 Running Unit Tests"
echo "--------------------"
python -m pytest tests/unit/ -v --tb=short

echo ""
echo "🔌 Running Integration Tests"
echo "-------------------------"
python -m pytest tests/integration/ -v --tb=short

echo ""
echo "📊 Generating Coverage Report"
echo "---------------------------"
python -m pytest tests/unit/ tests/integration/ \
    --cov=core \
    --cov=api \
    --cov-report=term-missing \
    --cov-report=html \
    --cov-report=term:skip-covered \
    -q

echo ""
echo "✅ Test Suite Complete!"
echo ""
echo "📈 Coverage Summary:"
python -m coverage report --skip-covered --precision=1

echo ""
echo "📂 HTML coverage report generated in: htmlcov/index.html"
echo ""
echo "🎯 Next Steps:"
echo "  1. Review coverage gaps in htmlcov/index.html"
echo "  2. Add more unit tests for uncovered code"
echo "  3. Implement e2e tests for user workflows"
echo "  4. Set up CI/CD pipeline integration"