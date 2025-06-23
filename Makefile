# StockTrader Development Makefile
# Provides convenient commands for full-stack development

.PHONY: help install dev start stop clean test lint docs

# Default target
help:
	@echo "🚀 StockTrader Development Commands"
	@echo ""
	@echo "Setup:"
	@echo "  make install     - Install all dependencies (backend + frontend)"
	@echo "  make install-be  - Install backend dependencies only"
	@echo "  make install-fe  - Install frontend dependencies only"
	@echo ""
	@echo "Development:"
	@echo "  make dev         - Start both backend and frontend servers"
	@echo "  make start       - Alias for 'make dev'"
	@echo "  make dev-be      - Start backend server only"
	@echo "  make dev-fe      - Start frontend server only"
	@echo ""
	@echo "Testing:"
	@echo "  make test        - Run all tests"
	@echo "  make test-be     - Run backend tests"
	@echo "  make test-fe     - Run frontend tests"
	@echo "  make lint        - Run linting on all code"
	@echo ""
	@echo "Utilities:"
	@echo "  make clean       - Clean build artifacts and caches"
	@echo "  make docs        - Open API documentation"
	@echo "  make status      - Check system status"

# Installation targets
install: install-be install-fe
	@echo "✅ All dependencies installed"

install-be:
	@echo "📦 Installing backend dependencies..."
	pip install -r requirements.txt

install-fe:
	@echo "📦 Installing frontend dependencies..."
	cd frontend && npm install

# Development targets
dev: start

start:
	@echo "🚀 Starting StockTrader development environment..."
	@echo "🐍 Backend: http://localhost:8000"
	@echo "📱 Frontend: http://localhost:3000"
	@echo "📖 API Docs: http://localhost:8000/docs"
	npm run dev

dev-be:
	@echo "🐍 Starting backend server..."
	source venv/Scripts/activate && uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

dev-fe:
	@echo "📱 Starting frontend server..."
	cd frontend && npm run dev

# Testing targets
test: test-be test-fe

test-be:
	@echo "🧪 Running backend tests..."
	pytest tests/ -v

test-fe:
	@echo "🧪 Running frontend tests..."
	cd frontend && npm test

# Linting targets
lint:
	@echo "🔍 Running backend linting..."
	ruff check .
	@echo "🔍 Running frontend linting..."
	cd frontend && npm run lint

# Utility targets
clean:
	@echo "🧹 Cleaning build artifacts..."
	rm -rf frontend/.next
	rm -rf frontend/node_modules/.cache
	rm -rf __pycache__
	rm -rf .pytest_cache
	find . -name "*.pyc" -delete

docs:
	@echo "📖 Opening API documentation..."
	open http://localhost:8000/docs || start http://localhost:8000/docs

status:
	@echo "📊 System Status Check"
	@echo "Python version: $(shell python --version)"
	@echo "Node version: $(shell node --version 2>/dev/null || echo 'Not installed')"
	@echo "Virtual env: $(shell which python | grep venv > /dev/null && echo 'Active' || echo 'Inactive')"
	@echo "Backend health: $(shell curl -s http://localhost:8000/api/v1/health | grep -o '"status":"[^"]*"' || echo 'Not running')"
