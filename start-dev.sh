#!/bin/bash
# 
# StockTrader Development Server Startup Script
# Starts both backend (FastAPI) and frontend (Next.js) servers
#

echo "🚀 Starting StockTrader Full-Stack Development Environment..."

# Check if we're in the correct directory
if [ ! -f "requirements.txt" ] || [ ! -d "frontend" ]; then
    echo "❌ Error: Please run this script from the project root directory"
    exit 1
fi

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "❌ Error: Virtual environment not found. Please run: python -m venv venv"
    exit 1
fi

# Check if frontend dependencies are installed
if [ ! -d "frontend/node_modules" ]; then
    echo "📦 Installing frontend dependencies..."
    cd frontend && npm install && cd ..
fi

echo "🔧 Activating Python virtual environment..."
source venv/Scripts/activate

echo "🐍 Starting Backend (FastAPI) on http://localhost:8000..."
echo "📱 Starting Frontend (Next.js) on http://localhost:3000..."
echo ""
echo "📖 API Documentation: http://localhost:8000/docs"
echo "🌐 Frontend Application: http://localhost:3000"
echo ""
echo "Press Ctrl+C to stop both servers"
echo ""

# Start both servers using concurrently-like approach
# Backend in background
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000 &
BACKEND_PID=$!

# Frontend in background
cd frontend && npm run dev &
FRONTEND_PID=$!

# Wait for Ctrl+C
trap 'echo ""; echo "🛑 Shutting down servers..."; kill $BACKEND_PID $FRONTEND_PID; exit' INT

# Keep script running
wait
