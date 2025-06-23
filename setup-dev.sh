#!/bin/bash
# StockTrader Development Environment Setup
# This script sets up the complete development environment

echo "🚀 Setting up StockTrader Development Environment..."

# Ensure Node.js is in PATH
export PATH="/c/Program Files/nodejs:$PATH"

# Verify tools are available
echo "📋 Checking prerequisites..."
if command -v node >/dev/null 2>&1; then
    echo "✅ Node.js $(node --version)"
else
    echo "❌ Node.js not found"
    exit 1
fi

if command -v npm >/dev/null 2>&1; then
    echo "✅ npm $(npm --version)"
else
    echo "❌ npm not found"
    exit 1
fi

if command -v python >/dev/null 2>&1; then
    echo "✅ Python $(python --version)"
else
    echo "❌ Python not found"
    exit 1
fi

# Check if we're in the right directory
if [ ! -f "package.json" ]; then
    echo "❌ Not in StockTrader root directory"
    echo "Please run this script from /c/dev/stocktrader2"
    exit 1
fi

echo "✅ All prerequisites met!"
echo ""
echo "🎯 Available development commands:"
echo "   npm run dev          - Start both frontend and backend"
echo "   npm run dev:backend  - Start backend only"
echo "   npm run dev:frontend - Start frontend only"
echo "   npm run install:all  - Install all dependencies"
echo "   npm run docs         - Open API documentation"
echo ""
echo "💡 To start development:"
echo "   npm run dev"
echo ""
