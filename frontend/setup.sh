#!/bin/bash
# Frontend Setup Script for StockTrader
# This script sets up the frontend development environment

set -e

echo "🚀 StockTrader Frontend Setup"
echo "=============================="

# Check if we're in the right directory
if [ ! -f "package.json" ]; then
    echo "❌ Error: package.json not found. Please run this script from the frontend directory."
    exit 1
fi

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check Node.js installation
echo "📦 Checking Node.js installation..."
if command_exists node; then
    NODE_VERSION=$(node --version)
    echo "✅ Node.js is installed: $NODE_VERSION"
    
    # Check if version is 18 or higher
    NODE_MAJOR=$(echo $NODE_VERSION | sed 's/v\([0-9]*\).*/\1/')
    if [ "$NODE_MAJOR" -lt 18 ]; then
        echo "⚠️  Warning: Node.js version 18+ is recommended. Current: $NODE_VERSION"
    fi
else
    echo "❌ Node.js is not installed."
    echo "📥 Installing Node.js using winget..."
    
    if command_exists winget; then
        winget install OpenJS.NodeJS
        echo "✅ Node.js installed via winget"
        echo "🔄 Please restart your terminal and run this script again"
        exit 0
    else
        echo "❌ winget not available. Please install Node.js manually:"
        echo "   1. Visit https://nodejs.org/"
        echo "   2. Download and install Node.js 18+"
        echo "   3. Restart your terminal"
        echo "   4. Run this script again"
        exit 1
    fi
fi

# Check npm
echo "📦 Checking npm installation..."
if command_exists npm; then
    NPM_VERSION=$(npm --version)
    echo "✅ npm is installed: $NPM_VERSION"
else
    echo "❌ npm is not available (should come with Node.js)"
    exit 1
fi

# Install dependencies
echo "📦 Installing frontend dependencies..."
if [ -d "node_modules" ]; then
    echo "📁 node_modules directory exists, checking for updates..."
    npm update
else
    echo "📥 Installing dependencies for the first time..."
    npm install
fi

# Check if installation was successful
if [ -d "node_modules" ]; then
    echo "✅ Dependencies installed successfully"
else
    echo "❌ Failed to install dependencies"
    exit 1
fi

# Create environment file if it doesn't exist
if [ ! -f ".env.local" ]; then
    echo "📝 Creating .env.local file..."
    cat > .env.local << EOF
# Frontend Environment Variables
NEXT_PUBLIC_API_URL=http://localhost:8000
NEXT_PUBLIC_WS_URL=ws://localhost:8000/ws
NEXT_PUBLIC_APP_NAME="StockTrader Pro"
NEXT_PUBLIC_APP_VERSION="1.0.0"
NODE_ENV=development
EOF
    echo "✅ Created .env.local file"
else
    echo "✅ .env.local file already exists"
fi

# Run type check
echo "🔍 Running type check..."
if npm run type-check; then
    echo "✅ TypeScript type check passed"
else
    echo "⚠️  TypeScript type check found issues (expected until dependencies are installed)"
fi

# Check if backend is running
echo "🔗 Checking backend connection..."
BACKEND_URL="${NEXT_PUBLIC_API_URL:-http://localhost:8000}"
if curl -s --connect-timeout 5 "$BACKEND_URL/health" >/dev/null 2>&1; then
    echo "✅ Backend is running at $BACKEND_URL"
else
    echo "⚠️  Backend is not running at $BACKEND_URL"
    echo "   Make sure to start the Python backend first:"
    echo "   cd /c/dev/stocktrader2"
    echo "   source venv/Scripts/activate"
    echo "   python -m uvicorn api.main:app --reload"
fi

echo ""
echo "🎉 Frontend setup complete!"
echo ""
echo "🚀 To start the development server:"
echo "   npm run dev"
echo ""
echo "📱 The application will be available at:"
echo "   http://localhost:3000"
echo ""
echo "📚 Available commands:"
echo "   npm run dev          # Start development server"
echo "   npm run build        # Build for production"
echo "   npm run start        # Start production server"
echo "   npm run lint         # Run ESLint"
echo "   npm run type-check   # Run TypeScript check"
echo ""
echo "✅ Setup completed successfully!"
