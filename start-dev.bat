@echo off
REM StockTrader Development Server Startup Script (Windows)
REM Starts both backend (FastAPI) and frontend (Next.js) servers

echo 🚀 Starting StockTrader Full-Stack Development Environment...

REM Check if we're in the correct directory
if not exist "requirements.txt" (
    echo ❌ Error: Please run this script from the project root directory
    exit /b 1
)

if not exist "frontend" (
    echo ❌ Error: Frontend directory not found
    exit /b 1
)

REM Check if virtual environment exists
if not exist "venv" (
    echo ❌ Error: Virtual environment not found. Please run: python -m venv venv
    exit /b 1
)

REM Check if frontend dependencies are installed
if not exist "frontend\node_modules" (
    echo 📦 Installing frontend dependencies...
    cd frontend && npm install && cd ..
)

echo 🔧 Activating Python virtual environment...
call venv\Scripts\activate

echo 🐍 Starting Backend (FastAPI) on http://localhost:8000...
echo 📱 Starting Frontend (Next.js) on http://localhost:3000...
echo.
echo 📖 API Documentation: http://localhost:8000/docs
echo 🌐 Frontend Application: http://localhost:3000
echo.
echo Press Ctrl+C to stop both servers
echo.

REM Start both servers
start "Backend" cmd /k "uvicorn api.main:app --reload --host 0.0.0.0 --port 8000"
timeout /t 3 >nul
start "Frontend" cmd /k "cd frontend && npm run dev"

echo ✅ Both servers started in separate windows
echo ℹ️  Close the server windows to stop the services
pause
