---
applyTo: '**'
---
# StockTrader Bot - Modern Full-Stack Architecture

### 🚀 **Quick Start Commands**
**Start Full-Stack Development:**
```bash
cd /c/dev/stocktrader2
source venv/Scripts/activate  # Activate Python environment
npm run dev                   # Start both frontend and backend
```

**Individual Services:**
```bash
npm run dev:backend           # FastAPI server on :8000
npm run dev:frontend          # Next.js app on :3000
npm run install:all           # Install all dependencies
```bash
npm run dev:backend           # FastAPI server on :8000
npm run dev:frontend          # Next.js app on :3000
npm run install:all           # Install all dependencies
```

**Access Points:**
- Frontend UI: http://localhost:3000
- Backend API: http://localhost:8000
- API Docs: http://localhost:8000/docs
- Health Check: http://localhost:8000/api/v1/health

### ⚡ **System Verification Commands**
**Test Complete System:**
```bash
cd /c/dev/stocktrader2
source venv/Scripts/activate

# Test enhanced market data service
python -c "
from api.services.market_data_service_enhanced import MarketDataService
from datetime import date, timedelta
service = MarketDataService()
end_date = date.today()
start_date = end_date - timedelta(days=5)
data = service.download_and_save_stock_data('AAPL', start_date, end_date, save_csv=False)
print(f'✅ Market data service: {len(data[\"AAPL\"])} rows fetched')
"

# Test API endpoint (server must be running)
curl -X POST "http://localhost:8000/api/v1/market-data/download" \
  -H "Content-Type: application/json" \
  -d '{"symbols": ["AAPL"], "start_date": "2025-06-18", "end_date": "2025-06-23", "interval": "1d", "save_csv": false}'
```

**Core Module Verification:**
```bash
cd /c/dev/stocktrader2
python -c "
from core.data_validator import validate_file_path, validate_dataframe
from train.deeplearning_trainer import PatternModelTrainer
from train.feature_engineering import compute_technical_features, add_candlestick_pattern_features
from patterns.factory import create_pattern_detector
from patterns.orchestrator import CandlestickPatterns
from security.authentication import create_jwt_token, get_api_credentials
from security.authorization import create_user_context, Role
print('✅ All core modules operational!')
"
```

## 🏗️ **Updated Architecture** (June 2025)

### ✅ **Operational Components**
- **Backend**: FastAPI with enhanced MarketDataService using yfinance
- **Frontend**: Next.js + TypeScript + Tailwind CSS + shadcn/ui
- **API Layer**: RESTful endpoints with Pydantic validation
- **Data Pipeline**: Market data download, validation, CSV storage
- **Core Logic**: ✅ **STABLE** - All Python modules (trading, indicators, ML, patterns)
- **Development**: npm-based concurrent development workflow

### 🔧 **Key Implementation Details**
- **Enhanced Router**: Using `market_data_enhanced.py` with correct method names
- **Service Layer**: `MarketDataService.download_and_save_stock_data()` method
- **Frontend Components**: React components for data download and file management
- **API Communication**: Complete frontend-backend integration working
- **Environment**: Windows/GitBash compatible with reproducible setup

## 📊 **Core System Status** (Stable Since December 2024)

### ✅ **ALL CORE MODULES OPERATIONAL**
- **✅ Core Data Validation**: `core/data_validator.py` - Standalone functions available and validated
- **✅ Deep Learning Training**: `train/deeplearning_trainer.py` - PyTorch models fully operational  
- **✅ Feature Engineering**: `train/feature_engineering.py` - 50+ technical features implemented
- **✅ Pattern Recognition**: `patterns/` - 18+ candlestick patterns with confidence scoring
- **✅ Technical Indicators**: `core/indicators/` - 10+ indicators with API integration
- **✅ Security Framework**: `security/` - Enterprise JWT authentication, role-based authorization, E*TRADE integration
- **✅ Import System**: All modules import without errors, ready for production use

**Comprehensive Testing:**
```bash
cd /c/dev/stocktrader2
python -c "
import pandas as pd
import numpy as np

# Test with sample data
dates = pd.date_range('2024-01-01', periods=50, freq='D')
data = pd.DataFrame({
    'Open': np.random.uniform(100, 110, 50),
    'High': np.random.uniform(105, 115, 50), 
    'Low': np.random.uniform(95, 105, 50),
    'Close': np.random.uniform(100, 110, 50),
    'Volume': np.random.randint(1000, 10000, 50)
}, index=dates)

# Test all major components
from patterns.orchestrator import CandlestickPatterns
detector = CandlestickPatterns()
results = detector.detect_all_patterns(data)
print(f'✅ Patterns: {len(results)} detections')

from core.technical_indicators import TechnicalIndicators
ti = TechnicalIndicators(data)
rsi = ti.calculate_rsi()
print(f'✅ Indicators: RSI for {len(rsi)} periods')

print('🎉 Full system verification complete!')
"
```

## 🕯️ **Candlestick Pattern Organization**

### Hybrid Organization Approach
The pattern system uses a **hybrid approach** documented in `patterns/detectors/README.md`:

**Individual Files** (Critical Patterns):
- `hammer.py` (85 lines) - Fundamental bullish reversal
- `doji.py` (76 lines) - Market indecision/reversal
- `engulfing.py` (72 lines) - Strong reversal signal  
- `morning_star.py` (92 lines) - 3-candle reversal

**Grouped Files** (Related Patterns):
- `bullish_patterns.py` (462 lines) - 9 bullish patterns
- `bearish_patterns.py` (262 lines) - 5 bearish patterns


## Project Overview
**Goal**: Create a modularized stocktrader bot system that downloads OHLCV data for given time periods and intervals, with modern web frontend and robust backend services.


### **Individual Service Development**
```bash
# Backend only (Python/FastAPI)
npm run dev:backend

# Frontend only (Node.js/Next.js)
npm run dev:frontend

# Install dependencies
npm run install:all
```

### **API Testing & Verification**
```bash
# Health check
curl http://localhost:8000/api/v1/health

# Market data download test
curl -X POST "http://localhost:8000/api/v1/market-data/download" \
  -H "Content-Type: application/json" \
  -d '{"symbols": ["AAPL"], "start_date": "2025-06-18", "end_date": "2025-06-23", "interval": "1d", "save_csv": false}'
```

## Environment & Platform Standards
- **Python Version**: 3.12 (Backend)
- **Node.js Version**: 18+ (Frontend)
- **Package Manager**: `uv` (Python), `npm`/`pnpm` (Node.js)
- **Linter**: `ruff` (Python), `eslint` (TypeScript)

### Operating System & Shell Requirements
- **Target OS**: Windows
- **Default Shell**: GitBash (`bash.exe`)
- **All terminal commands MUST be bash-compatible**

### Terminal Command Standards
- **No emojis in bash commands**
- **No special characters in bash commands**
- **Use `which python` and `which pip` to verify virtual environment activation**
- **Use absolute paths**
- **Allow commands to complete execution fully before determining if they are hanging**
- **Wait at least 30-60 seconds for package installation commands (pip install) to complete**
- **Wait at least 10-15 seconds for Python import commands to complete**
- **Python module imports (especially Streamlit, pandas, plotly) can take 5-10 seconds to load**
- **Use `cd /c/dev/stocktrader2` to ensure correct project directory**
- **Handle file path separators correctly for Windows (use forward slashes in GitBash)**
- **Escape spaces in file paths or use quotes when necessary**

#### Pattern Detection Issues
**Problem**: Pattern detection not working
**Solution**:
```bash
# Test with sample data
python -c "
import pandas as pd, numpy as np
data = pd.DataFrame({
    'Open': [100, 101, 102], 'High': [105, 106, 107],
    'Low': [99, 100, 101], 'Close': [104, 105, 106], 'Volume': [1000]*3
})
from patterns.orchestrator import CandlestickPatterns
detector = CandlestickPatterns()
print('Pattern detection working:', len(detector.detect_all_patterns(data)) >= 0)
"
```

### Development Environment Setup

#### Backend (FastAPI) Setup
```bash
cd /c/dev/stocktrader2
python -m venv venv
source venv/Scripts/activate
pip install -r requirements.txt
# Install additional FastAPI dependencies
pip install fastapi uvicorn websockets sqlalchemy
```

#### Frontend (Next.js) Setup
```bash
cd /c/dev/stocktrader2
npx create-next-app@latest frontend --typescript --tailwind --eslint --app --use-npm
cd frontend
npm install zustand swr @radix-ui/react-select @radix-ui/react-dialog
npm install -D @types/node
```

#### Running the Application
**Backend (FastAPI)**:
```bash
cd /c/dev/stocktrader2
source venv/Scripts/activate
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

**Frontend (Next.js)**:
```bash
cd /c/dev/stocktrader2/frontend
npm run dev
```

### Project Structure

```plaintext
stocktrader2/
│
├── api/                              # FastAPI backend application
│   ├── __init__.py                   # API package initialization
│   ├── main.py                       # FastAPI application entry point
│   ├── dependencies.py               # Dependency injection and common dependencies
│   ├── middleware.py                 # Custom middleware (CORS, logging, etc.)
│   ├── routers/                      # API route modules
│   │   ├── __init__.py               # Router package initialization
│   │   ├── auth.py                   # Authentication endpoints
│   │   ├── market_data.py            # Market data endpoints (OHLCV, indicators)
│   │   ├── trading.py                # Trading operations endpoints
│   │   ├── portfolio.py              # Portfolio management endpoints
│   │   ├── websocket.py              # WebSocket endpoints for real-time data
│   │   └── admin.py                  # Administrative endpoints
│   ├── models/                       # Pydantic models for API
│   │   ├── __init__.py               # Models package initialization
│   │   ├── auth.py                   # Authentication models
│   │   ├── market_data.py            # Market data request/response models
│   │   ├── trading.py                # Trading operation models
│   │   └── portfolio.py              # Portfolio models
│   ├── services/                     # Business logic services
│   │   ├── __init__.py               # Services package initialization
│   │   ├── market_data_service.py    # Market data orchestration
│   │   ├── trading_service.py        # Trading operations orchestration
│   │   ├── websocket_service.py      # WebSocket connection management
│   │   └── notification_service.py   # Real-time notifications
│   └── database/                     # Database models and operations (if needed)
│       ├── __init__.py               # Database package initialization
│       ├── models.py                 # SQLAlchemy models
│       ├── crud.py                   # Database CRUD operations
│       └── connection.py             # Database connection management
│
├── frontend/                         # Next.js frontend application
│   ├── package.json                  # Node.js dependencies
│   ├── next.config.js                # Next.js configuration
│   ├── tailwind.config.js            # Tailwind CSS configuration
│   ├── tsconfig.json                 # TypeScript configuration
│   ├── src/                          # Source code
│   │   ├── app/                      # App Router pages and layouts
│   │   │   ├── layout.tsx            # Root layout
│   │   │   ├── page.tsx              # Home page
│   │   │   ├── dashboard/            # Trading dashboard pages
│   │   │   ├── portfolio/            # Portfolio management pages
│   │   │   └── settings/             # Configuration pages
│   │   ├── components/               # Reusable UI components
│   │   │   ├── ui/                   # shadcn/ui components
│   │   │   ├── charts/               # Chart components (candlestick, indicators)
│   │   │   ├── trading/              # Trading-specific components
│   │   │   └── layout/               # Layout components
│   │   ├── lib/                      # Utility libraries and configurations
│   │   │   ├── api.ts                # API client configuration
│   │   │   ├── websocket.ts          # WebSocket client
│   │   │   ├── utils.ts              # General utilities
│   │   │   └── types.ts              # TypeScript type definitions
│   │   ├── stores/                   # State management (Zustand stores)
│   │   │   ├── market-data.ts        # Market data state
│   │   │   ├── trading.ts            # Trading operations state
│   │   │   ├── portfolio.ts          # Portfolio state
│   │   │   └── auth.ts               # Authentication state
│   │   └── hooks/                    # Custom React hooks
│   │       ├── use-market-data.ts    # Market data hooks
│   │       ├── use-websocket.ts      # WebSocket hooks
│   │       └── use-trading.ts        # Trading operation hooks
│   └── public/                       # Static assets
│
├── core/                             # ✅ STABLE - Core trading logic modules 
│   ├── validation/                   # ✅ Enterprise validation system
│   │   ├── dataframe_validation_logic.py    # Data validation implementation
│   │   ├── validation_config.py              # Validation configuration
│   │   ├── validation_models.py              # Validation data models
│   │   └── validation_results.py             # Validation result handling
│   ├── indicators/                   # ✅ Technical indicator suite (10+ indicators)
│   │   ├── __init__.py               # Exports all indicators for easy import
│   │   ├── base.py                   # Basic indicators (SMA, EMA) and utilities
│   │   ├── rsi.py                    # Relative Strength Index implementation
│   │   ├── macd.py                   # MACD (Moving Average Convergence Divergence)
│   │   ├── bollinger_bands.py        # Bollinger Bands volatility indicator
│   │   ├── stochastic.py             # Stochastic Oscillator (%K, %D)
│   │   ├── williams_r.py             # Williams %R momentum oscillator
│   │   ├── cci.py                    # Commodity Channel Index
│   │   ├── vwap.py                   # VWAP and On-Balance Volume (OBV)
│   │   └── adx.py                    # ADX, +DI, -DI, and ATR indicators
│   ├── data_validator.py             # ✅ Centralized validation with standalone functions
│   ├── etrade_candlestick_bot.py     # Trading engine logic
│   ├── etrade_client.py              # E*TRADE API client
│   ├── risk_manager_v2.py            # Advanced risk management logic
│   └── technical_indicators.py       # Core technical indicator calculations
│
├── utils/                            # Utility modules
├── security/                         # ✅ STABLE - Enterprise-grade security package
│   ├── authentication.py            # JWT tokens, credential management, password hashing
│   ├── authorization.py             # Role-based access control, E*TRADE permissions
│   ├── etrade_security.py           # E*TRADE credential management, session validation
│   ├── encryption.py                # Cryptographic operations, token generation
│   └── utils.py                     # Input validation, sanitization, path security
├── patterns/                         # ✅ STABLE - Pattern recognition system (18+ patterns)
│   ├── detectors/                    # Candlestick pattern detectors (hybrid organization)
│   │   ├── README.md                 # ✅ Comprehensive pattern documentation & organization
│   │   ├── __init__.py               # Pattern imports and exports  
│   │   ├── hammer.py                 # ✅ Hammer pattern (individual - 85 lines)
│   │   ├── doji.py                   # ✅ Doji pattern (individual - 76 lines)
│   │   ├── engulfing.py              # ✅ Bullish Engulfing pattern (individual - 72 lines)
│   │   ├── morning_star.py           # ✅ Morning Star pattern (individual - 92 lines)
│   │   ├── bullish_patterns.py       # ✅ 9 bullish patterns (grouped - 462 lines)
│   │   └── bearish_patterns.py       # ✅ 5 bearish patterns (grouped - 262 lines)
│   ├── base.py                       # Pattern detection base classes
│   ├── orchestrator.py               # ✅ CandlestickPatterns orchestrator
│   ├── factory.py                    # ✅ Pattern detector factory
│   └── pattern_utils.py              # Pattern utilities
│
├── train/                            # ✅ STABLE - ML training pipeline
│   ├── deeplearning_trainer.py       # ✅ PyTorch neural network training
│   ├── feature_engineering.py        # ✅ 50+ technical features
│   ├── ml_trainer.py                 # Classical ML training
│   ├── model_manager.py              # Model versioning and management
│   └── ml_config.py                  # ML configuration settings
├── models/                           # Saved ML models and artifacts
├── data/                             # Data storage directory
├── logs/                             # Application logs
├── tests/                            # Unit & integration tests
│   ├── api/                          # Backend API tests
│   ├── frontend/                     # Frontend component tests
│   └── integration/                  # End-to-end integration tests
├── docs/                             # Documentation
├── examples/                         # Example scripts and configurations
├── .env.example                      # Example environment configuration
├── requirements.txt                  # Python dependencies
├── docker-compose.yml                # Docker composition for development
├── README.md                         # Project documentation
└── LICENSE                           # License file
```


### File and Module Guidelines
- **Module Organization**: Use the established structure with clear separation between frontend/backend
- **Testing**: All test scripts should be saved in the `tests/` directory
- **API Integration**: Core Python modules accessible via FastAPI services
- **Frontend Components**: Reusable UI components in `frontend/src/components/`
- **State Management**: Use Zustand stores for client-side state
- **Real-time Data**: WebSocket connections for live market data and trading updates

### Code Quality Standards
- **Function Length**: Keep modules focused (~300 lines max)
- **Single Responsibility**: Each module should have one clear purpose
- **Type Hints**: Use Python type annotations (backend) and TypeScript (frontend)
- **Docstrings**: Google-style docstrings for all public functions
- **API Documentation**: FastAPI auto-generated OpenAPI documentation
- **Component Documentation**: JSDoc comments for React components

## Architecture Guidelines

### Backend (FastAPI) Standards
- **Route Organization**: Group related endpoints in router modules
- **Dependency Injection**: Use FastAPI dependencies for common operations
- **Error Handling**: Consistent error responses with proper HTTP status codes
- **Validation**: Pydantic models for request/response validation
- **WebSocket**: Real-time data streaming for market updates
- **Security**: JWT authentication, rate limiting, input validation

### Frontend (Next.js) Standards
- **Component Structure**: Functional components with TypeScript
- **State Management**: Zustand for global state, local state for component-specific data
- **Data Fetching**: SWR for server state management and caching
- **Styling**: Tailwind CSS with shadcn/ui components
- **Real-time Updates**: WebSocket integration for live data
- **Performance**: Code splitting, lazy loading, optimized bundle size

### API Design Principles
- **RESTful**: Standard HTTP methods and status codes
- **Consistent Naming**: snake_case for Python, camelCase for TypeScript/JavaScript
- **Versioning**: API versioning strategy (/api/v1/)
- **Documentation**: OpenAPI/Swagger documentation
- **Real-time**: WebSocket endpoints for streaming data

## Core Features

### Market Data Management
- **OHLCV Data**: Download and process stock data for various time periods
- **Comprehensive Technical Indicators Suite**:
  - **Momentum Indicators**: RSI, Stochastic Oscillator, Williams %R, CCI
  - **Trend Following**: MACD, ADX, ATR
  - **Volatility Indicators**: Bollinger Bands, ATR
  - **Volume Indicators**: VWAP, On-Balance Volume (OBV)
  - **Basic Indicators**: SMA, EMA with various period configurations
- **Real-time Feeds**: Live market data streaming via WebSocket
- **Data Validation**: Comprehensive validation of market data integrity

## 📈 Technical Indicators Architecture

### Indicator Implementation Standards
All technical indicators follow these standards:
- **Type Safety**: Full type hints and Pydantic models for API integration
- **Error Resilience**: Graceful fallbacks when pandas_ta fails
- **Index Agnostic**: Support both RangeIndex and DatetimeIndex
- **Warning Suppression**: Clean execution without pandas_ta warnings
- **Documentation**: Comprehensive docstrings with parameter descriptions
- **Testing**: Functionally verified with both index types


### API Integration Pattern
```python
from core.technical_indicators import TechnicalIndicators

# All indicators accessible through main class
ti = TechnicalIndicators(ohlcv_data)

# Momentum indicators
rsi = ti.calculate_rsi(period=14)
stoch = ti.calculate_stochastic(k_period=14, d_period=3)
williams_r = ti.calculate_williams_r(period=14)
cci = ti.calculate_cci(period=20)

# Trend indicators
macd = ti.calculate_macd(fast=12, slow=26, signal=9)
adx = ti.calculate_adx(period=14)

# Volatility indicators
bb = ti.calculate_bollinger_bands(period=20, std_dev=2)

# Volume indicators
vwap = ti.calculate_vwap()
obv = ti.calculate_obv()
```

### Trading Operations
- **Order Management**: Place, modify, cancel orders via E*TRADE API
- **Risk Management**: Position sizing, stop-loss, risk assessment
- **Portfolio Tracking**: Real-time portfolio value and performance metrics
- **Pattern Recognition**: Candlestick pattern detection and alerts

### Security Framework
- **JWT Authentication**: Modern token-based authentication with python-jose
- **Role-Based Authorization**: 5-tier access control (Guest/Viewer/Trader/Analyst/Admin)
- **E*TRADE Integration**: Secure credential management and session validation
- **Input Validation**: XSS protection, sanitization, and secure data handling
- **Audit Logging**: Comprehensive security event tracking and monitoring
- **Cryptographic Operations**: Secure token generation, password hashing, data encryption

### User Interface
- **Dashboard**: Real-time market overview with charts and indicators
- **Trading Panel**: Order placement and portfolio management
- **Analytics**: Technical analysis charts with interactive indicators
- **Settings**: Configuration for trading parameters and notifications

## Development Workflow

### Backend Development
1. **API First**: Design API endpoints and models before implementation
2. **Service Layer**: Implement business logic in service modules
3. **Testing**: Unit tests for services, integration tests for API endpoints
4. **Documentation**: Update OpenAPI schemas and endpoint documentation

### Frontend Development
1. **Component Library**: Build reusable UI components with shadcn/ui
2. **State Management**: Define Zustand stores for application state
3. **API Integration**: Use SWR for data fetching and caching
4. **Real-time**: Implement WebSocket connections for live updates

### Integration Testing
- **End-to-End**: Test complete user workflows
- **API Testing**: Validate API contracts and responses
- **WebSocket Testing**: Test real-time data streaming
- **Performance**: Monitor API response times and frontend performance

## Security & Best Practices
- **Authentication**: JWT-based authentication with secure token handling
- **API Security**: Rate limiting, input validation, CORS configuration
- **Data Protection**: Encrypt sensitive data, secure API keys in environment variables
- **Frontend Security**: XSS protection, secure cookie handling, CSP headers
- **Error Handling**: Don't expose sensitive information in error messages
- **Logging**: Comprehensive logging without exposing sensitive data
- **Environment**: Separate configurations for development, staging, production

## Migration Strategy
1. **Phase 1**: Set up FastAPI backend with core market data endpoints
2. **Phase 2**: Create Next.js frontend with basic dashboard functionality
3. **Phase 3**: Implement WebSocket for real-time data streaming
4. **Phase 4**: Migrate trading operations and portfolio management
5. **Phase 5**: Add advanced features (ML models, pattern recognition)
6. **Phase 6**: Performance optimization and production deployment

`

## Environment Variables
```bash
# Backend
ETRADE_CLIENT_KEY=your_etrade_client_key
ETRADE_CLIENT_SECRET=your_etrade_client_secret
JWT_SECRET_KEY=your_jwt_secret
DATABASE_URL=sqlite:///./stocktrader.db
ENVIRONMENT=development

# Frontend
NEXT_PUBLIC_API_URL=http://localhost:8000
NEXT_PUBLIC_WS_URL=ws://localhost:8000/ws
```
