# StockTrader Bot - Modern Full-Stack Trading Platform

A next-generation stocktrader bot system that combines sophisticated technical analysis with modern web technologies. Built with a decoupled architecture featuring a FastAPI backend and Next.js frontend for real-time market data processing and trading operations.

## 🚀 Project Overview

**Goal**: Create a modularized stocktrader bot system that downloads OHLCV data for given time periods and intervals, with a modern web frontend and robust backend services.

**Architecture**: Full-stack web application with clear frontend-backend separation
- **Frontend**: Next.js + TypeScript + shadcn/ui + Zustand + SWR
- **Backend**: FastAPI + Pydantic + WebSockets + SQLAlchemy
- **Core Logic**: Advanced Python modules for trading, indicators, ML, and risk management

## ✨ Key Features

### 📊 Market Data Management
- **OHLCV Data Processing**: Download and process stock data for various time periods and intervals
- **Comprehensive Technical Indicators Suite**: 
  - **Momentum Indicators**: RSI, Stochastic Oscillator, Williams %R, CCI
  - **Trend Following**: MACD, ADX, ATR
  - **Volatility Indicators**: Bollinger Bands, ATR
  - **Volume Indicators**: VWAP, On-Balance Volume (OBV)
  - **Basic Indicators**: SMA, EMA with various period configurations
- **Real-time Feeds**: Live market data streaming via WebSocket connections
- **Data Validation**: Enterprise-grade validation system ensuring data integrity

### 🤖 Trading Operations
- **Order Management**: Place, modify, and cancel orders via E*TRADE API integration
- **Risk Management**: Advanced position sizing, stop-loss, and risk assessment
- **Portfolio Tracking**: Real-time portfolio value and performance metrics
- **Pattern Recognition**: ML-powered candlestick pattern detection and alerts

### 💻 Modern User Interface
- **Real-time Dashboard**: Interactive market overview with live charts and indicators
- **Trading Panel**: Intuitive order placement and portfolio management interface
- **Analytics Suite**: Technical analysis charts with interactive indicators
- **Configuration**: Comprehensive settings for trading parameters and notifications

### 🧠 Machine Learning Pipeline
- **Feature Engineering**: Automated technical feature extraction
- **Model Training**: Classic ML and deep learning model training
- **Pattern Detection**: Neural network-based pattern recognition
- **Live Inference**: Real-time ML predictions for trading signals

## � Technology Stack

### Backend
- **FastAPI**: High-performance Python web framework
- **Pydantic**: Data validation and serialization
- **WebSockets**: Real-time bidirectional communication
- **SQLAlchemy**: Database ORM (if database storage needed)
- **pandas/numpy**: Data processing and analysis

### Frontend
- **Next.js 14**: React framework with App Router
- **TypeScript**: Type-safe JavaScript development
- **shadcn/ui**: Beautiful and accessible UI components
- **Zustand**: Lightweight state management
- **SWR**: Data fetching and caching
- **Tailwind CSS**: Utility-first CSS framework

## 📋 System Requirements

- **Python**: 3.12+
- **Node.js**: 18+
- **Git**: Version control
- **E*TRADE Developer Account**: API keys for trading operations

## 🚀 Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/stocktrader2.git
cd stocktrader2
```

### 2. Backend Setup (FastAPI)
```bash
# Create and activate virtual environment
python -m venv venv
source venv/Scripts/activate  # Windows GitBash
# source venv/bin/activate    # Linux/macOS

# Install Python dependencies
pip install -r requirements.txt

# Install additional FastAPI dependencies
pip install fastapi uvicorn websockets sqlalchemy
```

### 3. Frontend Setup (Next.js)
```bash
# Create Next.js frontend
npx create-next-app@latest frontend --typescript --tailwind --eslint --app --use-npm

# Navigate to frontend directory
cd frontend

# Install additional dependencies
npm install zustand swr @radix-ui/react-select @radix-ui/react-dialog recharts
npm install -D @types/node
```

### 4. Environment Configuration
```bash
# Copy example environment file
cp .env.example .env

# Edit .env file with your E*TRADE API keys and configuration
# ETRADE_CLIENT_KEY=your_etrade_client_key
# ETRADE_CLIENT_SECRET=your_etrade_client_secret
# JWT_SECRET_KEY=your_jwt_secret
```

### 5. Run the Application

**Backend (Terminal 1)**:
```bash
cd /c/dev/stocktrader2
source venv/Scripts/activate
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

**Frontend (Terminal 2)**:
```bash
cd /c/dev/stocktrader2/frontend
npm run dev
```

**Access the Application**:
- Frontend: http://localhost:3000
- Backend API: http://localhost:8000
- API Documentation: http://localhost:8000/docs

## 📈 Technical Indicators Suite

The StockTrader Bot includes a comprehensive suite of technical indicators built for robust analysis and API integration. All indicators are optimized for performance, include proper error handling, and support both pandas RangeIndex and DatetimeIndex.

### 🎯 Momentum Indicators

#### RSI (Relative Strength Index)
**File**: `core/indicators/rsi.py`
- **Purpose**: Measures the speed and change of price movements (0-100 scale)
- **Usage**: Identify overbought (>70) and oversold (<30) conditions
- **Implementation**: Uses pandas_ta with fallback to manual calculation
- **Parameters**: Configurable period (default: 14)

#### Stochastic Oscillator
**File**: `core/indicators/stochastic.py`
- **Purpose**: Compares closing price to price range over given period
- **Usage**: Identify momentum reversals and overbought/oversold levels
- **Returns**: %K and %D lines for comprehensive analysis
- **Parameters**: K period (14), D period (3), smoothing (3)

#### Williams %R
**File**: `core/indicators/williams_r.py`
- **Purpose**: Momentum oscillator that moves between 0 and -100
- **Usage**: Identify overbought (>-20) and oversold (<-80) conditions
- **Implementation**: Inverse of Fast Stochastic Oscillator
- **Parameters**: Configurable period (default: 14)

#### CCI (Commodity Channel Index)
**File**: `core/indicators/cci.py`
- **Purpose**: Measures deviation from average price (typically ±100 range)
- **Usage**: Identify cyclical trends and reversal points
- **Implementation**: Uses typical price and mean deviation
- **Parameters**: Configurable period (default: 20)

### 📊 Trend Following Indicators

#### MACD (Moving Average Convergence Divergence)
**File**: `core/indicators/macd.py`
- **Purpose**: Shows relationship between two moving averages
- **Usage**: Identify trend changes and momentum shifts
- **Returns**: MACD line, signal line, and histogram
- **Parameters**: Fast (12), slow (26), signal (9) periods

#### ADX (Average Directional Index)
**File**: `core/indicators/adx.py`
- **Purpose**: Measures trend strength regardless of direction
- **Usage**: Values >25 indicate strong trend, <20 indicate weak trend
- **Returns**: ADX, +DI, and -DI for comprehensive trend analysis
- **Parameters**: Configurable period (default: 14)

### 📏 Volatility Indicators

#### Bollinger Bands
**File**: `core/indicators/bollinger_bands.py`
- **Purpose**: Volatility bands around moving average
- **Usage**: Identify overbought/oversold conditions and volatility changes
- **Returns**: Upper band, middle (SMA), lower band, and bandwidth
- **Parameters**: Period (20), standard deviations (2)

#### ATR (Average True Range)
**File**: `core/indicators/adx.py` (included with ADX)
- **Purpose**: Measures market volatility
- **Usage**: Position sizing and stop-loss placement
- **Implementation**: Exponential moving average of true range
- **Parameters**: Configurable period (default: 14)

### 📊 Volume Indicators

#### VWAP (Volume Weighted Average Price)
**File**: `core/indicators/vwap.py`
- **Purpose**: Average price weighted by volume
- **Usage**: Intraday benchmark for institutional trading
- **Features**: Robust datetime index handling with fallback support
- **Note**: Works with both RangeIndex and DatetimeIndex data

#### OBV (On-Balance Volume)
**File**: `core/indicators/vwap.py` (included with VWAP)
- **Purpose**: Cumulative volume based on price direction
- **Usage**: Confirm price trends with volume analysis
- **Implementation**: Running total of volume on up/down days

### 🔄 Basic Moving Averages

#### Simple Moving Average (SMA)
**File**: `core/indicators/base.py`
- **Purpose**: Average price over specified period
- **Usage**: Trend identification and support/resistance levels
- **Implementation**: Efficient pandas rolling mean
- **Parameters**: Configurable period

#### Exponential Moving Average (EMA)
**File**: `core/indicators/base.py`
- **Purpose**: Weighted average giving more importance to recent prices
- **Usage**: More responsive trend following than SMA
- **Implementation**: Pandas exponential weighted mean
- **Parameters**: Configurable period and smoothing factor

### 🛠️ Indicator Usage Examples

```python
from core.technical_indicators import TechnicalIndicators
import pandas as pd

# Initialize with OHLCV data
ti = TechnicalIndicators(data)

# Calculate individual indicators
rsi = ti.calculate_rsi(period=14)
macd_data = ti.calculate_macd(fast=12, slow=26, signal=9)
bb_data = ti.calculate_bollinger_bands(period=20, std_dev=2)

# Momentum indicators
stoch_data = ti.calculate_stochastic(k_period=14, d_period=3)
williams_r = ti.calculate_williams_r(period=14)
cci = ti.calculate_cci(period=20)

# Trend and volatility
adx_data = ti.calculate_adx(period=14)
vwap = ti.calculate_vwap()
obv = ti.calculate_obv()

# All indicators are designed for FastAPI integration
# and include proper error handling and validation
```

### 🔧 API Integration Ready

All technical indicators are:
- **Type-safe**: Full type hints and validation
- **Error-resilient**: Graceful fallbacks when pandas_ta fails
- **Index-agnostic**: Work with both RangeIndex and DatetimeIndex
- **Warning-free**: Suppressed non-critical pandas_ta warnings
- **Documented**: Comprehensive docstrings and parameter documentation
- **Tested**: Functionally verified with both index types## 📁 Project Structure

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
│   └── src/                          # Source code
│       ├── app/                      # App Router pages and layouts
│       ├── components/               # Reusable UI components
│       ├── lib/                      # Utility libraries and configurations
│       ├── stores/                   # State management (Zustand stores)
│       └── hooks/                    # Custom React hooks
│
├── core/                             # Core trading logic modules
│   ├── validation/                   # Validation logic modules
│   ├── indicators/                   # Technical indicator modules
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
│   ├── data_validator.py             # Centralized data validation services
│   ├── etrade_candlestick_bot.py     # Trading engine logic
│   ├── etrade_client.py              # E*TRADE API client
│   ├── exceptions.py                 # Custom application exceptions
│   ├── risk_manager_v2.py            # Advanced risk management logic
│   └── technical_indicators.py       # Core technical indicator calculations
│
├── utils/                            # Utility modules
├── security/                         # Enterprise-grade security package
├── patterns/                         # Pattern recognition modules
├── train/                            # Machine learning training pipeline
├── models/                           # Saved ML models and artifacts
├── data/                             # Data storage directory
├── logs/                             # Application logs
├── tests/                            # Unit & integration tests
├── docs/                             # Documentation
├── .env.example                      # Example environment configuration
├── requirements.txt                  # Python dependencies
└── README.md                         # Project documentation
```

## 🔄 Development Workflow

### API-First Development
1. **Design API Endpoints**: Define FastAPI routes and Pydantic models
2. **Implement Services**: Build business logic in service layer
3. **Create Frontend Components**: Build React components consuming the API
4. **WebSocket Integration**: Implement real-time data streaming
5. **Testing**: Unit tests for backend services, component tests for frontend

### Real-time Data Flow
1. **Market Data Service** fetches OHLCV data from external APIs
2. **Technical Indicators** are calculated using core Python modules
3. **WebSocket Service** broadcasts real-time updates to connected clients
4. **Frontend** receives updates and updates the UI reactively
5. **Trading Engine** processes signals and executes trades via E*TRADE API

## 🧪 Testing

### Backend Testing
```bash
# Run Python tests
pytest tests/api/ -v
pytest tests/core/ -v
```

### Frontend Testing
```bash
# Run React component tests
cd frontend
npm test
```

### Integration Testing
```bash
# Run end-to-end tests
pytest tests/integration/ -v
```

## 📚 API Documentation

The FastAPI backend automatically generates interactive API documentation:

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **OpenAPI Schema**: http://localhost:8000/openapi.json

## 🚀 Deployment

### Development
- Backend: `uvicorn api.main:app --reload`
- Frontend: `npm run dev`

### Production
- Backend: `uvicorn api.main:app --host 0.0.0.0 --port 8000`
- Frontend: `npm run build && npm start`

## 🔒 Security Features

- **JWT Authentication**: Secure token-based authentication
- **Input Validation**: Comprehensive request validation with Pydantic
- **Rate Limiting**: API endpoint protection against abuse
- **CORS Configuration**: Secure cross-origin resource sharing
- **Environment Variables**: Secure configuration management

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m "Add amazing feature"`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

Please ensure your code follows the project's coding standards and includes appropriate tests.

## 📄 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

## 🆘 Support

For questions, issues, or feature requests:

1. Check the [Documentation](docs/)
2. Search [Issues](https://github.com/your-username/stocktrader2/issues)
3. Create a new issue if needed

## 🎯 Roadmap

- [x] **Phase 0**: Core technical indicators suite with comprehensive implementation
  - [x] Momentum indicators (RSI, Stochastic, Williams %R, CCI)
  - [x] Trend indicators (MACD, ADX, ATR)
  - [x] Volatility indicators (Bollinger Bands)
  - [x] Volume indicators (VWAP, OBV)
  - [x] All indicators optimized for API integration
- [ ] **Phase 1**: FastAPI backend with core market data endpoints
- [ ] **Phase 2**: Next.js frontend with basic dashboard functionality  
- [ ] **Phase 3**: WebSocket implementation for real-time data streaming
- [ ] **Phase 4**: Trading operations and portfolio management migration
- [ ] **Phase 5**: Advanced features (ML models, pattern recognition)
- [ ] **Phase 6**: Performance optimization and production deployment

---

**Built with ❤️ using modern web technologies for next-generation trading applications.**
