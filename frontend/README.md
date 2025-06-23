# StockTrader Frontend

A modern Next.js frontend for the StockTrader application with TypeScript, Tailwind CSS, and real-time trading features.

## 🚀 Quick Start

### Prerequisites

- Node.js 18+ 
- npm or yarn
- Backend API running on http://localhost:8000

### Installation

1. **Install Node.js** (if not already installed):
   ```bash
   # Using winget (Windows)
   winget install OpenJS.NodeJS
   
   # Or download from https://nodejs.org/
   ```

2. **Navigate to frontend directory**:
   ```bash
   cd /c/dev/stocktrader2/frontend
   ```

3. **Install dependencies**:
   ```bash
   npm install
   ```

4. **Start development server**:
   ```bash
   npm run dev
   ```

5. **Open browser**:
   - Navigate to http://localhost:3000

## 📁 Project Structure

```
frontend/
├── src/
│   ├── app/                    # Next.js App Router
│   │   ├── layout.tsx          # Root layout
│   │   ├── page.tsx            # Home page
│   │   ├── globals.css         # Global styles
│   │   ├── dashboard/          # Dashboard pages
│   │   ├── portfolio/          # Portfolio pages
│   │   └── settings/           # Settings pages
│   ├── components/             # Reusable components
│   │   ├── ui/                 # Base UI components
│   │   ├── charts/             # Chart components
│   │   ├── trading/            # Trading components
│   │   └── layout/             # Layout components
│   ├── lib/                    # Utilities and configurations
│   │   ├── api.ts              # API client
│   │   ├── types.ts            # TypeScript types
│   │   ├── utils.ts            # Utility functions
│   │   └── websocket.ts        # WebSocket client
│   ├── stores/                 # Zustand state stores
│   │   ├── market-data.ts      # Market data state
│   │   ├── trading.ts          # Trading state
│   │   ├── portfolio.ts        # Portfolio state
│   │   └── auth.ts             # Authentication state
│   └── hooks/                  # Custom React hooks
│       ├── use-market-data.ts  # Market data hooks
│       ├── use-websocket.ts    # WebSocket hooks
│       └── use-trading.ts      # Trading hooks
├── public/                     # Static assets
├── package.json                # Dependencies and scripts
├── tailwind.config.js          # Tailwind configuration
├── tsconfig.json               # TypeScript configuration
└── next.config.js              # Next.js configuration
```

## 🎨 Features

### Core Features
- **Real-time Dashboard**: Live market data and portfolio tracking
- **Interactive Charts**: Candlestick charts with technical indicators
- **Pattern Recognition**: Visual pattern alerts and notifications
- **Order Management**: Place, modify, and track trading orders
- **Portfolio Analytics**: Comprehensive portfolio performance metrics

### Technical Features
- **TypeScript**: Full type safety across the application
- **Tailwind CSS**: Utility-first styling with custom trading themes
- **shadcn/ui**: Modern, accessible UI components
- **Zustand**: Lightweight state management
- **SWR**: Data fetching with caching and revalidation
- **WebSocket**: Real-time data streaming
- **Responsive Design**: Works on desktop, tablet, and mobile

## 🛠️ Development

### Available Scripts

```bash
npm run dev          # Start development server
npm run build        # Build for production
npm run start        # Start production server
npm run lint         # Run ESLint
npm run type-check   # Run TypeScript type checking
```

### Environment Variables

Create `.env.local` with:

```env
NEXT_PUBLIC_API_URL=http://localhost:8000
NEXT_PUBLIC_WS_URL=ws://localhost:8000/ws
NEXT_PUBLIC_APP_NAME="StockTrader Pro"
```

### Code Style

The project uses:
- **ESLint**: Code linting
- **Prettier**: Code formatting
- **TypeScript**: Type checking

## 🎯 Architecture

### State Management
- **Zustand stores** for global state
- **SWR** for server state caching
- **Local state** for component-specific data

### API Integration
- RESTful API client with automatic token handling
- WebSocket client for real-time updates
- Error handling and retry logic

### Component Structure
- **Base UI components** in `components/ui/`
- **Feature components** organized by domain
- **Layout components** for consistent structure

## 🔧 Backend Integration

The frontend is designed to work with the StockTrader Python backend:

- **Authentication**: JWT token-based authentication
- **Market Data**: Real-time quotes, historical data, indicators
- **Trading**: Order placement, portfolio management
- **Patterns**: Candlestick pattern recognition alerts

## 📱 Responsive Design

The application is fully responsive with:
- **Mobile-first approach**
- **Breakpoint system**: sm, md, lg, xl, 2xl
- **Touch-friendly** trading interface
- **Adaptive charts** for different screen sizes

## 🚀 Deployment

### Production Build

```bash
npm run build
npm run start
```

### Docker Deployment

```dockerfile
FROM node:18-alpine
WORKDIR /app
COPY package*.json ./
RUN npm ci --only=production
COPY . .
RUN npm run build
EXPOSE 3000
CMD ["npm", "start"]
```

## 🤝 Contributing

1. Follow the established code structure
2. Use TypeScript for all new components
3. Add proper error handling
4. Test with real backend integration
5. Ensure responsive design

## 📄 License

This project is part of the StockTrader application suite.
