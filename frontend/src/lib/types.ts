// Market Data Types
export interface Quote {
  symbol: string
  price: number
  change: number
  changePercent: number
  volume: number
  marketCap?: number
  pe?: number
  lastUpdate: string
}

export interface HistoricalData {
  symbol: string
  data: OHLCV[]
  period: string
}

export interface OHLCV {
  timestamp: string
  open: number
  high: number
  low: number
  close: number
  volume: number
}

// Technical Indicators
export interface TechnicalIndicator {
  name: string
  values: number[]
  timestamps: string[]
  parameters: Record<string, any>
}

export interface IndicatorResponse {
  symbol: string
  indicators: Record<string, TechnicalIndicator>
}

// Pattern Recognition
export interface PatternResult {
  name: string
  detected: boolean
  confidence: number
  timestamp: string
  type: 'bullish' | 'bearish' | 'neutral'
  description: string
}

export interface PatternResponse {
  symbol: string
  patterns: PatternResult[]
  lastUpdate: string
}

// Trading Types
export interface Position {
  symbol: string
  quantity: number
  averagePrice: number
  currentPrice: number
  marketValue: number
  unrealizedPnL: number
  unrealizedPnLPercent: number
  side: 'long' | 'short'
}

export interface Order {
  id: string
  symbol: string
  side: 'buy' | 'sell'
  type: 'market' | 'limit' | 'stop' | 'stop_limit'
  quantity: number
  price?: number
  stopPrice?: number
  status: 'pending' | 'filled' | 'cancelled' | 'rejected'
  timestamp: string
  filledQuantity?: number
  averageFillPrice?: number
}

export interface Portfolio {
  totalValue: number
  totalGainLoss: number
  totalGainLossPercent: number
  buyingPower: number
  cashBalance: number
  positions: Position[]
  dayGainLoss: number
  dayGainLossPercent: number
}

export interface Account {
  accountId: string
  accountType: string
  accountValue: number
  buyingPower: number
  cashBalance: number
  dayTradeCount: number
  dayTradingBuyingPower: number
}

// Authentication Types
export interface User {
  id: string
  username: string
  email: string
  firstName: string
  lastName: string
  role: 'viewer' | 'trader' | 'analyst' | 'admin'
  permissions: string[]
  lastLogin: string
}

export interface AuthResponse {
  user: User
  token: string
  refreshToken: string
  expiresIn: number
}

// WebSocket Types
export interface WebSocketMessage {
  type: string
  data: any
  timestamp: string
}

export interface MarketDataUpdate {
  type: 'quote' | 'trade' | 'level1' | 'level2'
  symbol: string
  data: any
  timestamp: string
}

export interface OrderUpdate {
  type: 'order_status' | 'fill' | 'cancel'
  orderId: string
  data: Partial<Order>
  timestamp: string
}

export interface Alert {
  id: string
  type: 'pattern' | 'price' | 'volume' | 'indicator'
  symbol: string
  message: string
  severity: 'info' | 'warning' | 'error'
  timestamp: string
  read: boolean
}

// API Response Types
export interface ApiResponse<T = any> {
  success: boolean
  data: T
  message?: string
  errors?: string[]
  timestamp: string
}

export interface PaginatedResponse<T = any> extends ApiResponse<T> {
  pagination: {
    page: number
    limit: number
    total: number
    totalPages: number
  }
}

// Chart Types
export interface ChartConfig {
  symbol: string
  timeframe: '1m' | '5m' | '15m' | '1h' | '1d' | '1w' | '1M'
  indicators: string[]
  patterns: boolean
  volume: boolean
}

export interface ChartData {
  symbol: string
  timeframe: string
  candles: OHLCV[]
  indicators?: Record<string, TechnicalIndicator>
  patterns?: PatternResult[]
  volume: number[]
}

// UI State Types
export interface UIState {
  theme: 'light' | 'dark' | 'system'
  sidebarCollapsed: boolean
  activeTab: string
  notifications: Alert[]
}

// Error Types
export interface ApiError {
  code: string
  message: string
  details?: any
  timestamp: string
}

export class TradingError extends Error {
  constructor(
    message: string,
    public code: string,
    public details?: any
  ) {
    super(message)
    this.name = 'TradingError'
  }
}
