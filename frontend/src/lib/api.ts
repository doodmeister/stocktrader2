// API Configuration
export const API_CONFIG = {
  BASE_URL: process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000',
  WS_URL: process.env.NEXT_PUBLIC_WS_URL || 'ws://localhost:8000/ws',
  TIMEOUT: 10000,
}

// Types matching backend models
export interface MarketDataRequest {
  symbols: string[]
  start_date: string
  end_date: string
  interval?: string
  save_csv?: boolean
}

export interface MarketDataResponse {
  status: string
  message: string
  symbols: string[]
  data_info: Record<string, {
    rows: number
    columns: string[]
    date_range: {
      start: string | null
      end: string | null
    }
    latest_close?: number
  }>
}

export interface LoadCSVRequest {
  file_path: string
  symbol?: string
}

export interface LoadCSVResponse {
  symbol: string
  file_path: string
  total_records: number
  start_date: string
  end_date: string
  data_summary: {
    first_price: number
    last_price: number
    price_change: number
    price_change_percent: number
    volume_avg: number
  }
}

export interface StockInfo {
  symbol: string
  company_name?: string
  current_price?: number
  previous_close?: number
  market_cap?: number
  volume?: number
  pe_ratio?: number
  dividend_yield?: number
  sector?: string
  industry?: string
}

export interface ValidationResponse {
  symbol: string
  is_valid: boolean
  info?: StockInfo
  error?: string
}

export interface MarketDataFile {
  file_path: string
  symbol: string
  period: string
  interval: string
  created_date: string
  file_size: number
}

// Technical Analysis types
export interface TechnicalAnalysisRequest {
  symbol: string
  data_source: string
  csv_file_path?: string
  period?: string
  include_indicators?: string[]
  rsi_period?: number
  macd_fast?: number
  macd_slow?: number
  macd_signal?: number
  bb_period?: number
  bb_std_dev?: number
  stoch_k_period?: number
  stoch_d_period?: number
  williams_r_period?: number
  cci_period?: number
  adx_period?: number
  sma_periods?: number[]
  ema_periods?: number[]
}

export interface IndicatorResult {
  name: string
  current_value: any
  signal: string
  strength: number
  data: any[]
}

export interface TechnicalAnalysisResponse {
  symbol: string
  analysis_timestamp: string
  data_period: string
  total_records: number
  indicators: IndicatorResult[]
  overall_signal: string
  signal_strength: number
}

// Pattern Detection types
export interface PatternDetectionRequest {
  symbol: string
  data_source: string
  csv_file_path?: string
  period?: string
  min_confidence?: number
  include_patterns?: string[]
  recent_only?: boolean
  lookback_days?: number
}

export interface PatternResult {
  name: string
  confidence: number
  signal: string
  start_index: number
  end_index: number
  description: string
  candles: any[]
  date?: string; // Add date field for compatibility with backend response
  pattern_name?: string;
  pattern_type?: string;
}

export interface PatternDetectionResponse {
  symbol: string
  analysis_timestamp: string
  data_period: string
  total_records: number
  patterns: PatternResult[]
  pattern_summary: {
    total_patterns: number
    bullish_patterns: number
    bearish_patterns: number
    neutral_patterns: number
  }
}

// API Endpoints
export const API_ENDPOINTS = {
  // Health
  HEALTH: '/api/v1/health',
    // Market Data  
  MARKET_DATA: {
    DOWNLOAD: '/api/v1/market-data/download',
    LOAD_CSV: '/api/v1/market-data/load-csv',
    LIST_FILES: '/api/v1/market-data/list-files',
    INFO: '/api/v1/market-data/info',
    VALIDATE: '/api/v1/market-data/validate',  },
  // Analysis endpoints
  ANALYSIS: {
    TECHNICAL: '/api/v1/analysis/technical-indicators',
    PATTERNS: '/api/v1/analysis/pattern-detection',
    OPENAI: '/api/v1/analysis/openai',
  },
}

// API Error class
export class APIError extends Error {
  constructor(public status: number, message: string, public response?: any) {
    super(message)
    this.name = 'APIError'
  }
}

// HTTP Client
class ApiClient {
  private baseUrl: string
  private timeout: number

  constructor(baseUrl: string = API_CONFIG.BASE_URL, timeout: number = API_CONFIG.TIMEOUT) {
    this.baseUrl = baseUrl
    this.timeout = timeout
  }

  private async request<T>(
    endpoint: string,
    options: RequestInit = {}
  ): Promise<T> {
    const url = `${this.baseUrl}${endpoint}`
    
    const config: RequestInit = {
      ...options,
      headers: {
        'Content-Type': 'application/json',
        ...options.headers,
      },
    }

    try {
      const controller = new AbortController()
      const timeoutId = setTimeout(() => controller.abort(), this.timeout)

      const response = await fetch(url, {
        ...config,
        signal: controller.signal,
      })

      clearTimeout(timeoutId)

      if (!response.ok) {
        let errorMessage = `HTTP ${response.status}: ${response.statusText}`
        try {
          const errorBody = await response.json()
          errorMessage = errorBody.detail || errorMessage
        } catch {
          // If we can't parse the error body, use the status text
        }
        throw new APIError(response.status, errorMessage, response)
      }

      return await response.json()
    } catch (error) {
      if (error instanceof APIError) {
        throw error
      }
      // Network or other errors
      throw new APIError(0, `Network error: ${error instanceof Error ? error.message : 'Unknown error'}`)
    }
  }

  // HTTP Methods
  async get<T>(endpoint: string, params?: Record<string, string>): Promise<T> {
    const url = params ? `${endpoint}?${new URLSearchParams(params)}` : endpoint
    return this.request<T>(url, { method: 'GET' })
  }

  async post<T>(endpoint: string, data?: any): Promise<T> {
    return this.request<T>(endpoint, {
      method: 'POST',
      body: data ? JSON.stringify(data) : undefined,
    })
  }

  async put<T>(endpoint: string, data?: any): Promise<T> {
    return this.request<T>(endpoint, {
      method: 'PUT',
      body: data ? JSON.stringify(data) : undefined,
    })
  }

  async delete<T>(endpoint: string): Promise<T> {
    return this.request<T>(endpoint, { method: 'DELETE' })
  }
}

// Export singleton instance
export const apiClient = new ApiClient()

// Market Data API functions
export const marketDataAPI = {
  /**
   * Download stock data from Yahoo Finance
   */
  async downloadStockData(request: MarketDataRequest): Promise<MarketDataResponse> {
    return apiClient.post<MarketDataResponse>(API_ENDPOINTS.MARKET_DATA.DOWNLOAD, request)
  },

  /**
   * Load existing CSV file
   */
  async loadCSVData(request: LoadCSVRequest): Promise<LoadCSVResponse> {
    return apiClient.post<LoadCSVResponse>(API_ENDPOINTS.MARKET_DATA.LOAD_CSV, request)
  },

  /**
   * List available CSV files
   */
  async listCSVFiles(): Promise<MarketDataFile[]> {
    return apiClient.get<MarketDataFile[]>(API_ENDPOINTS.MARKET_DATA.LIST_FILES)
  },

  /**
   * Get stock information
   */
  async getStockInfo(symbol: string): Promise<StockInfo> {
    return apiClient.get<StockInfo>(`${API_ENDPOINTS.MARKET_DATA.INFO}/${encodeURIComponent(symbol)}`)
  },

  /**
   * Validate stock symbol
   */
  async validateSymbol(symbol: string): Promise<ValidationResponse> {
    return apiClient.get<ValidationResponse>(`${API_ENDPOINTS.MARKET_DATA.VALIDATE}/${encodeURIComponent(symbol)}`)
  },
}

// Health check API
export const healthAPI = {
  /**
   * Check if the backend is healthy
   */
  async checkHealth(): Promise<{ status: string; timestamp: string; version?: string }> {
    return apiClient.get(API_ENDPOINTS.HEALTH)
  },
}

// Analysis API
export const analysisAPI = {
  /**
   * Run technical analysis on stock data
   */
  async runTechnicalAnalysis(request: TechnicalAnalysisRequest): Promise<TechnicalAnalysisResponse> {
    return apiClient.post<TechnicalAnalysisResponse>(API_ENDPOINTS.ANALYSIS.TECHNICAL, request)
  },

  /**
   * Detect candlestick patterns
   */
  async detectPatterns(request: PatternDetectionRequest): Promise<PatternDetectionResponse> {
    return apiClient.post<PatternDetectionResponse>(API_ENDPOINTS.ANALYSIS.PATTERNS, request)
  },
}

// Utility functions
export const utils = {
  /**
   * Format currency values
   */
  formatCurrency(value: number): string {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
    }).format(value)
  },

  /**
   * Format percentage values
   */
  formatPercentage(value: number): string {
    return `${value >= 0 ? '+' : ''}${value.toFixed(2)}%`
  },

  /**
   * Format large numbers (market cap, volume)
   */
  formatLargeNumber(value: number): string {
    if (value >= 1e12) {
      return `${(value / 1e12).toFixed(1)}T`
    } else if (value >= 1e9) {
      return `${(value / 1e9).toFixed(1)}B`
    } else if (value >= 1e6) {
      return `${(value / 1e6).toFixed(1)}M`
    } else if (value >= 1e3) {
      return `${(value / 1e3).toFixed(1)}K`
    }
    return value.toString()
  },

  /**
   * Get period display name
   */
  getPeriodDisplayName(period: string): string {
    const periodMap: Record<string, string> = {
      '1d': '1 Day',
      '5d': '5 Days',
      '1mo': '1 Month',
      '3mo': '3 Months',
      '6mo': '6 Months',
      '1y': '1 Year',
      '2y': '2 Years',
      '5y': '5 Years',
      '10y': '10 Years',
      'ytd': 'Year to Date',
      'max': 'All Time',
    }
    return periodMap[period] || period
  },

  /**
   * Get interval display name
   */
  getIntervalDisplayName(interval: string): string {
    const intervalMap: Record<string, string> = {
      '1m': '1 Minute',
      '2m': '2 Minutes',
      '5m': '5 Minutes',
      '15m': '15 Minutes',
      '30m': '30 Minutes',
      '60m': '1 Hour',
      '90m': '90 Minutes',
      '1h': '1 Hour',
      '1d': '1 Day',
      '5d': '5 Days',
      '1wk': '1 Week',
      '1mo': '1 Month',
      '3mo': '3 Months',
    }
    return intervalMap[interval] || interval
  },
}

// Legacy API functions (keep for compatibility, but these won't work with current backend)
export const api = {
  // Market Data - Updated to use actual backend
  downloadStockData: marketDataAPI.downloadStockData,
  loadCSVData: marketDataAPI.loadCSVData,
  listCSVFiles: marketDataAPI.listCSVFiles,
  getStockInfo: marketDataAPI.getStockInfo,
  validateSymbol: marketDataAPI.validateSymbol,
  
  // Health
  checkHealth: healthAPI.checkHealth,
  
  // Analysis
  runTechnicalAnalysis: analysisAPI.runTechnicalAnalysis,
  detectPatterns: analysisAPI.detectPatterns,
}
