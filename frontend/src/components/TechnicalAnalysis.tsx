'use client'

import { useState } from 'react'
import { analysisAPI, utils, APIError } from '@/lib/api'
import type { TechnicalAnalysisRequest, TechnicalAnalysisResponse, TechnicalIndicatorData } from '@/lib/api'

interface TechnicalAnalysisProps {
  symbol: string
  className?: string
  onAnalysisComplete?: (response: TechnicalAnalysisResponse) => void
}

export default function TechnicalAnalysis({ symbol, className = '', onAnalysisComplete }: TechnicalAnalysisProps) {
  const [isAnalyzing, setIsAnalyzing] = useState(false)
  const [analysisResult, setAnalysisResult] = useState<TechnicalAnalysisResponse | null>(null)
  const [error, setError] = useState<string | null>(null)
  
  const handleRunAnalysis = async () => {
    if (!symbol) {
      setError('No symbol provided for analysis')
      return
    }

    setIsAnalyzing(true)
    setError(null)
    setAnalysisResult(null)

    try {
      const request: TechnicalAnalysisRequest = {
        symbol: symbol.trim().toUpperCase(),
        indicators: ['rsi', 'macd', 'bollinger_bands', 'sma', 'ema'],
        rsi_period: 14,
        macd_fast: 12,
        macd_slow: 26,
        macd_signal: 9,
        bb_period: 20,
        bb_std: 2.0,
        sma_period: 20,
        ema_period: 20
      }
      
      const response = await analysisAPI.runTechnicalAnalysis(request)
      setAnalysisResult(response)
      onAnalysisComplete?.(response)
    } catch (err) {
      console.error('Technical analysis failed:', err)
      setError(err instanceof APIError ? err.message : 'Analysis failed')
    } finally {
      setIsAnalyzing(false)
    }
  }

  const getSignalColor = (signal?: string) => {
    switch (signal?.toLowerCase()) {
      case 'buy':
        return 'text-bullish'
      case 'sell':
        return 'text-bearish'
      default:
        return 'text-muted-foreground'
    }
  }

  const getSignalStrengthColor = (strength?: string) => {
    switch (strength?.toLowerCase()) {
      case 'strong':
        return 'text-bullish font-semibold'
      case 'medium':
        return 'text-secondary-foreground font-medium'
      case 'weak':
        return 'text-muted-foreground'
      default:
        return 'text-foreground'
    }
  }

  const formatCompositeSignal = (signal?: number) => {
    if (signal === undefined || signal === null) return 'N/A'
    
    if (signal > 0.6) return 'Strong Bullish'
    if (signal > 0.3) return 'Bullish'
    if (signal > 0.1) return 'Weak Bullish'
    if (signal < -0.6) return 'Strong Bearish'
    if (signal < -0.3) return 'Bearish'
    if (signal < -0.1) return 'Weak Bearish'
    return 'Neutral'
  }

  const renderIndicator = (name: string, indicator: TechnicalIndicatorData) => (
    <div key={name} className="p-3 bg-background/50 rounded-md border">
      <div className="flex justify-between items-start mb-2">
        <h4 className="font-medium text-sm">{indicator.name}</h4>
        <span className={`text-xs px-2 py-1 rounded ${getSignalColor(indicator.signal)} bg-current/10`}>
          {indicator.signal || 'Hold'}
        </span>
      </div>
      
      <div className="space-y-1 text-xs text-muted-foreground">
        {indicator.current_value !== undefined && (
          <div>
            <span className="font-medium">Current:</span> {
              typeof indicator.current_value === 'number' 
                ? indicator.current_value.toFixed(2)
                : 'N/A'
            }
          </div>
        )}
        
        {indicator.metadata && Object.entries(indicator.metadata).map(([key, value]) => {
          if (key === 'period' || key === 'fast' || key === 'slow') {
            return (
              <div key={key}>
                <span className="font-medium capitalize">{key}:</span> {value}
              </div>
            )
          }
          if (key === 'overbought' || key === 'oversold') {
            return (
              <div key={key}>
                <span className="font-medium capitalize">{key}:</span> {value}
              </div>
            )
          }
          if (key === 'upper_band' || key === 'lower_band' && typeof value === 'number') {
            return (
              <div key={key}>
                <span className="font-medium">{key.replace('_', ' ')}:</span> {value.toFixed(2)}
              </div>
            )
          }
          return null
        })}
      </div>
    </div>
  )

  return (
    <div className={`trading-card ${className}`}>
      <div className="flex justify-between items-center mb-4">
        <h2 className="text-lg font-semibold">Technical Analysis</h2>
        <button
          onClick={handleRunAnalysis}
          disabled={!symbol || isAnalyzing}
          className={`px-4 py-2 rounded-md font-medium transition-colors ${
            symbol && !isAnalyzing
              ? 'bg-primary text-primary-foreground hover:bg-primary/90'
              : 'bg-muted text-muted-foreground cursor-not-allowed'
          }`}
        >
          {isAnalyzing ? (
            <span className="flex items-center gap-2">
              <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-current"></div>
              Analyzing...
            </span>
          ) : (
            'Run Analysis'
          )}
        </button>
      </div>

      {/* Error Display */}
      {error && (
        <div className="p-3 bg-bearish/10 border border-bearish/20 rounded-md text-bearish text-sm mb-4">
          {error}
        </div>
      )}

      {/* Analysis Results */}
      {analysisResult && (
        <div className="space-y-4">
          {/* Overall Signal */}
          <div className="p-4 bg-accent/10 border border-accent/20 rounded-md">
            <h3 className="font-semibold mb-2">Overall Analysis</h3>
            <div className="grid grid-cols-2 gap-4 text-sm">
              <div>
                <span className="text-muted-foreground">Symbol:</span> 
                <span className="font-medium ml-2">{analysisResult.symbol}</span>
              </div>
              <div>
                <span className="text-muted-foreground">Current Price:</span> 
                <span className="font-medium ml-2">
                  {analysisResult.current_price ? utils.formatCurrency(analysisResult.current_price) : 'N/A'}
                </span>
              </div>
              <div>
                <span className="text-muted-foreground">Composite Signal:</span> 
                <span className={`font-medium ml-2 ${getSignalStrengthColor(analysisResult.signal_strength)}`}>
                  {formatCompositeSignal(analysisResult.composite_signal)}
                </span>
              </div>
              <div>
                <span className="text-muted-foreground">Signal Strength:</span> 
                <span className={`font-medium ml-2 ${getSignalStrengthColor(analysisResult.signal_strength)}`}>
                  {analysisResult.signal_strength || 'N/A'}
                </span>
              </div>
              {analysisResult.price_change !== undefined && (
                <div>
                  <span className="text-muted-foreground">Price Change:</span> 
                  <span className={`font-medium ml-2 ${analysisResult.price_change >= 0 ? 'text-bullish' : 'text-bearish'}`}>
                    {utils.formatPercentage(analysisResult.price_change_percent || 0)}
                  </span>
                </div>
              )}
            </div>
          </div>

          {/* Individual Indicators */}
          <div>
            <h3 className="font-semibold mb-3">Technical Indicators</h3>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
              {Object.entries(analysisResult.indicators).map(([name, indicator]) =>
                renderIndicator(name, indicator)
              )}
            </div>
          </div>

          {/* Data Info */}
          <div className="text-xs text-muted-foreground">
            <div>Analysis Time: {new Date(analysisResult.analysis_time).toLocaleString()}</div>
            <div>Data Points: {analysisResult.data_info.rows?.toLocaleString() || 'N/A'}</div>
            {analysisResult.data_info.date_range && (
              <div>
                Date Range: {analysisResult.data_info.date_range.start} to {analysisResult.data_info.date_range.end}
              </div>
            )}
          </div>
        </div>
      )}

      {/* Initial State */}
      {!analysisResult && !error && !isAnalyzing && (
        <div className="text-center py-8 text-muted-foreground">
          <p>Click "Run Analysis" to perform technical analysis on {symbol || 'the selected symbol'}</p>
          <p className="text-xs mt-2">Analysis includes RSI, MACD, Bollinger Bands, SMA, and EMA</p>
        </div>
      )}
    </div>
  )
}
