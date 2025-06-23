'use client'

import { useState, useEffect } from 'react'
import { marketDataAPI, utils, APIError } from '@/lib/api'
import type { MarketDataRequest, MarketDataResponse, ValidationResponse } from '@/lib/api'

interface MarketDataDownloadProps {
  onDownloadComplete?: (response: MarketDataResponse) => void
  className?: string
}

export default function MarketDataDownload({ onDownloadComplete, className = '' }: MarketDataDownloadProps) {
  const [formData, setFormData] = useState<MarketDataRequest>({
    symbol: '',
    period: '1y',
    interval: '1d',
    save_csv: true,
  })
  
  const [validation, setValidation] = useState<ValidationResponse | null>(null)
  const [isValidating, setIsValidating] = useState(false)
  const [isDownloading, setIsDownloading] = useState(false)
  const [downloadResult, setDownloadResult] = useState<MarketDataResponse | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [validationTimer, setValidationTimer] = useState<NodeJS.Timeout | null>(null)

  // Available options
  const periods = [
    { value: '1d', label: '1 Day' },
    { value: '5d', label: '5 Days' },
    { value: '1mo', label: '1 Month' },
    { value: '3mo', label: '3 Months' },
    { value: '6mo', label: '6 Months' },
    { value: '1y', label: '1 Year' },
    { value: '2y', label: '2 Years' },
    { value: '5y', label: '5 Years' },
    { value: '10y', label: '10 Years' },
    { value: 'ytd', label: 'Year to Date' },
    { value: 'max', label: 'All Time' },
  ]

  const intervals = [
    { value: '1m', label: '1 Minute' },
    { value: '2m', label: '2 Minutes' },
    { value: '5m', label: '5 Minutes' },
    { value: '15m', label: '15 Minutes' },
    { value: '30m', label: '30 Minutes' },
    { value: '60m', label: '1 Hour' },
    { value: '90m', label: '90 Minutes' },
    { value: '1h', label: '1 Hour' },
    { value: '1d', label: '1 Day' },
    { value: '5d', label: '5 Days' },
    { value: '1wk', label: '1 Week' },
    { value: '1mo', label: '1 Month' },
    { value: '3mo', label: '3 Months' },
  ]

  // Debounced symbol validation
  useEffect(() => {
    if (validationTimer) {
      clearTimeout(validationTimer)
    }

    if (formData.symbol.trim().length > 0) {
      setValidationTimer(setTimeout(async () => {
        await validateSymbol(formData.symbol.trim())
      }, 500))
    } else {
      setValidation(null)
    }

    return () => {
      if (validationTimer) {
        clearTimeout(validationTimer)
      }
    }
  }, [formData.symbol])

  const validateSymbol = async (symbol: string) => {
    if (!symbol || symbol.length === 0) return

    setIsValidating(true)
    setError(null)

    try {
      const result = await marketDataAPI.validateSymbol(symbol)
      setValidation(result)
    } catch (err) {
      console.error('Symbol validation failed:', err)
      setValidation({
        symbol,
        is_valid: false,
        error: err instanceof APIError ? err.message : 'Validation failed'
      })
    } finally {
      setIsValidating(false)
    }
  }

  const handleInputChange = (field: keyof MarketDataRequest, value: string | boolean) => {
    setFormData(prev => ({
      ...prev,
      [field]: value
    }))
    
    // Clear previous results when form changes
    if (field !== 'symbol') {
      setDownloadResult(null)
      setError(null)
    }
  }

  const handleDownload = async () => {
    if (!formData.symbol.trim()) {
      setError('Please enter a stock symbol')
      return
    }

    if (validation && !validation.is_valid) {
      setError('Please enter a valid stock symbol')
      return
    }

    setIsDownloading(true)
    setError(null)
    setDownloadResult(null)

    try {
      const response = await marketDataAPI.downloadStockData({
        ...formData,
        symbol: formData.symbol.trim().toUpperCase()
      })
      
      setDownloadResult(response)
      onDownloadComplete?.(response)
    } catch (err) {
      console.error('Download failed:', err)
      setError(err instanceof APIError ? err.message : 'Download failed')
    } finally {
      setIsDownloading(false)
    }
  }

  const isFormValid = formData.symbol.trim().length > 0 && validation?.is_valid

  return (
    <div className={`trading-card ${className}`}>
      <h2 className="text-lg font-semibold mb-4">Download Market Data</h2>
      
      <div className="space-y-4">
        {/* Symbol Input */}
        <div>
          <label className="block text-sm font-medium mb-2">
            Stock Symbol
          </label>
          <div className="relative">
            <input
              type="text"
              value={formData.symbol}
              onChange={(e) => handleInputChange('symbol', e.target.value.toUpperCase())}
              placeholder="Enter symbol (e.g., AAPL, MSFT)"
              className="w-full px-3 py-2 border border-border rounded-md bg-background text-foreground placeholder-muted-foreground focus:outline-none focus:ring-2 focus:ring-primary focus:border-transparent"
            />
            {isValidating && (
              <div className="absolute right-3 top-1/2 transform -translate-y-1/2">
                <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-primary"></div>
              </div>
            )}
          </div>
          
          {/* Validation Status */}
          {validation && (
            <div className="mt-2 text-sm">
              {validation.is_valid ? (
                <div className="text-bullish flex items-center gap-2">
                  <span>✓</span>
                  <span>Valid symbol: {validation.info?.company_name || validation.symbol}</span>
                  {validation.info?.current_price && (
                    <span className="text-muted-foreground">
                      ({utils.formatCurrency(validation.info.current_price)})
                    </span>
                  )}
                </div>
              ) : (
                <div className="text-bearish flex items-center gap-2">
                  <span>✗</span>
                  <span>Invalid symbol: {validation.error || 'Symbol not found'}</span>
                </div>
              )}
            </div>
          )}
        </div>

        {/* Period Selection */}
        <div>
          <label className="block text-sm font-medium mb-2">
            Time Period
          </label>
          <select
            value={formData.period}
            onChange={(e) => handleInputChange('period', e.target.value)}
            className="w-full px-3 py-2 border border-border rounded-md bg-background text-foreground focus:outline-none focus:ring-2 focus:ring-primary focus:border-transparent"
          >
            {periods.map(period => (
              <option key={period.value} value={period.value}>
                {period.label}
              </option>
            ))}
          </select>
        </div>

        {/* Interval Selection */}
        <div>
          <label className="block text-sm font-medium mb-2">
            Data Interval
          </label>
          <select
            value={formData.interval}
            onChange={(e) => handleInputChange('interval', e.target.value)}
            className="w-full px-3 py-2 border border-border rounded-md bg-background text-foreground focus:outline-none focus:ring-2 focus:ring-primary focus:border-transparent"
          >
            {intervals.map(interval => (
              <option key={interval.value} value={interval.value}>
                {interval.label}
              </option>
            ))}
          </select>
        </div>

        {/* Save CSV Option */}
        <div className="flex items-center gap-2">
          <input
            type="checkbox"
            id="save-csv"
            checked={formData.save_csv}
            onChange={(e) => handleInputChange('save_csv', e.target.checked)}
            className="rounded border-border text-primary focus:ring-primary focus:ring-offset-0"
          />
          <label htmlFor="save-csv" className="text-sm font-medium">
            Save data as CSV file
          </label>
        </div>

        {/* Download Button */}
        <button
          onClick={handleDownload}
          disabled={!isFormValid || isDownloading}
          className={`w-full py-2 px-4 rounded-md font-medium transition-colors ${
            isFormValid && !isDownloading
              ? 'bg-primary text-primary-foreground hover:bg-primary/90'
              : 'bg-muted text-muted-foreground cursor-not-allowed'
          }`}
        >
          {isDownloading ? (
            <span className="flex items-center justify-center gap-2">
              <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-current"></div>
              Downloading...
            </span>
          ) : (
            'Download Data'
          )}
        </button>

        {/* Error Display */}
        {error && (
          <div className="p-3 bg-bearish/10 border border-bearish/20 rounded-md text-bearish text-sm">
            {error}
          </div>
        )}

        {/* Success Display */}
        {downloadResult && (
          <div className="p-4 bg-bullish/10 border border-bullish/20 rounded-md">
            <h3 className="font-medium text-bullish mb-2">Download Complete!</h3>
            <div className="text-sm space-y-1">
              <div>Symbol: {downloadResult.symbol}</div>
              <div>Records: {downloadResult.total_records.toLocaleString()}</div>
              <div>Period: {downloadResult.start_date} to {downloadResult.end_date}</div>
              {downloadResult.csv_file_path && (
                <div>File: {downloadResult.csv_file_path}</div>
              )}
              <div className="mt-2 p-2 bg-background/50 rounded text-xs">
                <div>Price: {utils.formatCurrency(downloadResult.data_summary.first_price)} → {utils.formatCurrency(downloadResult.data_summary.last_price)}</div>
                <div>Change: {utils.formatPercentage(downloadResult.data_summary.price_change_percent)}</div>
                <div>Avg Volume: {utils.formatLargeNumber(downloadResult.data_summary.volume_avg)}</div>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}
