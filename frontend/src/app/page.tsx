'use client'

import { useState, useEffect } from 'react'
import MarketDataDownload from '@/components/MarketDataDownload'
import CSVFileManager from '@/components/CSVFileManager'
import BackendStatus from '@/components/BackendStatus'
import TechnicalAnalysisWithCharts from '@/components/TechnicalAnalysisWithCharts'
import PriceChart from '@/components/charts/PriceChart'
import type { MarketDataResponse, LoadCSVResponse, TechnicalAnalysisResponse } from '@/lib/api'

export default function Home() {
  const [mounted, setMounted] = useState(false)
  const [isBackendHealthy, setIsBackendHealthy] = useState(false)
  const [currentData, setCurrentData] = useState<MarketDataResponse | LoadCSVResponse | null>(null)
  const [currentSymbol, setCurrentSymbol] = useState<string | null>(null)
  const [showAnalysis, setShowAnalysis] = useState(false)
  const [showCharts, setShowCharts] = useState(false)

  useEffect(() => {
    setMounted(true)
  }, [])

  if (!mounted) {
    return null
  }
  const handleDataDownloaded = (response: MarketDataResponse) => {
    setCurrentData(response)
    // Extract symbol from the first symbol in the array
    const symbol = response.symbols && response.symbols.length > 0 ? response.symbols[0] : null
    setCurrentSymbol(symbol)
    setShowAnalysis(false) // Reset analysis view when new data is loaded
    setShowCharts(false) // Reset chart view when new data is loaded
  }

  const handleFileLoaded = (response: LoadCSVResponse) => {
    setCurrentData(response)
    setCurrentSymbol(response.symbol)
    setShowAnalysis(false) // Reset analysis view when new data is loaded
    setShowCharts(false) // Reset chart view when new data is loaded
  }

  const handleRunAnalysis = () => {
    if (currentSymbol) {
      setShowAnalysis(true)
      setShowCharts(false) // Hide charts when showing analysis
    }
  }

  const handleViewCharts = () => {
    if (currentSymbol) {
      setShowCharts(true)
      setShowAnalysis(false) // Hide analysis when showing charts
    }
  }

  return (
    <main className="min-h-screen bg-background">
      {/* Header */}
      <header className="border-b border-border bg-card/50 backdrop-blur supports-[backdrop-filter]:bg-card/60">
        <div className="container mx-auto px-4 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-4">
              <h1 className="text-2xl font-bold text-foreground">
                StockTrader Pro
              </h1>
              <span className="text-sm text-muted-foreground">
                Market Data & Analysis Platform
              </span>
            </div>
            <div className="flex items-center space-x-4">
              <BackendStatus onStatusChange={setIsBackendHealthy} />
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <div className="container mx-auto px-4 py-6">
        {!isBackendHealthy && (
          <div className="mb-6 p-4 bg-yellow-50 border border-yellow-200 rounded-lg">
            <div className="flex items-center gap-2 text-yellow-800">
              <span>⚠️</span>
              <span className="font-medium">Backend Connection Issue</span>
            </div>
            <p className="text-sm text-yellow-700 mt-1">
              Make sure the FastAPI backend is running on http://localhost:8000. 
              You can start it by running: <code className="bg-yellow-100 px-1 rounded">python -m uvicorn api.main:app --reload</code>
            </p>
          </div>
        )}

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Market Data Tools */}
          <div className="lg:col-span-2 space-y-6">
            {/* Market Data Download */}
            <MarketDataDownload 
              onDownloadComplete={handleDataDownloaded}
              className="w-full"
            />            {/* Current Data Display */}
            {currentData && (
              <div className="trading-card">
                {'symbol' in currentData ? (
                  // LoadCSVResponse display
                  <>
                    <h2 className="text-lg font-semibold mb-4">Current Data: {currentData.symbol}</h2>
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                      <div className="space-y-2">
                        <div className="text-sm">
                          <span className="text-muted-foreground">Records:</span>
                          <span className="ml-2 font-medium">{currentData.total_records?.toLocaleString() || 'N/A'}</span>
                        </div>
                        <div className="text-sm">
                          <span className="text-muted-foreground">Period:</span>
                          <span className="ml-2 font-medium">{currentData.start_date} to {currentData.end_date}</span>
                        </div>
                        <div className="text-sm">
                          <span className="text-muted-foreground">File:</span>
                          <span className="ml-2 font-medium text-xs">{currentData.file_path}</span>
                        </div>
                      </div>
                      <div className="space-y-2">
                        <div className="text-sm">
                          <span className="text-muted-foreground">Price Change:</span>
                          <span className={`ml-2 font-medium ${currentData.data_summary.price_change_percent >= 0 ? 'text-bullish' : 'text-bearish'}`}>
                            {currentData.data_summary.price_change_percent >= 0 ? '+' : ''}
                            {currentData.data_summary.price_change_percent.toFixed(2)}%
                          </span>
                        </div>
                        <div className="text-sm">
                          <span className="text-muted-foreground">Price Range:</span>
                          <span className="ml-2 font-medium">
                            ${currentData.data_summary.first_price.toFixed(2)} → ${currentData.data_summary.last_price.toFixed(2)}
                          </span>
                        </div>
                        <div className="text-sm">
                          <span className="text-muted-foreground">Avg Volume:</span>
                          <span className="ml-2 font-medium">
                            {(currentData.data_summary.volume_avg / 1000000).toFixed(1)}M
                          </span>
                        </div>
                      </div>
                    </div>
                  </>
                ) : (
                  // MarketDataResponse display
                  <>
                    <h2 className="text-lg font-semibold mb-4">Market Data Downloaded</h2>
                    <div className="space-y-4">
                      <div className="text-sm">
                        <span className="text-muted-foreground">Status:</span>
                        <span className="ml-2 font-medium">{currentData.status}</span>
                      </div>
                      <div className="text-sm">
                        <span className="text-muted-foreground">Message:</span>
                        <span className="ml-2 font-medium">{currentData.message}</span>
                      </div>
                      <div className="text-sm">
                        <span className="text-muted-foreground">Symbols:</span>
                        <span className="ml-2 font-medium">{currentData.symbols.join(', ')}</span>
                      </div>
                      {Object.entries(currentData.data_info).map(([symbol, info]) => (
                        <div key={symbol} className="border-l-2 border-primary pl-4">
                          <div className="font-medium text-sm mb-2">{symbol}</div>
                          <div className="text-xs space-y-1 text-muted-foreground">
                            <div>Rows: {info.rows}</div>
                            <div>Date Range: {info.date_range.start} to {info.date_range.end}</div>
                            {info.latest_close && (
                              <div>Latest Close: ${info.latest_close.toFixed(2)}</div>
                            )}
                          </div>
                        </div>
                      ))}
                    </div>
                  </>
                )}
              </div>
            )}

            {/* Analysis Actions */}
            <div className="trading-card">
              <h2 className="text-lg font-semibold mb-4">Analysis Tools</h2>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">                <button 
                  onClick={handleRunAnalysis}
                  className="trading-button bg-primary text-primary-foreground hover:bg-primary/90 disabled:opacity-50"
                  disabled={!currentData || !isBackendHealthy || !currentSymbol}
                >
                  Run Technical Analysis
                </button>
                <button 
                  className="trading-button bg-secondary text-secondary-foreground hover:bg-secondary/90 disabled:opacity-50"
                  disabled={!currentData || !isBackendHealthy}
                >
                  Detect Patterns
                </button>
                <button 
                  className="trading-button bg-accent text-accent-foreground hover:bg-accent/90 disabled:opacity-50"
                  disabled={!currentData || !isBackendHealthy}
                >
                  Generate AI Report
                </button>
                <button 
                  onClick={handleViewCharts}
                  className="trading-button bg-blue-600 text-white hover:bg-blue-700 disabled:opacity-50"
                  disabled={!currentData || !isBackendHealthy}
                >
                  📊 View Charts
                </button>
              </div>
              {!currentData && (
                <p className="text-sm text-muted-foreground mt-2">
                  Load or download market data to enable analysis tools
                </p>
              )}            </div>
          </div>

          {/* Technical Analysis Section */}
          {showAnalysis && currentSymbol && (
            <div className="col-span-1 lg:col-span-2">
              <TechnicalAnalysisWithCharts 
                symbol={currentSymbol}
                onAnalysisComplete={(result: TechnicalAnalysisResponse) => {
                  console.log('Technical analysis completed:', result)
                }}
              />
            </div>
          )}

          {/* Price Chart Section */}
          {showCharts && currentSymbol && currentData && (
            <div className="col-span-1 lg:col-span-2">
              <div className="trading-card">
                <div className="flex justify-between items-center mb-4">
                  <h2 className="text-lg font-semibold">Price Chart - {currentSymbol}</h2>
                  <button
                    onClick={() => setShowCharts(false)}
                    className="px-3 py-1 text-xs rounded-md border bg-background hover:bg-accent transition-colors"
                  >
                    Close Chart
                  </button>
                </div>
                
                <div className="text-center py-8 text-muted-foreground">
                  <p>📈 Interactive price charts coming soon!</p>
                  <p className="text-xs mt-2">Chart integration with market data in development.</p>
                  <p className="text-xs mt-1">
                    Current data: {currentSymbol} ({('symbol' in currentData) ? 'CSV file' : 'Downloaded data'})
                  </p>
                </div>
              </div>
            </div>
          )}

          {/* Sidebar */}
          <div className="space-y-6">
            {/* CSV File Manager */}
            <CSVFileManager 
              onFileLoaded={handleFileLoaded}
              className="w-full"
            />

            {/* Market Overview */}
            <div className="trading-card">
              <h2 className="text-lg font-semibold mb-4">Market Overview</h2>
              <div className="space-y-3">
                <div className="p-3 bg-muted/30 rounded-lg">
                  <div className="text-sm text-muted-foreground">S&P 500</div>
                  <div className="text-lg font-bold">4,567.89</div>
                  <div className="text-sm price-up">+1.23 (+0.27%)</div>
                </div>
                <div className="p-3 bg-muted/30 rounded-lg">
                  <div className="text-sm text-muted-foreground">NASDAQ</div>
                  <div className="text-lg font-bold">14,234.56</div>
                  <div className="text-sm price-down">-23.45 (-0.16%)</div>
                </div>
                <div className="p-3 bg-muted/30 rounded-lg">
                  <div className="text-sm text-muted-foreground">DOW</div>
                  <div className="text-lg font-bold">34,123.78</div>
                  <div className="text-sm price-up">+45.67 (+0.13%)</div>
                </div>
              </div>
            </div>            {/* Quick Stats */}
            {currentData && 'symbol' in currentData && (
              <div className="trading-card">
                <h2 className="text-lg font-semibold mb-4">Quick Stats</h2>
                <div className="space-y-2 text-sm">
                  <div className="flex justify-between">
                    <span className="text-muted-foreground">Symbol:</span>
                    <span className="font-medium">{currentData.symbol}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-muted-foreground">Data Points:</span>
                    <span className="font-medium">{currentData.total_records}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-muted-foreground">Date Range:</span>
                    <span className="font-medium text-xs">
                      {new Date(currentData.start_date).toLocaleDateString()} - {new Date(currentData.end_date).toLocaleDateString()}
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-muted-foreground">Performance:</span>
                    <span className={`font-medium ${currentData.data_summary.price_change_percent >= 0 ? 'text-bullish' : 'text-bearish'}`}>
                      {currentData.data_summary.price_change_percent >= 0 ? '+' : ''}{currentData.data_summary.price_change_percent.toFixed(2)}%
                    </span>
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </main>
  )
}
