'use client'

import { useState, useEffect } from 'react'
import { marketDataAPI, utils, APIError } from '@/lib/api'
import type { MarketDataFile, LoadCSVResponse } from '@/lib/api'

interface CSVFileManagerProps {
  onFileLoaded?: (response: LoadCSVResponse) => void
  className?: string
}

export default function CSVFileManager({ onFileLoaded, className = '' }: CSVFileManagerProps) {
  const [files, setFiles] = useState<MarketDataFile[]>([])
  const [isLoading, setIsLoading] = useState(true)
  const [isLoadingFile, setIsLoadingFile] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [loadResult, setLoadResult] = useState<LoadCSVResponse | null>(null)

  useEffect(() => {
    loadFileList()
  }, [])
  const loadFileList = async () => {
    setIsLoading(true)
    setError(null)

    try {
      const fileList = await marketDataAPI.listCSVFiles()
      setFiles(Array.isArray(fileList) ? fileList : [])
    } catch (err) {
      console.error('Failed to load file list:', err)
      setError(err instanceof APIError ? err.message : 'Failed to load files')
      setFiles([]) // Ensure files is always an array even on error
    } finally {
      setIsLoading(false)
    }
  }

  const handleLoadFile = async (filePath: string, symbol?: string) => {
    setIsLoadingFile(filePath)
    setError(null)
    setLoadResult(null)

    try {
      const response = await marketDataAPI.loadCSVData({
        file_path: filePath,
        symbol
      })
      
      setLoadResult(response)
      onFileLoaded?.(response)
    } catch (err) {
      console.error('Failed to load file:', err)
      setError(err instanceof APIError ? err.message : 'Failed to load file')
    } finally {
      setIsLoadingFile(null)
    }
  }

  const formatFileSize = (bytes: number): string => {
    if (bytes === 0) return '0 Bytes'
    const k = 1024
    const sizes = ['Bytes', 'KB', 'MB', 'GB']
    const i = Math.floor(Math.log(bytes) / Math.log(k))
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i]
  }

  const formatDate = (dateString: string): string => {
    try {
      return new Date(dateString).toLocaleDateString('en-US', {
        year: 'numeric',
        month: 'short',
        day: 'numeric',
        hour: '2-digit',
        minute: '2-digit'
      })
    } catch {
      return dateString
    }
  }

  if (isLoading) {
    return (
      <div className={`trading-card ${className}`}>
        <h2 className="text-lg font-semibold mb-4">CSV Files</h2>
        <div className="flex items-center justify-center py-8">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary"></div>
        </div>
      </div>
    )
  }

  return (
    <div className={`trading-card ${className}`}>
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-lg font-semibold">CSV Files</h2>
        <button
          onClick={loadFileList}
          className="text-sm text-primary hover:text-primary/80 transition-colors"
        >
          Refresh
        </button>
      </div>

      {error && (
        <div className="mb-4 p-3 bg-bearish/10 border border-bearish/20 rounded-md text-bearish text-sm">
          {error}
        </div>
      )}

      {loadResult && (
        <div className="mb-4 p-4 bg-bullish/10 border border-bullish/20 rounded-md">
          <h3 className="font-medium text-bullish mb-2">File Loaded Successfully!</h3>
          <div className="text-sm space-y-1">
            <div>Symbol: {loadResult.symbol}</div>
            <div>Records: {loadResult.total_records.toLocaleString()}</div>
            <div>Period: {loadResult.start_date} to {loadResult.end_date}</div>
            <div className="mt-2 p-2 bg-background/50 rounded text-xs">
              <div>Price: {utils.formatCurrency(loadResult.data_summary.first_price)} → {utils.formatCurrency(loadResult.data_summary.last_price)}</div>
              <div>Change: {utils.formatPercentage(loadResult.data_summary.price_change_percent)}</div>
              <div>Avg Volume: {utils.formatLargeNumber(loadResult.data_summary.volume_avg)}</div>
            </div>
          </div>
        </div>
      )}      {Array.isArray(files) && files.length === 0 ? (
        <div className="text-center py-8 text-muted-foreground">
          <div className="text-4xl mb-2">📁</div>
          <div className="text-sm">No CSV files found</div>
          <div className="text-xs mt-1">Download some market data to get started</div>
        </div>
      ) : (
        <div className="space-y-2">
          {Array.isArray(files) && files.map((file, index) => (
            <div
              key={index}
              className="p-3 border border-border rounded-lg hover:bg-muted/30 transition-colors"
            >
              <div className="flex items-center justify-between">
                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-2 mb-1">
                    <span className="font-medium text-sm">{file.symbol}</span>
                    <span className="text-xs text-muted-foreground bg-muted px-2 py-0.5 rounded">
                      {utils.getPeriodDisplayName(file.period)}
                    </span>
                    <span className="text-xs text-muted-foreground bg-muted px-2 py-0.5 rounded">
                      {utils.getIntervalDisplayName(file.interval)}
                    </span>
                  </div>
                  
                  <div className="text-xs text-muted-foreground space-y-1">
                    <div>Created: {formatDate(file.created_date)}</div>
                    <div>Size: {formatFileSize(file.file_size)}</div>
                    <div className="truncate" title={file.file_path}>
                      Path: {file.file_path}
                    </div>
                  </div>
                </div>
                
                <button
                  onClick={() => handleLoadFile(file.file_path, file.symbol)}
                  disabled={isLoadingFile === file.file_path}
                  className="ml-3 px-3 py-1 text-xs font-medium bg-primary text-primary-foreground rounded hover:bg-primary/90 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
                >
                  {isLoadingFile === file.file_path ? (
                    <span className="flex items-center gap-1">
                      <div className="animate-spin rounded-full h-3 w-3 border border-current border-t-transparent"></div>
                      Loading...
                    </span>
                  ) : (
                    'Load'
                  )}
                </button>
              </div>            </div>
          ))}
        </div>
      )}
    </div>
  )
}
