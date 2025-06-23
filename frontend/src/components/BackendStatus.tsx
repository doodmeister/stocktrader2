'use client'

import { useState, useEffect } from 'react'
import { healthAPI, APIError } from '@/lib/api'

interface BackendStatusProps {
  className?: string
  onStatusChange?: (isHealthy: boolean) => void
}

export default function BackendStatus({ className = '', onStatusChange }: BackendStatusProps) {
  const [status, setStatus] = useState<'checking' | 'healthy' | 'unhealthy'>('checking')
  const [lastCheck, setLastCheck] = useState<Date | null>(null)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    checkHealth()
    
    // Check health every 30 seconds
    const interval = setInterval(checkHealth, 30000)
    return () => clearInterval(interval)
  }, [])

  const checkHealth = async () => {
    try {
      await healthAPI.checkHealth()
      setStatus('healthy')
      setError(null)
      onStatusChange?.(true)
    } catch (err) {
      console.error('Health check failed:', err)
      setStatus('unhealthy')
      setError(err instanceof APIError ? err.message : 'Backend unreachable')
      onStatusChange?.(false)
    } finally {
      setLastCheck(new Date())
    }
  }

  const getStatusColor = () => {
    switch (status) {
      case 'checking':
        return 'text-yellow-500'
      case 'healthy':
        return 'text-bullish'
      case 'unhealthy':
        return 'text-bearish'
      default:
        return 'text-muted-foreground'
    }
  }

  const getStatusIcon = () => {
    switch (status) {
      case 'checking':
        return <div className="animate-spin rounded-full h-2 w-2 border border-current border-t-transparent"></div>
      case 'healthy':
        return '●'
      case 'unhealthy':
        return '●'
      default:
        return '○'
    }
  }

  const getStatusText = () => {
    switch (status) {
      case 'checking':
        return 'Checking...'
      case 'healthy':
        return 'Connected'
      case 'unhealthy':
        return 'Disconnected'
      default:
        return 'Unknown'
    }
  }

  return (
    <div className={`flex items-center gap-2 text-sm ${className}`}>
      <span className={`flex items-center ${getStatusColor()}`}>
        {getStatusIcon()}
      </span>
      <span className={getStatusColor()}>
        Backend: {getStatusText()}
      </span>
      {lastCheck && (
        <span className="text-xs text-muted-foreground">
          ({lastCheck.toLocaleTimeString()})
        </span>
      )}
      {error && status === 'unhealthy' && (
        <span className="text-xs text-bearish" title={error}>
          ({error.split(':')[0]})
        </span>
      )}
    </div>
  )
}
