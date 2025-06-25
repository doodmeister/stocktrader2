'use client'

import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ReferenceLine } from 'recharts'
import type { TechnicalIndicatorData } from '@/lib/api'

interface TechnicalIndicatorChartProps {
  indicator: TechnicalIndicatorData
  indicatorName: string
  height?: number
  className?: string
}

export default function TechnicalIndicatorChart({ 
  indicator, 
  indicatorName, 
  height = 200, 
  className = '' 
}: TechnicalIndicatorChartProps) {
  
  // Prepare data for chart
  const chartData = indicator.values?.map((value, index) => ({
    index: index + 1,
    value: typeof value === 'number' && !isNaN(value) ? value : null,
    name: `Point ${index + 1}`
  })).filter(point => point.value !== null && point.value !== undefined) || []

  // Get indicator-specific configuration
  const getIndicatorConfig = (name: string) => {
    switch (name.toLowerCase()) {
      case 'rsi':
        return {
          color: '#8884d8',
          referenceLines: [
            { value: 70, color: '#ef4444', label: 'Overbought (70)' },
            { value: 30, color: '#22c55e', label: 'Oversold (30)' }
          ],
          yDomain: [0, 100],
          formatValue: (value: number) => `${value.toFixed(1)}%`
        }
      case 'sma':
      case 'ema':
        return {
          color: name === 'sma' ? '#f59e0b' : '#06b6d4',
          referenceLines: [],
          yDomain: ['dataMin - 5', 'dataMax + 5'],
          formatValue: (value: number) => `$${value.toFixed(2)}`
        }
      case 'macd':
        return {
          color: '#8b5cf6',
          referenceLines: [{ value: 0, color: '#6b7280', label: 'Zero Line' }],
          yDomain: ['dataMin - 0.1', 'dataMax + 0.1'],
          formatValue: (value: number) => value.toFixed(3)
        }
      case 'bollinger_bands':
        return {
          color: '#ec4899',
          referenceLines: [],
          yDomain: ['dataMin - 10', 'dataMax + 10'],
          formatValue: (value: number) => `$${value.toFixed(2)}`
        }
      default:
        return {
          color: '#64748b',
          referenceLines: [],
          yDomain: ['auto', 'auto'],
          formatValue: (value: number) => value.toFixed(2)
        }
    }
  }

  const config = getIndicatorConfig(indicatorName)

  if (!chartData.length) {
    return (
      <div className={`flex flex-col items-center justify-center bg-background/50 rounded-md border ${className}`} style={{ height }}>
        <p className="text-muted-foreground text-sm mb-2">No chart data available for {indicator.name}</p>
        <p className="text-xs text-muted-foreground">
          {indicator.values && indicator.values.length > 0 
            ? `${indicator.values.length} data points, but all values are null/invalid`
            : 'No data values provided'}
        </p>
      </div>
    )
  }

  return (
    <div className={`bg-background/50 rounded-md border p-3 ${className}`}>
      <div className="flex justify-between items-center mb-2">
        <h4 className="font-medium text-sm">{indicator.name}</h4>
        <div className="flex items-center gap-2">
          <span className={`text-xs px-2 py-1 rounded ${
            indicator.signal === 'Buy' ? 'text-bullish bg-bullish/10' :
            indicator.signal === 'Sell' ? 'text-bearish bg-bearish/10' :
            'text-muted-foreground bg-muted/50'
          }`}>
            {indicator.signal || 'Hold'}
          </span>
          {indicator.current_value !== undefined && (
            <span className="text-xs text-muted-foreground">
              Current: {config.formatValue(indicator.current_value)}
            </span>
          )}
        </div>
      </div>
      
      <ResponsiveContainer width="100%" height={height}>
        <LineChart data={chartData} margin={{ top: 5, right: 10, left: 10, bottom: 5 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
          <XAxis 
            dataKey="index"
            tick={{ fontSize: 10, fill: 'hsl(var(--muted-foreground))' }}
            axisLine={{ stroke: 'hsl(var(--border))' }}
          />
          <YAxis 
            domain={config.yDomain}
            tick={{ fontSize: 10, fill: 'hsl(var(--muted-foreground))' }}
            axisLine={{ stroke: 'hsl(var(--border))' }}
          />
          <Tooltip
            contentStyle={{
              backgroundColor: 'hsl(var(--background))',
              border: '1px solid hsl(var(--border))',
              borderRadius: '6px',
              fontSize: '12px'
            }}
            labelFormatter={(label) => `Point ${label}`}
            formatter={(value: number) => [config.formatValue(value), indicator.name]}
          />
          
          {/* Reference lines for indicators like RSI */}
          {config.referenceLines.map((line, index) => (
            <ReferenceLine
              key={index}
              y={line.value}
              stroke={line.color}
              strokeDasharray="2 2"
              strokeWidth={1}
            />
          ))}
          
          <Line
            type="monotone"
            dataKey="value"
            stroke={config.color}
            strokeWidth={2}
            dot={false}
            activeDot={{ r: 4, stroke: config.color, strokeWidth: 2 }}
          />
        </LineChart>
      </ResponsiveContainer>

      {/* Show metadata */}
      {indicator.metadata && (
        <div className="mt-2 text-xs text-muted-foreground flex flex-wrap gap-3">
          {Object.entries(indicator.metadata).map(([key, value]) => {
            if (['period', 'fast', 'slow', 'signal_period'].includes(key)) {
              return (
                <span key={key}>
                  <span className="font-medium capitalize">{key.replace('_', ' ')}:</span> {value}
                </span>
              )
            }
            if (['overbought', 'oversold'].includes(key)) {
              return (
                <span key={key}>
                  <span className="font-medium capitalize">{key}:</span> {value}
                </span>
              )
            }
            return null
          })}
        </div>
      )}
    </div>
  )
}
