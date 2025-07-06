'use client'

import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ReferenceLine, Legend } from 'recharts'
import type { IndicatorResult } from '@/lib/api'

interface TechnicalIndicatorChartProps {
  indicator: IndicatorResult
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
  
  // Prepare data for chart based on the actual API response structure
  let chartData: any[] = []
  const isBollingerBands = indicator.name.toLowerCase().includes('bollinger')
  const isMACD = indicator.name.toLowerCase() === 'macd'
  const isStochastic = indicator.name.toLowerCase() === 'stochastic'
  const isADX = indicator.name.toLowerCase() === 'adx'

  // Transform the API data structure to chart format
  if (indicator.data && Array.isArray(indicator.data)) {
    chartData = indicator.data.map((item, index) => {
      // Handle different data structures
      if (typeof item === 'object' && item !== null) {
        // For indicators like Bollinger Bands that have multiple values
        if ('upper_band' in item && 'middle_band' in item && 'lower_band' in item) {
          return {
            index: index + 1,
            upper: typeof item.upper_band === 'number' ? item.upper_band : null,
            middle: typeof item.middle_band === 'number' ? item.middle_band : null,
            lower: typeof item.lower_band === 'number' ? item.lower_band : null,
            name: item.date || `Point ${index + 1}`
          }
        }
        // For indicators like MACD that have multiple components
        else if ('macd' in item && 'signal' in item && 'histogram' in item) {
          return {
            index: index + 1,
            macd: typeof item.macd === 'number' ? item.macd : null,
            signal: typeof item.signal === 'number' ? item.signal : null,
            histogram: typeof item.histogram === 'number' ? item.histogram : null,
            name: item.date || `Point ${index + 1}`
          }
        }
        // For indicators like Stochastic with k and d values
        else if ('k' in item && 'd' in item) {
          return {
            index: index + 1,
            k: typeof item.k === 'number' ? item.k : null,
            d: typeof item.d === 'number' ? item.d : null,
            name: item.date || `Point ${index + 1}`
          }
        }
        // For ADX with multiple components
        else if ('adx' in item && 'plus_di' in item && 'minus_di' in item) {
          return {
            index: index + 1,
            adx: typeof item.adx === 'number' ? item.adx : null,
            plus_di: typeof item.plus_di === 'number' ? item.plus_di : null,
            minus_di: typeof item.minus_di === 'number' ? item.minus_di : null,
            name: item.date || `Point ${index + 1}`
          }
        }
        // For simple indicators with a single value
        else if ('value' in item) {
          return {
            index: index + 1,
            value: typeof item.value === 'number' ? item.value : null,
            name: item.date || `Point ${index + 1}`
          }
        }
      }
      
      // Fallback for simple numeric values
      return {
        index: index + 1,
        value: typeof item === 'number' ? item : null,
        name: `Point ${index + 1}`
      }
    }).filter(point => {
      // For multi-value indicators like MACD, check if at least one value is valid
      if (isMACD) {
        return (typeof point.macd === 'number' && !isNaN(point.macd)) ||
               (typeof point.signal === 'number' && !isNaN(point.signal)) ||
               (typeof point.histogram === 'number' && !isNaN(point.histogram))
      }
      
      // For Bollinger Bands, check if at least one band value is valid
      if (isBollingerBands) {
        return (typeof point.upper === 'number' && !isNaN(point.upper)) ||
               (typeof point.middle === 'number' && !isNaN(point.middle)) ||
               (typeof point.lower === 'number' && !isNaN(point.lower))
      }
      
      // For Stochastic, check if k or d is valid
      if (isStochastic) {
        return (typeof point.k === 'number' && !isNaN(point.k)) ||
               (typeof point.d === 'number' && !isNaN(point.d))
      }
      
      // For ADX, check if any component is valid
      if (isADX) {
        return (typeof point.adx === 'number' && !isNaN(point.adx)) ||
               (typeof point.plus_di === 'number' && !isNaN(point.plus_di)) ||
               (typeof point.minus_di === 'number' && !isNaN(point.minus_di))
      }
      
      // For simple indicators, check if value is valid
      return typeof point.value === 'number' && !isNaN(point.value)
    })
  }

  // Debug logging for processed data
  console.log('Processed chartData:', {
    indicatorName: indicator.name,
    originalLength: indicator.data?.length || 0,
    processedLength: chartData.length,
    sampleProcessed: chartData.slice(0, 3),
    lastProcessed: chartData.slice(-3)
  })

  // Get indicator-specific configuration
  const getIndicatorConfig = (name: string) => {
    switch (name.toLowerCase()) {
      case 'rsi':
        return {
          color: '#3b82f6',
          referenceLines: [
            { value: 70, color: '#ef4444', label: 'Overbought' },
            { value: 30, color: '#22c55e', label: 'Oversold' }
          ],
          yDomain: [0, 100],
          formatValue: (value: number) => value.toFixed(2)
        }
      case 'williams %r':
        return {
          color: '#f59e0b',
          referenceLines: [
            { value: -20, color: '#ef4444', label: 'Overbought' },
            { value: -80, color: '#22c55e', label: 'Oversold' }
          ],
          yDomain: [-100, 0],
          formatValue: (value: number) => value.toFixed(2)
        }
      case 'cci':
        return {
          color: '#10b981',
          referenceLines: [
            { value: 100, color: '#ef4444', label: 'Overbought' },
            { value: -100, color: '#22c55e', label: 'Oversold' }
          ],
          yDomain: ['dataMin - 20', 'dataMax + 20'],
          formatValue: (value: number) => value.toFixed(2)
        }
      case 'macd':
        return {
          color: '#8b5cf6',
          referenceLines: [{ value: 0, color: '#6b7280', label: 'Zero Line' }],
          yDomain: ['dataMin - 0.1', 'dataMax + 0.1'],
          formatValue: (value: number) => value.toFixed(3)
        }
      case 'stochastic':
        return {
          color: '#ec4899',
          referenceLines: [
            { value: 80, color: '#ef4444', label: 'Overbought' },
            { value: 20, color: '#22c55e', label: 'Oversold' }
          ],
          yDomain: [0, 100],
          formatValue: (value: number) => value.toFixed(2)
        }
      case 'adx':
        return {
          color: '#f97316',
          referenceLines: [{ value: 25, color: '#6b7280', label: 'Strong Trend' }],
          yDomain: [0, 100],
          formatValue: (value: number) => value.toFixed(2)
        }
      case 'bollinger bands':
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

  const config = getIndicatorConfig(indicator.name)

  if (!chartData.length) {
    return (
      <div className={`flex flex-col items-center justify-center bg-background/50 rounded-md border ${className}`} style={{ height }}>
        <p className="text-muted-foreground text-sm mb-2">No chart data available for {indicator.name}</p>
        <p className="text-xs text-muted-foreground">
          {indicator.data && indicator.data.length > 0 
            ? `${indicator.data.length} data points, but all values are null/invalid`
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
            indicator.signal === 'bullish' ? 'text-bullish bg-bullish/10' :
            indicator.signal === 'bearish' ? 'text-bearish bg-bearish/10' :
            'text-muted-foreground bg-muted/50'
          }`}>
            {indicator.signal || 'neutral'}
          </span>
          {indicator.current_value !== undefined && indicator.current_value !== null && (
            <span className="text-xs text-muted-foreground">
              Current: {(() => {
                if (typeof indicator.current_value === 'number') {
                  return config.formatValue(indicator.current_value)
                } else if (typeof indicator.current_value === 'object' && indicator.current_value !== null) {
                  // For multi-value indicators, show only the most relevant value
                  if (isBollingerBands) {
                    return `Price: ${config.formatValue(indicator.current_value.current_price)}`
                  } else if (isMACD) {
                    return `MACD: ${config.formatValue(indicator.current_value.macd)}`
                  } else if (isStochastic) {
                    return `%K: ${config.formatValue(indicator.current_value.k)}`
                  } else if (isADX) {
                    return `ADX: ${config.formatValue(indicator.current_value.adx)}`
                  } else {
                    // fallback: show first key/value
                    const key = Object.keys(indicator.current_value)[0]
                    return `${key}: ${config.formatValue(indicator.current_value[key])}`
                  }
                } else {
                  return String(indicator.current_value)
                }
              })()}
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
          
          {/* Legend for multi-line charts */}
          {(isBollingerBands || isMACD || isStochastic || isADX) && (
            <Legend 
              wrapperStyle={{ fontSize: '11px', marginTop: '5px' }}
              formatter={(value) => {
                const labelMap: Record<string, string> = {
                  'upper': 'Upper Band',
                  'middle': 'Middle Band', 
                  'lower': 'Lower Band',
                  'macd': 'MACD',
                  'signal': 'Signal',
                  'histogram': 'Histogram',
                  'k': '%K',
                  'd': '%D',
                  'adx': 'ADX',
                  'plus_di': '+DI',
                  'minus_di': '-DI'
                }
                return labelMap[value] || value
              }}
            />
          )}
          
          <Tooltip
            contentStyle={{
              backgroundColor: 'hsl(var(--background))',
              border: '1px solid hsl(var(--border))',
              borderRadius: '6px',
              fontSize: '12px'
            }}
            labelFormatter={(label) => `Point ${label}`}
            formatter={(value: number, name: string) => {
              const labelMap: Record<string, string> = {
                'upper': 'Upper Band',
                'middle': 'Middle Band', 
                'lower': 'Lower Band',
                'macd': 'MACD',
                'signal': 'Signal',
                'histogram': 'Histogram',
                'k': '%K',
                'd': '%D',
                'adx': 'ADX',
                'plus_di': '+DI',
                'minus_di': '-DI',
                'value': indicator.name
              }
              return [config.formatValue(value), labelMap[name] || name]
            }}
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
          
          {/* Conditional rendering based on indicator type */}
          {isBollingerBands ? (
            <>
              <Line type="monotone" dataKey="upper" stroke="#ef4444" strokeWidth={2} dot={false} name="upper" />
              <Line type="monotone" dataKey="middle" stroke="#f59e0b" strokeWidth={2} dot={false} name="middle" />
              <Line type="monotone" dataKey="lower" stroke="#22c55e" strokeWidth={2} dot={false} name="lower" />
            </>
          ) : isMACD ? (
            <>
              <Line type="monotone" dataKey="macd" stroke="#8b5cf6" strokeWidth={2} dot={false} name="macd" />
              <Line type="monotone" dataKey="signal" stroke="#f59e0b" strokeWidth={2} dot={false} name="signal" />
              <Line type="monotone" dataKey="histogram" stroke="#10b981" strokeWidth={1} dot={false} name="histogram" />
            </>
          ) : isStochastic ? (
            <>
              <Line type="monotone" dataKey="k" stroke="#ec4899" strokeWidth={2} dot={false} name="k" />
              <Line type="monotone" dataKey="d" stroke="#3b82f6" strokeWidth={2} dot={false} name="d" />
            </>
          ) : isADX ? (
            <>
              <Line type="monotone" dataKey="adx" stroke="#f97316" strokeWidth={2} dot={false} name="adx" />
              <Line type="monotone" dataKey="plus_di" stroke="#22c55e" strokeWidth={2} dot={false} name="plus_di" />
              <Line type="monotone" dataKey="minus_di" stroke="#ef4444" strokeWidth={2} dot={false} name="minus_di" />
            </>
          ) : (
            <Line
              type="monotone"
              dataKey="value"
              stroke={config.color}
              strokeWidth={2}
              dot={false}
              activeDot={{ r: 4, stroke: config.color, strokeWidth: 2 }}
            />
          )}
        </LineChart>
      </ResponsiveContainer>

      {/* Show current values */}
      <div className="mt-2 text-xs text-muted-foreground">
        <span className="font-medium">Signal:</span> {indicator.signal} 
        <span className="ml-3 font-medium">Strength:</span> {(indicator.strength * 100).toFixed(1)}%
      </div>
    </div>
  )
}
