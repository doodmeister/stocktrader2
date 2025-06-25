'use client'

import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Area, AreaChart } from 'recharts'

interface PriceChartProps {
  data: Array<{
    timestamp: string
    open: number
    high: number
    low: number
    close: number
    volume: number
  }>
  height?: number
  className?: string
  showVolume?: boolean
}

export default function PriceChart({ 
  data, 
  height = 300, 
  className = '',
  showVolume = false 
}: PriceChartProps) {
  
  // Prepare data for chart
  const chartData = data.map((item, index) => ({
    index: index + 1,
    ...item,
    timestamp: new Date(item.timestamp).toLocaleDateString(),
    priceChange: index > 0 ? item.close - data[index - 1].close : 0
  }))

  const formatPrice = (value: number) => `$${value.toFixed(2)}`
  const formatVolume = (value: number) => {
    if (value >= 1000000) return `${(value / 1000000).toFixed(1)}M`
    if (value >= 1000) return `${(value / 1000).toFixed(1)}K`
    return value.toString()
  }

  if (!chartData.length) {
    return (
      <div className={`flex items-center justify-center bg-background/50 rounded-md border ${className}`} style={{ height }}>
        <p className="text-muted-foreground text-sm">No price data available</p>
      </div>
    )
  }

  const latestPrice = chartData[chartData.length - 1]
  const priceChange = latestPrice.priceChange
  const isPositive = priceChange >= 0

  return (
    <div className={`bg-background/50 rounded-md border p-3 ${className}`}>
      <div className="flex justify-between items-center mb-2">
        <h4 className="font-medium text-sm">Price Chart</h4>
        <div className="flex items-center gap-2">
          <span className="text-sm font-medium">
            {formatPrice(latestPrice.close)}
          </span>
          <span className={`text-xs px-2 py-1 rounded ${
            isPositive ? 'text-bullish bg-bullish/10' : 'text-bearish bg-bearish/10'
          }`}>
            {isPositive ? '+' : ''}{formatPrice(priceChange)}
          </span>
        </div>
      </div>
      
      <ResponsiveContainer width="100%" height={height}>
        <AreaChart data={chartData} margin={{ top: 5, right: 10, left: 10, bottom: 5 }}>
          <defs>
            <linearGradient id="priceGradient" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor="#3b82f6" stopOpacity={0.3}/>
              <stop offset="95%" stopColor="#3b82f6" stopOpacity={0.1}/>
            </linearGradient>
          </defs>
          <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
          <XAxis 
            dataKey="timestamp"
            tick={{ fontSize: 10, fill: 'hsl(var(--muted-foreground))' }}
            axisLine={{ stroke: 'hsl(var(--border))' }}
          />
          <YAxis 
            domain={['dataMin - 5', 'dataMax + 5']}
            tick={{ fontSize: 10, fill: 'hsl(var(--muted-foreground))' }}
            axisLine={{ stroke: 'hsl(var(--border))' }}
            tickFormatter={formatPrice}
          />
          <Tooltip
            contentStyle={{
              backgroundColor: 'hsl(var(--background))',
              border: '1px solid hsl(var(--border))',
              borderRadius: '6px',
              fontSize: '12px'
            }}
            labelFormatter={(label) => `Date: ${label}`}
            formatter={(value: number, name: string) => {
              if (name === 'close') return [formatPrice(value), 'Close']
              if (name === 'volume') return [formatVolume(value), 'Volume']
              return [formatPrice(value), name]
            }}
          />
          
          <Area
            type="monotone"
            dataKey="close"
            stroke="#3b82f6"
            strokeWidth={2}
            fill="url(#priceGradient)"
            dot={false}
            activeDot={{ r: 4, stroke: '#3b82f6', strokeWidth: 2 }}
          />
        </AreaChart>
      </ResponsiveContainer>

      {showVolume && (
        <div className="mt-3 border-t pt-3">
          <h5 className="text-xs font-medium text-muted-foreground mb-2">Volume</h5>
          <ResponsiveContainer width="100%" height={80}>
            <AreaChart data={chartData} margin={{ top: 0, right: 10, left: 10, bottom: 0 }}>
              <defs>
                <linearGradient id="volumeGradient" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#8b5cf6" stopOpacity={0.3}/>
                  <stop offset="95%" stopColor="#8b5cf6" stopOpacity={0.1}/>
                </linearGradient>
              </defs>
              <XAxis dataKey="timestamp" hide />
              <YAxis hide tickFormatter={formatVolume} />
              <Tooltip
                contentStyle={{
                  backgroundColor: 'hsl(var(--background))',
                  border: '1px solid hsl(var(--border))',
                  borderRadius: '6px',
                  fontSize: '12px'
                }}
                formatter={(value: number) => [formatVolume(value), 'Volume']}
              />
              <Area
                type="monotone"
                dataKey="volume"
                stroke="#8b5cf6"
                strokeWidth={1}
                fill="url(#volumeGradient)"
                dot={false}
              />
            </AreaChart>
          </ResponsiveContainer>
        </div>
      )}

      <div className="mt-2 grid grid-cols-4 gap-2 text-xs">
        <div>
          <span className="text-muted-foreground">Open:</span>
          <div className="font-medium">{formatPrice(latestPrice.open)}</div>
        </div>
        <div>
          <span className="text-muted-foreground">High:</span>
          <div className="font-medium">{formatPrice(latestPrice.high)}</div>
        </div>
        <div>
          <span className="text-muted-foreground">Low:</span>
          <div className="font-medium">{formatPrice(latestPrice.low)}</div>
        </div>
        <div>
          <span className="text-muted-foreground">Volume:</span>
          <div className="font-medium">{formatVolume(latestPrice.volume)}</div>
        </div>
      </div>
    </div>
  )
}
