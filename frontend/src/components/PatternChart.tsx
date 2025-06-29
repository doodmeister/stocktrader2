// New PatternChart.tsx for overlaying detected patterns on a candlestick chart using react-financial-charts
// This file will be implemented to visualize OHLCV data and detected patterns.

import React from 'react';
import {
  ChartCanvas,
  Chart,
  XAxis,
  YAxis,
  CrossHairCursor,
  CandlestickSeries,
  Annotate,
  SvgPathAnnotation,
} from "react-financial-charts";
import { scaleTime } from "d3-scale";
import type { PatternResult } from '@/lib/api';

interface OHLCV {
  date: Date;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

interface PatternChartProps {
  ohlcv: OHLCV[];
  patterns: PatternResult[];
}

// Use a single color and marker path for all detected patterns
const markerColor = '#2563eb'; // blue
const markerPath = 'M0,0 L10,0 L5,-10 Z'; // triangle up

// Move PatternMarkerAnnotation above PatternChart
function PatternMarkerAnnotation(props: any) {
  // Make marker larger and more visible, and add a label above
  return (
    <g>
      {/* Pattern label above marker */}
      {props.x !== undefined && props.y !== undefined && props.tooltip && (
        <text
          x={props.x}
          y={props.y - 12}
          textAnchor="middle"
          fontSize="12"
          fontWeight="bold"
          fill={props.fill}
          stroke="#fff"
          strokeWidth={0.5}
          paintOrder="stroke"
        >
          {props.tooltip.split('(')[0].trim()}
        </text>
      )}
      <SvgPathAnnotation
        {...props}
        strokeWidth={2}
        opacity={0.95}
      />
      {/* Debug: show a small circle at the marker position */}
      {props.x !== undefined && props.y !== undefined && (
        <circle cx={props.x} cy={props.y} r={6} fill={props.fill} opacity={0.4} />
      )}
    </g>
  );
}

export function PatternChart({ ohlcv, patterns }: PatternChartProps) {
  if (!ohlcv || ohlcv.length === 0) return <div>No OHLCV data for chart.</div>;

  const data = ohlcv.map(d => ({
    ...d,
    date: d.date instanceof Date ? d.date : new Date(d.date),
  }));

  // Build a map from date string to candle for fast lookup
  const candleByDate: Record<string, OHLCV> = {};
  data.forEach((d, idx) => {
    // Use ISO string for robust matching
    candleByDate[d.date.toISOString()] = d;
  });

  // Try to match patterns by date (prefer pattern.date, fallback to start_index)
  const annotations = patterns.map(pattern => {
    // Prefer pattern.date if available, else fallback to start_index
    const patternDate = (pattern as any).date ? new Date((pattern as any).date) : null;
    let candle: OHLCV | undefined;
    if (patternDate && !isNaN(patternDate.getTime())) {
      candle = candleByDate[patternDate.toISOString()];
    } else if (typeof pattern.start_index === 'number') {
      candle = data[pattern.start_index];
    }
    // Defensive: only return annotation if candle and candle.high are valid numbers
    if (candle && candle.date && typeof candle.high === 'number' && !isNaN(candle.high)) {
      return {
        x: candle.date,
        name: pattern.name,
        confidence: pattern.confidence,
      };
    }
    return null;
  }).filter(Boolean) as Array<{
    x: Date;
    name: string;
    confidence: number;
  }>;

  // Only show candles where a pattern is detected
  const patternDates = new Set(annotations.map(a => a.x.getTime()));
  const filteredData = data.filter(d => patternDates.has(d.date.getTime()) && typeof d.high === 'number' && !isNaN(d.high));

  const width = 800;

  return (
    <div style={{ width: '100%', height: 400 }}>
      <ChartCanvas
        height={400}
        width={width}
        ratio={1}
        margin={{ left: 50, right: 50, top: 10, bottom: 30 }}
        seriesName="PatternChart"
        data={filteredData}
        xAccessor={(d: OHLCV) => d.date}
        xScale={scaleTime() as any}
        displayXAccessor={(d: OHLCV) => d.date}
      >
        <Chart id={1} yExtents={(d: OHLCV) => [d.high, d.low]}>
          <XAxis showGridLines />
          <YAxis showGridLines />
          <CandlestickSeries />
          {annotations.map((a, i) => (
            <Annotate
              key={i}
              with={PatternMarkerAnnotation}
              when={(d: OHLCV) => d.date && a.x && d.date instanceof Date && a.x instanceof Date && d.date.getTime() === a.x.getTime() && typeof d.high === 'number' && !isNaN(d.high)}
              usingProps={{
                y: ({ yScale, datum }: any) => typeof datum.high === 'number' && !isNaN(datum.high) ? yScale(datum.high) - 20 : 0, // move marker higher above candle
                fill: markerColor,
                stroke: markerColor,
                path: () => markerPath,
                tooltip: `${a.name} (${(a.confidence * 100).toFixed(1)}%)`,
              }}
            />
          ))}
        </Chart>
        <CrossHairCursor />
      </ChartCanvas>
    </div>
  );
}

export default PatternChart;
