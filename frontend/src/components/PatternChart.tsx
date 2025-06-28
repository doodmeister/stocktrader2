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

const markerPath = {
  bullish: 'M0,0 L10,0 L5,-10 Z', // triangle up
  bearish: 'M0,0 L10,0 L5,10 Z', // triangle down
  neutral: 'M0,0 L10,0 L5,-5 L0,0 Z', // diamond
};

const markerColor = {
  bullish: '#16a34a',
  bearish: '#dc2626',
  neutral: '#facc15',
};

function getPatternType(signal: string) {
  if (signal.toLowerCase().includes('bullish')) return 'bullish';
  if (signal.toLowerCase().includes('bearish')) return 'bearish';
  return 'neutral';
}

function PatternMarkerAnnotation(props: any) {
  return <SvgPathAnnotation {...props} />;
}

export function PatternChart({ ohlcv, patterns }: PatternChartProps) {
  if (!ohlcv || ohlcv.length === 0) return <div>No OHLCV data for chart.</div>;

  const data = ohlcv.map(d => ({
    ...d,
    date: d.date instanceof Date ? d.date : new Date(d.date),
  }));

  const annotations = patterns.map(pattern => {
    const type = getPatternType(pattern.signal || pattern.name || '');
    const candleIdx = pattern.start_index ?? 0;
    const candle = data[candleIdx];
    return candle && candle.date ? {
      x: candle.date,
      type,
      name: pattern.name,
      confidence: pattern.confidence,
    } : null;
  }).filter(Boolean) as Array<{
    x: Date;
    type: keyof typeof markerColor;
    name: string;
    confidence: number;
  }>;

  const width = 800;

  return (
    <div style={{ width: '100%', height: 400 }}>
      <ChartCanvas
        height={400}
        width={width}
        ratio={1}
        margin={{ left: 50, right: 50, top: 10, bottom: 30 }}
        seriesName="PatternChart"
        data={data}
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
              when={(d: OHLCV) => d.date && a.x && d.date.getTime() === a.x.getTime()}
              usingProps={{
                y: ({ yScale, datum }: any) => yScale(datum.high) - 10,
                fill: markerColor[a.type],
                stroke: markerColor[a.type],
                path: markerPath[a.type],
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
