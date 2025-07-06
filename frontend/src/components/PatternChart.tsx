// New PatternChart.tsx for overlaying detected patterns on a candlestick chart using react-financial-charts
// This file will be implemented to visualize OHLCV data and detected patterns.

import React, { useState, useMemo } from 'react';
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


// Pattern marker color and shape map (enterprise-grade)
const PATTERN_MARKERS: Record<string, { color: string; path: string; label: string; type: 'bullish' | 'bearish' | 'neutral' }> = {
  'hammer':      { color: '#22c55e', path: 'M0,0 L10,0 L5,-10 Z', label: 'Hammer', type: 'bullish' }, // green triangle up
  'doji':        { color: '#64748b', path: 'M-5,0 L5,0 M0,-5 L0,5', label: 'Doji', type: 'neutral' }, // gray cross
  'engulfing':   { color: '#22c55e', path: 'M0,0 L10,0 L5,-10 Z', label: 'Engulfing', type: 'bullish' },
  'morning star':{ color: '#22c55e', path: 'M0,0 L10,0 L5,-10 Z', label: 'Morning Star', type: 'bullish' },
  'shooting star':{ color: '#ef4444', path: 'M0,0 L10,0 L5,10 Z', label: 'Shooting Star', type: 'bearish' }, // red triangle down
  'hanging man': { color: '#ef4444', path: 'M0,0 L10,0 L5,10 Z', label: 'Hanging Man', type: 'bearish' },
  'bearish engulfing': { color: '#ef4444', path: 'M0,0 L10,0 L5,10 Z', label: 'Bearish Engulfing', type: 'bearish' },
  'bullish engulfing': { color: '#22c55e', path: 'M0,0 L10,0 L5,-10 Z', label: 'Bullish Engulfing', type: 'bullish' },
  'dark cloud cover': { color: '#ef4444', path: 'M0,0 L10,0 L5,10 Z', label: 'Dark Cloud', type: 'bearish' },
  'piercing line': { color: '#22c55e', path: 'M0,0 L10,0 L5,-10 Z', label: 'Piercing Line', type: 'bullish' },
  'evening star': { color: '#ef4444', path: 'M0,0 L10,0 L5,10 Z', label: 'Evening Star', type: 'bearish' },
  // fallback
  'default_bullish': { color: '#22c55e', path: 'M0,0 L10,0 L5,-10 Z', label: 'Bullish', type: 'bullish' },
  'default_bearish': { color: '#ef4444', path: 'M0,0 L10,0 L5,10 Z', label: 'Bearish', type: 'bearish' },
  'default_neutral': { color: '#64748b', path: 'M-5,0 L5,0 M0,-5 L0,5', label: 'Neutral', type: 'neutral' },
};

function getPatternMarker(patternName: string, patternType?: string): { color: string; path: string; label: string; type: string } {
  const key = patternName?.toLowerCase();
  if (PATTERN_MARKERS[key]) return PATTERN_MARKERS[key];
  if (patternType === 'bullish') return PATTERN_MARKERS['default_bullish'];
  if (patternType === 'bearish') return PATTERN_MARKERS['default_bearish'];
  if (patternType === 'neutral') return PATTERN_MARKERS['default_neutral'];
  return PATTERN_MARKERS['default_neutral'];
}

// Move PatternMarkerAnnotation above PatternChart

function PatternMarkerAnnotation(props: any) {
  // Add marker and label above, color-coded by pattern type
  // Improved: Increase y offset for label, and add a semi-transparent background for readability
  const labelYOffset = 28; // Increased offset for better visibility above marker
  const labelFontSize = 13;
  const labelPadding = 4;
  const labelText = props.label;
  const labelWidth = labelText ? labelText.length * (labelFontSize * 0.65) + labelPadding * 2 : 0;
  return (
    <g>
      {props.x !== undefined && props.y !== undefined && props.tooltip && (
        <g>
          {/* Background rectangle for label readability */}
          <rect
            x={props.x - labelWidth / 2}
            y={props.y - labelYOffset - labelFontSize - 2}
            width={labelWidth}
            height={labelFontSize + labelPadding}
            fill="#fff"
            opacity={0.85}
            rx={4}
          />
          <text
            x={props.x}
            y={props.y - labelYOffset}
            textAnchor="middle"
            fontSize={labelFontSize}
            fontWeight="bold"
            fill={props.fill}
            stroke="#fff"
            strokeWidth={0.5}
            paintOrder="stroke"
            style={{ pointerEvents: 'none' }}
          >
            {labelText}
          </text>
        </g>
      )}
      <SvgPathAnnotation
        {...props}
        strokeWidth={2}
        opacity={0.95}
      />
    </g>
  );
}
export function PatternChart({ ohlcv, patterns }: PatternChartProps) {
  if (!ohlcv || ohlcv.length === 0) return <div>No OHLCV data for chart.</div>;

  const data = useMemo(() => ohlcv.map(d => ({
    ...d,
    date: d.date instanceof Date ? d.date : new Date(d.date),
  })), [ohlcv]);

  // Build a map from date string to candle for fast lookup
  const candleByDate: Record<string, OHLCV> = useMemo(() => {
    const map: Record<string, OHLCV> = {};
    data.forEach((d) => {
      map[d.date.toISOString()] = d;
    });
    return map;
  }, [data]);

  // Get all unique pattern keys present in this data
  const uniquePatternKeys = useMemo(() => {
    const keys = new Set<string>();
    patterns.forEach(p => {
      const name = (p.pattern_name || p.name || '').toLowerCase();
      if (PATTERN_MARKERS[name]) {
        keys.add(name);
      } else if (p.pattern_type === 'bullish') {
        keys.add('default_bullish');
      } else if (p.pattern_type === 'bearish') {
        keys.add('default_bearish');
      } else {
        keys.add('default_neutral');
      }
    });
    return Array.from(keys);
  }, [patterns]);

  // State: which patterns are enabled
  const [enabledPatterns, setEnabledPatterns] = useState<Record<string, boolean>>(() => {
    const initial: Record<string, boolean> = {};
    uniquePatternKeys.forEach(k => { initial[k] = true; });
    return initial;
  });

  // Handler for toggling pattern
  const handleTogglePattern = (key: string) => {
    setEnabledPatterns(prev => ({ ...prev, [key]: !prev[key] }));
  };

  // Group patterns by candle date for stacking/offsetting, filter by enabled
  const patternGroups: Record<number, Array<any>> = useMemo(() => {
    const groups: Record<number, Array<any>> = {};
    patterns.forEach(pattern => {
      const patternName = (pattern.pattern_name || pattern.name || '').toLowerCase();
      let markerKey = patternName;
      if (!PATTERN_MARKERS[markerKey]) {
        if (pattern.pattern_type === 'bullish') markerKey = 'default_bullish';
        else if (pattern.pattern_type === 'bearish') markerKey = 'default_bearish';
        else markerKey = 'default_neutral';
      }
      if (!enabledPatterns[markerKey]) return; // skip if not enabled
      const patternDate = (pattern as any).date ? new Date((pattern as any).date) : null;
      let candle: OHLCV | undefined;
      if (patternDate && !isNaN(patternDate.getTime())) {
        candle = candleByDate[patternDate.toISOString()];
      } else if (typeof pattern.start_index === 'number') {
        candle = data[pattern.start_index];
      }
      if (candle && candle.date && typeof candle.high === 'number' && !isNaN(candle.high)) {
        const key = candle.date.getTime();
        if (!groups[key]) groups[key] = [];
        groups[key].push({
          x: candle.date,
          name: pattern.pattern_name || pattern.name,
          confidence: pattern.confidence,
          type: pattern.pattern_type || pattern.signal,
          markerKey,
        });
      }
    });
    return groups;
  }, [patterns, data, candleByDate, enabledPatterns]);

  // Build annotation list with vertical offset for stacking
  const annotations: Array<{ x: Date; name: string; confidence: number; type: string; offset: number; markerKey: string }> = useMemo(() => {
    const ann: Array<any> = [];
    Object.values(patternGroups).forEach((group: any[]) => {
      group.forEach((p, i) => {
        ann.push({ ...p, offset: i });
      });
    });
    return ann;
  }, [patternGroups]);

  // Only show candles where a pattern is detected
  const patternDates = useMemo(() => new Set(annotations.map(a => a.x.getTime())), [annotations]);
  const filteredData = useMemo(() => data.filter(d => patternDates.has(d.date.getTime()) && typeof d.high === 'number' && !isNaN(d.high)), [data, patternDates]);

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
          {annotations.map((a, i) => {
            const marker = getPatternMarker(a.name, a.type);
            return (
              <Annotate
                key={i}
                with={PatternMarkerAnnotation}
                when={(d: OHLCV) => d.date && a.x && d.date instanceof Date && a.x instanceof Date && d.date.getTime() === a.x.getTime() && typeof d.high === 'number' && !isNaN(d.high)}
                usingProps={{
                  y: ({ yScale, datum }: any) => typeof datum.high === 'number' && !isNaN(datum.high) ? yScale(datum.high) - 20 - (a.offset * 18) : 0, // stack markers
                  fill: marker.color,
                  stroke: marker.color,
                  path: () => marker.path,
                  tooltip: `${a.name} (${(a.confidence * 100).toFixed(1)}%)`,
                  label: marker.label,
                }}
              />
            );
          })}
        </Chart>
        <CrossHairCursor />
      </ChartCanvas>

      {/* Enterprise-grade legend for pattern markers with toggles */}
      <div className="flex flex-wrap gap-4 mt-2 text-xs items-center">
        <span className="font-semibold">Legend:</span>
        {uniquePatternKeys.map((key) => {
          const marker = PATTERN_MARKERS[key];
          return (
            <label key={key} className="flex items-center gap-1 cursor-pointer select-none">
              <input
                type="checkbox"
                checked={!!enabledPatterns[key]}
                onChange={() => handleTogglePattern(key)}
                className="accent-current"
                style={{ accentColor: marker.color }}
              />
              <svg width="18" height="18" style={{ verticalAlign: 'middle' }}>
                <path d={marker.path} fill={marker.color} stroke={marker.color} strokeWidth={2} opacity={0.95} transform="translate(4,10)" />
              </svg>
              <span style={{ color: marker.color }}>{marker.label}</span>
            </label>
          );
        })}
        <span className="ml-2 text-muted-foreground">(toggle to show/hide pattern overlays)</span>
      </div>
    </div>
  );
}

export default PatternChart;
