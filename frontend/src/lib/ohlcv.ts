import Papa, { ParseResult } from 'papaparse';
import { apiClient, API_ENDPOINTS } from './api';

export interface OHLCV {
  date: Date;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

// Robust loader: handles case-insensitive headers, missing/invalid data, and logs errors
type HeaderMap = Record<string, string>;

export async function loadOHLCVFromCSV(csvUrl: string): Promise<OHLCV[]> {
  return new Promise((resolve, reject) => {
    if (!csvUrl) {
      reject(new Error('No CSV file path provided'));
      return;
    }
    Papa.parse(csvUrl, {
      download: true,
      header: true,
      skipEmptyLines: true,
      complete: (results: ParseResult<any>) => {
        if (results.errors && results.errors.length > 0) {
          reject(new Error('CSV parse error: ' + results.errors.map(e => e.message).join('; ')));
          return;
        }
        const data = results.data as any[];
        if (!data || data.length === 0) {
          reject(new Error('CSV file is empty or malformed'));
          return;
        }
        // Build header map (case-insensitive)
        const firstRow = data[0];
        const headerMap: HeaderMap = {};
        for (const key of Object.keys(firstRow)) {
          const lower = key.toLowerCase();
          if (["date", "datetime", "timestamp"].includes(lower)) headerMap["date"] = key;
          if (lower === "open") headerMap["open"] = key;
          if (lower === "high") headerMap["high"] = key;
          if (lower === "low") headerMap["low"] = key;
          if (lower === "close") headerMap["close"] = key;
          if (lower === "volume") headerMap["volume"] = key;
        }
        const required = ["date", "open", "high", "low", "close", "volume"];
        for (const req of required) {
          if (!headerMap[req]) {
            reject(new Error(`Missing required column: ${req}`));
            return;
          }
        }
        const ohlcv: OHLCV[] = [];
        for (const row of data) {
          try {
            const dateVal = row[headerMap["date"]];
            const date = dateVal ? new Date(dateVal) : null;
            if (!date || isNaN(date.getTime())) continue;
            const open = parseFloat(row[headerMap["open"]]);
            const high = parseFloat(row[headerMap["high"]]);
            const low = parseFloat(row[headerMap["low"]]);
            const close = parseFloat(row[headerMap["close"]]);
            const volume = parseFloat(row[headerMap["volume"]]);
            if ([open, high, low, close, volume].some(v => isNaN(v))) continue;
            ohlcv.push({ date, open, high, low, close, volume });
          } catch (err) {
            // Skip malformed row
            continue;
          }
        }
        if (ohlcv.length === 0) {
          reject(new Error('No valid OHLCV rows found in CSV'));
          return;
        }
        resolve(ohlcv);
      },
      error: (err: any) => reject(new Error('PapaParse error: ' + err.message)),
    });
  });
}

// New: Load OHLCV via backend API (recommended for security)
export async function loadOHLCVFromAPI(filename: string): Promise<OHLCV[]> {
  if (!filename) throw new Error('No CSV filename provided');

  // The API returns the data directly as an array
  const ohlcvData = await apiClient.get<any[]>(`/api/v1/market-data/ohlcv-json/${filename}`);

  if (!ohlcvData || !Array.isArray(ohlcvData)) {
    throw new Error('No OHLCV data in API response or response is not an array');
  }

  // Convert to typed OHLCV[]
  return ohlcvData.map(row => ({
    date: new Date(row.date),
    open: Number(row.open),
    high: Number(row.high),
    low: Number(row.low),
    close: Number(row.close),
    volume: Number(row.volume),
  })).filter(row => !isNaN(row.date.getTime()) && [row.open, row.high, row.low, row.close, row.volume].every(v => !isNaN(v)));
}
