import { type ClassValue, clsx } from "clsx"
import { twMerge } from "tailwind-merge"

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs))
}

export function formatCurrency(value: number, currency = 'USD'): string {
  return new Intl.NumberFormat('en-US', {
    style: 'currency',
    currency,
  }).format(value)
}

export function formatPercentage(value: number): string {
  return new Intl.NumberFormat('en-US', {
    style: 'percent',
    minimumFractionDigits: 2,
    maximumFractionDigits: 2,
  }).format(value / 100)
}

export function formatNumber(value: number): string {
  return new Intl.NumberFormat('en-US').format(value)
}

export function getPriceChangeColor(change: number): string {
  if (change > 0) return 'text-bullish'
  if (change < 0) return 'text-bearish'
  return 'text-neutral'
}

export function getPriceChangeIcon(change: number): string {
  if (change > 0) return '↗'
  if (change < 0) return '↘'
  return '→'
}
