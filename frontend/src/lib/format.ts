import type { ExerciseAnalysis } from '../types'

export const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'

/** "Bench Press (Barbell)" -> { name: "Bench Press", equipment: "Barbell" } */
export function splitExerciseName(full: string) {
  const match = full.match(/^(.*?)\s*\(([^)]+)\)\s*$/)
  return match ? { name: match[1], equipment: match[2] } : { name: full, equipment: null }
}

export function lbs(value: number | undefined, digits = 0) {
  if (value === undefined || Number.isNaN(value)) return '—'
  return value.toLocaleString('en-US', { maximumFractionDigits: digits, minimumFractionDigits: digits })
}

export function signedPct(fraction: number | undefined, digits = 1) {
  if (fraction === undefined) return '—'
  const pct = fraction * 100
  const sign = pct > 0 ? '+' : pct < 0 ? '−' : '±'
  return `${sign}${Math.abs(pct).toFixed(digits)}%`
}

export function signed(value: number | undefined, digits = 1) {
  if (value === undefined) return '—'
  const sign = value > 0 ? '+' : value < 0 ? '−' : '±'
  return `${sign}${Math.abs(value).toFixed(digits)}`
}

const shortDate = new Intl.DateTimeFormat('en-US', { month: 'short', day: 'numeric' })
const longDate = new Intl.DateTimeFormat('en-US', { month: 'short', day: 'numeric', year: 'numeric' })

/** API dates are "YYYY-MM-DD"; parse as local midnight so they don't shift a day. */
export function parseDay(iso: string) {
  const [y, m, d] = iso.split('-').map(Number)
  return new Date(y, m - 1, d).getTime()
}

export const fmtShort = (t: number) => shortDate.format(t)
export const fmtLong = (t: number) => longDate.format(t)

export type Trend = 'up' | 'flat' | 'down' | 'inactive' | 'unknown'

const TREND_BY_STATUS = { progressing: 'up', plateau: 'flat', declining: 'down', inactive: 'inactive' } as const

export function trendOf(ex: ExerciseAnalysis): Trend {
  if (ex.plateau.status) return TREND_BY_STATUS[ex.plateau.status]
  // older API responses only carried the boolean
  if (ex.plateau.is_plateau !== undefined) return ex.plateau.is_plateau ? 'flat' : 'up'
  return 'unknown'
}

export const TREND_LABEL: Record<Trend, string> = {
  up: 'Progressing',
  flat: 'Plateau',
  down: 'Declining',
  inactive: 'Inactive',
  unknown: 'Building baseline',
}

export function currentE1rm(ex: ExerciseAnalysis) {
  return ex.outlook.current_smoothed_e1rm_lbs ?? ex.trajectory[ex.trajectory.length - 1]?.rolling_e1rm
}
