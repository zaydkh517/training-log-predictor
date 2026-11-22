export type TrajectoryPoint = { date: string; e1rm: number; rolling_e1rm: number }

export type Plateau = {
  status?: 'progressing' | 'plateau' | 'declining' | 'inactive'
  is_plateau?: boolean
  percent_change?: number
  threshold_pct?: number
  window_days?: number
  window_start?: string
  rolling_e1rm_now?: number
  rolling_e1rm_window_start?: number
  days_since_last_session?: number
  error?: string
}

export type Outlook = {
  current_smoothed_e1rm_lbs?: number
  raw_last_session_e1rm_lbs?: number
  your_rate_lbs_per_week?: number
  rate_window?: string
  outlook_3mo_range_lbs?: [number, number]
  range_basis?: string
  error?: string
}

export type ExerciseAnalysis = {
  trajectory: TrajectoryPoint[]
  plateau: Plateau
  outlook: Outlook
}

export type Analysis = { exercises: Record<string, ExerciseAnalysis> }

export type PlannedSet = {
  set_number: number
  weight?: number
  predicted_reps?: number
  note?: string
}
