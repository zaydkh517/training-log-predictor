import type { ExerciseAnalysis } from '../types'
import { useCountUp } from '../hooks/useCountUp'
import { TREND_LABEL, fmtLong, lbs, machineLabel, parseDay, signed, signedPct, splitExerciseName, trendOf } from '../lib/format'

type Props = { exercise: string; data: ExerciseAnalysis }

function LiftHeader({ exercise, data }: Props) {
  const { name, equipment } = splitExerciseName(exercise)
  const { plateau, outlook, trajectory } = data
  const trend = trendOf(data)
  const current = outlook.current_smoothed_e1rm_lbs ?? trajectory[trajectory.length - 1]?.rolling_e1rm
  const animated = useCountUp(current)
  const range = outlook.outlook_3mo_range_lbs
  const first = trajectory[0]
  const last = trajectory[trajectory.length - 1]

  return (
    <header className="lift-header">
      <div className="lift-header-top">
        <div>
          <p className="eyebrow">
            {equipment && <>{equipment} · </>}
            {trajectory.length} sessions
            {first && last && <> · {fmtLong(parseDay(first.date))} – {fmtLong(parseDay(last.date))}</>}
            {(data.machine_count ?? 1) > 1 && data.current_machine !== undefined && (
              <> · stats for {machineLabel(data.current_machine)} of {data.machine_count}</>
            )}
          </p>
          <h1 className="lift-title">{name}</h1>
        </div>
        <div className={`status status-${trend}`}>
          <span className="status-dot" aria-hidden />
          {TREND_LABEL[trend]}
        </div>
      </div>

      <dl className="stat-strip">
        <div className="stat stat-hero">
          <dt>Estimated 1RM</dt>
          <dd>
            <span className="display">{lbs(animated)}</span>
            <span className="unit mono">LB</span>
          </dd>
          <span className="stat-note mono">28-day smoothed</span>
        </div>
        <div className="stat">
          <dt>Last {plateau.window_days ?? 28} days</dt>
          <dd className={`trend-${trend}`}>
            <span className="display-sm">{signedPct(plateau.percent_change)}</span>
          </dd>
          <span className="stat-note mono">
            {plateau.error ??
              (trend === 'inactive'
                ? `last trained ${plateau.days_since_last_session} days ago`
                : `stall line ${signedPct(plateau.threshold_pct)} for your training age`)}
          </span>
        </div>
        <div className="stat">
          <dt>Rate</dt>
          <dd>
            <span className="display-sm">{signed(outlook.your_rate_lbs_per_week)}</span>
            <span className="unit mono">LB/WK</span>
          </dd>
          <span className="stat-note mono">{outlook.error ?? outlook.rate_window}</span>
        </div>
        <div className="stat">
          <dt>3-month outlook</dt>
          <dd>
            {range ? (
              <>
                <span className="display-sm accent">{lbs(range[0])}–{lbs(range[1])}</span>
                <span className="unit mono">LB</span>
              </>
            ) : (
              <span className="display-sm dim">—</span>
            )}
          </dd>
          <span className="stat-note mono">{range ? 'heuristic range' : outlook.error}</span>
        </div>
      </dl>

      {trend === 'flat' && (
        <p className="callout">
          <strong>Stalled.</strong> Your smoothed e1RM moved {signedPct(plateau.percent_change)} over the last{' '}
          {plateau.window_days} days, under the {signedPct(plateau.threshold_pct)} expected at your training age.
          Common fixes: change the rep range, add a back-off set, or take a deload week.
        </p>
      )}
      {trend === 'down' && (
        <p className="callout">
          <strong>Dropping.</strong> Your smoothed e1RM is down {signedPct(plateau.percent_change)} over the last{' '}
          {plateau.window_days} days. If you're not in a cut, check sleep, recovery and total training volume.
        </p>
      )}
    </header>
  )
}

export default LiftHeader
