import { useMemo, useState } from 'react'
import {
  Area, CartesianGrid, ComposedChart, Line, ReferenceArea, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis,
  type TooltipProps,
} from 'recharts'
import type { ExerciseAnalysis } from '../types'
import { fmtLong, fmtShort, lbs, machineLabel, parseDay } from '../lib/format'

type Props = { exercise: string; data: ExerciseAnalysis }

type ChartPoint = {
  t: number
  e1rm?: number
  rolling?: number
  band?: [number, number]
  median?: number
  projected?: boolean
}

const DAY = 86_400_000
const RANGES = [
  { key: '3M', days: 90 },
  { key: '6M', days: 182 },
  { key: 'ALL', days: Infinity },
] as const
type RangeKey = (typeof RANGES)[number]['key']

const css = (name: string) =>
  typeof window === 'undefined' ? '' : getComputedStyle(document.documentElement).getPropertyValue(name).trim()

function ChartTooltip({ active, payload }: TooltipProps<number, string>) {
  if (!active || !payload?.length) return null
  const p = payload[0].payload as ChartPoint
  return (
    <div className="chart-tip">
      <div className="chart-tip-date mono">{p.projected ? `Projected · ${fmtLong(p.t)}` : fmtLong(p.t)}</div>
      {p.rolling !== undefined && (
        <div className="chart-tip-row"><i className="swatch swatch-line" />Smoothed<b className="mono">{lbs(p.rolling, 1)}</b></div>
      )}
      {p.e1rm !== undefined && (
        <div className="chart-tip-row"><i className="swatch swatch-dot" />Session e1RM<b className="mono">{lbs(p.e1rm, 1)}</b></div>
      )}
      {p.projected && p.band && (
        <div className="chart-tip-row"><i className="swatch swatch-band" />80% range<b className="mono">{lbs(p.band[0])}–{lbs(p.band[1])}</b></div>
      )}
      {p.projected && p.median !== undefined && (
        <div className="chart-tip-row"><i className="swatch swatch-median" />Median forecast<b className="mono">{lbs(p.median)}</b></div>
      )}
    </div>
  )
}

function TrajectoryChart({ exercise, data }: Props) {
  const { trajectory, outlook, plateau } = data
  const [range, setRange] = useState<RangeKey>('ALL')
  // a machine choice belongs to one lift; switching lifts falls back to that lift's current machine
  const [picked, setPicked] = useState<{ exercise: string; machine: number } | null>(null)
  const current = data.current_machine ?? 0
  const machineCount = data.machine_count ?? 1
  const machine = picked?.exercise === exercise ? picked.machine : current
  const onCurrent = machine === current

  const { points, lastT, endT, plateauStartT } = useMemo(() => {
    const all: ChartPoint[] = trajectory
      .filter((p) => (p.machine ?? 0) === machine)
      .map((p) => ({ t: parseDay(p.date), e1rm: p.e1rm, rolling: p.rolling_e1rm }))
    const lastT = all[all.length - 1]?.t ?? 0
    const days = RANGES.find((r) => r.key === range)!.days
    const points = all.filter((p) => lastT - p.t <= days * DAY)

    let endT = lastT
    const band = onCurrent ? outlook.outlook_3mo_range_lbs : undefined
    if (band && outlook.current_smoothed_e1rm_lbs !== undefined && points.length) {
      const anchor = outlook.current_smoothed_e1rm_lbs
      points[points.length - 1] = { ...points[points.length - 1], band: [anchor, anchor], median: anchor }
      endT = lastT + 90 * DAY
      points.push({ t: endT, band, median: outlook.outlook_3mo_median_lbs, projected: true })
    }
    // shade the window the plateau check actually compared across
    const plateauStartT = onCurrent && plateau.is_plateau && plateau.window_start ? parseDay(plateau.window_start) : undefined
    return { points, lastT, endT, plateauStartT }
  }, [trajectory, outlook, plateau, range, machine, onCurrent])

  const firstT = points[0]?.t ?? 0
  const accent = css('--accent') || '#ff2d46'
  const warn = css('--warn') || '#ffb020'
  const dim = css('--text-dim') || '#5e5e66'
  const line = css('--line') || '#222226'
  const muted = css('--text-muted') || '#8b8b93'

  return (
    <section className="panel chart-panel">
      <div className="panel-head">
        <div>
          <h2 className="label">Strength trajectory</h2>
          <ul className="legend mono">
            <li><i className="swatch swatch-line" />Smoothed e1RM</li>
            <li><i className="swatch swatch-dot" />Session best</li>
            {onCurrent && outlook.outlook_3mo_range_lbs && <><li><i className="swatch swatch-band" />3-mo 80% range</li><li><i className="swatch swatch-median" />Median forecast</li></>}
            {plateauStartT !== undefined && <li><i className="swatch swatch-warn" />{plateau.status === 'declining' ? 'Decline' : 'Plateau'} window</li>}
          </ul>
        </div>
        <div className="chart-controls">
        {machineCount > 1 && (
          <div className="segmented" role="tablist" aria-label="Machine">
            {Array.from({ length: machineCount }, (_, id) => (
              <button
                key={id}
                type="button"
                role="tab"
                aria-selected={machine === id}
                title={id === current ? `${machineLabel(id)} (current)` : machineLabel(id)}
                className={machine === id ? 'is-on' : ''}
                onClick={() => setPicked({ exercise, machine: id })}
              >
                {String.fromCharCode(65 + id)}{id === current && <span className="tab-dot" aria-hidden />}
              </button>
            ))}
          </div>
        )}
        <div className="segmented" role="tablist" aria-label="Time range">
          {RANGES.map((r) => (
            <button
              key={r.key}
              type="button"
              role="tab"
              aria-selected={range === r.key}
              className={range === r.key ? 'is-on' : ''}
              onClick={() => setRange(r.key)}
            >
              {r.key}
            </button>
          ))}
        </div>
        </div>
      </div>
      {machineCount > 1 && (
        <p className="machine-note">
          Logged on {machineCount} different machines. Pounds aren&rsquo;t comparable between them, so each is
          tracked on its own. Showing {machineLabel(machine)}{onCurrent ? ' (current)' : ''}.
        </p>
      )}

      <div className="chart-wrap">
        <ResponsiveContainer width="100%" height="100%">
          {/* remount per lift/range: Recharts' tween between series of different lengths can stall */}
          <ComposedChart key={`${exercise}:${machine}:${range}`} data={points} margin={{ top: 12, right: 8, bottom: 0, left: -8 }}>
            <defs>
              <linearGradient id="rollFill" x1="0" x2="0" y1="0" y2="1">
                <stop offset="0" stopColor={accent} stopOpacity={0.22} />
                <stop offset="1" stopColor={accent} stopOpacity={0} />
              </linearGradient>
              <pattern id="bandHatch" width="6" height="6" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
                <rect width="6" height="6" fill={accent} fillOpacity={0.06} />
                <line x1="0" y1="0" x2="0" y2="6" stroke={accent} strokeOpacity={0.4} strokeWidth={1.5} />
              </pattern>
            </defs>
            <XAxis
              dataKey="t"
              type="number"
              scale="time"
              domain={[firstT, endT]}
              tickFormatter={fmtShort}
              tick={{ fill: muted, fontSize: 11, fontFamily: 'var(--font-mono)' }}
              axisLine={{ stroke: line }}
              tickLine={false}
              minTickGap={40}
              tickMargin={10}
            />
            <YAxis
              domain={[(min: number) => Math.floor((min * 0.96) / 5) * 5, (max: number) => Math.ceil((max * 1.02) / 5) * 5]}
              tick={{ fill: muted, fontSize: 11, fontFamily: 'var(--font-mono)' }}
              axisLine={false}
              tickLine={false}
              width={52}
              tickCount={6}
            />
            <CartesianGrid vertical={false} stroke={line} />
            {plateauStartT !== undefined && plateauStartT >= firstT && (
              <ReferenceArea x1={plateauStartT} x2={lastT} fill={warn} fillOpacity={0.08} stroke={warn} strokeOpacity={0.35} strokeDasharray="2 3" />
            )}
            <ReferenceLine
              x={lastT}
              stroke={dim}
              strokeDasharray="3 4"
              label={{ value: 'LAST SESSION', position: 'insideTopRight', fill: muted, fontSize: 10, fontFamily: 'var(--font-mono)' }}
            />
            <Tooltip content={<ChartTooltip />} cursor={{ stroke: dim, strokeDasharray: '2 3' }} />
            <Area dataKey="band" type="monotone" stroke={accent} strokeOpacity={0.6} strokeDasharray="4 4" fill="url(#bandHatch)" isAnimationActive={false} connectNulls />
            <Line dataKey="median" stroke={accent} strokeWidth={1.5} strokeDasharray="2 4" dot={false} isAnimationActive={false} connectNulls />
            <Area dataKey="rolling" type="monotone" stroke={accent} strokeWidth={2.5} fill="url(#rollFill)" animationDuration={700} activeDot={{ r: 5, fill: accent, stroke: '#000', strokeWidth: 2 }} />
            <Line dataKey="e1rm" stroke="none" isAnimationActive={false} dot={{ r: 2.5, fill: dim, stroke: 'none' }} activeDot={{ r: 4, fill: '#fff' }} />
          </ComposedChart>
        </ResponsiveContainer>
      </div>

      {onCurrent && outlook.range_basis && <p className="footnote">Forecast: {outlook.range_basis}.</p>}
    </section>
  )
}

export default TrajectoryChart
