import { useMemo, useRef, useState } from 'react'
import type { Analysis } from '../types'
import { currentE1rm, lbs, signedPct, splitExerciseName, trendOf } from '../lib/format'

type Props = {
  analysis: Analysis
  selected: string
  onSelect: (exercise: string) => void
}

function Sparkline({ values, flat }: { values: number[]; flat: boolean }) {
  if (values.length < 2) return <svg className="spark" />
  const min = Math.min(...values)
  const max = Math.max(...values)
  const span = max - min || 1
  const pts = values.map((v, i) => `${(i / (values.length - 1)) * 64},${20 - ((v - min) / span) * 18 - 1}`)
  return (
    <svg className={`spark${flat ? ' is-flat' : ''}`} viewBox="0 0 64 20" preserveAspectRatio="none" aria-hidden>
      <polyline points={pts.join(' ')} fill="none" vectorEffect="non-scaling-stroke" />
    </svg>
  )
}

function ExerciseList({ analysis, selected, onSelect }: Props) {
  const [query, setQuery] = useState('')
  const listRef = useRef<HTMLUListElement>(null)

  const rows = useMemo(
    () =>
      Object.entries(analysis.exercises)
        .map(([key, ex]) => ({ key, ex, ...splitExerciseName(key), sessions: ex.trajectory.length }))
        .sort((a, b) => b.sessions - a.sessions),
    [analysis],
  )
  const visible = rows.filter((r) => r.key.toLowerCase().includes(query.trim().toLowerCase()))
  const plateaued = rows.filter((r) => r.ex.plateau.is_plateau).length

  function handleKey(e: React.KeyboardEvent) {
    if (e.key !== 'ArrowDown' && e.key !== 'ArrowUp') return
    e.preventDefault()
    const idx = visible.findIndex((r) => r.key === selected)
    const next = visible[Math.max(0, Math.min(visible.length - 1, idx + (e.key === 'ArrowDown' ? 1 : -1)))]
    if (next) {
      onSelect(next.key)
      listRef.current?.querySelector<HTMLElement>(`[data-key="${CSS.escape(next.key)}"]`)?.focus()
    }
  }

  return (
    <aside className="lift-nav">
      <div className="lift-nav-head">
        <h2 className="label">Lifts <span className="mono dim">{rows.length}</span></h2>
        {plateaued > 0 && <span className="mono tiny warn">{plateaued} stalled</span>}
      </div>
      {rows.length > 6 && (
        <input
          className="lift-search"
          type="search"
          placeholder="Filter lifts…"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
        />
      )}
      <ul className="lift-list" role="listbox" aria-label="Exercises" ref={listRef} onKeyDown={handleKey}>
        {visible.map(({ key, ex, name, equipment, sessions }, i) => {
          const trend = trendOf(ex)
          const active = key === selected
          return (
            <li key={key} style={{ animationDelay: `${Math.min(i, 10) * 0.035}s` }}>
              <button
                type="button"
                role="option"
                aria-selected={active}
                data-key={key}
                tabIndex={active ? 0 : -1}
                className={`lift-row${active ? ' is-active' : ''}`}
                onClick={() => onSelect(key)}
              >
                <span className="lift-row-name">
                  {name}
                  <span className="lift-row-sub mono">{equipment ? `${equipment} · ` : ''}{sessions} sess</span>
                </span>
                <span className="lift-row-right">
                  <span className="lift-row-value mono">{lbs(currentE1rm(ex))}</span>
                  <span className={`lift-row-delta mono trend-${trend}`}>
                    {trend === 'unknown' ? 'new' : signedPct(ex.plateau.percent_change)}
                  </span>
                </span>
                <Sparkline values={ex.trajectory.map((p) => p.rolling_e1rm)} flat={trend === 'flat'} />
              </button>
            </li>
          )
        })}
        {visible.length === 0 && <li className="lift-empty">No lifts match “{query}”</li>}
      </ul>
    </aside>
  )
}

export default ExerciseList
