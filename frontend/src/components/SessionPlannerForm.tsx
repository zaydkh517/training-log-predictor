import { useState } from 'react'
import { API_URL, lbs } from '../lib/format'
import type { PlannedSet, RepModelParams } from '../types'

type Props = {
  exercise: string
  rollingE1rm: number
  repModel?: RepModelParams
}

function Stepper({ label, value, onChange, step, min, max, unit }: {
  label: string
  value: string
  onChange: (v: string) => void
  step: number
  min: number
  max?: number
  unit?: string
}) {
  const bump = (dir: 1 | -1) => {
    const n = (Number(value) || 0) + dir * step
    onChange(String(Math.max(min, max !== undefined ? Math.min(max, n) : n)))
  }
  return (
    <div className="field">
      <label>{label}</label>
      <div className="stepper">
        <button type="button" onClick={() => bump(-1)} aria-label={`Decrease ${label}`}>−</button>
        <input
          type="number"
          inputMode="decimal"
          value={value}
          min={min}
          max={max}
          onChange={(e) => onChange(e.target.value)}
          required
        />
        {unit && <span className="stepper-unit mono">{unit}</span>}
        <button type="button" onClick={() => bump(1)} aria-label={`Increase ${label}`}>+</button>
      </div>
    </div>
  )
}

function SessionPlannerForm({ exercise, rollingE1rm, repModel }: Props) {
  // default to ~75% of e1RM: a typical working weight, so the first click is useful
  const [totalSets, setTotalSets] = useState('4')
  const [firstSetWeight, setFirstSetWeight] = useState(String(5 * Math.round((rollingE1rm * 0.75) / 5)))
  const [targetMinReps, setTargetMinReps] = useState('6')
  const [targetMaxReps, setTargetMaxReps] = useState('10')
  const [sets, setSets] = useState<PlannedSet[] | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [loading, setLoading] = useState(false)

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault()
    setLoading(true)
    setError(null)
    try {
      const res = await fetch(`${API_URL}/plan-session`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          exercise,
          rolling_e1rm: rollingE1rm,
          total_sets: Number(totalSets),
          first_set_weight: Number(firstSetWeight),
          target_min_reps: Number(targetMinReps),
          target_max_reps: Number(targetMaxReps),
          rep_model: repModel,
        }),
      })
      const data = await res.json()
      if (Array.isArray(data.detail)) {
        setError(String(data.detail[0]?.msg ?? 'Invalid input').replace(/^Value error, /, ''))
        setSets(null)
      } else if (data.error) {
        setError(data.error)
        setSets(null)
      } else {
        setSets(data.sets)
      }
    } catch {
      setError("Couldn't reach the planner. Try again in a moment.")
    } finally {
      setLoading(false)
    }
  }

  const minR = Number(targetMinReps)
  const maxR = Number(targetMaxReps)
  const heaviest = Math.max(rollingE1rm, ...(sets ?? []).map((s) => s.weight ?? 0))
  const volume = (sets ?? []).reduce((sum, s) => sum + (s.weight ?? 0) * (s.predicted_reps ?? 0), 0)

  return (
    <section className="panel planner">
      <div className="panel-head">
        <div>
          <h2 className="label">Next session planner</h2>
          <p className="panel-sub">
            Set 1 uses the Epley formula. Each later set carries your previous set&rsquo;s strength to the new
            weight, then applies a learned fatigue correction{' '}
            {repModel?.source === 'personal'
              ? <>fitted to <b className="accent">{repModel.n_sets} of your sets</b>.</>
              : <>from the default model (log more sets to personalize it).</>}
          </p>
        </div>
      </div>

      <div className="planner-grid">
        <form className="planner-form" onSubmit={handleSubmit}>
          <Stepper label="Working sets" value={totalSets} onChange={setTotalSets} step={1} min={1} max={10} />
          <Stepper label="Starting weight" value={firstSetWeight} onChange={setFirstSetWeight} step={5} min={5} unit="LB" />
          <div className="field">
            <label>Target reps</label>
            <div className="rep-range">
              <input type="number" inputMode="numeric" aria-label="Minimum reps" value={targetMinReps} min={1} onChange={(e) => setTargetMinReps(e.target.value)} required />
              <span className="mono dim">to</span>
              <input type="number" inputMode="numeric" aria-label="Maximum reps" value={targetMaxReps} min={1} onChange={(e) => setTargetMaxReps(e.target.value)} required />
            </div>
          </div>
          <button type="submit" className="btn btn-primary btn-block" disabled={loading}>
            {loading ? <span className="loading-dots">Planning</span> : <>Build session <span aria-hidden>→</span></>}
          </button>
          <p className="fine-print mono">Based on e1RM {lbs(rollingE1rm, 1)} lb</p>
        </form>

        <div className="planner-output" aria-live="polite">
          {error && <p className="error-text" role="alert">{error}</p>}
          {!sets && !error && (
            <div className="planner-empty">
              <span className="display-sm dim">{totalSets || '–'} × {targetMinReps}–{targetMaxReps}</span>
              <p>Set your starting weight and rep target, then build the session to get weights for every set.</p>
            </div>
          )}
          {sets && (
            <>
              <ol className="set-list">
                {sets.map((s, i) => {
                  const failed = s.weight == null
                  const inRange = !failed && s.predicted_reps! >= minR && s.predicted_reps! <= maxR
                  return (
                    <li key={s.set_number} className={`set-row${failed ? ' is-failed' : ''}`} style={{ animationDelay: `${i * 0.06}s` }}>
                      <span className="set-num mono">{String(s.set_number).padStart(2, '0')}</span>
                      {failed ? (
                        <span className="set-note">{s.note}</span>
                      ) : (
                        <>
                          <span className="set-load">
                            <span className="set-weight"><b className="display-xs">{lbs(s.weight)}</b><span className="unit mono">LB</span></span>
                            <span className="set-bar"><i style={{ width: `${(s.weight! / heaviest) * 100}%` }} /></span>
                            <span className="set-pct mono">{Math.round((s.weight! / rollingE1rm) * 100)}% 1RM</span>
                          </span>
                          <span className={`set-reps${inRange ? '' : ' is-off'}`}>
                            <span className="mono dim">×</span><b className="display-xs">{s.predicted_reps}</b>
                          </span>
                          {s.note && <span className="set-note">{s.note}</span>}
                        </>
                      )}
                    </li>
                  )
                })}
              </ol>
              <div className="planner-total mono">
                <span>Predicted volume</span>
                <b>{lbs(volume)} lb</b>
              </div>
            </>
          )}
        </div>
      </div>
    </section>
  )
}

export default SessionPlannerForm
