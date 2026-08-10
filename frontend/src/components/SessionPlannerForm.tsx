import { useState } from 'react'
const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'

type Props = {
  exercise: string
  rollingE1rm: number
}

function SessionPlannerForm({ exercise, rollingE1rm }: Props) {
  const [totalSets, setTotalSets] = useState('3')
  const [firstSetWeight, setFirstSetWeight] = useState('')
  const [targetMinReps, setTargetMinReps] = useState('6')
  const [targetMaxReps, setTargetMaxReps] = useState('10')
  const [result, setResult] = useState<any>(null)

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault()

    const body = {
      exercise,
      rolling_e1rm: rollingE1rm,
      total_sets: Number(totalSets),
      first_set_weight: Number(firstSetWeight),
      target_min_reps: Number(targetMinReps),
      target_max_reps: Number(targetMaxReps),
    }

    const res = await fetch(`${API_URL}/plan-session`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    })
    const data = await res.json()
    setResult(data)
  }

  return (
    <div>
      <h2>Plan a session: {exercise}</h2>
      <form onSubmit={handleSubmit}>
        <div className="field">
          <label>Total sets: </label>
          <input type="number" value={totalSets} onChange={(e) => setTotalSets(e.target.value)} min="1" required />
        </div>
        <div className="field">
          <label>Starting weight (lbs): </label>
          <input type="number" value={firstSetWeight} onChange={(e) => setFirstSetWeight(e.target.value)} min="0" required />
        </div>
        <div className="field">
          <label>Target rep range: </label>
          <input type="number" value={targetMinReps} onChange={(e) => setTargetMinReps(e.target.value)} min="1" required />
          {' to '}
          <input type="number" value={targetMaxReps} onChange={(e) => setTargetMaxReps(e.target.value)} min="1" required />
        </div>
        <button type="submit">Plan session</button>
      </form>

      {result && result.detail && <p className="error-text">Error: {result.detail[0].msg}</p>}
      {result && result.error && <p className="error-text">Error: {result.error}</p>}
      {result && result.sets && (
        <ul className="set-list">
          {result.sets.map((set: any) => (
            <li key={set.set_number} className={set.weight == null ? 'set-row set-row--note' : 'set-row'}>
              <span>Set {set.set_number}</span>
              <span>{set.weight != null ? `${set.weight} lbs × ${set.predicted_reps} reps` : set.note}</span>
              {set.weight != null && set.note && <span className="set-row-note">{set.note}</span>}
            </li>
          ))}
        </ul>
      )}
    </div>
  )
}

export default SessionPlannerForm