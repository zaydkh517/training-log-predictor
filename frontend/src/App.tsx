import { useMemo, useState } from 'react'
import UploadForm from './components/UploadForm'
import ExerciseList from './components/ExerciseList'
import LiftHeader from './components/LiftHeader'
import TrajectoryChart from './components/TrajectoryChart'
import SessionPlannerForm from './components/SessionPlannerForm'
import type { Analysis } from './types'
import { currentE1rm } from './lib/format'

function pickDefault(analysis: Analysis) {
  // open on the most-logged lift, so there's something to look at right away
  return Object.entries(analysis.exercises).sort((a, b) => b[1].trajectory.length - a[1].trajectory.length)[0]?.[0] ?? null
}

function App() {
  const [analysis, setAnalysis] = useState<Analysis | null>(null)
  const [source, setSource] = useState<'upload' | 'sample'>('upload')
  const [selected, setSelected] = useState<string | null>(null)
  const [showUpload, setShowUpload] = useState(true)

  function handleAnalyzed(data: Analysis, from: 'upload' | 'sample') {
    setAnalysis(data)
    setSource(from)
    setSelected(pickDefault(data))
    setShowUpload(false)
    window.scrollTo({ top: 0 })
  }

  const summary = useMemo(() => {
    if (!analysis) return null
    const days = new Set<string>()
    for (const ex of Object.values(analysis.exercises)) ex.trajectory.forEach((p) => days.add(p.date))
    return { lifts: Object.keys(analysis.exercises).length, days: days.size }
  }, [analysis])

  const current = analysis && selected ? analysis.exercises[selected] : null
  const e1rm = current ? currentE1rm(current) : undefined

  return (
    <div className="shell">
      <nav className="topbar">
        <a className="brand" href="/" onClick={(e) => { e.preventDefault(); if (analysis) setShowUpload(false) }}>
          <span className="brand-mark" aria-hidden><i /><i /><i /></span>
          <span className="brand-name">Training Log <b>Predictor</b></span>
        </a>
        {analysis && !showUpload && summary && (
          <div className="topbar-right">
            <span className="pill mono">
              <span className={`pill-dot${source === 'sample' ? ' is-sample' : ''}`} />
              {source === 'sample' ? 'Sample data' : 'Your data'} · {summary.lifts} lifts · {summary.days} training days
            </span>
            <button type="button" className="btn btn-ghost btn-sm" onClick={() => setShowUpload(true)}>New upload</button>
          </div>
        )}
      </nav>

      {showUpload || !analysis ? (
        <main className="page page-landing">
          <UploadForm onAnalyzed={handleAnalyzed} onCancel={analysis ? () => setShowUpload(false) : undefined} />
        </main>
      ) : (
        <main className="page dashboard">
          <ExerciseList analysis={analysis} selected={selected ?? ''} onSelect={setSelected} />
          {current && selected && (
            <div className="lift-detail">
              <LiftHeader exercise={selected} data={current} />
              <TrajectoryChart exercise={selected} data={current} />
              {e1rm !== undefined && <SessionPlannerForm key={selected} exercise={selected} rollingE1rm={e1rm} repModel={analysis.rep_model} />}
            </div>
          )}
        </main>
      )}
    </div>
  )
}

export default App
