import { useRef, useState } from 'react'
import { API_URL } from '../lib/format'
import type { Analysis } from '../types'

type Props = {
  onAnalyzed: (data: Analysis, source: 'upload' | 'sample') => void
  onCancel?: () => void
}

type Source = 'strong' | 'hevy'

const SOURCES: { key: Source; app: string; hint: string }[] = [
  { key: 'strong', app: 'Strong', hint: 'Settings → Export Data' },
  { key: 'hevy', app: 'Hevy', hint: 'Settings → Export & Import Data' },
]

function DropZone({ app, hint, file, onFile }: {
  app: string
  hint: string
  file: File | null
  onFile: (f: File | null) => void
}) {
  const inputRef = useRef<HTMLInputElement>(null)
  const [dragging, setDragging] = useState(false)

  return (
    <label
      className={`dropzone${dragging ? ' is-dragging' : ''}${file ? ' has-file' : ''}`}
      onDragOver={(e) => { e.preventDefault(); setDragging(true) }}
      onDragLeave={() => setDragging(false)}
      onDrop={(e) => {
        e.preventDefault()
        setDragging(false)
        const dropped = e.dataTransfer.files?.[0]
        if (dropped) onFile(dropped)
      }}
    >
      <input
        ref={inputRef}
        type="file"
        accept=".csv"
        className="visually-hidden"
        onChange={(e) => onFile(e.target.files?.[0] ?? null)}
      />
      <span className="dropzone-app">{app}</span>
      {file ? (
        <>
          <span className="dropzone-file mono">{file.name}</span>
          <span className="dropzone-meta mono">{(file.size / 1024).toFixed(1)} KB · ready</span>
          <button
            type="button"
            className="dropzone-clear"
            aria-label={`Remove ${app} file`}
            onClick={(e) => {
              e.preventDefault()
              onFile(null)
              if (inputRef.current) inputRef.current.value = ''
            }}
          >
            ✕
          </button>
        </>
      ) : (
        <>
          <span className="dropzone-cta">Drop CSV or <u>browse</u></span>
          <span className="dropzone-meta">{hint}</span>
        </>
      )}
    </label>
  )
}

function UploadForm({ onAnalyzed, onCancel }: Props) {
  const [files, setFiles] = useState<Record<Source, File | null>>({ strong: null, hevy: null })
  const [loading, setLoading] = useState<false | 'upload' | 'sample'>(false)
  const [error, setError] = useState<string | null>(null)

  async function analyze(form: FormData, source: 'upload' | 'sample') {
    setLoading(source)
    setError(null)
    try {
      const res = await fetch(`${API_URL}/analyze`, { method: 'POST', body: form })
      const data = await res.json().catch(() => ({}))
      if (!res.ok) {
        setError(typeof data.detail === 'string' ? data.detail : 'Something went wrong analyzing that file.')
        return
      }
      onAnalyzed(data as Analysis, source)
    } catch {
      setError("Couldn't reach the analysis server. If it's been idle it may be waking up, so try again in a few seconds.")
    } finally {
      setLoading(false)
    }
  }

  function handleSubmit(e: React.FormEvent) {
    e.preventDefault()
    const form = new FormData()
    if (files.strong) form.append('strong_file', files.strong)
    if (files.hevy) form.append('hevy_file', files.hevy)
    analyze(form, 'upload')
  }

  async function handleSample() {
    setLoading('sample')
    setError(null)
    try {
      const blob = await (await fetch('/sample-strong.csv')).blob()
      const form = new FormData()
      form.append('strong_file', new File([blob], 'sample-strong.csv', { type: 'text/csv' }))
      await analyze(form, 'sample')
    } catch {
      setError("Couldn't load the sample dataset.")
      setLoading(false)
    }
  }

  const hasFile = Boolean(files.strong || files.hevy)

  return (
    <section className="hero">
      <div className="hero-copy">
        <p className="eyebrow">Strength forecasting for Strong + Hevy</p>
        <h1 className="hero-title">
          Know where your lifts<br />are <span className="accent">headed.</span>
        </h1>
        <p className="hero-sub">
          Drop in your workout export. Get an estimated 1RM trend for every lift, plateau
          calls, a 3-month outlook, and set-by-set weights for your next session.
        </p>

        <form className="upload" onSubmit={handleSubmit}>
          <div className="dropzones">
            {SOURCES.map((s) => (
              <DropZone
                key={s.key}
                app={s.app}
                hint={s.hint}
                file={files[s.key]}
                onFile={(f) => setFiles((prev) => ({ ...prev, [s.key]: f }))}
              />
            ))}
          </div>

          <div className="upload-actions">
            <button type="submit" className="btn btn-primary" disabled={!hasFile || loading !== false}>
              {loading === 'upload' ? <span className="loading-dots">Analyzing</span> : <>Analyze history <span aria-hidden>→</span></>}
            </button>
            <button type="button" className="btn btn-ghost" onClick={handleSample} disabled={loading !== false}>
              {loading === 'sample' ? <span className="loading-dots">Loading sample</span> : 'Try sample data'}
            </button>
            {onCancel && (
              <button type="button" className="btn btn-text" onClick={onCancel} disabled={loading !== false}>
                Back to dashboard
              </button>
            )}
          </div>
          {error && <p className="error-text" role="alert">{error}</p>}
          <p className="fine-print">Files are processed in memory and never stored.</p>
        </form>
      </div>

      <HeroGraphic />

      <ul className="spec-row">
        <li><span className="spec-k mono">e1RM</span><span>Epley estimate per session, 28-day rolling average</span></li>
        <li><span className="spec-k mono">PLATEAU</span><span>Flags lifts under +2% across the last 4 sessions</span></li>
        <li><span className="spec-k mono">OUTLOOK</span><span>Damped trend, capped by training-age growth limits</span></li>
        <li><span className="spec-k mono">PLANNER</span><span>Ridge model predicts reps set by set to hit your range</span></li>
      </ul>
    </section>
  )
}

/** Illustrative trend + projection band, drawn on load. */
function HeroGraphic() {
  return (
    <figure className="hero-graphic" aria-hidden>
      <div className="hero-graphic-head mono">
        <span>SQUAT · e1RM</span>
        <span className="accent">▲ 6.2%</span>
      </div>
      <div className="hero-graphic-value">
        <span className="display">315</span><span className="unit mono">LB</span>
      </div>
      <svg viewBox="0 0 400 200" preserveAspectRatio="none">
        <defs>
          <linearGradient id="heroFill" x1="0" x2="0" y1="0" y2="1">
            <stop offset="0" stopColor="var(--accent)" stopOpacity="0.28" />
            <stop offset="1" stopColor="var(--accent)" stopOpacity="0" />
          </linearGradient>
          <pattern id="heroHatch" width="6" height="6" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
            <line x1="0" y1="0" x2="0" y2="6" stroke="var(--accent)" strokeOpacity="0.35" strokeWidth="2" />
          </pattern>
        </defs>
        {[40, 80, 120, 160].map((y) => (
          <line key={y} x1="0" x2="400" y1={y} y2={y} stroke="var(--line)" />
        ))}
        <path className="hero-band" d="M290 78 L400 30 L400 70 L290 78 Z" fill="url(#heroHatch)" />
        <path
          d="M0 170 C40 165 60 150 90 146 S140 128 170 120 S220 104 250 92 S280 80 290 78 L290 200 L0 200 Z"
          fill="url(#heroFill)"
          className="hero-fill"
        />
        <path
          className="hero-line"
          d="M0 170 C40 165 60 150 90 146 S140 128 170 120 S220 104 250 92 S280 80 290 78"
          fill="none"
          stroke="var(--accent)"
          strokeWidth="3"
        />
        {[[20, 176], [55, 158], [80, 152], [112, 132], [140, 134], [168, 114], [196, 116], [226, 96], [254, 98], [280, 76]].map(([x, y], i) => (
          <circle key={i} cx={x} cy={y} r="3" fill="var(--text-dim)" className="hero-dot" style={{ animationDelay: `${0.3 + i * 0.06}s` }} />
        ))}
        <line x1="290" x2="290" y1="0" y2="200" stroke="var(--line-strong)" strokeDasharray="3 4" />
      </svg>
      <figcaption className="hero-graphic-foot mono">
        <span>MAR</span><span>JUN</span><span>SEP</span><span className="accent">+3 MO</span>
      </figcaption>
    </figure>
  )
}

export default UploadForm
