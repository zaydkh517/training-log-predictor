import { useState } from 'react'
const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'

type Props = {
  onAnalyzed: (data: any) => void
}

function UploadForm({ onAnalyzed }: Props) {
  const [strongFile, setStrongFile] = useState<File | null>(null)
  const [hevyFile, setHevyFile] = useState<File | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault()
    setLoading(true)
    setError(null)

    const formData = new FormData()
    if (strongFile) formData.append('strong_file', strongFile)
    if (hevyFile) formData.append('hevy_file', hevyFile)

    const res = await fetch(`${API_URL}/analyze`, {
      method: 'POST',
      body: formData,
    })
    const data = await res.json()

    setLoading(false)

    if (!res.ok) {
      setError(data.detail ?? 'Something went wrong analyzing that file.')
      return
    }
    onAnalyzed(data)
  }

  return (
    <form onSubmit={handleSubmit}>
      <div className="field">
        <label>Strong CSV: </label>
        <input
          type="file"
          accept=".csv"
          onChange={(e) => setStrongFile(e.target.files?.[0] ?? null)}
        />
      </div>
      <div className="field">
        <label>Hevy CSV: </label>
        <input
          type="file"
          accept=".csv"
          onChange={(e) => setHevyFile(e.target.files?.[0] ?? null)}
        />
      </div>
      <button type="submit" disabled={loading}>
        {loading ? 'Analyzing...' : 'Analyze'}
      </button>
      {error && <p className="error-text">{error}</p>}
    </form>
  )
}

export default UploadForm
