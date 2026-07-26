import { useState } from 'react'
import UploadForm from './components/UploadForm'

function App() {
  const [analysis, setAnalysis] = useState<any>(null)

  return (
    <div>
      <h1>Training Log Progression Predictor</h1>
      <UploadForm onAnalyzed={setAnalysis} />
      {analysis && <pre>{JSON.stringify(analysis, null, 2)}</pre>}
    </div>
  )
}

export default App
