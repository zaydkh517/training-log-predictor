import { useState } from 'react'
import UploadForm from './components/UploadForm'
import TrajectoryChart from './components/TrajectoryChart'
import SessionPlannerForm from './components/SessionPlannerForm'

function App() {
  const [analysis, setAnalysis] = useState<any>(null)
  const [selectedExercise, setSelectedExercise] = useState<string | null>(null)
  const [showUpload, setShowUpload] = useState(true)

  function handleAnalyzed(data: any) {
    setAnalysis(data)
    setShowUpload(false)
  }

  return (
    <div className="app">
      <header className="app-header">
        <h1>Training Log Progression Predictor</h1>
        <p className="app-subtitle">Upload your Strong or Hevy history to see where your lifts are headed.</p>
      </header>

      {showUpload ? (
        <div className="card">
          <UploadForm onAnalyzed={handleAnalyzed} />
        </div>
      ) : (
        <button className="link-button" onClick={() => setShowUpload(true)}>Upload different files</button>
      )}

      {analysis && (
        <select onChange={(e) => setSelectedExercise(e.target.value)}>
          <option value="">-- pick an exercise --</option>
          {Object.keys(analysis.exercises).map((exercise) => (
            <option key={exercise} value={exercise}>
              {exercise}
            </option>
          ))}
        </select>
      )}

      {analysis && selectedExercise && (
        <div className="card">
          <TrajectoryChart
            exercise={selectedExercise}
            trajectory={analysis.exercises[selectedExercise].trajectory}
            plateau={analysis.exercises[selectedExercise].plateau}
            outlook={analysis.exercises[selectedExercise].outlook}
          />
        </div>
      )}
      {analysis && selectedExercise && (
        <div className="card">
          <SessionPlannerForm
            exercise={selectedExercise}
            rollingE1rm={analysis.exercises[selectedExercise].outlook.current_smoothed_e1rm_lbs}
          />
        </div>
      )}
    </div>
  )
}

export default App
