import { useState } from 'react'
import UploadForm from './components/UploadForm'
import TrajectoryChart from './components/TrajectoryChart'

function App() {
  const [analysis, setAnalysis] = useState<any>(null)
  const [selectedExercise, setSelectedExercise] = useState<string | null>(null)

  return (
    <div>
      <h1>Training Log Progression Predictor</h1>
      <UploadForm onAnalyzed={setAnalysis} />
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
        <TrajectoryChart
          exercise={selectedExercise}
          trajectory={analysis.exercises[selectedExercise].trajectory}
          plateau={analysis.exercises[selectedExercise].plateau}
          outlook={analysis.exercises[selectedExercise].outlook}
        />
      )}
    </div>
  )
}

export default App
