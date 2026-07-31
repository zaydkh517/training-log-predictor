import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts'

type Props = {
  exercise: string
  trajectory: { date: string; e1rm: number; rolling_e1rm: number }[]
  plateau: { is_plateau?: boolean; percent_change?: number }
  outlook: {
    current_smoothed_e1rm_lbs?: number
    outlook_6mo_range_lbs?: [number, number]
    error?: string
  }
}

type ChartPoint = {
  date: string
  e1rm?: number
  rolling_e1rm?: number
  low?: number
  high?: number
}

function buildChartData(trajectory: Props['trajectory'], outlook: Props['outlook']): ChartPoint[] {
  const data: ChartPoint[] = [...trajectory]

  if (outlook.outlook_6mo_range_lbs && outlook.current_smoothed_e1rm_lbs !== undefined) {
    const [low, high] = outlook.outlook_6mo_range_lbs
    const anchor = outlook.current_smoothed_e1rm_lbs

    data[data.length - 1] = { ...data[data.length - 1], low: anchor, high: anchor }
    data.push({ date: '+6 months', low, high })
  }

  return data
}

function TrajectoryChart({ exercise, trajectory, plateau, outlook }: Props) {
  const chartData = buildChartData(trajectory, outlook)

  return (
    <div>
      <h2>{exercise}</h2>
      {plateau.is_plateau && <span className="badge badge--warning">Plateaued</span>}
      <ResponsiveContainer width="100%" height={300}>
        <LineChart data={chartData}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="date" />
          <YAxis />
          <Tooltip />
          <Line type="monotone" dataKey="e1rm" stroke="#8884d8" />
          <Line type="monotone" dataKey="rolling_e1rm" stroke="#82ca9d" />
          <Line type="monotone" dataKey="low" stroke="#f4a623" strokeDasharray="5 5" />
          <Line type="monotone" dataKey="high" stroke="#f4a623" strokeDasharray="5 5" />
        </LineChart>
      </ResponsiveContainer>
    </div>
  )
}

export default TrajectoryChart