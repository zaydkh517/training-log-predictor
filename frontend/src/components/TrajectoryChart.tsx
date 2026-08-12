import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts'

type Props = {
  exercise: string
  trajectory: { date: string; e1rm: number; rolling_e1rm: number }[]
  plateau: { is_plateau?: boolean; percent_change?: number }
  outlook: {
    current_smoothed_e1rm_lbs?: number
    outlook_3mo_range_lbs?: [number, number]
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

  if (outlook.outlook_3mo_range_lbs && outlook.current_smoothed_e1rm_lbs !== undefined) {
    const [low, high] = outlook.outlook_3mo_range_lbs
    const anchor = outlook.current_smoothed_e1rm_lbs

    data[data.length - 1] = { ...data[data.length - 1], low: anchor, high: anchor }
    data.push({ date: '+3 months', low, high })
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
          <CartesianGrid strokeDasharray="3 3" stroke="#262c35" />
          <XAxis dataKey="date" tick={{ fill: '#9aa3af', fontSize: 12 }} stroke="#333b47" />
          <YAxis tick={{ fill: '#9aa3af', fontSize: 12 }} stroke="#333b47" />
          <Tooltip
            contentStyle={{ background: '#171b21', border: '1px solid #333b47', borderRadius: 8 }}
            labelStyle={{ color: '#e8eaed' }}
          />
          <Line type="monotone" dataKey="e1rm" stroke="#5c6674" dot={false} />
          <Line type="monotone" dataKey="rolling_e1rm" stroke="#f2a33c" strokeWidth={2} dot={false} />
          <Line type="monotone" dataKey="low" stroke="#f2a33c" strokeOpacity={0.5} strokeDasharray="5 5" dot={false} />
          <Line type="monotone" dataKey="high" stroke="#f2a33c" strokeOpacity={0.5} strokeDasharray="5 5" dot={false} />
        </LineChart>
      </ResponsiveContainer>
    </div>
  )
}

export default TrajectoryChart