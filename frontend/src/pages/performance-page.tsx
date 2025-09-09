import { useState, useEffect } from 'react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, AreaChart, Area } from 'recharts'
import { DollarSign, TrendingUp, TrendingDown, Target, Activity, Brain } from 'lucide-react'

interface PerformanceDataPoint {
  date: string
  profit_loss: number
  running_balance: number
  cumulative_profit: number
}

interface BrierScoreDataPoint {
  date: string
  avg_brier_score: number
  prediction_count: number
}

interface PerformanceHistory {
  data_points: PerformanceDataPoint[]
  brier_score_points: BrierScoreDataPoint[]
  total_profit_loss: number
  total_bets: number
  win_rate: number
  roi: number
  best_day: number
  worst_day: number
  avg_brier_score: number
  total_predictions_scored: number
}

export function PerformancePage() {
  const [performanceData, setPerformanceData] = useState<PerformanceHistory | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    fetchPerformanceData()
  }, [])

  const fetchPerformanceData = async () => {
    try {
      setLoading(true)
      const response = await fetch('http://localhost:8001/api/performance/history')
      if (!response.ok) {
        throw new Error('Failed to fetch performance data')
      }
      const data = await response.json()
      setPerformanceData(data)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An error occurred')
    } finally {
      setLoading(false)
    }
  }

  const formatDate = (dateString: string) => {
    const date = new Date(dateString)
    return date.toLocaleDateString('en-US', { month: 'short', day: 'numeric' })
  }

  const formatCurrency = (value: number) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
    }).format(value)
  }

  const formatPercentage = (value: number) => {
    return `${value.toFixed(1)}%`
  }

  const formatBrierScore = (value: number) => {
    return value.toFixed(3)
  }

  const getBrierScoreRating = (score: number) => {
    if (score <= 0.15) return { rating: 'Excellent', color: 'text-green-600' }
    if (score <= 0.20) return { rating: 'Good', color: 'text-blue-600' }
    if (score <= 0.25) return { rating: 'Fair', color: 'text-yellow-600' }
    return { rating: 'Poor', color: 'text-red-600' }
  }

  if (loading) {
    return (
      <div className="container mx-auto p-6">
        <div className="flex items-center justify-center h-64">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary"></div>
        </div>
      </div>
    )
  }

  if (error) {
    return (
      <div className="container mx-auto p-6">
        <Card className="border-destructive">
          <CardContent className="p-6">
            <p className="text-destructive">Error loading performance data: {error}</p>
          </CardContent>
        </Card>
      </div>
    )
  }

  if (!performanceData) {
    return null
  }

  // Transform data for charts
  const chartData = performanceData.data_points.map(point => ({
    date: formatDate(point.date),
    balance: point.running_balance,
    cumulative: point.cumulative_profit,
    dailyPL: point.profit_loss,
  }))

  const brierChartData = performanceData.brier_score_points.map(point => ({
    date: formatDate(point.date),
    brierScore: point.avg_brier_score,
    predictionCount: point.prediction_count,
  }))

  const brierRating = getBrierScoreRating(performanceData.avg_brier_score)

  return (
    <div className="container mx-auto p-6 space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold tracking-tight">Performance Analysis</h1>
          <p className="text-muted-foreground">
            Track your betting performance and profitability over time
          </p>
        </div>
      </div>

      {/* Performance Stats Cards */}
      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-6">
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Total P&L</CardTitle>
            <DollarSign className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className={`text-2xl font-bold ${performanceData.total_profit_loss >= 0 ? 'text-green-600' : 'text-red-600'}`}>
              {formatCurrency(performanceData.total_profit_loss)}
            </div>
            <p className="text-xs text-muted-foreground">
              {performanceData.total_bets} total bets
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Win Rate</CardTitle>
            <Target className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">
              {formatPercentage(performanceData.win_rate)}
            </div>
            <p className="text-xs text-muted-foreground">
              Success percentage
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">ROI</CardTitle>
            <Activity className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className={`text-2xl font-bold ${performanceData.roi >= 0 ? 'text-green-600' : 'text-red-600'}`}>
              {formatPercentage(performanceData.roi)}
            </div>
            <p className="text-xs text-muted-foreground">
              Return on investment
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Model Accuracy</CardTitle>
            <Brain className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className={`text-2xl font-bold ${brierRating.color}`}>
              {formatBrierScore(performanceData.avg_brier_score)}
            </div>
            <p className="text-xs text-muted-foreground">
              Brier Score ({brierRating.rating})
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Best Day</CardTitle>
            <TrendingUp className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold text-green-600">
              {formatCurrency(performanceData.best_day)}
            </div>
            <p className="text-xs text-muted-foreground">
              Highest daily profit
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Worst Day</CardTitle>
            <TrendingDown className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold text-red-600">
              {formatCurrency(performanceData.worst_day)}
            </div>
            <p className="text-xs text-muted-foreground">
              Biggest daily loss
            </p>
          </CardContent>
        </Card>
      </div>

      {/* Charts */}
      <div className="grid gap-6 md:grid-cols-1 lg:grid-cols-2">
        {/* Balance Chart */}
        <Card>
          <CardHeader>
            <CardTitle>Bankroll Balance</CardTitle>
            <CardDescription>
              Your account balance over time
            </CardDescription>
          </CardHeader>
          <CardContent>
            <ResponsiveContainer width="100%" height={300}>
              <AreaChart data={chartData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis 
                  dataKey="date" 
                  tick={{ fontSize: 12 }}
                  tickLine={{ stroke: '#666' }}
                />
                <YAxis 
                  tick={{ fontSize: 12 }}
                  tickLine={{ stroke: '#666' }}
                  tickFormatter={(value) => `$${value}`}
                />
                <Tooltip 
                  formatter={(value: number) => [formatCurrency(value), 'Balance']}
                  labelFormatter={(label) => `Date: ${label}`}
                  contentStyle={{
                    backgroundColor: 'hsl(var(--popover))',
                    border: '1px solid hsl(var(--border))',
                    borderRadius: '6px',
                  }}
                />
                <Area 
                  type="monotone" 
                  dataKey="balance" 
                  stroke="hsl(var(--primary))" 
                  fill="hsl(var(--primary))" 
                  fillOpacity={0.1}
                  strokeWidth={2}
                />
              </AreaChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>

        {/* Cumulative Profit Chart */}
        <Card>
          <CardHeader>
            <CardTitle>Cumulative Profit/Loss</CardTitle>
            <CardDescription>
              Your total profit or loss accumulation
            </CardDescription>
          </CardHeader>
          <CardContent>
            <ResponsiveContainer width="100%" height={300}>
              <LineChart data={chartData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis 
                  dataKey="date" 
                  tick={{ fontSize: 12 }}
                  tickLine={{ stroke: '#666' }}
                />
                <YAxis 
                  tick={{ fontSize: 12 }}
                  tickLine={{ stroke: '#666' }}
                  tickFormatter={(value) => `$${value}`}
                />
                <Tooltip 
                  formatter={(value: number) => [formatCurrency(value), 'Cumulative P&L']}
                  labelFormatter={(label) => `Date: ${label}`}
                  contentStyle={{
                    backgroundColor: 'hsl(var(--popover))',
                    border: '1px solid hsl(var(--border))',
                    borderRadius: '6px',
                  }}
                />
                <Line 
                  type="monotone" 
                  dataKey="cumulative" 
                  stroke={performanceData.total_profit_loss >= 0 ? '#22c55e' : '#ef4444'}
                  strokeWidth={3}
                  dot={{ fill: performanceData.total_profit_loss >= 0 ? '#22c55e' : '#ef4444', strokeWidth: 2, r: 4 }}
                />
              </LineChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>
      </div>

      {/* Brier Score Chart - Full Width */}
      <Card>
        <CardHeader>
          <CardTitle>Model Accuracy (Brier Score)</CardTitle>
          <CardDescription>
            Scientific measure of prediction accuracy over time. Lower scores indicate better model performance.
            Scores: Excellent (≤0.15), Good (≤0.20), Fair (≤0.25), Poor ({'>'}0.25)
          </CardDescription>
        </CardHeader>
        <CardContent>
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={brierChartData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis 
                dataKey="date" 
                tick={{ fontSize: 12 }}
                tickLine={{ stroke: '#666' }}
              />
              <YAxis 
                tick={{ fontSize: 12 }}
                tickLine={{ stroke: '#666' }}
                domain={[0, 0.5]}
                tickFormatter={(value) => value.toFixed(2)}
              />
              <Tooltip 
                formatter={(value: number, name: string) => {
                  if (name === 'brierScore') {
                    const rating = getBrierScoreRating(value)
                    return [formatBrierScore(value) + ` (${rating.rating})`, 'Brier Score']
                  }
                  return [value, name]
                }}
                labelFormatter={(label) => `Date: ${label}`}
                contentStyle={{
                  backgroundColor: 'hsl(var(--popover))',
                  border: '1px solid hsl(var(--border))',
                  borderRadius: '6px',
                }}
              />
              <Line 
                type="monotone" 
                dataKey="brierScore" 
                stroke="#8b5cf6"
                strokeWidth={3}
                dot={{ fill: '#8b5cf6', strokeWidth: 2, r: 4 }}
              />
            </LineChart>
          </ResponsiveContainer>
          <div className="mt-4 text-sm text-muted-foreground">
            {performanceData.total_predictions_scored > 0 ? (
              `Average Brier Score: ${formatBrierScore(performanceData.avg_brier_score)} (${brierRating.rating}) • ${performanceData.total_predictions_scored} predictions scored`
            ) : (
              'No predictions have been scored yet. Predictions are automatically scored after games are completed.'
            )}
          </div>
        </CardContent>
      </Card>

      {/* Performance Insights */}
      <Card>
        <CardHeader>
          <CardTitle>Performance Insights</CardTitle>
          <CardDescription>
            AI-powered analysis of your betting performance with V3 ensemble model metrics
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          {performanceData.total_bets === 0 ? (
            <div className="text-center py-8">
              <Activity className="h-12 w-12 mx-auto text-muted-foreground mb-4" />
              <h3 className="text-lg font-semibold mb-2">No Betting History Yet</h3>
              <p className="text-muted-foreground">
                Start placing bets to see your performance analysis here. 
                The V3 ensemble system will track your progress and provide detailed insights.
              </p>
            </div>
          ) : (
            <div className="grid gap-4 md:grid-cols-3">
              <div className="space-y-2">
                <h4 className="font-semibold text-green-600">Strengths</h4>
                <ul className="text-sm space-y-1">
                  {performanceData.win_rate > 55 && (
                    <li>• Strong win rate above 55%</li>
                  )}
                  {performanceData.roi > 5 && (
                    <li>• Excellent ROI performance</li>
                  )}
                  {performanceData.avg_brier_score <= 0.20 && performanceData.total_predictions_scored > 0 && (
                    <li>• High model accuracy (Brier ≤ 0.20)</li>
                  )}
                  {performanceData.best_day > 50 && (
                    <li>• Capable of significant daily profits</li>
                  )}
                  {performanceData.win_rate <= 55 && performanceData.roi <= 5 && performanceData.best_day <= 50 && (
                    <li>• Consistent betting approach</li>
                  )}
                </ul>
              </div>
              <div className="space-y-2">
                <h4 className="font-semibold text-amber-600">Areas for Improvement</h4>
                <ul className="text-sm space-y-1">
                  {performanceData.win_rate < 50 && (
                    <li>• Focus on improving win rate</li>
                  )}
                  {performanceData.roi < 0 && (
                    <li>• Review bankroll management strategy</li>
                  )}
                  {performanceData.avg_brier_score > 0.25 && performanceData.total_predictions_scored > 0 && (
                    <li>• Model accuracy needs improvement</li>
                  )}
                  {Math.abs(performanceData.worst_day) > performanceData.best_day && (
                    <li>• Consider reducing position sizes</li>
                  )}
                  {performanceData.total_bets < 20 && (
                    <li>• Build larger sample size for better analysis</li>
                  )}
                </ul>
              </div>
              <div className="space-y-2">
                <h4 className="font-semibold text-blue-600">V3 Ensemble Model</h4>
                <ul className="text-sm space-y-1">
                  <li>• LightGBM + XGBoost ensemble</li>
                  <li>• Advanced feature engineering</li>
                  <li>• Automated Brier score tracking</li>
                  {performanceData.total_predictions_scored > 0 ? (
                    <li>• {performanceData.total_predictions_scored} predictions scored</li>
                  ) : (
                    <li>• Automated scoring pending</li>
                  )}
                  <li>• Daily performance auditing</li>
                </ul>
              </div>
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  )
}