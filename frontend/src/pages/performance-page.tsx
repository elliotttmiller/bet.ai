import { useState, useEffect } from 'react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, AreaChart, Area } from 'recharts'
import { DollarSign, TrendingUp, TrendingDown, Target, Activity, Brain, Award } from 'lucide-react'

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

interface EloDataPoint {
  date: string
  elo_rating: number
  team_name: string
  sport: string
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
  average_clv: number | null
  average_clv_by_sport: { [sport: string]: number }
  elo_history_sample: EloDataPoint[]
}

interface Team {
  team_id: number
  team_name: string
  sport: string
}

export function PerformancePage() {
  const [performanceData, setPerformanceData] = useState<PerformanceHistory | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [selectedTeam, setSelectedTeam] = useState<string>('Lakers')
  const [teams, setTeams] = useState<Team[]>([])

  useEffect(() => {
    fetchPerformanceData()
    fetchTeams()
  }, [])

  const fetchPerformanceData = async () => {
    try {
      setLoading(true)
      const response = await fetch('http://localhost:8000/api/performance/history')
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

  const fetchTeams = async () => {
    try {
      // For now, create mock teams based on the sample data we know exists
      const mockTeams: Team[] = [
        { team_id: 1, team_name: 'Lakers', sport: 'NBA' },
        { team_id: 2, team_name: 'Warriors', sport: 'NBA' },
        { team_id: 3, team_name: 'Celtics', sport: 'NBA' },
        { team_id: 4, team_name: 'Heat', sport: 'NBA' },
        { team_id: 5, team_name: 'Patriots', sport: 'NFL' },
        { team_id: 6, team_name: 'Bills', sport: 'NFL' },
        { team_id: 7, team_name: 'Chiefs', sport: 'NFL' },
        { team_id: 8, team_name: 'Cowboys', sport: 'NFL' },
        { team_id: 9, team_name: 'Yankees', sport: 'MLB' },
        { team_id: 10, team_name: 'Red Sox', sport: 'MLB' },
        { team_id: 11, team_name: 'Dodgers', sport: 'MLB' },
        { team_id: 12, team_name: 'Giants', sport: 'MLB' }
      ]
      setTeams(mockTeams)
    } catch (err) {
      console.error('Failed to fetch teams:', err)
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

  const formatCLV = (value: number) => {
    return `${value > 0 ? '+' : ''}${value.toFixed(1)}%`
  }

  const getCLVRating = (clv: number) => {
    if (clv >= 3) return { rating: 'Exceptional', color: 'text-green-600' }
    if (clv >= 1) return { rating: 'Good', color: 'text-blue-600' }
    if (clv >= 0) return { rating: 'Positive', color: 'text-green-500' }
    if (clv >= -2) return { rating: 'Fair', color: 'text-yellow-600' }
    return { rating: 'Poor', color: 'text-red-600' }
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

      {/* Performance Stats Cards - Now including CLV KPIs */}
      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
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
            <CardTitle className="text-sm font-medium">Average CLV</CardTitle>
            <Award className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className={`text-2xl font-bold ${performanceData.average_clv !== null ? (performanceData.average_clv >= 0 ? 'text-green-600' : 'text-red-600') : 'text-gray-500'}`}>
              {performanceData.average_clv !== null ? formatCLV(performanceData.average_clv) : 'N/A'}
            </div>
            <p className="text-xs text-muted-foreground">
              {performanceData.average_clv !== null ? getCLVRating(performanceData.average_clv).rating : 'No CLV data yet'}
            </p>
          </CardContent>
        </Card>
      </div>

      {/* CLV by Sport Cards */}
      {performanceData.average_clv_by_sport && Object.keys(performanceData.average_clv_by_sport).length > 0 && (
        <div>
          <h2 className="text-2xl font-bold tracking-tight mb-4">Closing Line Value by Sport</h2>
          <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
            {Object.entries(performanceData.average_clv_by_sport).map(([sport, clv]) => {
              const clvRating = getCLVRating(clv)
              const sportEmoji = sport === 'NBA' ? '🏀' : sport === 'NFL' ? '🏈' : '⚾'
              return (
                <Card key={sport}>
                  <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                    <CardTitle className="text-sm font-medium">{sport} CLV</CardTitle>
                    <span className="text-lg">{sportEmoji}</span>
                  </CardHeader>
                  <CardContent>
                    <div className={`text-2xl font-bold ${clvRating.color}`}>
                      {formatCLV(clv)}
                    </div>
                    <p className="text-xs text-muted-foreground">
                      {clvRating.rating} performance
                    </p>
                  </CardContent>
                </Card>
              )
            })}
          </div>
        </div>
      )}

      {/* Model Accuracy and Best/Worst Days */}
      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
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

      {/* Elo Rating Chart */}
      <Card>
        <CardHeader>
          <div className="flex items-center justify-between">
            <div>
              <CardTitle>Team Elo Rating Over Time</CardTitle>
              <CardDescription>
                Dynamic team strength ratings calculated by the V5 Elo Engine
              </CardDescription>
            </div>
            <Select value={selectedTeam} onValueChange={setSelectedTeam}>
              <SelectTrigger className="w-40">
                <SelectValue placeholder="Select team" />
              </SelectTrigger>
              <SelectContent>
                {teams.map((team) => (
                  <SelectItem key={team.team_id} value={team.team_name}>
                    {team.sport === 'NBA' ? '🏀' : team.sport === 'NFL' ? '🏈' : '⚾'} {team.team_name}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>
        </CardHeader>
        <CardContent>
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={performanceData.elo_history_sample.filter(point => point.team_name === selectedTeam).map(point => ({
              date: formatDate(point.date),
              elo: Math.round(point.elo_rating),
              team: point.team_name,
              sport: point.sport
            }))}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis 
                dataKey="date" 
                tick={{ fontSize: 12 }}
                tickLine={{ stroke: '#666' }}
              />
              <YAxis 
                tick={{ fontSize: 12 }}
                tickLine={{ stroke: '#666' }}
                domain={[1200, 1800]}
                tickFormatter={(value) => value.toString()}
              />
              <Tooltip 
                formatter={(value: number) => [value, 'Elo Rating']}
                labelFormatter={(label) => `Date: ${label}`}
                contentStyle={{
                  backgroundColor: 'hsl(var(--popover))',
                  border: '1px solid hsl(var(--border))',
                  borderRadius: '6px',
                }}
              />
              <Line 
                type="monotone" 
                dataKey="elo" 
                stroke="#f59e0b"
                strokeWidth={3}
                dot={{ fill: '#f59e0b', strokeWidth: 2, r: 4 }}
              />
            </LineChart>
          </ResponsiveContainer>
          <div className="mt-4 text-sm text-muted-foreground">
            {performanceData.elo_history_sample.length > 0 ? (
              `Dynamic Elo ratings track team strength over time. Higher ratings (>1600) indicate stronger teams. Current sample: ${performanceData.elo_history_sample.length} data points.`
            ) : (
              'No Elo rating history available. The Dynamic Elo Engine will populate this data as games are processed.'
            )}
          </div>
        </CardContent>
      </Card>

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
          <CardTitle>V5 Quantitative Performance Analysis</CardTitle>
          <CardDescription>
            AI-powered analysis with Dynamic Elo Engine, CLV validation, and enhanced ensemble models
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          {performanceData.total_bets === 0 ? (
            <div className="text-center py-8">
              <Activity className="h-12 w-12 mx-auto text-muted-foreground mb-4" />
              <h3 className="text-lg font-semibold mb-2">No Betting History Yet</h3>
              <p className="text-muted-foreground">
                Start placing bets to see your V5 quantitative performance analysis here. 
                The enhanced system will track CLV, Elo dynamics, and provide professional-grade insights.
              </p>
            </div>
          ) : (
            <div className="grid gap-4 md:grid-cols-3">
              <div className="space-y-2">
                <h4 className="font-semibold text-green-600">V5 Strengths</h4>
                <ul className="text-sm space-y-1">
                  {performanceData.win_rate > 55 && (
                    <li>• Strong win rate above 55%</li>
                  )}
                  {performanceData.roi > 5 && (
                    <li>• Excellent ROI performance</li>
                  )}
                  {performanceData.average_clv !== null && performanceData.average_clv > 1 && (
                    <li>• Positive CLV indicating sustainable edge</li>
                  )}
                  {performanceData.avg_brier_score <= 0.20 && performanceData.total_predictions_scored > 0 && (
                    <li>• High V5 ensemble model accuracy</li>
                  )}
                  {performanceData.best_day > 50 && (
                    <li>• Capable of significant daily profits</li>
                  )}
                  {performanceData.win_rate <= 55 && performanceData.roi <= 5 && performanceData.best_day <= 50 && (
                    <li>• Consistent V5 quantitative approach</li>
                  )}
                </ul>
              </div>
              <div className="space-y-2">
                <h4 className="font-semibold text-amber-600">Optimization Areas</h4>
                <ul className="text-sm space-y-1">
                  {performanceData.win_rate < 50 && (
                    <li>• Focus on Elo differential analysis</li>
                  )}
                  {performanceData.roi < 0 && (
                    <li>• Review quantitative position sizing</li>
                  )}
                  {performanceData.average_clv !== null && performanceData.average_clv < 0 && (
                    <li>• Improve line shopping and timing</li>
                  )}
                  {performanceData.avg_brier_score > 0.25 && performanceData.total_predictions_scored > 0 && (
                    <li>• V5 ensemble model recalibration needed</li>
                  )}
                  {Math.abs(performanceData.worst_day) > performanceData.best_day && (
                    <li>• Consider Kelly criterion position sizing</li>
                  )}
                  {performanceData.total_bets < 20 && (
                    <li>• Build sample size for statistical significance</li>
                  )}
                </ul>
              </div>
              <div className="space-y-2">
                <h4 className="font-semibold text-blue-600">V5 System Features</h4>
                <ul className="text-sm space-y-1">
                  <li>• 🏆 Dynamic Elo Rating Engine</li>
                  <li>• 💎 CLV validation system</li>
                  <li>• 🤖 Elo-enhanced ensemble models</li>
                  <li>• 📊 Professional-grade analytics</li>
                  {performanceData.total_predictions_scored > 0 ? (
                    <li>• 🎯 {performanceData.total_predictions_scored} predictions scored</li>
                  ) : (
                    <li>• ⏳ Automated scoring pending</li>
                  )}
                  <li>• ⚡ Real-time performance tracking</li>
                </ul>
              </div>
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  )
}