import { useState, useEffect } from "react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { 
  DollarSign, 
  TrendingUp, 
  Target, 
  BarChart3,
  Wallet,
  Activity
} from "lucide-react"

interface DashboardStats {
  current_balance: number
  total_profit_loss: number
  roi: number
  win_rate: number
  total_bets: number
  pending_bets: number
}

export function DashboardStats() {
  const [stats, setStats] = useState<DashboardStats | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    fetchStats()
  }, [])

  const fetchStats = async () => {
    try {
      setLoading(true)
      const response = await fetch('http://localhost:8001/api/dashboard/stats')
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`)
      }
      const data = await response.json()
      setStats(data)
    } catch (error) {
      console.error('Error fetching stats:', error)
      setError(error instanceof Error ? error.message : 'Failed to fetch stats')
    } finally {
      setLoading(false)
    }
  }

  const formatCurrency = (amount: number) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD'
    }).format(amount)
  }

  const formatPercentage = (value: number) => {
    return `${value.toFixed(1)}%`
  }

  const getProfitLossColor = (value: number) => {
    if (value > 0) return "text-green-600"
    if (value < 0) return "text-red-600"
    return "text-muted-foreground"
  }

  const getROIBadge = (roi: number) => {
    if (roi > 10) return <Badge variant="default" className="bg-green-600">Excellent</Badge>
    if (roi > 5) return <Badge variant="default" className="bg-blue-600">Good</Badge>
    if (roi > 0) return <Badge variant="secondary">Positive</Badge>
    return <Badge variant="destructive">Negative</Badge>
  }

  if (loading) {
    return (
      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
        {[1, 2, 3, 4].map((i) => (
          <Card key={i}>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">Loading...</CardTitle>
              <div className="h-4 w-4 animate-pulse bg-muted rounded" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold animate-pulse bg-muted h-6 w-20 rounded" />
            </CardContent>
          </Card>
        ))}
      </div>
    )
  }

  if (error || !stats) {
    return (
      <Card className="col-span-full">
        <CardContent className="pt-6">
          <div className="text-center text-muted-foreground">
            Error loading dashboard stats: {error}
          </div>
        </CardContent>
      </Card>
    )
  }

  return (
    <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
      <Card>
        <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
          <CardTitle className="text-sm font-medium">Current Balance</CardTitle>
          <Wallet className="h-4 w-4 text-muted-foreground" />
        </CardHeader>
        <CardContent>
          <div className="text-2xl font-bold">
            {formatCurrency(stats.current_balance)}
          </div>
          <p className="text-xs text-muted-foreground">
            Available funds
          </p>
        </CardContent>
      </Card>

      <Card>
        <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
          <CardTitle className="text-sm font-medium">Total P&L</CardTitle>
          <TrendingUp className="h-4 w-4 text-muted-foreground" />
        </CardHeader>
        <CardContent>
          <div className={`text-2xl font-bold ${getProfitLossColor(stats.total_profit_loss)}`}>
            {formatCurrency(stats.total_profit_loss)}
          </div>
          <p className="text-xs text-muted-foreground">
            All-time profit/loss
          </p>
        </CardContent>
      </Card>

      <Card>
        <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
          <CardTitle className="text-sm font-medium">ROI</CardTitle>
          <BarChart3 className="h-4 w-4 text-muted-foreground" />
        </CardHeader>
        <CardContent>
          <div className="flex items-center space-x-2">
            <div className={`text-2xl font-bold ${getProfitLossColor(stats.roi)}`}>
              {formatPercentage(stats.roi)}
            </div>
            {getROIBadge(stats.roi)}
          </div>
          <p className="text-xs text-muted-foreground">
            Return on investment
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
            {formatPercentage(stats.win_rate)}
          </div>
          <p className="text-xs text-muted-foreground">
            {stats.total_bets} total bets, {stats.pending_bets} pending
          </p>
        </CardContent>
      </Card>
    </div>
  )
}