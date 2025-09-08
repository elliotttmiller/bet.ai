import { useState, useEffect } from "react"
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table"
import { Badge } from "@/components/ui/badge"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { TrendingUp, TrendingDown, Target } from "lucide-react"

interface Prediction {
  prediction_id: number
  matchup: string
  sport: string
  league: string
  game_date: string
  team_a: string
  team_b: string
  predicted_pick: string
  predicted_odds: number
  confidence_score: number
  projected_score?: string
  calculated_edge?: number
  created_at: string
}

export function PredictionTable() {
  const [predictions, setPredictions] = useState<Prediction[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    fetchPredictions()
  }, [])

  const fetchPredictions = async () => {
    try {
      setLoading(true)
      const response = await fetch('http://localhost:8001/api/predictions')
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`)
      }
      const data = await response.json()
      setPredictions(data)
    } catch (error) {
      console.error('Error fetching predictions:', error)
      setError(error instanceof Error ? error.message : 'Failed to fetch predictions')
    } finally {
      setLoading(false)
    }
  }

  const formatOdds = (odds: number) => {
    return odds > 0 ? `+${odds}` : `${odds}`
  }

  const formatDate = (dateString: string) => {
    return new Date(dateString).toLocaleDateString('en-US', {
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit'
    })
  }

  const getConfidenceBadge = (confidence: number) => {
    if (confidence >= 70) return <Badge variant="default" className="bg-green-600">High</Badge>
    if (confidence >= 60) return <Badge variant="secondary">Medium</Badge>
    return <Badge variant="outline">Low</Badge>
  }

  const getEdgeIcon = (edge?: number) => {
    if (!edge) return null
    return edge > 3 ? <TrendingUp className="h-4 w-4 text-green-500" /> : <TrendingDown className="h-4 w-4 text-yellow-500" />
  }

  if (loading) {
    return (
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center">
            <Target className="mr-2 h-5 w-5" />
            AI Predictions
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="flex items-center justify-center py-8">
            <div className="text-muted-foreground">Loading predictions...</div>
          </div>
        </CardContent>
      </Card>
    )
  }

  if (error) {
    return (
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center">
            <Target className="mr-2 h-5 w-5" />
            AI Predictions
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="flex flex-col items-center justify-center py-8">
            <div className="text-muted-foreground mb-4">Error: {error}</div>
            <Button onClick={fetchPredictions} variant="outline">
              Retry
            </Button>
          </div>
        </CardContent>
      </Card>
    )
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center justify-between">
          <div className="flex items-center">
            <Target className="mr-2 h-5 w-5" />
            AI Predictions
          </div>
          <Button onClick={fetchPredictions} variant="outline" size="sm">
            Refresh
          </Button>
        </CardTitle>
      </CardHeader>
      <CardContent>
        {predictions.length === 0 ? (
          <div className="flex items-center justify-center py-8">
            <div className="text-muted-foreground">No predictions available</div>
          </div>
        ) : (
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>Matchup</TableHead>
                <TableHead>League</TableHead>
                <TableHead>Pick</TableHead>
                <TableHead>Odds</TableHead>
                <TableHead>Confidence</TableHead>
                <TableHead>Edge</TableHead>
                <TableHead>Score</TableHead>
                <TableHead>Game Date</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {predictions.map((prediction) => (
                <TableRow key={prediction.prediction_id}>
                  <TableCell className="font-medium">
                    {prediction.matchup}
                  </TableCell>
                  <TableCell>
                    <Badge variant="outline">{prediction.league}</Badge>
                  </TableCell>
                  <TableCell className="font-semibold text-primary">
                    {prediction.predicted_pick}
                  </TableCell>
                  <TableCell>
                    <code className="text-sm font-mono">
                      {formatOdds(prediction.predicted_odds)}
                    </code>
                  </TableCell>
                  <TableCell>
                    <div className="flex items-center space-x-2">
                      {getConfidenceBadge(prediction.confidence_score)}
                      <span className="text-sm text-muted-foreground">
                        {prediction.confidence_score.toFixed(1)}%
                      </span>
                    </div>
                  </TableCell>
                  <TableCell>
                    <div className="flex items-center space-x-1">
                      {getEdgeIcon(prediction.calculated_edge)}
                      <span className="text-sm">
                        {prediction.calculated_edge ? `${prediction.calculated_edge.toFixed(1)}%` : 'N/A'}
                      </span>
                    </div>
                  </TableCell>
                  <TableCell className="text-sm text-muted-foreground">
                    {prediction.projected_score || 'N/A'}
                  </TableCell>
                  <TableCell className="text-sm text-muted-foreground">
                    {formatDate(prediction.game_date)}
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        )}
      </CardContent>
    </Card>
  )
}