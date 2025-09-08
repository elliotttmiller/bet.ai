import ConfidenceGauge from './ConfidenceGauge'
import ResultTag from './ResultTag'
import { Button } from "@/components/ui/button"
import { Card, CardContent } from "@/components/ui/card"
import './PredictionRow.css'

function PredictionRow({ prediction, onTrackPrediction }) {
  const handleTrackPrediction = () => {
    if (onTrackPrediction) {
      onTrackPrediction(prediction)
    }
  }

  // Format date for display
  const formatGameDate = (dateString) => {
    try {
      const date = new Date(dateString)
      return date.toLocaleDateString('en-US', {
        weekday: 'short',
        month: 'short', 
        day: 'numeric',
        hour: '2-digit',
        minute: '2-digit'
      })
    } catch (error) {
      return dateString
    }
  }

  // Format odds display
  const formatOdds = (odds) => {
    return odds > 0 ? `+${odds}` : `${odds}`
  }

  return (
    <Card className="prediction-row bg-card border-border hover:bg-card/80 transition-colors">
      <CardContent className="p-6">
        <div className="flex flex-col lg:flex-row lg:items-center gap-6">
          {/* Matchup Section */}
          <div className="prediction-matchup flex-1">
            <div className="matchup-header flex items-center gap-2 mb-2">
              <span className="sport-badge px-2 py-1 rounded text-xs font-medium bg-primary/10 text-primary">
                {prediction.sport}
              </span>
              <span className="game-date text-sm text-muted-foreground">
                {formatGameDate(prediction.game_date)}
              </span>
            </div>
            
            <div className="matchup-teams text-lg font-semibold text-foreground mb-2">
              {prediction.matchup}
            </div>
            
            <div className="predicted-pick text-primary font-medium mb-1">
              🎯 {prediction.predicted_pick}
            </div>
            
            {prediction.projected_score && (
              <div className="projected-score text-sm text-muted-foreground">
                📊 {prediction.projected_score}
              </div>
            )}
          </div>

          {/* Analysis Section */}
          <div className="prediction-analysis flex items-center gap-4">
            <ConfidenceGauge 
              confidence={prediction.confidence_score} 
              size={70}
            />
            <ResultTag 
              confidence={prediction.confidence_score}
              animated={true}
            />
          </div>

          {/* Actions Section */}
          <div className="prediction-actions flex flex-col items-end gap-2">
            <Button 
              onClick={handleTrackPrediction}
              className="bg-primary hover:bg-primary/90 text-primary-foreground"
            >
              📈 Track
            </Button>
            
            <div className="odds-display text-sm text-muted-foreground">
              Odds: {formatOdds(prediction.predicted_odds)}
            </div>
            
            {prediction.calculated_edge && prediction.calculated_edge > 0 && (
              <div className="calculated-edge text-xs text-green-400 font-medium">
                +{prediction.calculated_edge.toFixed(1)}% Edge
              </div>
            )}
          </div>
        </div>
      </CardContent>
    </Card>
  )
}

export default PredictionRow