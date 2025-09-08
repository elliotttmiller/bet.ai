import ConfidenceGauge from './ConfidenceGauge'
import ResultTag from './ResultTag'
import './PredictionRow.css'

function PredictionRow({ prediction, onLogBet }) {
  const handleLogBet = () => {
    if (onLogBet) {
      onLogBet(prediction)
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
    <div className="prediction-row">
      {/* Matchup Section */}
      <div className="prediction-matchup">
        <div className="matchup-header">
          <span className="sport-badge">{prediction.sport}</span>
          <span className="game-date">
            {formatGameDate(prediction.game_date)}
          </span>
        </div>
        
        <div className="matchup-teams">
          {prediction.matchup}
        </div>
        
        <div className="predicted-pick">
          🎯 {prediction.predicted_pick}
        </div>
        
        {prediction.projected_score && (
          <div className="projected-score">
            📊 {prediction.projected_score}
          </div>
        )}
      </div>

      {/* Analysis Section */}
      <div className="prediction-analysis">
        <div className="analysis-content">
          <ConfidenceGauge 
            confidence={prediction.confidence_score} 
            size={70}
          />
          <ResultTag 
            confidence={prediction.confidence_score}
            animated={true}
          />
        </div>
      </div>

      {/* Actions Section */}
      <div className="prediction-actions">
        <button 
          className="bet-button"
          onClick={handleLogBet}
        >
          💰 Log Bet
        </button>
        
        <div className="odds-display">
          Odds: {formatOdds(prediction.predicted_odds)}
        </div>
        
        {prediction.calculated_edge && prediction.calculated_edge > 0 && (
          <div className="calculated-edge">
            +{prediction.calculated_edge.toFixed(1)}% Edge
          </div>
        )}
      </div>
    </div>
  )
}

export default PredictionRow