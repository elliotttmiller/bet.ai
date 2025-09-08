import { useState } from 'react'
import './PredictionCard.css'

function PredictionCard({ prediction, onLogBet }) {
  const [isLoading, setIsLoading] = useState(false)

  const formatDate = (dateString) => {
    const date = new Date(dateString)
    return date.toLocaleDateString('en-US', {
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit'
    })
  }

  const getConfidenceColor = (confidence) => {
    if (confidence >= 80) return '#10b981' // Green
    if (confidence >= 70) return '#f59e0b' // Yellow
    return '#ef4444' // Red
  }

  const getConfidenceLabel = (confidence) => {
    if (confidence >= 80) return 'High'
    if (confidence >= 70) return 'Medium'
    return 'Low'
  }

  const handleLogBet = async () => {
    setIsLoading(true)
    try {
      await onLogBet(prediction)
    } finally {
      setIsLoading(false)
    }
  }

  return (
    <div className="prediction-card">
      {/* Header Section */}
      <div className="prediction-header">
        <div className="game-datetime">
          {formatDate(prediction.game_date)}
        </div>
        <div className="sport-tag">
          {prediction.league}
        </div>
      </div>

      {/* Matchup Section */}
      <div className="matchup-section">
        <div className="teams-container">
          <div className="team">
            <div className="team-logo-placeholder">
              {prediction.team_a.split(' ').pop()[0]}
            </div>
            <span className="team-name">{prediction.team_a}</span>
          </div>
          <div className="vs-divider">VS</div>
          <div className="team">
            <div className="team-logo-placeholder">
              {prediction.team_b.split(' ').pop()[0]}
            </div>
            <span className="team-name">{prediction.team_b}</span>
          </div>
        </div>
      </div>

      {/* Prediction Section */}
      <div className="prediction-section">
        <div className="primary-pick">
          <div className="pick-text">{prediction.predicted_pick}</div>
          <div className="odds-text">{prediction.predicted_odds > 0 ? '+' : ''}{prediction.predicted_odds}</div>
        </div>

        <div className="confidence-container">
          <div className="confidence-label">
            Confidence: {getConfidenceLabel(prediction.confidence_score)}
          </div>
          <div className="confidence-bar-container">
            <div
              className="confidence-bar"
              style={{
                width: `${prediction.confidence_score}%`,
                backgroundColor: getConfidenceColor(prediction.confidence_score)
              }}
            />
          </div>
          <div className="confidence-percentage">
            {prediction.confidence_score.toFixed(1)}%
          </div>
        </div>

        <div className="supporting-data">
          <div className="data-item">
            <span className="data-label">Projected Score:</span>
            <span className="data-value">{prediction.projected_score || 'TBD'}</span>
          </div>
          <div className="data-item">
            <span className="data-label">Calculated Edge:</span>
            <span className="data-value">
              {prediction.calculated_edge ? `+${prediction.calculated_edge.toFixed(1)}%` : 'N/A'}
            </span>
          </div>
        </div>

        {/* Call-to-Action */}
        <button
          className={`log-bet-btn ${isLoading ? 'loading' : ''}`}
          onClick={handleLogBet}
          disabled={isLoading}
        >
          {isLoading ? '🔄 Logging...' : '📝 Log Bet'}
        </button>
      </div>
    </div>
  )
}

export default PredictionCard