import { useState, useEffect } from 'react'
import './BetModal.css'

function BetModal({ isOpen, onClose, prediction, onConfirm }) {
  const [stake, setStake] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState('')

  // Reset form when modal opens/closes
  useEffect(() => {
    if (isOpen) {
      setStake('')
      setError('')
    }
  }, [isOpen])

  const calculatePayout = () => {
    const stakeAmount = parseFloat(stake) || 0
    if (stakeAmount === 0) return 0

    const odds = prediction?.predicted_odds || 0
    if (odds > 0) {
      return stakeAmount * (odds / 100)
    } else {
      return stakeAmount / (Math.abs(odds) / 100)
    }
  }

  const handleSubmit = async (e) => {
    e.preventDefault()
    setError('')
    
    const stakeAmount = parseFloat(stake)
    
    if (!stakeAmount || stakeAmount <= 0) {
      setError('Please enter a valid stake amount')
      return
    }

    if (stakeAmount > 10000) {
      setError('Maximum stake is $10,000')
      return
    }

    setIsLoading(true)
    
    try {
      await onConfirm({
        matchup: prediction.matchup,
        bet_type: prediction.predicted_pick,
        stake: stakeAmount,
        odds: prediction.predicted_odds
      })
      
      onClose()
    } catch (err) {
      setError(err.message || 'Failed to place bet')
    } finally {
      setIsLoading(false)
    }
  }

  if (!isOpen) return null

  return (
    <div className="bet-modal-overlay" onClick={onClose}>
      <div className="bet-modal" onClick={(e) => e.stopPropagation()}>
        <div className="bet-modal-header">
          <h2>Confirm Bet</h2>
          <button className="close-btn" onClick={onClose}>×</button>
        </div>

        <div className="bet-modal-content">
          <div className="prediction-summary">
            <div className="matchup-info">
              <h3>{prediction?.matchup}</h3>
              <div className="pick-info">
                <span className="pick">{prediction?.predicted_pick}</span>
                <span className="odds">
                  {prediction?.predicted_odds > 0 ? '+' : ''}
                  {prediction?.predicted_odds}
                </span>
              </div>
            </div>
            <div className="confidence-info">
              <span>Confidence: {prediction?.confidence_score?.toFixed(1)}%</span>
            </div>
          </div>

          <form onSubmit={handleSubmit} className="bet-form">
            <div className="form-group">
              <label htmlFor="stake">Stake Amount</label>
              <div className="stake-input-container">
                <span className="currency-symbol">$</span>
                <input
                  id="stake"
                  type="number"
                  step="0.01"
                  min="0.01"
                  max="10000"
                  value={stake}
                  onChange={(e) => setStake(e.target.value)}
                  placeholder="0.00"
                  className="stake-input"
                  autoFocus
                />
              </div>
            </div>

            {stake && (
              <div className="payout-calculation">
                <div className="calc-row">
                  <span>Stake:</span>
                  <span>${parseFloat(stake).toFixed(2)}</span>
                </div>
                <div className="calc-row">
                  <span>Potential Profit:</span>
                  <span className="profit">${calculatePayout().toFixed(2)}</span>
                </div>
                <div className="calc-row total">
                  <span>Total Payout:</span>
                  <span>${(parseFloat(stake) + calculatePayout()).toFixed(2)}</span>
                </div>
              </div>
            )}

            {error && (
              <div className="error-message">
                {error}
              </div>
            )}

            <div className="modal-actions">
              <button
                type="button"
                className="cancel-btn"
                onClick={onClose}
                disabled={isLoading}
              >
                Cancel
              </button>
              <button
                type="submit"
                className="confirm-btn"
                disabled={!stake || isLoading}
              >
                {isLoading ? 'Placing Bet...' : 'Confirm Bet'}
              </button>
            </div>
          </form>
        </div>
      </div>
    </div>
  )
}

export default BetModal