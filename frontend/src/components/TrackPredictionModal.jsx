import { useState, useEffect } from 'react'
import { Button } from "@/components/ui/button"
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog"
import { Input } from "@/components/ui/input"
import { Textarea } from "@/components/ui/textarea"
import { Label } from "@/components/ui/label"

function TrackPredictionModal({ isOpen, onClose, prediction, onConfirm }) {
  const [stake, setStake] = useState('')
  const [notes, setNotes] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState('')

  // Reset form when modal opens/closes
  useEffect(() => {
    if (isOpen) {
      setStake('')
      setNotes('')
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
        odds: prediction.predicted_odds,
        notes: notes
      })
      
      onClose()
    } catch (err) {
      setError(err.message || 'Failed to track prediction')
    } finally {
      setIsLoading(false)
    }
  }

  return (
    <Dialog open={isOpen} onOpenChange={onClose}>
      <DialogContent className="sm:max-w-[500px] bg-card border-border">
        <DialogHeader>
          <DialogTitle className="text-foreground">Track Prediction</DialogTitle>
          <DialogDescription className="text-muted-foreground">
            Add this prediction to your analytical tracking system
          </DialogDescription>
        </DialogHeader>

        <div className="space-y-6">
          {/* Prediction Summary */}
          <div className="p-4 rounded-lg bg-secondary/50 border border-border">
            <div className="space-y-2">
              <h3 className="font-semibold text-foreground">{prediction?.matchup}</h3>
              <div className="flex items-center justify-between">
                <span className="text-primary font-medium">{prediction?.predicted_pick}</span>
                <span className="text-muted-foreground">
                  {prediction?.predicted_odds > 0 ? '+' : ''}
                  {prediction?.predicted_odds}
                </span>
              </div>
              <div className="text-sm text-muted-foreground">
                Confidence: {prediction?.confidence_score?.toFixed(1)}%
              </div>
            </div>
          </div>

          {/* Form */}
          <form onSubmit={handleSubmit} className="space-y-4">
            <div className="space-y-2">
              <Label htmlFor="stake" className="text-foreground">Stake Amount</Label>
              <div className="relative">
                <span className="absolute left-3 top-1/2 transform -translate-y-1/2 text-muted-foreground">$</span>
                <Input
                  id="stake"
                  type="number"
                  step="0.01"
                  min="0.01"
                  max="10000"
                  value={stake}
                  onChange={(e) => setStake(e.target.value)}
                  placeholder="0.00"
                  className="pl-7 bg-background border-border text-foreground"
                  autoFocus
                />
              </div>
            </div>

            <div className="space-y-2">
              <Label htmlFor="notes" className="text-foreground">Analysis Notes (Optional)</Label>
              <Textarea
                id="notes"
                value={notes}
                onChange={(e) => setNotes(e.target.value)}
                placeholder="Enter your analysis, reasoning, or additional notes..."
                className="bg-background border-border text-foreground resize-none"
                rows={3}
              />
            </div>

            {stake && (
              <div className="p-3 rounded-lg bg-secondary/30 border border-border space-y-2">
                <div className="flex justify-between text-sm">
                  <span className="text-muted-foreground">Stake:</span>
                  <span className="text-foreground">${parseFloat(stake).toFixed(2)}</span>
                </div>
                <div className="flex justify-between text-sm">
                  <span className="text-muted-foreground">Potential Profit:</span>
                  <span className="text-green-400">${calculatePayout().toFixed(2)}</span>
                </div>
                <div className="flex justify-between text-sm font-semibold border-t border-border pt-2">
                  <span className="text-foreground">Total Payout:</span>
                  <span className="text-foreground">${(parseFloat(stake) + calculatePayout()).toFixed(2)}</span>
                </div>
              </div>
            )}

            {error && (
              <div className="p-3 rounded-lg bg-destructive/10 border border-destructive/20 text-destructive text-sm">
                {error}
              </div>
            )}

            <div className="flex space-x-3 pt-4">
              <Button
                type="button"
                variant="outline"
                onClick={onClose}
                disabled={isLoading}
                className="flex-1"
              >
                Cancel
              </Button>
              <Button
                type="submit"
                disabled={!stake || isLoading}
                className="flex-1"
              >
                {isLoading ? 'Tracking...' : 'Track Prediction'}
              </Button>
            </div>
          </form>
        </div>
      </DialogContent>
    </Dialog>
  )
}

export default TrackPredictionModal