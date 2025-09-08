import { useState, useEffect } from 'react'
import { 
  Card, 
  CardContent, 
  CardHeader,
  Button,
  Input,
  Textarea,
  Badge,
  Progress,
  Modal,
  ModalContent,
  ModalHeader,
  ModalBody,
  ModalFooter,
  BackgroundGradient,
  ShimmerButton
} from '../components/ui'

const API_BASE = 'http://localhost:8000'

function DashboardPage() {
  const [dashboardStats, setDashboardStats] = useState({
    current_balance: 0,
    total_profit_loss: 0,
    roi: 0,
    win_rate: 0,
    total_bets: 0,
    pending_bets: 0
  })
  const [predictions, setPredictions] = useState([])
  const [filteredPredictions, setFilteredPredictions] = useState([])
  const [filterSport, setFilterSport] = useState("all")
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')
  
  // Modal state for tracking predictions
  const [isModalOpen, setIsModalOpen] = useState(false)
  const [selectedPrediction, setSelectedPrediction] = useState(null)
  const [stake, setStake] = useState("")
  const [notes, setNotes] = useState("")
  const [trackingLoading, setTrackingLoading] = useState(false)

  // Fetch dashboard data
  const fetchDashboardStats = async () => {
    try {
      const response = await fetch(`${API_BASE}/api/dashboard/stats`)
      if (!response.ok) throw new Error('Failed to fetch dashboard stats')
      const data = await response.json()
      setDashboardStats(data)
    } catch (err) {
      setError(`Failed to load dashboard statistics: ${err.message}`)
      console.error('Dashboard stats error:', err)
    }
  }

  // Fetch predictions
  const fetchPredictions = async () => {
    try {
      const response = await fetch(`${API_BASE}/api/predictions`)
      if (!response.ok) throw new Error('Failed to fetch predictions')
      const data = await response.json()
      setPredictions(data)
      setFilteredPredictions(data)
    } catch (err) {
      setError(`Failed to load predictions: ${err.message}`)
      console.error('Predictions error:', err)
    }
  }

  // Load data on component mount
  useEffect(() => {
    const loadData = async () => {
      setLoading(true)
      await Promise.all([fetchDashboardStats(), fetchPredictions()])
      setLoading(false)
    }
    loadData()
  }, [])

  // Apply sport filter
  useEffect(() => {
    let filtered = [...predictions]
    if (filterSport !== "all") {
      filtered = filtered.filter(p => p.sport === filterSport)
    }
    setFilteredPredictions(filtered)
  }, [predictions, filterSport])

  // Handle track prediction
  const handleTrackPrediction = (prediction) => {
    setSelectedPrediction(prediction)
    setStake("")
    setNotes("")
    setIsModalOpen(true)
  }

  // Handle prediction tracking confirmation
  const handleTrackingConfirm = async () => {
    if (!selectedPrediction || !stake) return

    setTrackingLoading(true)
    try {
      const response = await fetch(`${API_BASE}/api/bets`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          matchup: selectedPrediction.matchup,
          bet_type: selectedPrediction.predicted_pick,
          stake: parseFloat(stake),
          odds: selectedPrediction.predicted_odds
        })
      })

      if (!response.ok) {
        const error = await response.json()
        throw new Error(error.detail || 'Failed to track bet')
      }

      await fetchDashboardStats()
      setIsModalOpen(false)
      // Could add toast notification here
    } catch (err) {
      alert(`Error tracking bet: ${err.message}`)
    } finally {
      setTrackingLoading(false)
    }
  }

  // Format currency
  const formatCurrency = (value) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD'
    }).format(value)
  }

  // Get confidence color
  const getConfidenceColor = (confidence) => {
    if (confidence >= 70) return "success"
    if (confidence >= 60) return "warning" 
    return "danger"
  }

  if (loading) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <div className="flex flex-col items-center gap-4">
          <div className="w-8 h-8 border-4 border-blue-500 border-t-transparent rounded-full animate-spin"></div>
          <p className="text-gray-400">Loading dashboard...</p>
        </div>
      </div>
    )
  }

  if (error) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <Card className="max-w-md bg-gray-900 border-gray-800">
          <CardContent className="text-center">
            <h2 className="text-2xl mb-2 text-white">⚠️ Error</h2>
            <p className="text-gray-400 mb-4">{error}</p>
            <Button variant="primary" onClick={() => window.location.reload()}>
              Retry
            </Button>
          </CardContent>
        </Card>
      </div>
    )
  }

  return (
    <div className="p-6 max-w-7xl mx-auto space-y-6">
      {/* Page Header */}
      <div className="space-y-2">
        <h1 className="text-3xl font-bold text-white">Analytics Dashboard</h1>
        <p className="text-gray-400">Real-time performance tracking and AI-powered predictions</p>
      </div>

      {/* KPI Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        <Card className="bg-gray-900 border-gray-800">
          <CardHeader className="pb-2">
            <h3 className="text-lg font-semibold text-white">Current Balance</h3>
          </CardHeader>
          <CardContent className="pt-0">
            <p className="text-2xl font-bold text-blue-400">{formatCurrency(dashboardStats.current_balance)}</p>
          </CardContent>
        </Card>

        <Card className="bg-gray-900 border-gray-800">
          <CardHeader className="pb-2">
            <h3 className="text-lg font-semibold text-white">Total P/L</h3>
          </CardHeader>
          <CardContent className="pt-0">
            <p className={`text-2xl font-bold ${dashboardStats.total_profit_loss >= 0 ? 'text-green-400' : 'text-red-400'}`}>
              {formatCurrency(dashboardStats.total_profit_loss)}
            </p>
          </CardContent>
        </Card>

        <Card className="bg-gray-900 border-gray-800">
          <CardHeader className="pb-2">
            <h3 className="text-lg font-semibold text-white">ROI</h3>
          </CardHeader>
          <CardContent className="pt-0">
            <p className={`text-2xl font-bold ${dashboardStats.roi >= 0 ? 'text-green-400' : 'text-red-400'}`}>
              {dashboardStats.roi.toFixed(1)}%
            </p>
          </CardContent>
        </Card>

        <Card className="bg-gray-900 border-gray-800">
          <CardHeader className="pb-2">
            <h3 className="text-lg font-semibold text-white">Win Rate</h3>
          </CardHeader>
          <CardContent className="pt-0">
            <p className="text-2xl font-bold text-blue-400">{dashboardStats.win_rate.toFixed(1)}%</p>
          </CardContent>
        </Card>

        <Card className="bg-gray-900 border-gray-800">
          <CardHeader className="pb-2">
            <h3 className="text-lg font-semibold text-white">Total Bets</h3>
          </CardHeader>
          <CardContent className="pt-0">
            <p className="text-2xl font-bold text-white">{dashboardStats.total_bets}</p>
          </CardContent>
        </Card>

        <Card className="bg-gray-900 border-gray-800">
          <CardHeader className="pb-2">
            <h3 className="text-lg font-semibold text-white">Pending Bets</h3>
          </CardHeader>
          <CardContent className="pt-0">
            <p className="text-2xl font-bold text-yellow-400">{dashboardStats.pending_bets}</p>
          </CardContent>
        </Card>
      </div>

      {/* Predictions Section */}
      <Card className="bg-gray-900 border-gray-800">
        <CardHeader className="flex justify-between items-center">
          <div>
            <h2 className="text-2xl font-bold text-white">🤖 AI Predictions</h2>
            <p className="text-gray-400">{filteredPredictions.length} predictions available</p>
          </div>
          <select
            value={filterSport}
            onChange={(e) => setFilterSport(e.target.value)}
            className="px-4 py-2 bg-gray-800 border border-gray-700 rounded-md text-white focus:outline-none focus:ring-2 focus:ring-blue-500"
          >
            <option value="all">All Sports</option>
            <option value="NBA">NBA</option>
            <option value="NFL">NFL</option>
          </select>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            {filteredPredictions.map((prediction) => (
              <BackgroundGradient 
                key={prediction.prediction_id}
                className="rounded-2xl bg-gray-800 p-6"
                containerClassName="rounded-2xl"
              >
                <div className="grid grid-cols-1 md:grid-cols-4 gap-4 items-center">
                  {/* Matchup */}
                  <div className="flex flex-col">
                    <p className="font-semibold text-white text-lg">{prediction.matchup}</p>
                    <p className="text-sm text-gray-400">{prediction.sport} • {prediction.game_date}</p>
                  </div>
                  
                  {/* Prediction */}
                  <div className="flex flex-col items-center">
                    <Badge variant="primary" className="mb-2">
                      {prediction.predicted_pick}
                    </Badge>
                    <span className="font-mono text-white text-lg">
                      {prediction.predicted_odds > 0 ? '+' : ''}{prediction.predicted_odds}
                    </span>
                  </div>
                  
                  {/* Confidence */}
                  <div className="flex flex-col items-center">
                    <Progress
                      value={prediction.confidence_score}
                      color={getConfidenceColor(prediction.confidence_score)}
                      className="w-24 mb-2"
                    />
                    <span className="text-sm text-gray-300">{prediction.confidence_score.toFixed(1)}%</span>
                  </div>
                  
                  {/* Action */}
                  <div className="flex justify-center">
                    <ShimmerButton
                      onClick={() => handleTrackPrediction(prediction)}
                      className="px-6 py-2"
                    >
                      Track Bet
                    </ShimmerButton>
                  </div>
                </div>
              </BackgroundGradient>
            ))}
          </div>
        </CardContent>
      </Card>

      {/* Track Prediction Modal */}
      <Modal isOpen={isModalOpen} onClose={() => setIsModalOpen(false)} size="md">
        <ModalContent className="bg-gray-900 border-gray-800">
          <ModalHeader>
            <h3 className="text-white">Track Prediction</h3>
          </ModalHeader>
          <ModalBody>
            {selectedPrediction && (
              <div className="space-y-4">
                <div className="p-4 bg-gray-800 rounded-lg">
                  <p className="font-semibold text-white">{selectedPrediction.matchup}</p>
                  <p className="text-gray-300">{selectedPrediction.predicted_pick}</p>
                  <p className="text-sm text-gray-400">
                    {selectedPrediction.predicted_odds > 0 ? '+' : ''}{selectedPrediction.predicted_odds} odds
                    • {selectedPrediction.confidence_score.toFixed(1)}% confidence
                  </p>
                </div>
                
                <Input
                  label="Stake Amount"
                  placeholder="Enter stake amount"
                  startContent="$"
                  value={stake}
                  onChange={(e) => setStake(e.target.value)}
                  type="number"
                  step="0.01"
                  min="0"
                  className="bg-gray-800 border-gray-700"
                />
                
                <Textarea
                  label="Notes (optional)"
                  placeholder="Add any notes about this bet..."
                  value={notes}
                  onChange={(e) => setNotes(e.target.value)}
                  className="bg-gray-800 border-gray-700"
                  rows={3}
                />
              </div>
            )}
          </ModalBody>
          <ModalFooter>
            <Button variant="outline" onClick={() => setIsModalOpen(false)}>
              Cancel
            </Button>
            <Button
              variant="primary"
              onClick={handleTrackingConfirm}
              loading={trackingLoading}
              disabled={!stake || parseFloat(stake) <= 0}
            >
              Track Bet
            </Button>
          </ModalFooter>
        </ModalContent>
      </Modal>
    </div>
  )
}

export default DashboardPage