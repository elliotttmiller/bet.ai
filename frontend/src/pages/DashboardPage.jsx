import { useState, useEffect } from 'react'
import { 
  Card, 
  CardBody, 
  CardHeader,
  Table, 
  TableHeader, 
  TableColumn, 
  TableBody, 
  TableRow, 
  TableCell,
  Progress,
  Button,
  Modal,
  ModalContent,
  ModalHeader,
  ModalBody,
  ModalFooter,
  Input,
  Textarea,
  Chip,
  Select,
  SelectItem,
  Spinner,
  useDisclosure
} from '@heroui/react'

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
  const { isOpen, onOpen, onClose } = useDisclosure()
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
    onOpen()
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
      onClose()
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
          <Spinner size="lg" />
          <p className="text-default-500">Loading dashboard...</p>
        </div>
      </div>
    )
  }

  if (error) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <Card className="max-w-md">
          <CardBody className="text-center">
            <h2 className="text-2xl mb-2">⚠️ Error</h2>
            <p className="text-default-500 mb-4">{error}</p>
            <Button color="primary" onClick={() => window.location.reload()}>
              Retry
            </Button>
          </CardBody>
        </Card>
      </div>
    )
  }

  return (
    <div className="p-6 max-w-7xl mx-auto space-y-6">
      {/* Page Header */}
      <div className="space-y-2">
        <h1 className="text-3xl font-bold">Analytics Dashboard</h1>
        <p className="text-default-500">Real-time performance tracking and AI-powered predictions</p>
      </div>

      {/* KPI Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        <Card>
          <CardHeader className="pb-2">
            <h3 className="text-lg font-semibold">Current Balance</h3>
          </CardHeader>
          <CardBody className="pt-0">
            <p className="text-2xl font-bold text-primary">{formatCurrency(dashboardStats.current_balance)}</p>
          </CardBody>
        </Card>

        <Card>
          <CardHeader className="pb-2">
            <h3 className="text-lg font-semibold">Total P/L</h3>
          </CardHeader>
          <CardBody className="pt-0">
            <p className={`text-2xl font-bold ${dashboardStats.total_profit_loss >= 0 ? 'text-success' : 'text-danger'}`}>
              {formatCurrency(dashboardStats.total_profit_loss)}
            </p>
          </CardBody>
        </Card>

        <Card>
          <CardHeader className="pb-2">
            <h3 className="text-lg font-semibold">ROI</h3>
          </CardHeader>
          <CardBody className="pt-0">
            <p className={`text-2xl font-bold ${dashboardStats.roi >= 0 ? 'text-success' : 'text-danger'}`}>
              {dashboardStats.roi.toFixed(1)}%
            </p>
          </CardBody>
        </Card>

        <Card>
          <CardHeader className="pb-2">
            <h3 className="text-lg font-semibold">Win Rate</h3>
          </CardHeader>
          <CardBody className="pt-0">
            <p className="text-2xl font-bold text-primary">{dashboardStats.win_rate.toFixed(1)}%</p>
          </CardBody>
        </Card>

        <Card>
          <CardHeader className="pb-2">
            <h3 className="text-lg font-semibold">Total Bets</h3>
          </CardHeader>
          <CardBody className="pt-0">
            <p className="text-2xl font-bold">{dashboardStats.total_bets}</p>
          </CardBody>
        </Card>

        <Card>
          <CardHeader className="pb-2">
            <h3 className="text-lg font-semibold">Pending Bets</h3>
          </CardHeader>
          <CardBody className="pt-0">
            <p className="text-2xl font-bold text-warning">{dashboardStats.pending_bets}</p>
          </CardBody>
        </Card>
      </div>

      {/* Predictions Section */}
      <Card>
        <CardHeader className="flex justify-between items-center">
          <div>
            <h2 className="text-2xl font-bold">🤖 AI Predictions</h2>
            <p className="text-default-500">{filteredPredictions.length} predictions available</p>
          </div>
          <Select
            label="Filter by Sport"
            placeholder="All Sports"
            selectedKeys={[filterSport]}
            onSelectionChange={(keys) => setFilterSport(Array.from(keys)[0] || "all")}
            className="w-48"
          >
            <SelectItem key="all" value="all">All Sports</SelectItem>
            <SelectItem key="NBA" value="NBA">NBA</SelectItem>
            <SelectItem key="NFL" value="NFL">NFL</SelectItem>
          </Select>
        </CardHeader>
        <CardBody>
          <Table aria-label="AI Predictions Table">
            <TableHeader>
              <TableColumn>MATCHUP</TableColumn>
              <TableColumn>PREDICTION</TableColumn>
              <TableColumn>ODDS</TableColumn>
              <TableColumn>CONFIDENCE</TableColumn>
              <TableColumn>ACTIONS</TableColumn>
            </TableHeader>
            <TableBody>
              {filteredPredictions.map((prediction) => (
                <TableRow key={prediction.prediction_id}>
                  <TableCell>
                    <div className="flex flex-col">
                      <p className="font-semibold">{prediction.matchup}</p>
                      <p className="text-small text-default-500">{prediction.sport} • {prediction.game_date}</p>
                    </div>
                  </TableCell>
                  <TableCell>
                    <Chip color="primary" variant="flat">
                      {prediction.predicted_pick}
                    </Chip>
                  </TableCell>
                  <TableCell>
                    <span className="font-mono">{prediction.predicted_odds > 0 ? '+' : ''}{prediction.predicted_odds}</span>
                  </TableCell>
                  <TableCell>
                    <div className="flex flex-col gap-1">
                      <Progress
                        value={prediction.confidence_score}
                        color={getConfidenceColor(prediction.confidence_score)}
                        className="w-20"
                        size="sm"
                      />
                      <span className="text-small">{prediction.confidence_score.toFixed(1)}%</span>
                    </div>
                  </TableCell>
                  <TableCell>
                    <Button
                      color="primary"
                      size="sm"
                      onPress={() => handleTrackPrediction(prediction)}
                    >
                      Track
                    </Button>
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </CardBody>
      </Card>

      {/* Track Prediction Modal */}
      <Modal isOpen={isOpen} onClose={onClose} size="md">
        <ModalContent>
          {(onClose) => (
            <>
              <ModalHeader>
                <h3>Track Prediction</h3>
              </ModalHeader>
              <ModalBody>
                {selectedPrediction && (
                  <div className="space-y-4">
                    <div className="p-4 bg-default-100 rounded-lg">
                      <p className="font-semibold">{selectedPrediction.matchup}</p>
                      <p className="text-default-600">{selectedPrediction.predicted_pick}</p>
                      <p className="text-small text-default-500">
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
                    />
                    
                    <Textarea
                      label="Notes (optional)"
                      placeholder="Add any notes about this bet..."
                      value={notes}
                      onChange={(e) => setNotes(e.target.value)}
                      maxRows={3}
                    />
                  </div>
                )}
              </ModalBody>
              <ModalFooter>
                <Button color="danger" variant="light" onPress={onClose}>
                  Cancel
                </Button>
                <Button
                  color="primary"
                  onPress={handleTrackingConfirm}
                  isLoading={trackingLoading}
                  isDisabled={!stake || parseFloat(stake) <= 0}
                >
                  Track Bet
                </Button>
              </ModalFooter>
            </>
          )}
        </ModalContent>
      </Modal>
    </div>
  )
}

export default DashboardPage