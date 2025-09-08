import { useState, useEffect } from 'react'
import KPIBar from '../components/KPIBar'
import FilterControls from '../components/FilterControls'
import PredictionRow from '../components/PredictionRow'
import BetModal from '../components/BetModal'
import './DashboardPage.css'

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
  const [filters, setFilters] = useState({})
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')
  
  // Modal state
  const [selectedPrediction, setSelectedPrediction] = useState(null)
  const [isModalOpen, setIsModalOpen] = useState(false)

  // Fetch dashboard data
  const fetchDashboardStats = async () => {
    try {
      const response = await fetch(`${API_BASE}/api/dashboard/stats`)
      if (!response.ok) throw new Error('Failed to fetch dashboard stats')
      const data = await response.json()
      setDashboardStats(data)
    } catch (err) {
      setError('Failed to load dashboard statistics')
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
      setError('Failed to load predictions')
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

  // Apply filters to predictions
  useEffect(() => {
    let filtered = [...predictions]

    if (filters.sport && filters.sport !== 'all') {
      filtered = filtered.filter(p => p.sport === filters.sport)
    }

    if (filters.sortBy) {
      switch (filters.sortBy) {
        case 'date_desc':
          filtered.sort((a, b) => new Date(b.created_at) - new Date(a.created_at))
          break
        case 'date_asc':
          filtered.sort((a, b) => new Date(a.created_at) - new Date(b.created_at))
          break
        case 'confidence_desc':
          filtered.sort((a, b) => b.confidence_score - a.confidence_score)
          break
        case 'confidence_asc':
          filtered.sort((a, b) => a.confidence_score - b.confidence_score)
          break
        default:
          break
      }
    }

    setFilteredPredictions(filtered)
  }, [predictions, filters])

  // Handle filter changes
  const handleFilterChange = (newFilters) => {
    setFilters(newFilters)
  }

  // Handle log bet
  const handleLogBet = (prediction) => {
    setSelectedPrediction(prediction)
    setIsModalOpen(true)
  }

  // Handle bet confirmation
  const handleBetConfirm = async (betData) => {
    try {
      const response = await fetch(`${API_BASE}/api/bets`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify(betData)
      })

      if (!response.ok) {
        const errorData = await response.json()
        throw new Error(errorData.detail || 'Failed to place bet')
      }

      // Refresh dashboard stats after successful bet
      await fetchDashboardStats()
      
      // Close modal
      setIsModalOpen(false)
      setSelectedPrediction(null)
      
      // Show success message
      alert('✅ Bet placed successfully!')
      
    } catch (err) {
      throw err // Re-throw to be handled by modal
    }
  }

  if (loading) {
    return (
      <div className="dashboard-page">
        <div className="loading-container">
          <div className="loading-spinner"></div>
          <p>Loading dashboard...</p>
        </div>
      </div>
    )
  }

  if (error) {
    return (
      <div className="dashboard-page">
        <div className="error-container">
          <h2>⚠️ Error</h2>
          <p>{error}</p>
          <button onClick={() => window.location.reload()}>
            Retry
          </button>
        </div>
      </div>
    )
  }

  return (
    <div className="dashboard-page">
      <div className="dashboard-container">
        {/* Page Header */}
        <div className="page-header">
          <h1>Dashboard</h1>
          <p>Real-time betting performance and AI predictions</p>
        </div>

        {/* KPI Bar */}
        <KPIBar stats={dashboardStats} />

        {/* Filter Controls */}
        <FilterControls onFilterChange={handleFilterChange} />

        {/* Predictions Section */}
        <div className="predictions-section">
          <div className="section-header">
            <h2>🤖 ML Predictions</h2>
            <span className="predictions-count">
              {filteredPredictions.length} prediction{filteredPredictions.length !== 1 ? 's' : ''}
            </span>
          </div>
          
          {filteredPredictions.length === 0 ? (
            <div className="empty-state">
              <h3>No predictions found</h3>
              <p>Try adjusting your filters or check back later for new ML predictions.</p>
            </div>
          ) : (
            <div className="predictions-list">
              {filteredPredictions.map((prediction) => (
                <PredictionRow
                  key={prediction.prediction_id}
                  prediction={prediction}
                  onLogBet={handleLogBet}
                />
              ))}
            </div>
          )}
        </div>

        {/* Bet Modal */}
        <BetModal
          isOpen={isModalOpen}
          onClose={() => {
            setIsModalOpen(false)
            setSelectedPrediction(null)
          }}
          prediction={selectedPrediction}
          onConfirm={handleBetConfirm}
        />
      </div>
    </div>
  )
}

export default DashboardPage