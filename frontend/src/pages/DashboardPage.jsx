import { useState, useEffect } from 'react'
import KPIBar from '../components/KPIBar'
import FilterControls from '../components/FilterControls'
import PredictionRow from '../components/PredictionRow'
import BetModal from '../components/BetModal'
import { api } from '../lib/parlant'
import './DashboardPage.css'

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

  // Fetch dashboard data using Parlant client
  const fetchDashboardStats = async () => {
    try {
      const data = await api.getDashboardStats()
      setDashboardStats(data)
    } catch (err) {
      setError('Failed to load dashboard statistics')
      console.error('Dashboard stats error:', err)
    }
  }

  // Fetch predictions using Parlant client
  const fetchPredictions = async () => {
    try {
      const data = await api.getPredictions()
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

  // Handle bet confirmation using Parlant client
  const handleBetConfirm = async (betData) => {
    try {
      await api.addBet(betData)

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
        <div className="dashboard-container">
          {/* Page Header */}
          <div className="page-header">
            <h1>Dashboard</h1>
            <p>Real-time betting performance and AI predictions</p>
          </div>

          {/* KPI Bar Skeleton */}
          <div className="skeleton-kpi-bar">
            <div className="skeleton-kpi-card"></div>
            <div className="skeleton-kpi-card"></div>
            <div className="skeleton-kpi-card"></div>
            <div className="skeleton-kpi-card"></div>
          </div>

          {/* Filter Controls Skeleton */}
          <div className="skeleton-filter-controls">
            <div className="skeleton-filter"></div>
            <div className="skeleton-filter"></div>
          </div>

          {/* Predictions Skeleton */}
          <div className="predictions-section">
            <div className="section-header">
              <h2>🤖 ML Predictions</h2>
              <span className="predictions-count skeleton-text"></span>
            </div>
            
            <div className="predictions-list">
              {[...Array(3)].map((_, index) => (
                <div key={index} className="skeleton-prediction-row">
                  <div className="skeleton-content">
                    <div className="skeleton-line long"></div>
                    <div className="skeleton-line medium"></div>
                    <div className="skeleton-line short"></div>
                  </div>
                  <div className="skeleton-gauge"></div>
                  <div className="skeleton-actions">
                    <div className="skeleton-button"></div>
                    <div className="skeleton-line short"></div>
                  </div>
                </div>
              ))}
            </div>
          </div>
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
              {filteredPredictions.map((prediction, index) => (
                <PredictionRow
                  key={prediction.prediction_id}
                  prediction={prediction}
                  onLogBet={handleLogBet}
                  className="fade-in-delayed"
                  style={{ animationDelay: `${index * 0.1}s` }}
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