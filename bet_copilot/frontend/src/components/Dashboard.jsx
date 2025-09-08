/**
 * Dashboard Component - Main container for the Bet Copilot application
 * 
 * Implements Core Protocol #2 (Systemic Cohesion) with unified design language
 * and Core Protocol #4 (Impeccable Craftsmanship) with clean, maintainable code.
 */

import React, { useState, useEffect } from 'react';
import KPI from './KPI';
import BetForm from './BetForm';
import BetList from './BetList';
import './Dashboard.css';

const Dashboard = () => {
  // State management for dashboard data
  const [dashboardData, setDashboardData] = useState({
    total_profit_loss: 0,
    roi: 0,
    win_rate: 0,
    total_bets: 0,
    pending_bets: 0,
    current_balance: 1000,
    recent_bets: []
  });
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  // API base URL - configured for development
  const API_BASE = 'http://localhost:8000';

  /**
   * Fetch dashboard statistics from the backend API
   */
  const fetchDashboardData = async () => {
    try {
      setLoading(true);
      setError(null);
      
      const response = await fetch(`${API_BASE}/api/dashboard/stats`);
      
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      
      const data = await response.json();
      setDashboardData(data);
    } catch (err) {
      console.error('Error fetching dashboard data:', err);
      setError('Failed to load dashboard data. Please ensure the backend server is running.');
      
      // Use mock data for development/demo purposes
      setDashboardData({
        total_profit_loss: 0,
        roi: 0,
        win_rate: 0,
        total_bets: 0,
        pending_bets: 0,
        current_balance: 1000,
        recent_bets: []
      });
    } finally {
      setLoading(false);
    }
  };

  /**
   * Handle new bet submission
   */
  const handleBetSubmit = async (betData) => {
    try {
      const response = await fetch(`${API_BASE}/api/bets`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(betData),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.message || 'Failed to create bet');
      }

      const result = await response.json();
      console.log('Bet created successfully:', result);
      
      // Refresh dashboard data
      await fetchDashboardData();
      
      return { success: true, message: 'Bet created successfully!' };
    } catch (err) {
      console.error('Error creating bet:', err);
      return { success: false, message: err.message };
    }
  };

  /**
   * Handle bet settlement
   */
  const handleBetSettle = async (betId, result) => {
    try {
      const response = await fetch(`${API_BASE}/api/bets/${betId}/settle`, {
        method: 'PUT',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ result }),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.message || 'Failed to settle bet');
      }

      const settlementResult = await response.json();
      console.log('Bet settled successfully:', settlementResult);
      
      // Refresh dashboard data
      await fetchDashboardData();
      
      return { success: true, message: `Bet settled as ${result}!` };
    } catch (err) {
      console.error('Error settling bet:', err);
      return { success: false, message: err.message };
    }
  };

  // Load dashboard data on component mount
  useEffect(() => {
    fetchDashboardData();
  }, []);

  if (loading) {
    return (
      <div className="dashboard">
        <div className="loading-container">
          <div className="loading-spinner"></div>
          <p>Loading dashboard...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="dashboard">
      {/* Header */}
      <header className="dashboard-header">
        <h1>🎯 Bet Copilot</h1>
        <p className="dashboard-subtitle">Advanced Betting Analytics & Performance Tracking</p>
        {error && (
          <div className="error-banner">
            <span className="error-icon">⚠️</span>
            {error}
          </div>
        )}
      </header>

      {/* KPI Section */}
      <section className="kpi-section">
        <h2>📊 Performance Overview</h2>
        <div className="kpi-grid">
          <KPI
            title="Current Balance"
            value={`$${dashboardData.current_balance.toFixed(2)}`}
            icon="💰"
            trend={dashboardData.total_profit_loss >= 0 ? 'positive' : 'negative'}
          />
          <KPI
            title="Total P/L"
            value={`$${dashboardData.total_profit_loss.toFixed(2)}`}
            icon="📈"
            trend={dashboardData.total_profit_loss >= 0 ? 'positive' : 'negative'}
          />
          <KPI
            title="ROI"
            value={`${dashboardData.roi.toFixed(1)}%`}
            icon="🎯"
            trend={dashboardData.roi >= 0 ? 'positive' : 'negative'}
          />
          <KPI
            title="Win Rate"
            value={`${dashboardData.win_rate.toFixed(1)}%`}
            icon="🏆"
            trend={dashboardData.win_rate >= 50 ? 'positive' : 'negative'}
          />
          <KPI
            title="Total Bets"
            value={dashboardData.total_bets.toString()}
            icon="📋"
            trend="neutral"
          />
          <KPI
            title="Pending Bets"
            value={dashboardData.pending_bets.toString()}
            icon="⏳"
            trend="neutral"
          />
        </div>
      </section>

      {/* Main Content Grid */}
      <div className="main-content">
        {/* Bet Form */}
        <section className="bet-form-section">
          <h2>🎲 Place New Bet</h2>
          <BetForm onSubmit={handleBetSubmit} />
        </section>

        {/* Recent Bets */}
        <section className="bet-list-section">
          <h2>📜 Recent Bets</h2>
          <BetList 
            bets={dashboardData.recent_bets} 
            onSettle={handleBetSettle}
          />
        </section>
      </div>

      {/* Footer */}
      <footer className="dashboard-footer">
        <p>Bet Copilot Alpha v1.0 | Built with React + FastAPI</p>
        <button 
          className="refresh-button"
          onClick={fetchDashboardData}
          disabled={loading}
        >
          🔄 Refresh Data
        </button>
      </footer>
    </div>
  );
};

export default Dashboard;