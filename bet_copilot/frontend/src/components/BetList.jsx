/**
 * BetList Component - Display and manage list of bets
 * 
 * Implements tabular data display with interactive settlement functionality,
 * following Core Protocols for performance and maintainability.
 */

import React, { useState } from 'react';
import PropTypes from 'prop-types';
import './BetList.css';

const BetList = ({ bets, onSettle }) => {
  const [settlingBet, setSettlingBet] = useState(null);
  const [message, setMessage] = useState(null);

  /**
   * Handle bet settlement
   */
  const handleSettle = async (betId, result) => {
    setSettlingBet(betId);
    setMessage(null);

    try {
      const settlementResult = await onSettle(betId, result);
      
      if (settlementResult.success) {
        setMessage({
          type: 'success',
          text: settlementResult.message
        });
      } else {
        setMessage({
          type: 'error',
          text: settlementResult.message
        });
      }
    } catch (error) {
      setMessage({
        type: 'error',
        text: 'Failed to settle bet'
      });
    } finally {
      setSettlingBet(null);
      
      // Clear message after 3 seconds
      setTimeout(() => setMessage(null), 3000);
    }
  };

  /**
   * Format date for display
   */
  const formatDate = (dateString) => {
    try {
      const date = new Date(dateString);
      return date.toLocaleDateString() + ' ' + date.toLocaleTimeString([], { 
        hour: '2-digit', 
        minute: '2-digit' 
      });
    } catch {
      return dateString;
    }
  };

  /**
   * Format odds for display
   */
  const formatOdds = (odds) => {
    if (odds > 0) {
      return `+${odds}`;
    }
    return odds.toString();
  };

  /**
   * Get status badge styling
   */
  const getStatusBadge = (status) => {
    const statusClass = status.toLowerCase();
    const statusEmoji = {
      pending: '⏳',
      won: '🏆',
      lost: '❌'
    };

    return (
      <span className={`status-badge status-badge--${statusClass}`}>
        {statusEmoji[statusClass]} {status}
      </span>
    );
  };

  /**
   * Get profit/loss styling
   */
  const getProfitLossDisplay = (profitLoss, status) => {
    if (status === 'Pending') {
      return <span className="profit-loss profit-loss--pending">-</span>;
    }

    const className = profitLoss >= 0 ? 'profit-loss--positive' : 'profit-loss--negative';
    const prefix = profitLoss >= 0 ? '+' : '';
    
    return (
      <span className={`profit-loss ${className}`}>
        {prefix}${profitLoss.toFixed(2)}
      </span>
    );
  };

  if (bets.length === 0) {
    return (
      <div className="bet-list">
        <div className="empty-state">
          <div className="empty-state-icon">📋</div>
          <h3>No bets yet</h3>
          <p>Place your first bet to get started with tracking your performance!</p>
        </div>
      </div>
    );
  }

  return (
    <div className="bet-list">
      {message && (
        <div className={`message message--${message.type}`}>
          <span className="message-icon">
            {message.type === 'success' ? '✅' : '❌'}
          </span>
          {message.text}
        </div>
      )}

      <div className="bet-table-container">
        <table className="bet-table">
          <thead>
            <tr>
              <th>Date</th>
              <th>Matchup</th>
              <th>Type</th>
              <th>Stake</th>
              <th>Odds</th>
              <th>Status</th>
              <th>P/L</th>
              <th>Actions</th>
            </tr>
          </thead>
          <tbody>
            {bets.map((bet) => (
              <tr key={bet.bet_id} className="bet-row">
                <td className="bet-date">
                  {formatDate(bet.bet_date)}
                </td>
                <td className="bet-matchup">
                  <strong>{bet.matchup}</strong>
                </td>
                <td className="bet-type">
                  {bet.bet_type}
                </td>
                <td className="bet-stake">
                  ${bet.stake.toFixed(2)}
                </td>
                <td className="bet-odds">
                  {formatOdds(bet.odds)}
                </td>
                <td className="bet-status">
                  {getStatusBadge(bet.status)}
                </td>
                <td className="bet-profit-loss">
                  {getProfitLossDisplay(bet.profit_loss, bet.status)}
                </td>
                <td className="bet-actions">
                  {bet.status === 'Pending' ? (
                    <div className="settlement-buttons">
                      <button
                        className="settle-button settle-button--won"
                        onClick={() => handleSettle(bet.bet_id, 'Won')}
                        disabled={settlingBet === bet.bet_id}
                      >
                        {settlingBet === bet.bet_id ? '⏳' : '🏆'} Won
                      </button>
                      <button
                        className="settle-button settle-button--lost"
                        onClick={() => handleSettle(bet.bet_id, 'Lost')}
                        disabled={settlingBet === bet.bet_id}
                      >
                        {settlingBet === bet.bet_id ? '⏳' : '❌'} Lost
                      </button>
                    </div>
                  ) : (
                    <span className="settled-indicator">Settled</span>
                  )}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <div className="bet-list-summary">
        <p>Showing {bets.length} recent bet{bets.length !== 1 ? 's' : ''}</p>
      </div>
    </div>
  );
};

BetList.propTypes = {
  bets: PropTypes.arrayOf(PropTypes.shape({
    bet_id: PropTypes.number.isRequired,
    matchup: PropTypes.string.isRequired,
    bet_type: PropTypes.string.isRequired,
    stake: PropTypes.number.isRequired,
    odds: PropTypes.number.isRequired,
    status: PropTypes.string.isRequired,
    profit_loss: PropTypes.number.isRequired,
    bet_date: PropTypes.string.isRequired
  })).isRequired,
  onSettle: PropTypes.func.isRequired
};

export default BetList;