/**
 * BetForm Component - Form for creating new bets
 * 
 * Implements form validation, user experience optimizations,
 * and adherence to Core Protocols for clean, maintainable code.
 */

import React, { useState } from 'react';
import PropTypes from 'prop-types';
import './BetForm.css';

const BetForm = ({ onSubmit }) => {
  const [formData, setFormData] = useState({
    matchup: '',
    bet_type: '',
    stake: '',
    odds: ''
  });
  const [loading, setLoading] = useState(false);
  const [message, setMessage] = useState(null);

  const betTypes = [
    'Moneyline',
    'Point Spread',
    'Over/Under',
    'Prop Bet',
    'Parlay',
    'Futures'
  ];

  /**
   * Handle form field changes
   */
  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData(prev => ({
      ...prev,
      [name]: value
    }));
    
    // Clear message when user starts typing
    if (message) {
      setMessage(null);
    }
  };

  /**
   * Validate form data
   */
  const validateForm = () => {
    const errors = [];

    if (!formData.matchup.trim()) {
      errors.push('Matchup is required');
    }

    if (!formData.bet_type) {
      errors.push('Bet type is required');
    }

    const stake = parseFloat(formData.stake);
    if (!formData.stake || isNaN(stake) || stake <= 0) {
      errors.push('Stake must be a positive number');
    }

    const odds = parseInt(formData.odds);
    if (!formData.odds || isNaN(odds)) {
      errors.push('Odds must be a valid number (American format)');
    }

    return errors;
  };

  /**
   * Handle form submission
   */
  const handleSubmit = async (e) => {
    e.preventDefault();
    
    const errors = validateForm();
    if (errors.length > 0) {
      setMessage({
        type: 'error',
        text: errors.join(', ')
      });
      return;
    }

    setLoading(true);
    setMessage(null);

    try {
      const betData = {
        matchup: formData.matchup.trim(),
        bet_type: formData.bet_type,
        stake: parseFloat(formData.stake),
        odds: parseInt(formData.odds)
      };

      const result = await onSubmit(betData);
      
      if (result.success) {
        setMessage({
          type: 'success',
          text: result.message
        });
        
        // Reset form on success
        setFormData({
          matchup: '',
          bet_type: '',
          stake: '',
          odds: ''
        });
      } else {
        setMessage({
          type: 'error',
          text: result.message
        });
      }
    } catch (error) {
      setMessage({
        type: 'error',
        text: 'An unexpected error occurred'
      });
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="bet-form">
      {message && (
        <div className={`message message--${message.type}`}>
          <span className="message-icon">
            {message.type === 'success' ? '✅' : '❌'}
          </span>
          {message.text}
        </div>
      )}

      <form onSubmit={handleSubmit} className="bet-form__form">
        <div className="form-row">
          <div className="form-group">
            <label htmlFor="matchup" className="form-label">
              Matchup *
            </label>
            <input
              id="matchup"
              name="matchup"
              type="text"
              value={formData.matchup}
              onChange={handleChange}
              placeholder="e.g., Lakers vs Warriors"
              className="form-input"
              disabled={loading}
              required
            />
          </div>

          <div className="form-group">
            <label htmlFor="bet_type" className="form-label">
              Bet Type *
            </label>
            <select
              id="bet_type"
              name="bet_type"
              value={formData.bet_type}
              onChange={handleChange}
              className="form-select"
              disabled={loading}
              required
            >
              <option value="">Select bet type</option>
              {betTypes.map(type => (
                <option key={type} value={type}>{type}</option>
              ))}
            </select>
          </div>
        </div>

        <div className="form-row">
          <div className="form-group">
            <label htmlFor="stake" className="form-label">
              Stake ($) *
            </label>
            <input
              id="stake"
              name="stake"
              type="number"
              step="0.01"
              min="0.01"
              value={formData.stake}
              onChange={handleChange}
              placeholder="100.00"
              className="form-input"
              disabled={loading}
              required
            />
          </div>

          <div className="form-group">
            <label htmlFor="odds" className="form-label">
              Odds (American) *
            </label>
            <input
              id="odds"
              name="odds"
              type="number"
              value={formData.odds}
              onChange={handleChange}
              placeholder="+150 or -110"
              className="form-input"
              disabled={loading}
              required
            />
            <small className="form-help">
              Positive for underdogs (+150), negative for favorites (-110)
            </small>
          </div>
        </div>

        <button
          type="submit"
          className="submit-button"
          disabled={loading}
        >
          {loading ? (
            <>
              <span className="loading-spinner-small"></span>
              Placing Bet...
            </>
          ) : (
            <>
              🎲 Place Bet
            </>
          )}
        </button>
      </form>
    </div>
  );
};

BetForm.propTypes = {
  onSubmit: PropTypes.func.isRequired
};

export default BetForm;