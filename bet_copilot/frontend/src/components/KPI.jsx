/**
 * KPI Component - Reusable component for displaying key performance indicators
 * 
 * Implements Core Protocol #2 (Systemic Cohesion) with consistent design
 * and Core Protocol #4 (Impeccable Craftsmanship) with clean, reusable code.
 */

import React from 'react';
import PropTypes from 'prop-types';
import './KPI.css';

const KPI = ({ title, value, icon, trend = 'neutral' }) => {
  return (
    <div className={`kpi-card kpi-card--${trend}`}>
      <div className="kpi-header">
        <span className="kpi-icon">{icon}</span>
        <h3 className="kpi-title">{title}</h3>
      </div>
      <div className="kpi-value">{value}</div>
      <div className={`kpi-trend kpi-trend--${trend}`}>
        {trend === 'positive' && <span className="trend-indicator">↗️</span>}
        {trend === 'negative' && <span className="trend-indicator">↘️</span>}
        {trend === 'neutral' && <span className="trend-indicator">➡️</span>}
      </div>
    </div>
  );
};

KPI.propTypes = {
  title: PropTypes.string.isRequired,
  value: PropTypes.string.isRequired,
  icon: PropTypes.string.isRequired,
  trend: PropTypes.oneOf(['positive', 'negative', 'neutral'])
};

export default KPI;