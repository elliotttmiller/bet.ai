import KPIStatCard from './KPIStatCard'
import './KPIBar.css'

function KPIBar({ stats }) {
  const formatCurrency = (value) => {
    const sign = value >= 0 ? '+' : '';
    return `${sign}$${value.toFixed(2)}`;
  }

  const formatPercentage = (value) => {
    const sign = value >= 0 ? '+' : '';
    return `${sign}${value.toFixed(1)}%`;
  }

  const getTrendColor = (value) => {
    if (value > 0) return 'positive';
    if (value < 0) return 'negative';
    return 'neutral';
  }

  return (
    <div className="kpi-bar">
      <div className="kpi-grid">
        <KPIStatCard
          title="Current Balance"
          value={stats.current_balance}
          formatter={(v) => `$${v.toFixed(2)}`}
        />
        <KPIStatCard
          title="Total P/L"
          value={stats.total_profit_loss}
          formatter={formatCurrency}
          trend={stats.total_profit_loss}
          trendColor={getTrendColor(stats.total_profit_loss)}
        />
        <KPIStatCard
          title="ROI"
          value={stats.roi}
          formatter={formatPercentage}
          trend={stats.roi}
          trendColor={getTrendColor(stats.roi)}
        />
        <KPIStatCard
          title="Win Rate"
          value={stats.win_rate}
          formatter={(v) => `${v.toFixed(1)}%`}
        />
        <KPIStatCard
          title="Total Bets"
          value={stats.total_bets}
        />
        <KPIStatCard
          title="Pending"
          value={stats.pending_bets}
          trendColor="neutral"
        />
      </div>
    </div>
  )
}

export default KPIBar