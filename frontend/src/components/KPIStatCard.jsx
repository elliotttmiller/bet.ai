import './KPIStatCard.css'

function KPIStatCard({ title, value, formatter = (v) => v, trend, trendColor = 'neutral' }) {
  return (
    <div className="kpi-stat-card">
      <div className="kpi-stat-header">
        <h3 className="kpi-stat-title">{title}</h3>
        {trend && (
          <span className={`kpi-trend ${trendColor}`}>
            {trend > 0 ? '↗' : trend < 0 ? '↘' : '→'}
          </span>
        )}
      </div>
      <div className="kpi-stat-value">
        {formatter(value)}
      </div>
    </div>
  )
}

export default KPIStatCard