import { useState } from 'react'
import './FilterControls.css'

function FilterControls({ onFilterChange }) {
  const [filters, setFilters] = useState({
    sport: 'all',
    status: 'all',
    sortBy: 'date_desc'
  })

  const handleFilterChange = (key, value) => {
    const newFilters = { ...filters, [key]: value }
    setFilters(newFilters)
    onFilterChange(newFilters)
  }

  return (
    <div className="filter-controls">
      <div className="filter-group">
        <label className="filter-label">Sport</label>
        <select
          className="filter-select"
          value={filters.sport}
          onChange={(e) => handleFilterChange('sport', e.target.value)}
        >
          <option value="all">All Sports</option>
          <option value="NBA">NBA</option>
          <option value="NFL">NFL</option>
          <option value="MLB">MLB</option>
          <option value="NHL">NHL</option>
        </select>
      </div>

      <div className="filter-group">
        <label className="filter-label">Status</label>
        <select
          className="filter-select"
          value={filters.status}
          onChange={(e) => handleFilterChange('status', e.target.value)}
        >
          <option value="all">All Status</option>
          <option value="pending">Pending</option>
          <option value="won">Won</option>
          <option value="lost">Lost</option>
        </select>
      </div>

      <div className="filter-group">
        <label className="filter-label">Sort By</label>
        <select
          className="filter-select"
          value={filters.sortBy}
          onChange={(e) => handleFilterChange('sortBy', e.target.value)}
        >
          <option value="date_desc">Newest First</option>
          <option value="date_asc">Oldest First</option>
          <option value="confidence_desc">High Confidence</option>
          <option value="confidence_asc">Low Confidence</option>
        </select>
      </div>

      <div className="filter-actions">
        <button
          className="filter-reset-btn"
          onClick={() => {
            const resetFilters = {
              sport: 'all',
              status: 'all',
              sortBy: 'date_desc'
            }
            setFilters(resetFilters)
            onFilterChange(resetFilters)
          }}
        >
          Reset
        </button>
      </div>
    </div>
  )
}

export default FilterControls