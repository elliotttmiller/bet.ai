/**
 * Bet Copilot Application Root Component
 * 
 * Implements Core Protocol #2 (Systemic Cohesion) with unified application structure
 * and Core Protocol #4 (Impeccable Craftsmanship) with clean architecture.
 */

import React from 'react'
import Dashboard from './components/Dashboard'
import './App.css'

function App() {
  return (
    <div className="app">
      <Dashboard />
    </div>
  )
}

export default App
