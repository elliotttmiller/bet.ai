import { Link, useLocation } from 'react-router-dom'
import './Navbar.css'

function Navbar() {
  const location = useLocation()

  return (
    <nav className="navbar">
      <div className="navbar-container">
        <div className="navbar-brand">
          <h1>🎯 Bet Copilot</h1>
          <span className="navbar-subtitle">Enterprise Analytics</span>
        </div>
        
        <div className="navbar-menu">
          <Link 
            to="/" 
            className={`navbar-item ${location.pathname === '/' ? 'active' : ''}`}
          >
            📊 Dashboard
          </Link>
          <Link 
            to="/betai" 
            className={`navbar-item ${location.pathname === '/betai' ? 'active' : ''}`}
          >
            🤖 BetAI
          </Link>
        </div>
      </div>
    </nav>
  )
}

export default Navbar