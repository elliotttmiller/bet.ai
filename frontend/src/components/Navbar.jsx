import { Link, useLocation } from 'react-router-dom'

function Navbar() {
  const location = useLocation()

  return (
    <nav className="w-full bg-gray-950 border-b border-gray-800">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex justify-between h-16">
          {/* Brand */}
          <div className="flex items-center gap-2">
            <span className="text-2xl">🎯</span>
            <div className="flex flex-col">
              <span className="text-xl font-bold text-white">Bet Copilot</span>
              <span className="text-xs text-gray-400">Enterprise Analytics</span>
            </div>
          </div>
          
          {/* Navigation */}
          <div className="flex items-center gap-4">
            <Link 
              to="/" 
              className={`flex items-center gap-2 px-3 py-2 rounded-md transition-colors ${
                location.pathname === '/' 
                  ? 'bg-blue-600 text-white' 
                  : 'text-gray-300 hover:bg-gray-800 hover:text-white'
              }`}
            >
              📊 Dashboard
            </Link>
            <Link 
              to="/betai" 
              className={`flex items-center gap-2 px-3 py-2 rounded-md transition-colors ${
                location.pathname === '/betai' 
                  ? 'bg-blue-600 text-white' 
                  : 'text-gray-300 hover:bg-gray-800 hover:text-white'
              }`}
            >
              🤖 BetAI
            </Link>
          </div>
        </div>
      </div>
    </nav>
  )
}

export default Navbar