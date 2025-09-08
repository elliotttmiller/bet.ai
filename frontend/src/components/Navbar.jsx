import { Link, useLocation } from 'react-router-dom'
import { Navbar as HeroNavbar, NavbarBrand, NavbarContent, NavbarItem } from '@heroui/react'

function Navbar() {
  const location = useLocation()

  return (
    <HeroNavbar maxWidth="full" className="bg-background border-b border-divider">
      <NavbarBrand className="gap-2">
        <span className="text-2xl">🎯</span>
        <div className="flex flex-col">
          <span className="text-xl font-bold text-foreground">Bet Copilot</span>
          <span className="text-xs text-default-500">Enterprise Analytics</span>
        </div>
      </NavbarBrand>
      
      <NavbarContent className="flex gap-4">
        <NavbarItem>
          <Link 
            to="/" 
            className={`flex items-center gap-2 px-3 py-2 rounded-medium transition-colors ${
              location.pathname === '/' 
                ? 'bg-primary text-primary-foreground' 
                : 'text-foreground hover:bg-default-100'
            }`}
          >
            📊 Dashboard
          </Link>
        </NavbarItem>
        <NavbarItem>
          <Link 
            to="/betai" 
            className={`flex items-center gap-2 px-3 py-2 rounded-medium transition-colors ${
              location.pathname === '/betai' 
                ? 'bg-primary text-primary-foreground' 
                : 'text-foreground hover:bg-default-100'
            }`}
          >
            🤖 BetAI
          </Link>
        </NavbarItem>
      </NavbarContent>
    </HeroNavbar>
  )
}

export default Navbar