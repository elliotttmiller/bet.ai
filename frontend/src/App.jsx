import { BrowserRouter as Router, Routes, Route } from 'react-router-dom'
import { HeroUIProvider } from '@heroui/react'
import Navbar from './components/Navbar'
import DashboardPage from './pages/DashboardPage'
import ChatPage from './pages/ChatPage'
import './App.css'

function App() {
  return (
    <HeroUIProvider theme={{ type: "dark" }}>
      <Router>
        <div className="app min-h-screen bg-background text-foreground">
          <Navbar />
          <main className="main-content">
            <Routes>
              <Route path="/" element={<DashboardPage />} />
              <Route path="/betai" element={<ChatPage />} />
            </Routes>
          </main>
        </div>
      </Router>
    </HeroUIProvider>
  )
}

export default App
