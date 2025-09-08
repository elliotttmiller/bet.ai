import { BrowserRouter as Router, Routes, Route } from 'react-router-dom'
import { SiteHeader } from '@/components/site-header'
import { DashboardPage } from '@/pages/dashboard-page'
import { ChatPage } from '@/pages/chat-page'
import { PerformancePage } from '@/pages/performance-page'
import './App.css'

function App() {
  return (
    <div className="dark min-h-screen bg-background font-sans antialiased">
      <div className="relative flex min-h-screen flex-col">
        <Router>
          <SiteHeader />
          <div className="flex-1">
            <Routes>
              <Route path="/" element={<DashboardPage />} />
              <Route path="/predictions" element={<DashboardPage />} />
              <Route path="/performance" element={<PerformancePage />} />
              <Route path="/betai" element={<ChatPage />} />
            </Routes>
          </div>
        </Router>
      </div>
    </div>
  )
}

export default App
