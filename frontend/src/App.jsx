import { BrowserRouter as Router, Routes, Route } from 'react-router-dom'
import Navbar from './components/Navbar'
import DashboardPage from './pages/DashboardPage'
import ChatPage from './pages/ChatPage'
import './App.css'

function App() {
  return (
    <Router>
      <div className="app min-h-screen bg-gray-900 text-white">
        <Navbar />
        <main className="main-content">
          <Routes>
            <Route path="/" element={<DashboardPage />} />
            <Route path="/betai" element={<ChatPage />} />
          </Routes>
        </main>
      </div>
    </Router>
  )
}

export default App
