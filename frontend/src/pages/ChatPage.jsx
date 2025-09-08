import { useState, useRef, useEffect } from 'react'
import './ChatPage.css'

const API_BASE = 'http://localhost:8000'

function ChatPage() {
  const [messages, setMessages] = useState([
    {
      id: 1,
      type: 'ai',
      content: 'Hello! I\'m BetAI, your expert sports betting analyst. I can help you with betting strategies, odds analysis, and sports predictions. What would you like to know?',
      timestamp: new Date()
    }
  ])
  const [inputMessage, setInputMessage] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState('')

  const messagesEndRef = useRef(null)
  const inputRef = useRef(null)

  // Auto-scroll to bottom when new messages are added
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  // Focus input on mount
  useEffect(() => {
    inputRef.current?.focus()
  }, [])

  const sendMessage = async (e) => {
    e.preventDefault()
    
    if (!inputMessage.trim() || isLoading) return

    const userMessage = {
      id: Date.now(),
      type: 'user',
      content: inputMessage.trim(),
      timestamp: new Date()
    }

    // Add user message
    setMessages(prev => [...prev, userMessage])
    setInputMessage('')
    setIsLoading(true)
    setError('')

    try {
      const response = await fetch(`${API_BASE}/api/betai/query`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          message: userMessage.content
        })
      })

      if (!response.ok) {
        throw new Error('Failed to get response from BetAI')
      }

      const data = await response.json()

      // Add AI response
      const aiMessage = {
        id: Date.now() + 1,
        type: 'ai',
        content: data.response,
        timestamp: new Date()
      }

      setMessages(prev => [...prev, aiMessage])

    } catch (err) {
      setError('Failed to connect to BetAI. Please ensure LM Studio is running on localhost:1234')
      console.error('Chat error:', err)
      
      // Add error message to chat
      const errorMessage = {
        id: Date.now() + 1,
        type: 'error',
        content: 'Sorry, I\'m having trouble connecting right now. Please make sure LM Studio is running and try again.',
        timestamp: new Date()
      }
      setMessages(prev => [...prev, errorMessage])
    } finally {
      setIsLoading(false)
    }
  }

  const formatTime = (timestamp) => {
    return timestamp.toLocaleTimeString('en-US', {
      hour: '2-digit',
      minute: '2-digit'
    })
  }

  const clearChat = () => {
    setMessages([
      {
        id: 1,
        type: 'ai',
        content: 'Hello! I\'m BetAI, your expert sports betting analyst. How can I help you today?',
        timestamp: new Date()
      }
    ])
    setError('')
  }

  return (
    <div className="chat-page">
      <div className="chat-container">
        {/* Chat Header */}
        <div className="chat-header">
          <div className="chat-title">
            <h1>🤖 BetAI Chat</h1>
            <p>Your AI-powered sports betting analyst</p>
          </div>
          <button className="clear-chat-btn" onClick={clearChat}>
            🗑️ Clear Chat
          </button>
        </div>

        {/* Messages Area */}
        <div className="messages-container">
          <div className="messages">
            {messages.map((message) => (
              <div
                key={message.id}
                className={`message ${message.type}`}
              >
                <div className="message-content">
                  <div className="message-text">
                    {message.content}
                  </div>
                  <div className="message-time">
                    {formatTime(message.timestamp)}
                  </div>
                </div>
                <div className="message-avatar">
                  {message.type === 'user' ? '👤' : message.type === 'error' ? '⚠️' : '🤖'}
                </div>
              </div>
            ))}

            {isLoading && (
              <div className="message ai loading">
                <div className="message-content">
                  <div className="typing-indicator">
                    <div className="typing-dots">
                      <span></span>
                      <span></span>
                      <span></span>
                    </div>
                    <div className="message-text">BetAI is thinking...</div>
                  </div>
                </div>
                <div className="message-avatar">🤖</div>
              </div>
            )}

            <div ref={messagesEndRef} />
          </div>
        </div>

        {/* Input Area */}
        <div className="input-container">
          {error && (
            <div className="error-banner">
              {error}
            </div>
          )}
          
          <form onSubmit={sendMessage} className="input-form">
            <div className="input-wrapper">
              <input
                ref={inputRef}
                type="text"
                value={inputMessage}
                onChange={(e) => setInputMessage(e.target.value)}
                placeholder="Ask BetAI about betting strategies, odds analysis, or predictions..."
                className="message-input"
                disabled={isLoading}
                maxLength={1000}
              />
              <button
                type="submit"
                className="send-btn"
                disabled={!inputMessage.trim() || isLoading}
              >
                {isLoading ? '⏳' : '📤'}
              </button>
            </div>
            <div className="input-footer">
              <span className="char-count">
                {inputMessage.length}/1000
              </span>
              <span className="input-hint">
                Press Enter to send
              </span>
            </div>
          </form>
        </div>
      </div>
    </div>
  )
}

export default ChatPage