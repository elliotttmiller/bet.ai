import { useState, useRef, useEffect } from 'react'
import { 
  Card, 
  CardHeader, 
  CardContent,
  Button,
  Input,
  Badge,
  TextGenerateEffect,
  ShimmerButton
} from '../components/ui'

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
  const [currentAiMessage, setCurrentAiMessage] = useState('')

  const messagesEndRef = useRef(null)
  const inputRef = useRef(null)

  // Auto-scroll to bottom when new messages are added
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, currentAiMessage])

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
    setCurrentAiMessage('')

    try {
      const response = await fetch(`${API_BASE}/api/betai/query`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: userMessage.content })
      })

      if (!response.ok) throw new Error('Failed to get AI response')
      
      const data = await response.json()

      // Set the AI message for TextGenerateEffect
      setCurrentAiMessage(data.response)

      // Add AI response to messages after a delay to allow animation to start
      setTimeout(() => {
        const aiMessage = {
          id: Date.now() + 1,
          type: 'ai',
          content: data.response,
          timestamp: new Date(),
          isAnimating: false
        }
        setMessages(prev => [...prev, aiMessage])
        setCurrentAiMessage('')
      }, data.response.split(' ').length * 100 + 500) // Delay based on message length

    } catch (err) {
      console.error('Chat error:', err)
      
      // Add error message to chat
      const errorMessage = {
        id: Date.now() + 1,
        type: 'error',
        content: 'Sorry, I\'m having trouble connecting right now. Please make sure LM Studio is running and try again.',
        timestamp: new Date()
      }
      setMessages(prev => [...prev, errorMessage])
      setCurrentAiMessage('')
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
    setCurrentAiMessage('')
  }

  return (
    <div className="p-6 max-w-4xl mx-auto">
      {/* Chat Header */}
      <div className="flex justify-between items-start mb-6">
        <div className="space-y-2">
          <h1 className="text-3xl font-bold flex items-center gap-2 text-white">
            🤖 BetAI Chat
          </h1>
          <p className="text-gray-400">Your AI-powered sports betting analyst</p>
        </div>
        <Button
          variant="danger"
          onClick={clearChat}
        >
          🗑️ Clear Chat
        </Button>
      </div>

      {/* Messages Card */}
      <Card className="mb-6 bg-gray-900 border-gray-800">
        <CardHeader>
          <h2 className="text-xl font-semibold text-white">Chat History</h2>
        </CardHeader>
        <CardContent>
          <div className="space-y-4 max-h-96 overflow-y-auto">
            {messages.map((message, index) => (
              <div key={message.id}>
                <div className={`flex gap-3 ${message.type === 'user' ? 'flex-row-reverse' : ''}`}>
                  <div className={`flex-shrink-0 w-8 h-8 rounded-full flex items-center justify-center text-sm ${
                    message.type === 'user' ? 'bg-blue-500 text-white' : 
                    message.type === 'error' ? 'bg-red-500 text-white' : 'bg-green-500 text-white'
                  }`}>
                    {message.type === 'user' ? '👤' : message.type === 'error' ? '⚠️' : '🤖'}
                  </div>
                  
                  <div className={`flex-1 max-w-[80%] ${message.type === 'user' ? 'text-right' : ''}`}>
                    <div className={`p-3 rounded-lg ${
                      message.type === 'user' 
                        ? 'bg-blue-600 text-white ml-auto' 
                        : message.type === 'error'
                        ? 'bg-red-800 text-red-200' 
                        : 'bg-gray-800 text-gray-100'
                    }`}>
                      <p className="text-sm whitespace-pre-wrap">{message.content}</p>
                    </div>
                    <div className={`text-xs text-gray-500 mt-1 ${
                      message.type === 'user' ? 'text-right' : ''
                    }`}>
                      {formatTime(message.timestamp)}
                    </div>
                  </div>
                </div>
                
                {index < messages.length - 1 && <div className="border-t border-gray-800 my-2" />}
              </div>
            ))}

            {/* Animating AI response */}
            {currentAiMessage && (
              <div className="flex gap-3">
                <div className="flex-shrink-0 w-8 h-8 rounded-full bg-green-500 text-white flex items-center justify-center text-sm">
                  🤖
                </div>
                <div className="flex-1">
                  <div className="p-3 rounded-lg bg-gray-800 text-gray-100">
                    <TextGenerateEffect
                      words={currentAiMessage}
                      className="text-sm whitespace-pre-wrap"
                    />
                  </div>
                </div>
              </div>
            )}

            {isLoading && !currentAiMessage && (
              <div className="flex gap-3">
                <div className="flex-shrink-0 w-8 h-8 rounded-full bg-green-500 text-white flex items-center justify-center text-sm">
                  🤖
                </div>
                <div className="flex-1">
                  <div className="p-3 rounded-lg bg-gray-800">
                    <div className="flex items-center gap-2">
                      <div className="flex gap-1">
                        <div className="w-2 h-2 bg-gray-400 rounded-full animate-pulse"></div>
                        <div className="w-2 h-2 bg-gray-400 rounded-full animate-pulse delay-150"></div>
                        <div className="w-2 h-2 bg-gray-400 rounded-full animate-pulse delay-300"></div>
                      </div>
                      <span className="text-sm text-gray-400">BetAI is thinking...</span>
                    </div>
                  </div>
                </div>
              </div>
            )}

            <div ref={messagesEndRef} />
          </div>
        </CardContent>
      </Card>

      {/* Input Area */}
      <Card className="bg-gray-900 border-gray-800">
        <CardContent>
          <form onSubmit={sendMessage} className="flex gap-3">
            <Input
              ref={inputRef}
              placeholder="Ask BetAI about betting strategies, odds analysis, or predictions..."
              value={inputMessage}
              onChange={(e) => setInputMessage(e.target.value)}
              disabled={isLoading}
              className="flex-1 bg-gray-800 border-gray-700 text-white placeholder-gray-400"
              endContent={
                <Badge variant="secondary">
                  {inputMessage.length}/1000
                </Badge>
              }
            />
            <ShimmerButton
              type="submit"
              disabled={!inputMessage.trim() || isLoading}
              className="px-6"
              loading={isLoading}
            >
              Send
            </ShimmerButton>
          </form>
          <p className="text-xs text-gray-400 mt-2">
            Press Enter to send your message to BetAI
          </p>
        </CardContent>
      </Card>
    </div>
  )
}

export default ChatPage