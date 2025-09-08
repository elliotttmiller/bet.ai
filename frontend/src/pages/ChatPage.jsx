import { useState, useRef, useEffect } from 'react'
import { 
  Card, 
  CardBody, 
  CardHeader,
  Input,
  Button,
  Avatar,
  Divider,
  Chip,
  Snippet
} from '@heroui/react'

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

    try {
      const response = await fetch(`${API_BASE}/api/betai/query`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: userMessage.content })
      })

      if (!response.ok) throw new Error('Failed to get AI response')
      
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
  }

  return (
    <div className="p-6 max-w-4xl mx-auto">
      {/* Chat Header */}
      <div className="flex justify-between items-start mb-6">
        <div className="space-y-2">
          <h1 className="text-3xl font-bold flex items-center gap-2">
            🤖 BetAI Chat
          </h1>
          <p className="text-default-500">Your AI-powered sports betting analyst</p>
        </div>
        <Button
          color="danger"
          variant="light"
          onPress={clearChat}
          startContent={<span>🗑️</span>}
        >
          Clear Chat
        </Button>
      </div>

      {/* Messages Card */}
      <Card className="mb-6">
        <CardHeader>
          <h2 className="text-xl font-semibold">Chat History</h2>
        </CardHeader>
        <CardBody>
          <div className="space-y-4 max-h-96 overflow-y-auto">
            {messages.map((message, index) => (
              <div key={message.id}>
                <div className={`flex gap-3 ${message.type === 'user' ? 'flex-row-reverse' : ''}`}>
                  <Avatar
                    icon={
                      message.type === 'user' ? '👤' : 
                      message.type === 'error' ? '⚠️' : '🤖'
                    }
                    className="flex-shrink-0"
                    size="sm"
                    color={message.type === 'user' ? 'primary' : message.type === 'error' ? 'danger' : 'success'}
                  />
                  
                  <div className={`flex-1 max-w-[80%] ${message.type === 'user' ? 'text-right' : ''}`}>
                    <div className={`p-3 rounded-lg ${
                      message.type === 'user' 
                        ? 'bg-primary text-primary-foreground ml-auto' 
                        : message.type === 'error'
                        ? 'bg-danger-50 text-danger' 
                        : 'bg-default-100'
                    }`}>
                      {message.type === 'ai' && message.content.length > 100 ? (
                        <Snippet className="w-full">
                          <pre className="whitespace-pre-wrap text-sm">{message.content}</pre>
                        </Snippet>
                      ) : (
                        <p className="text-sm whitespace-pre-wrap">{message.content}</p>
                      )}
                    </div>
                    <div className={`text-xs text-default-400 mt-1 ${
                      message.type === 'user' ? 'text-right' : ''
                    }`}>
                      {formatTime(message.timestamp)}
                    </div>
                  </div>
                </div>
                
                {index < messages.length - 1 && <Divider className="my-2" />}
              </div>
            ))}

            {isLoading && (
              <div className="flex gap-3">
                <Avatar
                  icon="🤖"
                  className="flex-shrink-0"
                  size="sm"
                  color="success"
                />
                <div className="flex-1">
                  <div className="p-3 rounded-lg bg-default-100">
                    <div className="flex items-center gap-2">
                      <div className="flex gap-1">
                        <div className="w-2 h-2 bg-default-400 rounded-full animate-pulse"></div>
                        <div className="w-2 h-2 bg-default-400 rounded-full animate-pulse delay-150"></div>
                        <div className="w-2 h-2 bg-default-400 rounded-full animate-pulse delay-300"></div>
                      </div>
                      <span className="text-sm text-default-500">BetAI is thinking...</span>
                    </div>
                  </div>
                </div>
              </div>
            )}

            <div ref={messagesEndRef} />
          </div>
        </CardBody>
      </Card>

      {/* Input Area */}
      <Card>
        <CardBody>
          <form onSubmit={sendMessage} className="flex gap-3">
            <Input
              ref={inputRef}
              placeholder="Ask BetAI about betting strategies, odds analysis, or predictions..."
              value={inputMessage}
              onChange={(e) => setInputMessage(e.target.value)}
              disabled={isLoading}
              maxLength={1000}
              className="flex-1"
              endContent={
                <Chip size="sm" variant="flat" color="default">
                  {inputMessage.length}/1000
                </Chip>
              }
            />
            <Button
              type="submit"
              color="primary"
              isLoading={isLoading}
              isDisabled={!inputMessage.trim() || isLoading}
              className="px-6"
            >
              Send
            </Button>
          </form>
          <p className="text-xs text-default-400 mt-2">
            Press Enter to send your message to BetAI
          </p>
        </CardBody>
      </Card>
    </div>
  )
}

export default ChatPage