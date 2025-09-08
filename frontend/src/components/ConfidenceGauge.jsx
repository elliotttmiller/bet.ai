import { useEffect, useState } from 'react'
import './ConfidenceGauge.css'

function ConfidenceGauge({ confidence, size = 60 }) {
  const [animatedConfidence, setAnimatedConfidence] = useState(0)
  
  // Ensure confidence is between 0-100
  const normalizedConfidence = Math.max(0, Math.min(100, confidence))
  
  // Animate to target confidence over 500ms
  useEffect(() => {
    const timer = setTimeout(() => {
      setAnimatedConfidence(normalizedConfidence)
    }, 100) // Small delay to ensure component is mounted
    
    return () => clearTimeout(timer)
  }, [normalizedConfidence])
  
  // Calculate angle for conic gradient (360 degrees = 100%)
  const angle = (animatedConfidence / 100) * 360
  
  // Determine confidence level and color
  let confidenceClass = 'confidence-low'
  if (normalizedConfidence >= 80) {
    confidenceClass = 'confidence-very-high'
  } else if (normalizedConfidence >= 65) {
    confidenceClass = 'confidence-high'
  } else if (normalizedConfidence >= 50) {
    confidenceClass = 'confidence-medium'
  }
  
  const gaugeStyle = {
    '--confidence-angle': `${angle}deg`,
    '--target-angle': `${(normalizedConfidence / 100) * 360}deg`,
    width: `${size}px`,
    height: `${size}px`
  }
  
  const innerStyle = {
    width: `${size * 0.75}px`,
    height: `${size * 0.75}px`,
    fontSize: `${size * 0.18}px`
  }
  
  return (
    <div className="confidence-gauge">
      <div 
        className={`gauge-container ${confidenceClass}`}
        style={gaugeStyle}
      >
        <div className="gauge-background">
          <div className="gauge-fill">
            <div className="gauge-inner" style={innerStyle}>
              {Math.round(normalizedConfidence)}%
            </div>
          </div>
        </div>
      </div>
      <div className="confidence-label">
        Confidence
      </div>
    </div>
  )
}

export default ConfidenceGauge