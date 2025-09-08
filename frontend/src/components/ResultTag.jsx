import './ResultTag.css'

function ResultTag({ 
  status = 'pending', 
  confidence = null, 
  animated = false,
  size = 'normal' 
}) {
  // Determine tag type and content based on status
  let tagClass = 'result-tag'
  let icon = ''
  let text = ''
  
  if (confidence !== null) {
    // This is a prediction confidence tag
    if (confidence >= 75) {
      tagClass += ' high-confidence'
      icon = '🔥'
      text = 'High'
    } else if (confidence >= 60) {
      tagClass += ' medium-confidence'
      icon = '⚡'
      text = 'Medium'
    } else {
      tagClass += ' low-confidence'
      icon = '🤔'
      text = 'Low'
    }
  } else {
    // This is a result status tag
    switch (status.toLowerCase()) {
      case 'won':
        tagClass += ' won'
        icon = '✅'
        text = 'Won'
        break
      case 'lost':
        tagClass += ' lost'
        icon = '❌'
        text = 'Lost'
        break
      case 'pushed':
        tagClass += ' pushed'
        icon = '➖'
        text = 'Push'
        break
      case 'pending':
      default:
        tagClass += ' pending'
        icon = '⏳'
        text = 'Pending'
        break
    }
  }
  
  if (animated) {
    tagClass += ' animate-in'
  }
  
  if (size === 'small') {
    tagClass += ' small'
  }
  
  return (
    <span className={tagClass}>
      <span className="result-tag-icon">{icon}</span>
      <span>{text}</span>
    </span>
  )
}

export default ResultTag