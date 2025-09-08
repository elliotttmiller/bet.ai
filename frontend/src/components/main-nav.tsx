import { cn } from "@/lib/utils"
import { Button } from "@/components/ui/button"
import { 
  BarChart3, 
  TrendingUp, 
  MessageSquare, 
  Target,
  Home
} from "lucide-react"
import { Link, useLocation } from "react-router-dom"

export function MainNav({
  className,
  ...props
}: React.HTMLAttributes<HTMLElement>) {
  const location = useLocation()

  return (
    <nav
      className={cn("flex items-center space-x-4 lg:space-x-6", className)}
      {...props}
    >
      <Link to="/" className="flex items-center space-x-2">
        <Target className="h-6 w-6" />
        <span className="font-bold">Bet.AI</span>
      </Link>
      <Link
        to="/"
        className={cn(
          "text-sm font-medium transition-colors hover:text-primary",
          location.pathname === "/" ? "text-foreground" : "text-muted-foreground"
        )}
      >
        <Home className="h-4 w-4 mr-2 inline" />
        Dashboard
      </Link>
      <Link
        to="/predictions"
        className={cn(
          "text-sm font-medium transition-colors hover:text-primary",
          location.pathname === "/predictions" ? "text-foreground" : "text-muted-foreground"
        )}
      >
        <TrendingUp className="h-4 w-4 mr-2 inline" />
        Predictions
      </Link>
      <Link
        to="/betai"
        className={cn(
          "text-sm font-medium transition-colors hover:text-primary",
          location.pathname === "/betai" ? "text-foreground" : "text-muted-foreground"
        )}
      >
        <MessageSquare className="h-4 w-4 mr-2 inline" />
        BetAI Chat
      </Link>
    </nav>
  )
}