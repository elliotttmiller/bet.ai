import { Card, CardContent } from "@/components/ui/card"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Filter } from "lucide-react"

interface FilterControlsProps {
  selectedSport: string
  onSportChange: (sport: string) => void
}

export function FilterControls({ selectedSport, onSportChange }: FilterControlsProps) {
  return (
    <Card className="border-border/50">
      <CardContent className="p-4">
        <div className="flex items-center space-x-4">
          <div className="flex items-center space-x-2">
            <Filter className="h-4 w-4 text-muted-foreground" />
            <span className="text-sm font-medium">Filter by Sport:</span>
          </div>
          
          <Select value={selectedSport} onValueChange={onSportChange}>
            <SelectTrigger className="w-32 h-9">
              <SelectValue placeholder="Select sport" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="NBA">
                <div className="flex items-center space-x-2">
                  <span className="w-2 h-2 rounded-full bg-orange-500"></span>
                  <span>NBA</span>
                </div>
              </SelectItem>
              <SelectItem value="NFL">
                <div className="flex items-center space-x-2">
                  <span className="w-2 h-2 rounded-full bg-green-500"></span>
                  <span>NFL</span>
                </div>
              </SelectItem>
              <SelectItem value="MLB">
                <div className="flex items-center space-x-2">
                  <span className="w-2 h-2 rounded-full bg-blue-500"></span>
                  <span>MLB</span>
                </div>
              </SelectItem>
            </SelectContent>
          </Select>
        </div>
      </CardContent>
    </Card>
  )
}