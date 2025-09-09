import { useState } from "react"
import { DashboardStats } from "@/components/dashboard-stats"
import { PredictionTable } from "@/components/prediction-table"
import { FilterControls } from "@/components/FilterControls"

export function DashboardPage() {
  const [selectedSport, setSelectedSport] = useState<string>("NBA")

  return (
    <div className="container mx-auto p-6 space-y-8">
      <div className="flex flex-col space-y-2">
        <h1 className="text-3xl font-bold tracking-tight">Dashboard</h1>
        <p className="text-muted-foreground">
          Overview of your betting performance and AI predictions.
        </p>
      </div>
      
      <DashboardStats />
      
      <FilterControls 
        selectedSport={selectedSport}
        onSportChange={setSelectedSport}
      />
      
      <PredictionTable sport={selectedSport} />
    </div>
  )
}