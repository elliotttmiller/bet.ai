import { DashboardStats } from "@/components/dashboard-stats"
import { PredictionTable } from "@/components/prediction-table"

export function DashboardPage() {
  return (
    <div className="container mx-auto p-6 space-y-8">
      <div className="flex flex-col space-y-2">
        <h1 className="text-3xl font-bold tracking-tight">Dashboard</h1>
        <p className="text-muted-foreground">
          Overview of your betting performance and AI predictions.
        </p>
      </div>
      
      <DashboardStats />
      
      <PredictionTable />
    </div>
  )
}