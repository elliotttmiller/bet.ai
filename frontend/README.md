# 🏆 Bet.AI Frontend - Multi-Sport Dashboard

> **React + TypeScript Multi-Sport Analytics Dashboard**

The frontend component of Bet.AI's multi-sport analytics platform, featuring dynamic sport filtering and real-time AI predictions.

## 🚀 Key Features

- **Multi-Sport Interface**: Dynamic sport selector (NBA, NFL, MLB)
- **Real-Time Data**: Live predictions with sport-specific API integration
- **Modern UI/UX**: Dark-themed design with Shadcn/UI components
- **Type-Safe API**: Full TypeScript integration with backend models
- **Responsive Design**: Mobile-first approach with professional styling

## 🏗️ Architecture

- **React 18** with TypeScript for type safety
- **Vite** for lightning-fast development and building
- **Shadcn/UI** for consistent, accessible component design
- **Sport-Aware State Management**: Reactive sport filtering
- **API Client**: Type-safe backend communication

## 📊 Multi-Sport Components

### FilterControls
Sport selector component with visual indicators:
```tsx
<FilterControls 
  selectedSport={selectedSport}
  onSportChange={setSelectedSport}
/>
```

### PredictionTable
Dynamic table that updates based on selected sport:
```tsx
<PredictionTable sport={selectedSport} />
```

### DashboardStats
Performance metrics (sport-agnostic):
```tsx
<DashboardStats />
```

## 🎯 Technology Stack

- **React 18** - Modern React with hooks and concurrent features
- **TypeScript** - Full type safety and IDE support
- **Vite** - Fast build tool and dev server
- **Shadcn/UI** - High-quality component library
- **Radix UI** - Accessible primitive components
- **Tailwind CSS** - Utility-first styling
- **Lucide React** - Beautiful icons

## ⚡ Quick Start

```bash
# Install dependencies
npm install

# Start development server
npm run dev

# Build for production
npm run build

# Preview production build
npm run preview
```

## 📱 Sport Filtering Implementation

The dashboard features seamless sport switching:

1. **State Management**: Single `selectedSport` state controls all components
2. **API Integration**: Dynamic API calls with sport parameter
3. **Real-Time Updates**: Automatic data refresh when sport changes
4. **Visual Feedback**: Sport-specific colors and indicators

## 🔧 Development Notes

- **Type-Safe API Client**: Auto-generated types from backend OpenAPI
- **Component Isolation**: Each sport component is self-contained
- **Performance Optimized**: Efficient re-renders with React best practices
- **Accessibility First**: WCAG compliant with keyboard navigation

---

**Bet.AI Frontend V2.0** - Multi-Sport React Dashboard
