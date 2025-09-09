# 🏆 Bet.AI - Multi-Sport Analytics Platform

> **Enterprise-Grade Multi-Sport Betting Performance Tracking & AI-Powered Analytics System**

Welcome to Bet.AI, a comprehensive multi-sport betting analytics platform built with modern web technologies following enterprise software development protocols. This repository contains the complete V2 build of the Bet Copilot system with advanced multi-sport intelligence.

## 🚀 Live Demo

![Multi-Sport Dashboard](https://github.com/user-attachments/assets/ec11d2f3-d1cc-4a54-a7ee-8e179c68edf0)

*Professional multi-sport dashboard with intelligent sport filtering and AI predictions*

## ⚡ Multi-Sport Architecture

**NEW: Multi-Sport Intelligence Engine** - The system now supports NBA, NFL, and MLB with sport-specific AI models and dynamic filtering:

- **🏀 NBA**: Advanced basketball analytics with ensemble LightGBM + XGBoost models
- **🏈 NFL**: Professional football predictions with sport-specific feature engineering  
- **⚾ MLB**: Baseball analytics (expandable architecture ready)
- **🎯 Sport-Aware UI**: Dynamic filtering with seamless sport switching
- **🤖 Sport-Specific AI Models**: Separate ensemble models trained for each sport

## 📁 Project Structure

```
bet.ai/                 # Multi-Sport Analytics Platform
├── backend/           # FastAPI + Python
│   ├── model_lgbm_nba.joblib    # NBA-specific LightGBM model
│   ├── model_xgb_nfl.joblib     # NFL-specific XGBoost model
│   └── main.py        # Sport-aware API endpoints
├── database/          # Sport-aware SQLite schema
├── frontend/          # React + Sport Filtering UI
└── README.md          # Enhanced multi-sport docs
```

## 🏗️ Architecture Highlights

- **Multi-Sport Database**: Sport column in all relevant tables for data isolation
- **Sport-Specific AI Models**: Separate ensemble models (LightGBM + XGBoost) for each sport
- **Dynamic API Endpoints**: `/api/predictions?sport=NBA` with sport parameter validation
- **Intelligent UI Filtering**: Sport selector with real-time data switching
- **Modern Tech Stack**: FastAPI, React, SQLite, Advanced ML Models
- **Performance Optimized**: Target hardware constraints with 8GB VRAM models

## 📊 Core Features

✅ **Multi-sport prediction engine** (NBA, NFL, MLB-ready)  
✅ **Sport-specific ensemble AI models**  
✅ **Dynamic sport filtering in dashboard**  
✅ **Real-time performance dashboards**  
✅ **Advanced feature engineering** (rolling averages, strength of schedule)  
✅ **American odds calculation engine**  
✅ **Sport-aware transaction ledger**  
✅ **Professional dark-themed UI**  
✅ **Complete API documentation**  

## 🎯 Technology Stack

- **Backend**: FastAPI, Pydantic, SQLite, LightGBM, XGBoost
- **Frontend**: React 18, TypeScript, Shadcn/UI, Sport Filtering
- **AI/ML**: Ensemble Models (LightGBM + XGBoost), Advanced Feature Engineering
- **Database**: Sport-aware SQLite schema with integrity constraints

## 🚀 Quick Start

1. **Initialize sport-aware database:**
   ```bash
   cd database && python create_db.py
   ```

2. **Train sport-specific models:**
   ```bash
   cd backend && python model_trainer.py NBA
   cd backend && python model_trainer.py NFL
   ```

3. **Start backend with multi-sport API:**
   ```bash
   cd backend && python main.py
   ```

4. **Start frontend with sport filtering:**
   ```bash
   cd frontend && npm install && npm run dev
   ```

## 🤖 Multi-Sport AI Intelligence

The system features state-of-the-art ensemble AI models:

- **Sport-Specific Training**: Each sport (NBA, NFL) has dedicated model files
- **Advanced Feature Engineering**: Rolling averages, strength of schedule, head-to-head analysis
- **Ensemble Architecture**: LightGBM + XGBoost combination for superior accuracy
- **Edge Calculation**: Automated +EV opportunity identification
- **Model Versioning**: `v3.0-ensemble-nba`, `v3.0-ensemble-nfl` tracking

## 📈 API Endpoints

### Multi-Sport Predictions
```bash
GET /api/predictions?sport=NBA&limit=10
GET /api/predictions?sport=NFL&limit=10
```

### Sport-Aware Betting
```bash
POST /api/bets
{
  "matchup": "Lakers vs Warriors",
  "bet_type": "ML",
  "stake": 100,
  "odds": -150,
  "sport": "NBA"
}
```

## 🔧 Development Notes

This V2 multi-sport build demonstrates:
- **Surgical Architecture Evolution**: Minimal changes for maximum functionality
- **Sport-Specific Intelligence**: Dedicated models per sport
- **Seamless UI Experience**: Dynamic filtering without page reloads  
- **Enterprise-Grade Code**: Maintains all existing quality standards
- **Scalable Design**: Ready for additional sports (MLB, NHL, etc.)

---

**Bet.AI V2.0** - Multi-Sport Intelligence Platform Built with React + FastAPI following Core Protocols V6
