# 🤖 Bet.AI Backend - Multi-Sport Intelligence Engine

> **FastAPI + Advanced ML Multi-Sport Analytics Backend**

The core intelligence engine of Bet.AI's multi-sport platform, featuring sport-specific AI models and comprehensive analytics.

## 🚀 Multi-Sport Architecture

### Sport-Specific AI Models
- **NBA Models**: `model_lgbm_nba.joblib`, `model_xgb_nba.joblib`, `model_nba.json`
- **NFL Models**: `model_lgbm_nfl.joblib`, `model_xgb_nfl.joblib`, `model_nfl.json`
- **Ensemble Approach**: LightGBM + XGBoost combination for superior accuracy

### Advanced Feature Engineering
- **Rolling Averages**: 5, 10, 20-game performance windows
- **Strength of Schedule**: Dynamic opponent quality assessment
- **Head-to-Head Analysis**: Historical matchup performance
- **Rest Days**: Recovery time impact on performance
- **Home Advantage**: Location-based performance adjustments

## 📊 API Endpoints

### Multi-Sport Predictions
```
GET /api/predictions?sport=NBA&limit=10
GET /api/predictions?sport=NFL&limit=10
```
**Response**: Sport-specific predictions with confidence scores and edge calculations

### Sport-Aware Betting
```
POST /api/bets
{
  "matchup": "Lakers vs Warriors",
  "bet_type": "ML", 
  "stake": 100,
  "odds": -150,
  "sport": "NBA"
}
```

### Dashboard Analytics
```
GET /api/dashboard/stats
```
**Response**: Comprehensive performance metrics

## 🤖 AI Model Training

### Train Sport-Specific Models
```bash
python model_trainer.py NBA    # Train NBA ensemble
python model_trainer.py NFL    # Train NFL ensemble
```

### Model Architecture
- **LightGBM**: Gradient boosting with early stopping
- **XGBoost**: Extreme gradient boosting with regularization
- **Ensemble**: Simple averaging of model outputs
- **Feature Count**: 22 engineered features per sport

## 🏗️ Technology Stack

- **FastAPI** - High-performance async API framework
- **LightGBM & XGBoost** - State-of-the-art gradient boosting
- **Pandas & NumPy** - Data manipulation and numerical computing
- **Scikit-learn** - ML utilities and preprocessing
- **Pydantic** - Data validation and settings management
- **APScheduler** - Autonomous model retraining
- **SQLite** - Sport-aware database schema

## 📈 Model Performance

### Training Results (Synthetic Data Demo)
- **Training Accuracy**: 85.2%
- **Test Accuracy**: 73.1%
- **Training AUC**: 0.936
- **Test AUC**: 0.816

### Top Features (LightGBM Importance)
1. **win_rate_5g**: Recent win rate (5 games)
2. **strength_of_schedule**: Opponent quality
3. **is_home**: Home field advantage
4. **point_diff_5g**: Scoring differential
5. **avg_points_20g**: Long-term scoring average

## 🔧 Data Pipeline

### Multi-Sport Data Ingestion
```bash
python data_pipeline.py
```
- **NBA Data**: TheSportsDB API integration
- **NFL Data**: Comprehensive team and game data
- **Automated Updates**: Scheduled data refresh

### Database Schema
- **Sport-Aware Tables**: All tables include sport column
- **Data Isolation**: Clean separation by sport
- **Referential Integrity**: Foreign key constraints

## 🎯 Autonomous Operation

### Scheduled Tasks
- **Daily Data Pipeline**: 3:00 AM automatic data refresh
- **Weekly Model Training**: Sunday 3:00 AM retraining
- **Daily Prediction Auditing**: 4:00 AM automated scoring

### Configuration
```python
# Environment variables
ENABLE_SCHEDULER=true
DATA_PIPELINE_HOUR=3
MODEL_TRAINING_HOUR=3
AUDIT_HOUR=4
```

## ⚡ Quick Start

### Dependencies
```bash
pip install -r requirements.txt
```

### Initialize Database
```bash
cd ../database && python create_db.py
```

### Train Models
```bash
python model_trainer.py NBA
python model_trainer.py NFL
```

### Start Server
```bash
python main.py
```
**Server**: http://localhost:8000
**Docs**: http://localhost:8000/docs

## 🔬 Advanced Features

### Edge Calculation
Automated +EV opportunity identification:
```python
edge = model_probability - implied_probability
```

### Brier Score Auditing
Probabilistic prediction accuracy measurement:
```python
brier_score = (predicted_probability - actual_outcome) ** 2
```

### Live Odds Integration
Real-time market odds comparison for edge calculation.

---

**Bet.AI Backend V2.0** - Multi-Sport Intelligence Engine with Advanced ML