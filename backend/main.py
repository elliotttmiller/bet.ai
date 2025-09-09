#!/usr/bin/env python3
"""
Bet Copilot FastAPI Backend
Enterprise-grade betting analytics API with AI integration
"""

import os
import sqlite3
import requests
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional, Dict, Any
from contextlib import contextmanager
import logging

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Bet Copilot API",
    description="Enterprise-grade betting analytics with AI integration and autonomous operation",
    version="1.0.0"
)

# Initialize scheduler for autonomous operation
scheduler = AsyncIOScheduler()

# CORS middleware for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Database path and API configuration
# Support both local development and containerized deployment
DB_PATH = Path(os.getenv("DB_PATH", str(Path(__file__).parent.parent / "database" / "bet_copilot.db")))
LM_STUDIO_API_URL = os.getenv("LM_STUDIO_API_URL", "http://localhost:1234/v1/chat/completions")
ODDS_API_KEY = os.getenv("ODDS_API_KEY", "")
ODDS_API_URL = os.getenv("ODDS_API_URL", "https://api.the-odds-api.com/v4")

# Pydantic models
class BetCreate(BaseModel):
    matchup: str = Field(..., min_length=1, max_length=200)
    bet_type: str = Field(..., min_length=1, max_length=100)
    stake: float = Field(..., gt=0)
    odds: int = Field(..., ge=-1000, le=1000)
    sport: str = Field(..., min_length=1, max_length=10)

class BetSettle(BaseModel):
    result: str = Field(..., pattern="^(Won|Lost)$")

class BetResponse(BaseModel):
    bet_id: int
    matchup: str
    bet_type: str
    stake: float
    odds: int
    sport: str
    status: str
    profit_loss: float
    odds_at_tracking: Optional[float] = None
    closing_line_odds: Optional[float] = None
    bet_date: str
    created_at: str
    updated_at: str

class DashboardStats(BaseModel):
    current_balance: float
    total_profit_loss: float
    roi: float
    win_rate: float
    total_bets: int
    pending_bets: int

class PredictionResponse(BaseModel):
    prediction_id: int
    matchup: str
    sport: str
    league: str
    game_date: str
    team_a: str
    team_b: str
    predicted_pick: str
    predicted_odds: int
    confidence_score: float
    projected_score: Optional[str]
    calculated_edge: Optional[float]
    created_at: str

class BetAIQuery(BaseModel):
    message: str = Field(..., min_length=1, max_length=1000)

class PerformanceDataPoint(BaseModel):
    date: str
    profit_loss: float
    running_balance: float
    cumulative_profit: float

class BrierScoreDataPoint(BaseModel):
    date: str
    avg_brier_score: float
    prediction_count: int

class PerformanceHistory(BaseModel):
    data_points: List[PerformanceDataPoint]
    brier_score_points: List[BrierScoreDataPoint]
    total_profit_loss: float
    total_bets: int
    win_rate: float
    roi: float
    best_day: float
    worst_day: float
    avg_brier_score: float
    total_predictions_scored: int
    average_clv: Optional[float] = None
    average_clv_by_sport: Dict[str, float] = {}
    elo_history_sample: List[Dict] = []

# Database context manager
@contextmanager
def get_db():
    """Database connection context manager."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()

def calculate_profit_loss(stake: float, odds: int, won: bool) -> float:
    """Calculate profit/loss for a bet based on American odds."""
    if not won:
        return -stake  # Full stake is lost
    
    if odds > 0:
        # Positive odds: profit = stake * (odds/100)
        return stake * (odds / 100)
    else:
        # Negative odds: profit = stake / (abs(odds)/100)
        return stake / (abs(odds) / 100)

def get_current_balance() -> float:
    """Get current balance from ledger."""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT running_balance FROM ledger ORDER BY entry_id DESC LIMIT 1")
        result = cursor.fetchone()
        return result[0] if result else 0.0

# V6 Autonomous Engine Functions with Dependency-Aware Execution
async def run_data_pipeline():
    """V6 Autonomous data pipeline execution (Step 1 of dependency chain)."""
    try:
        logger.info("🤖 Starting V6 autonomous data pipeline...")
        pipeline_path = Path(__file__).parent / "data_pipeline.py"
        result = subprocess.run([sys.executable, str(pipeline_path)], 
                              capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            logger.info("✅ V6 Data pipeline completed successfully")
            return True
        else:
            logger.error(f"❌ V6 Data pipeline failed: {result.stderr}")
            return False
    except Exception as e:
        logger.error(f"❌ V6 Data pipeline error: {e}")
        return False

async def run_player_impact_calculation():
    """V6 Player Impact Model execution (Step 2 of dependency chain)."""
    try:
        logger.info("🤖 Starting V6 Player Impact Model calculation...")
        impact_path = Path(__file__).parent / "player_impact_model.py"
        result = subprocess.run([sys.executable, str(impact_path)], 
                              capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            logger.info("✅ V6 Player Impact Model completed successfully")
            return True
        else:
            logger.error(f"❌ V6 Player Impact Model failed: {result.stderr}")
            return False
    except Exception as e:
        logger.error(f"❌ V6 Player Impact Model error: {e}")
        return False

async def run_elo_calculation():
    """V6 Elo Rating Engine execution (Step 3 of dependency chain)."""
    try:
        logger.info("🤖 Starting V6 Dynamic Elo Rating Engine...")
        elo_path = Path(__file__).parent / "elo_calculator.py"
        result = subprocess.run([sys.executable, str(elo_path)], 
                              capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            logger.info("✅ V6 Elo Rating Engine completed successfully")
            return True
        else:
            logger.error(f"❌ V6 Elo Rating Engine failed: {result.stderr}")
            return False
    except Exception as e:
        logger.error(f"❌ V6 Elo Rating Engine error: {e}")
        return False

async def run_daily_intelligence_chain():
    """V6 Daily Intelligence Chain: data_pipeline → player_impact → elo_calculator."""
    try:
        logger.info("🚀 Starting V6 Daily Intelligence Chain (Dependency-Aware Execution)...")
        
        # Step 1: Data Pipeline
        data_success = await run_data_pipeline()
        if not data_success:
            logger.error("❌ Daily Intelligence Chain failed at Step 1 (Data Pipeline)")
            return False
        
        # Step 2: Player Impact Calculation (depends on data pipeline)
        impact_success = await run_player_impact_calculation()
        if not impact_success:
            logger.error("❌ Daily Intelligence Chain failed at Step 2 (Player Impact)")
            return False
        
        # Step 3: Elo Rating Calculation (depends on completed games)
        elo_success = await run_elo_calculation()
        if not elo_success:
            logger.error("❌ Daily Intelligence Chain failed at Step 3 (Elo Ratings)")
            return False
        
        logger.info("✅ V6 Daily Intelligence Chain completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"❌ V6 Daily Intelligence Chain error: {e}")
        return False

async def run_model_training():
    """V6 Model training execution with full dependency chain (Step 4)."""
    try:
        logger.info("🤖 Starting V6 Enhanced Model Training Pipeline...")
        
        # First ensure daily intelligence chain is up to date
        chain_success = await run_daily_intelligence_chain()
        if not chain_success:
            logger.warning("⚠️ Daily Intelligence Chain had issues, proceeding with model training...")
        
        # Run V6 Model Training (now with player impact features)
        logger.info("   Running V6 Player-Impact-Aware Model Training...")
        trainer_path = Path(__file__).parent / "model_trainer.py"
        result = subprocess.run([sys.executable, str(trainer_path)], 
                              capture_output=True, text=True, timeout=600)
        
        if result.returncode == 0:
            logger.info("✅ V6 Enhanced Model Training Pipeline completed successfully")
            return True
        else:
            logger.error(f"❌ V6 Model training failed: {result.stderr}")
            return False
    except Exception as e:
        logger.error(f"❌ V6 Model training pipeline error: {e}")
        return False

async def run_weekly_training_chain():
    """V6 Weekly Training Chain: daily_intelligence_chain → model_training."""
    try:
        logger.info("🚀 Starting V6 Weekly Training Chain (Full Dependency Pipeline)...")
        
        # Execute full model training with dependencies
        training_success = await run_model_training()
        
        if training_success:
            logger.info("✅ V6 Weekly Training Chain completed successfully!")
        else:
            logger.error("❌ V6 Weekly Training Chain failed")
        
        return training_success
        
    except Exception as e:
        logger.error(f"❌ V6 Weekly Training Chain error: {e}")
        return False

def calculate_brier_score(predicted_probability: float, actual_outcome: bool) -> float:
    """Calculate Brier score for a probabilistic prediction."""
    outcome_value = 1.0 if actual_outcome else 0.0
    return (predicted_probability - outcome_value) ** 2

async def update_closing_line_odds():
    """Fetch and update closing line odds for tracked games (runs every 15 minutes)."""
    try:
        logger.info("🔍 Updating closing line odds for tracked games...")
        
        with get_db() as conn:
            cursor = conn.cursor()
            
            # Find bets with odds_at_tracking but no closing_line_odds that are near game time
            cursor.execute("""
                SELECT bet_id, matchup, sport, odds_at_tracking, bet_date
                FROM bets 
                WHERE odds_at_tracking IS NOT NULL 
                  AND closing_line_odds IS NULL
                  AND status = 'Pending'
                  AND datetime(bet_date) <= datetime('now', '+4 hours')
                  AND datetime(bet_date) >= datetime('now', '-1 hour')
            """)
            
            pending_bets = cursor.fetchall()
            
            if not pending_bets:
                logger.info("   No bets requiring closing line odds updates")
                return
            
            logger.info(f"   Found {len(pending_bets)} bets requiring closing odds updates")
            
            # For demonstration, simulate closing line odds (in production, call odds API)
            import random
            random.seed(42)
            
            for bet in pending_bets:
                bet_id, matchup, sport, odds_at_tracking, bet_date = bet
                
                # Simulate realistic closing line movement (±10% typical)
                original_odds = odds_at_tracking
                movement_factor = 1 + random.uniform(-0.10, 0.10)
                closing_odds = round(original_odds * movement_factor)
                
                # Ensure odds stay in reasonable bounds
                if closing_odds > 0:
                    closing_odds = max(100, min(1000, closing_odds))
                else:
                    closing_odds = max(-1000, min(-100, closing_odds))
                
                # Update closing line odds
                cursor.execute("""
                    UPDATE bets 
                    SET closing_line_odds = ?, updated_at = ?
                    WHERE bet_id = ?
                """, (float(closing_odds), datetime.now().isoformat(), bet_id))
                
                logger.info(f"   Updated bet #{bet_id}: {matchup} - Tracking: {odds_at_tracking:+.0f}, Closing: {closing_odds:+.0f}")
            
            conn.commit()
            logger.info("✅ Closing line odds update complete")
        
    except Exception as e:
        logger.error(f"❌ Closing line odds update failed: {e}")

async def audit_settled_predictions():
    """Audit settled predictions and calculate Brier scores."""
    try:
        logger.info("🔍 Starting automated prediction auditing...")
        
        with get_db() as conn:
            cursor = conn.cursor()
            
            # Find predictions that need auditing (game date has passed but still pending)
            current_time = datetime.now().isoformat()
            cursor.execute("""
                SELECT p.prediction_id, p.matchup, p.confidence_score, p.predicted_pick, 
                       p.game_date, p.team_a, p.team_b
                FROM predictions p
                LEFT JOIN bets b ON p.matchup = b.matchup 
                WHERE p.game_date < ? 
                AND (b.status = 'Pending' OR b.status IS NULL)
                AND p.created_at >= datetime('now', '-30 days')
            """, (current_time,))
            
            pending_predictions = cursor.fetchall()
            
            if not pending_predictions:
                logger.info("   No predictions require auditing")
                return
            
            logger.info(f"   Found {len(pending_predictions)} predictions to audit")
            
            audited_count = 0
            
            for prediction in pending_predictions:
                prediction_id = prediction[0]
                matchup = prediction[1]
                confidence_score = prediction[2]
                predicted_pick = prediction[3]
                game_date = prediction[4]
                team_a = prediction[5]
                team_b = prediction[6]
                
                # For demo purposes, simulate fetching game results
                # In production, this would call the sports API
                actual_outcome = simulate_game_result(team_a, team_b, predicted_pick)
                
                # Convert confidence to probability (0-1 scale)
                predicted_probability = confidence_score / 100.0
                
                # Calculate Brier score
                brier_score = calculate_brier_score(predicted_probability, actual_outcome)
                
                # Update any related bets
                cursor.execute("""
                    SELECT bet_id FROM bets WHERE matchup = ? AND status = 'Pending'
                """, (matchup,))
                
                related_bets = cursor.fetchall()
                
                for bet_row in related_bets:
                    bet_id = bet_row[0]
                    
                    # Update bet status and Brier score
                    new_status = 'Won' if actual_outcome else 'Lost'
                    
                    cursor.execute("""
                        UPDATE bets 
                        SET status = ?, brier_score = ?, updated_at = ?
                        WHERE bet_id = ?
                    """, (new_status, brier_score, current_time, bet_id))
                    
                    # Calculate and update profit/loss if won
                    if actual_outcome:
                        cursor.execute("SELECT stake, odds FROM bets WHERE bet_id = ?", (bet_id,))
                        bet_data = cursor.fetchone()
                        if bet_data:
                            stake, odds = bet_data
                            profit = calculate_profit_loss(stake, odds, True)
                            cursor.execute("""
                                UPDATE bets SET profit_loss = ? WHERE bet_id = ?
                            """, (profit, bet_id))
                            
                            # Update ledger
                            current_balance = get_current_balance()
                            new_balance = current_balance + profit
                            cursor.execute("""
                                INSERT INTO ledger (timestamp, transaction_type, amount, running_balance, related_bet_id, description)
                                VALUES (?, 'Bet Settled', ?, ?, ?, ?)
                            """, (current_time, profit, new_balance, bet_id, f"Auto-settled winnings for bet #{bet_id}"))
                
                audited_count += 1
            
            conn.commit()
            
            logger.info(f"✅ Auditing complete! Processed {audited_count} predictions")
            
    except Exception as e:
        logger.error(f"❌ Auditing error: {e}")

def simulate_game_result(team_a: str, team_b: str, predicted_pick: str) -> bool:
    """Simulate game result for demo purposes."""
    import random
    
    # Simple simulation based on team names and prediction
    # In production, this would fetch actual results from sports API
    
    # Create some deterministic randomness based on team names
    seed = hash(f"{team_a}{team_b}") % 1000
    random.seed(seed)
    
    # Simulate with 60% accuracy for demonstration
    if "Lakers" in predicted_pick or "Yankees" in predicted_pick or "Patriots" in predicted_pick:
        return random.random() < 0.65  # Slightly favor certain teams
    else:
        return random.random() < 0.55
    """Audit settled predictions and calculate Brier scores."""
    try:
        logger.info("🔍 Starting automated prediction auditing...")
        
        with get_db() as conn:
            cursor = conn.cursor()
            
            # Find predictions that need auditing (game date has passed but still pending)
            current_time = datetime.now().isoformat()
            cursor.execute("""
                SELECT p.prediction_id, p.matchup, p.confidence_score, p.predicted_pick, 
                       p.game_date, p.team_a, p.team_b
                FROM predictions p
                LEFT JOIN bets b ON p.matchup = b.matchup 
                WHERE p.game_date < ? 
                AND (b.status = 'Pending' OR b.status IS NULL)
                AND p.created_at >= datetime('now', '-30 days')
            """, (current_time,))
            
            pending_predictions = cursor.fetchall()
            
            if not pending_predictions:
                logger.info("   No predictions require auditing")
                return
            
            logger.info(f"   Found {len(pending_predictions)} predictions to audit")
            
            audited_count = 0
            
            for prediction in pending_predictions:
                prediction_id = prediction[0]
                matchup = prediction[1]
                confidence_score = prediction[2]
                predicted_pick = prediction[3]
                game_date = prediction[4]
                team_a = prediction[5]
                team_b = prediction[6]
                
                # For demo purposes, simulate fetching game results
                # In production, this would call the sports API
                actual_outcome = simulate_game_result(team_a, team_b, predicted_pick)
                
                # Convert confidence to probability (0-1 scale)
                predicted_probability = confidence_score / 100.0
                
                # Calculate Brier score
                brier_score = calculate_brier_score(predicted_probability, actual_outcome)
                
                # Update any related bets
                cursor.execute("""
                    SELECT bet_id FROM bets WHERE matchup = ? AND status = 'Pending'
                """, (matchup,))
                
                related_bets = cursor.fetchall()
                
                for bet_row in related_bets:
                    bet_id = bet_row[0]
                    
                    # Update bet status and Brier score
                    new_status = 'Won' if actual_outcome else 'Lost'
                    
                    cursor.execute("""
                        UPDATE bets 
                        SET status = ?, brier_score = ?, updated_at = ?
                        WHERE bet_id = ?
                    """, (new_status, brier_score, current_time, bet_id))
                    
                    # Calculate and update profit/loss if won
                    if actual_outcome:
                        cursor.execute("SELECT stake, odds FROM bets WHERE bet_id = ?", (bet_id,))
                        bet_data = cursor.fetchone()
                        if bet_data:
                            stake, odds = bet_data
                            profit = calculate_profit_loss(stake, odds, True)
                            cursor.execute("""
                                UPDATE bets SET profit_loss = ? WHERE bet_id = ?
                            """, (profit, bet_id))
                            
                            # Update ledger
                            current_balance = get_current_balance()
                            new_balance = current_balance + profit
                            cursor.execute("""
                                INSERT INTO ledger (timestamp, transaction_type, amount, running_balance, related_bet_id, description)
                                VALUES (?, 'Bet Settled', ?, ?, ?, ?)
                            """, (current_time, profit, new_balance, bet_id, f"Auto-settled winnings for bet #{bet_id}"))
                
                audited_count += 1
            
            conn.commit()
            
            logger.info(f"✅ Auditing complete! Processed {audited_count} predictions")
            
    except Exception as e:
        logger.error(f"❌ Auditing error: {e}")

def simulate_game_result(team_a: str, team_b: str, predicted_pick: str) -> bool:
    """Simulate game result for demo purposes."""
    import random
    
    # Simple simulation based on team names and prediction
    # In production, this would fetch actual results from sports API
    
    # Create some deterministic randomness based on team names
    seed = hash(f"{team_a}{team_b}") % 1000
    random.seed(seed)
    
    # Simulate with 60% accuracy for demonstration
    if "Lakers" in predicted_pick or "Yankees" in predicted_pick or "Patriots" in predicted_pick:
        return random.random() < 0.65  # Slightly favor certain teams
    else:
        return random.random() < 0.55

@app.on_event("startup")
async def startup_event():
    """Initialize autonomous scheduling on startup."""
    if os.getenv("ENABLE_SCHEDULER", "true").lower() == "true":
        try:
            # V6 Enhanced Scheduling: Schedule daily intelligence chain at 3 AM
            data_hour = int(os.getenv("DATA_PIPELINE_HOUR", "3"))
            scheduler.add_job(
                run_daily_intelligence_chain,
                CronTrigger(hour=data_hour, minute=0),
                id="v6_daily_intelligence_chain",
                replace_existing=True
            )
            
            # V6 Enhanced Scheduling: Schedule weekly training chain on Sunday at 3 AM
            training_day = int(os.getenv("MODEL_TRAINING_DAY", "6"))  # 6 = Sunday
            training_hour = int(os.getenv("MODEL_TRAINING_HOUR", "3"))
            scheduler.add_job(
                run_weekly_training_chain,
                CronTrigger(day_of_week=training_day, hour=training_hour, minute=0),
                id="v6_weekly_training_chain",
                replace_existing=True
            )
            
            # Schedule daily auditing at 4 AM (after data pipeline)
            audit_hour = int(os.getenv("AUDIT_HOUR", "4"))
            scheduler.add_job(
                audit_settled_predictions,
                CronTrigger(hour=audit_hour, minute=0),
                id="daily_prediction_audit",
                replace_existing=True
            )
            
            # Schedule closing line odds updates every 15 minutes
            scheduler.add_job(
                update_closing_line_odds,
                CronTrigger(minute="*/15"),
                id="closing_line_odds_update", 
                replace_existing=True
            )
            
            scheduler.start()
            logger.info("🤖 V6 Autonomous Scheduling Engine Initialized")
            logger.info(f"📅 V6 Daily Intelligence Chain scheduled daily at {data_hour:02d}:00")
            logger.info(f"   └─ Sequence: Data Pipeline → Player Impact → Elo Ratings")
            logger.info(f"📅 V6 Weekly Training Chain scheduled weekly on day {training_day} at {training_hour:02d}:00")
            logger.info(f"   └─ Sequence: Daily Chain → Player-Impact-Aware Model Training")
            logger.info(f"📅 Prediction auditing scheduled daily at {audit_hour:02d}:00")
            logger.info(f"📅 Closing line odds updates every 15 minutes")
        except Exception as e:
            logger.error(f"❌ Failed to initialize scheduler: {e}")
    else:
        logger.info("⏸️ Autonomous scheduling disabled by configuration")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup scheduler on shutdown."""
    if scheduler.running:
        scheduler.shutdown()
        logger.info("🛑 Autonomous scheduling engine stopped")

# API Endpoints

@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Bet Copilot API",
        "version": "1.0.0",
        "status": "active",
        "endpoints": {
            "dashboard": "/api/dashboard/stats",
            "bets": "/api/bets",
            "predictions": "/api/predictions",
            "betai": "/api/betai/query"
        }
    }

@app.get("/api/dashboard/stats", response_model=DashboardStats)
async def get_dashboard_stats():
    """Get real-time dashboard statistics via database aggregation."""
    with get_db() as conn:
        cursor = conn.cursor()
        
        # Get current balance
        current_balance = get_current_balance()
        
        # Calculate total P/L from settled bets
        cursor.execute("""
            SELECT COALESCE(SUM(profit_loss), 0) as total_pl
            FROM bets 
            WHERE status IN ('Won', 'Lost')
        """)
        total_profit_loss = cursor.fetchone()[0]
        
        # Calculate total stakes for ROI
        cursor.execute("SELECT COALESCE(SUM(stake), 0) FROM bets")
        total_stakes = cursor.fetchone()[0]
        roi = (total_profit_loss / total_stakes * 100) if total_stakes > 0 else 0.0
        
        # Calculate win rate
        cursor.execute("SELECT COUNT(*) FROM bets WHERE status IN ('Won', 'Lost')")
        settled_count = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM bets WHERE status = 'Won'")
        won_count = cursor.fetchone()[0]
        
        win_rate = (won_count / settled_count * 100) if settled_count > 0 else 0.0
        
        # Count total and pending bets
        cursor.execute("SELECT COUNT(*) FROM bets")
        total_bets = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM bets WHERE status = 'Pending'")
        pending_bets = cursor.fetchone()[0]
        
        return DashboardStats(
            current_balance=current_balance,
            total_profit_loss=total_profit_loss,
            roi=roi,
            win_rate=win_rate,
            total_bets=total_bets,
            pending_bets=pending_bets
        )

@app.get("/api/bets", response_model=List[BetResponse])
async def get_bets(limit: int = 20):
    """Get recent bets."""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT * FROM bets 
            ORDER BY created_at DESC 
            LIMIT ?
        """, (limit,))
        
        rows = cursor.fetchall()
        return [BetResponse(**dict(row)) for row in rows]

@app.post("/api/bets", response_model=BetResponse)
async def create_bet(bet: BetCreate):
    """Create a new bet."""
    current_balance = get_current_balance()
    
    if current_balance < bet.stake:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Insufficient funds. Current balance: ${current_balance:.2f}"
        )
    
    with get_db() as conn:
        cursor = conn.cursor()
        now = datetime.now().isoformat()
        
        # Insert bet with odds_at_tracking for CLV calculation
        cursor.execute("""
            INSERT INTO bets (matchup, bet_type, stake, odds, sport, odds_at_tracking, bet_date, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (bet.matchup, bet.bet_type, bet.stake, bet.odds, bet.sport.upper(), float(bet.odds), now, now, now))
        
        bet_id = cursor.lastrowid
        
        # Update ledger
        new_balance = current_balance - bet.stake
        cursor.execute("""
            INSERT INTO ledger (timestamp, transaction_type, amount, running_balance, related_bet_id, description)
            VALUES (?, 'Bet Placed', ?, ?, ?, ?)
        """, (now, -bet.stake, new_balance, bet_id, f"Stake for bet #{bet_id}: {bet.matchup}"))
        
        conn.commit()
        
        # Return created bet
        cursor.execute("SELECT * FROM bets WHERE bet_id = ?", (bet_id,))
        row = cursor.fetchone()
        return BetResponse(**dict(row))

@app.put("/api/bets/{bet_id}/settle", response_model=BetResponse)
async def settle_bet(bet_id: int, settle: BetSettle):
    """Settle a pending bet."""
    with get_db() as conn:
        cursor = conn.cursor()
        
        # Get bet
        cursor.execute("SELECT * FROM bets WHERE bet_id = ?", (bet_id,))
        bet_row = cursor.fetchone()
        
        if not bet_row:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Bet not found"
            )
        
        bet_dict = dict(bet_row)
        
        if bet_dict["status"] != "Pending":
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Bet is already settled as {bet_dict['status']}"
            )
        
        # Calculate profit/loss
        won = (settle.result == "Won")
        profit_loss = calculate_profit_loss(bet_dict["stake"], bet_dict["odds"], won)
        
        now = datetime.now().isoformat()
        
        # Update bet
        cursor.execute("""
            UPDATE bets 
            SET status = ?, profit_loss = ?, updated_at = ?
            WHERE bet_id = ?
        """, (settle.result, profit_loss, now, bet_id))
        
        # Update ledger if won
        if won and profit_loss > 0:
            current_balance = get_current_balance()
            new_balance = current_balance + profit_loss
            cursor.execute("""
                INSERT INTO ledger (timestamp, transaction_type, amount, running_balance, related_bet_id, description)
                VALUES (?, 'Bet Settled', ?, ?, ?, ?)
            """, (now, profit_loss, new_balance, bet_id, f"Winnings for bet #{bet_id}: {bet_dict['matchup']}"))
        
        conn.commit()
        
        # Return updated bet
        cursor.execute("SELECT * FROM bets WHERE bet_id = ?", (bet_id,))
        row = cursor.fetchone()
        return BetResponse(**dict(row))

async def fetch_live_odds(sport: str = "upcoming") -> List[Dict[str, Any]]:
    """Fetch live odds from The Odds API."""
    if not ODDS_API_KEY or ODDS_API_KEY == "your_odds_api_key_here":
        logger.warning("⚠️ No valid odds API key configured, using mock data")
        return []
    
    try:
        # Get upcoming games with odds
        url = f"{ODDS_API_URL}/sports/{sport}/odds"
        params = {
            "apiKey": ODDS_API_KEY,
            "regions": "us",
            "markets": "h2h,spreads,totals",
            "oddsFormat": "american",
            "dateFormat": "iso"
        }
        
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        
        return response.json()
    except Exception as e:
        logger.error(f"❌ Failed to fetch live odds: {e}")
        return []

def calculate_implied_probability(american_odds: int) -> float:
    """Convert American odds to implied probability."""
    if american_odds > 0:
        return 100 / (american_odds + 100) * 100
    else:
        return abs(american_odds) / (abs(american_odds) + 100) * 100

def calculate_edge(model_probability: float, market_odds: int) -> Optional[float]:
    """Calculate betting edge (+EV percentage)."""
    try:
        implied_probability = calculate_implied_probability(market_odds)
        edge = model_probability - implied_probability
        return round(edge, 2)
    except:
        return None

def calculate_clv(odds_at_tracking: float, closing_line_odds: float) -> Optional[float]:
    """
    Calculate Closing Line Value (CLV) - the gold standard metric for betting edge.
    CLV = ((1 / Closing Odds Implied Probability) - (1 / Tracking Odds Implied Probability)) * 100
    """
    try:
        if odds_at_tracking is None or closing_line_odds is None:
            return None
        
        tracking_implied = calculate_implied_probability(int(odds_at_tracking)) / 100
        closing_implied = calculate_implied_probability(int(closing_line_odds)) / 100
        
        # CLV calculation: positive CLV means we got better odds than closing
        clv = ((1 / closing_implied) - (1 / tracking_implied)) / (1 / tracking_implied) * 100
        return round(clv, 2)
    except:
        return None

@app.get("/api/predictions", response_model=List[PredictionResponse])
async def get_predictions(sport: str, limit: int = 10):
    """Get V6 AI-generated lineup-aware predictions with live odds integration and player impact analysis."""
    # Validate sport parameter
    valid_sports = ["NBA", "NFL", "MLB"]
    sport = sport.upper()
    if sport not in valid_sports:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid sport. Must be one of: {', '.join(valid_sports)}"
        )
    
    with get_db() as conn:
        cursor = conn.cursor()
        
        # Get existing predictions for the specified sport
        cursor.execute("""
            SELECT * FROM predictions 
            WHERE sport = ?
            ORDER BY created_at DESC 
            LIMIT ?
        """, (sport, limit,))
        
        rows = cursor.fetchall()
        
        # If no predictions exist, generate new ones using V6 lineup-aware models
        if not rows:
            logger.info(f"No {sport} predictions found, generating new V6 lineup-aware predictions...")
            try:
                trainer_path = Path(__file__).parent / "model_trainer.py"
                result = subprocess.run([sys.executable, str(trainer_path), sport], 
                                      capture_output=True, text=True, timeout=120)
                
                if result.returncode == 0:
                    logger.info(f"✅ {sport} V6 lineup-aware model training completed successfully")
                    cursor.execute("""
                        SELECT * FROM predictions 
                        WHERE sport = ?
                        ORDER BY created_at DESC 
                        LIMIT ?
                    """, (sport, limit,))
                    rows = cursor.fetchall()
                else:
                    logger.warning("⚠️ V6 Model training failed, using enhanced fallback predictions")
                    # Generate V6 enhanced fallback predictions with player impact simulation
                    fallback_predictions = [
                        {
                            "matchup": "Lakers vs Warriors",
                            "sport": "NBA",
                            "league": "NBA", 
                            "game_date": (datetime.now() + timedelta(days=1)).isoformat(),
                            "team_a": "Lakers",
                            "team_b": "Warriors",
                            "predicted_pick": "Lakers ML",
                            "predicted_odds": -125,
                            "confidence_score": 74.8,
                            "projected_score": "Lakers 118, Warriors 114",
                            "calculated_edge": 6.2,
                            "created_at": datetime.now().isoformat(),
                            "model_version": "v6.0-player-impact-ensemble-fallback"
                        },
                        {
                            "matchup": "Patriots vs Bills", 
                            "sport": "NFL",
                            "league": "NFL",
                            "game_date": (datetime.now() + timedelta(days=2)).isoformat(),
                            "team_a": "Patriots",
                            "team_b": "Bills",
                            "predicted_pick": "Bills -3.5",
                            "predicted_odds": -108,
                            "confidence_score": 71.3,
                            "projected_score": "Bills 28, Patriots 21",
                            "calculated_edge": 4.7,
                            "created_at": datetime.now().isoformat(),
                            "model_version": "v6.0-player-impact-ensemble-fallback"
                        },
                        {
                            "matchup": "Yankees vs Red Sox",
                            "sport": "MLB", 
                            "league": "MLB",
                            "game_date": (datetime.now() + timedelta(days=1)).isoformat(),
                            "team_a": "Yankees",
                            "team_b": "Red Sox",
                            "predicted_pick": "Yankees ML",
                            "predicted_odds": -140,
                            "confidence_score": 77.1,
                            "projected_score": "Yankees 9, Red Sox 6",
                            "calculated_edge": 7.8,
                            "created_at": datetime.now().isoformat(),
                            "model_version": "v6.0-player-impact-ensemble-fallback"
                        }
                    ]
                    
                    # Insert fallback predictions
                    for pred in fallback_predictions:
                        cursor.execute("""
                            INSERT INTO predictions (
                                matchup, sport, league, game_date, team_a, team_b,
                                predicted_pick, predicted_odds, confidence_score,
                                projected_score, calculated_edge, created_at, model_version
                            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """, (
                            pred["matchup"], pred["sport"], pred["league"], pred["game_date"],
                            pred["team_a"], pred["team_b"], pred["predicted_pick"],
                            pred["predicted_odds"], pred["confidence_score"], pred["projected_score"],
                            pred["calculated_edge"], pred["created_at"], pred["model_version"]
                        ))
                    
                    conn.commit()
                    
                    # Fetch the newly created predictions
                    cursor.execute("""
                        SELECT * FROM predictions 
                        ORDER BY created_at DESC 
                        LIMIT ?
                    """, (limit,))
                    rows = cursor.fetchall()
                
            except Exception as e:
                logger.error(f"❌ Error generating ensemble predictions: {e}")
        
        # Convert to prediction objects and enhance with live odds
        predictions = []
        live_odds_data = await fetch_live_odds()
        
        for row in rows:
            prediction = PredictionResponse(**dict(row))
            
            # Try to match with live odds and recalculate edge
            if live_odds_data:
                for game in live_odds_data:
                    # Simple matching logic (could be enhanced)
                    game_teams = f"{game.get('home_team', '')} vs {game.get('away_team', '')}"
                    if any(team in prediction.matchup for team in [game.get('home_team', ''), game.get('away_team', '')]):
                        # Find best odds for our prediction type
                        best_odds = None
                        for bookmaker in game.get('bookmakers', []):
                            for market in bookmaker.get('markets', []):
                                if market['key'] == 'h2h':  # Money line market
                                    for outcome in market['outcomes']:
                                        if prediction.predicted_pick.startswith(outcome['name']):
                                            if best_odds is None or abs(outcome['price']) < abs(best_odds):
                                                best_odds = outcome['price']
                        
                        if best_odds:
                            # Recalculate edge with live market odds
                            edge = calculate_edge(prediction.confidence_score, best_odds)
                            if edge is not None:
                                prediction.calculated_edge = edge
                                prediction.predicted_odds = best_odds
                        break
            
            predictions.append(prediction)
        
        # Sort by calculated edge (highest first) for +EV opportunities
        predictions.sort(key=lambda x: x.calculated_edge if x.calculated_edge else 0, reverse=True)
        
        return predictions

@app.get("/api/performance/history", response_model=PerformanceHistory)
async def get_performance_history():
    """Get enhanced performance data with CLV metrics and Elo ratings for visualization."""
    with get_db() as conn:
        cursor = conn.cursor()
        
        # Get daily profit/loss aggregation from bets
        cursor.execute("""
            SELECT 
                DATE(created_at) as bet_date,
                SUM(CASE WHEN status = 'Won' THEN profit_loss WHEN status = 'Lost' THEN -stake ELSE 0 END) as daily_pl,
                COUNT(*) as bet_count
            FROM bets 
            WHERE status IN ('Won', 'Lost')
            GROUP BY DATE(created_at)
            ORDER BY bet_date ASC
        """)
        
        daily_results = cursor.fetchall()
        
        # Get daily Brier score aggregation
        cursor.execute("""
            SELECT 
                DATE(updated_at) as score_date,
                AVG(brier_score) as avg_brier_score,
                COUNT(*) as prediction_count
            FROM bets 
            WHERE brier_score IS NOT NULL
            GROUP BY DATE(updated_at)
            ORDER BY score_date ASC
        """)
        
        daily_brier_scores = cursor.fetchall()
        
        # Get overall statistics
        cursor.execute("""
            SELECT 
                COALESCE(SUM(profit_loss), 0) as total_pl,
                COUNT(*) as total_bets,
                COUNT(CASE WHEN status = 'Won' THEN 1 END) as wins,
                COUNT(CASE WHEN status = 'Lost' THEN 1 END) as losses,
                COALESCE(SUM(stake), 0) as total_stakes,
                COALESCE(AVG(brier_score), 0) as avg_brier_score,
                COUNT(CASE WHEN brier_score IS NOT NULL THEN 1 END) as scored_predictions
            FROM bets 
            WHERE status IN ('Won', 'Lost')
        """)
        
        stats = cursor.fetchone()
        
        # Calculate CLV metrics
        cursor.execute("""
            SELECT 
                odds_at_tracking, 
                closing_line_odds,
                sport
            FROM bets 
            WHERE odds_at_tracking IS NOT NULL 
              AND closing_line_odds IS NOT NULL
        """)
        
        clv_data = cursor.fetchall()
        
        # Calculate overall and sport-specific CLV
        clv_values = []
        clv_by_sport = {}
        
        for odds_tracking, odds_closing, sport in clv_data:
            clv = calculate_clv(odds_tracking, odds_closing)
            if clv is not None:
                clv_values.append(clv)
                if sport not in clv_by_sport:
                    clv_by_sport[sport] = []
                clv_by_sport[sport].append(clv)
        
        average_clv = sum(clv_values) / len(clv_values) if clv_values else None
        average_clv_by_sport = {
            sport: sum(values) / len(values) 
            for sport, values in clv_by_sport.items()
        }
        
        # Get sample Elo history for Lakers (team_id=1) or first available team
        cursor.execute("""
            SELECT er.date, er.elo_rating, t.team_name, t.sport
            FROM elo_ratings er
            JOIN teams t ON er.team_id = t.team_id
            WHERE t.team_name = 'Lakers' OR er.team_id = 1
            ORDER BY er.date DESC
            LIMIT 30
        """)
        
        elo_history_data = cursor.fetchall()
        elo_history_sample = [
            {
                'date': row[0],
                'elo_rating': row[1],
                'team_name': row[2],
                'sport': row[3]
            }
            for row in elo_history_data
        ]
        
        # Calculate performance metrics
        total_profit_loss = stats[0] if stats else 0.0
        total_bets = stats[1] if stats else 0
        wins = stats[2] if stats else 0
        losses = stats[3] if stats else 0
        total_stakes = stats[4] if stats else 0
        avg_brier_score = stats[5] if stats else 0.0
        total_predictions_scored = stats[6] if stats else 0
        
        win_rate = (wins / (wins + losses) * 100) if (wins + losses) > 0 else 0.0
        roi = (total_profit_loss / total_stakes * 100) if total_stakes > 0 else 0.0
        
        # Build time series data for P&L
        data_points = []
        cumulative_profit = 0.0
        running_balance = 1000.0  # Starting balance
        best_day = 0.0
        worst_day = 0.0
        
        for row in daily_results:
            bet_date, daily_pl, bet_count = row
            cumulative_profit += daily_pl
            running_balance += daily_pl
            
            if daily_pl > best_day:
                best_day = daily_pl
            if daily_pl < worst_day:
                worst_day = daily_pl
            
            data_points.append(PerformanceDataPoint(
                date=bet_date,
                profit_loss=daily_pl,
                running_balance=running_balance,
                cumulative_profit=cumulative_profit
            ))
        
        # Build time series data for Brier scores
        brier_score_points = []
        for row in daily_brier_scores:
            score_date, daily_avg_brier, prediction_count = row
            brier_score_points.append(BrierScoreDataPoint(
                date=score_date,
                avg_brier_score=daily_avg_brier,
                prediction_count=prediction_count
            ))
        
        # If no historical data, create sample points with current balance
        if not data_points:
            current_balance = get_current_balance()
            data_points.append(PerformanceDataPoint(
                date=datetime.now().strftime('%Y-%m-%d'),
                profit_loss=0.0,
                running_balance=current_balance,
                cumulative_profit=current_balance - 1000.0
            ))
        
        # If no Brier score data, create sample point
        if not brier_score_points:
            brier_score_points.append(BrierScoreDataPoint(
                date=datetime.now().strftime('%Y-%m-%d'),
                avg_brier_score=0.25,  # Sample Brier score
                prediction_count=0
            ))
        
        return PerformanceHistory(
            data_points=data_points,
            brier_score_points=brier_score_points,
            total_profit_loss=total_profit_loss,
            total_bets=total_bets,
            win_rate=win_rate,
            roi=roi,
            best_day=best_day,
            worst_day=worst_day,
            avg_brier_score=avg_brier_score,
            total_predictions_scored=total_predictions_scored,
            average_clv=average_clv,
            average_clv_by_sport=average_clv_by_sport,
            elo_history_sample=elo_history_sample
        )

@app.post("/api/betai/query")
async def query_betai(query: BetAIQuery):
    """V6 RAG-powered AI query with Player Impact Intelligence, Elo ratings, CLV validation, and comprehensive quantitative context."""
    try:
        # V6 Enhanced context retrieval with Player Impact, Elo and CLV insights
        context_data = []
        
        with get_db() as conn:
            cursor = conn.cursor()
            
            # V6 NEW: Get top player impact ratings and matchups
            cursor.execute("""
                SELECT p.player_name, t.team_name, p.sport, p.position, p.impact_rating
                FROM players p
                JOIN teams t ON p.team_id = t.team_id
                WHERE p.impact_rating > 0
                ORDER BY p.impact_rating DESC
                LIMIT 10
            """)
            top_players = cursor.fetchall()
            if top_players:
                context_data.append("⭐ V6 Top Player Impact Ratings (Player-Level Intelligence):")
                for player_name, team_name, sport, position, impact in top_players:
                    impact_tier = "🌟" if impact > 70 else "⚡" if impact > 50 else "📊"
                    context_data.append(f"{impact_tier} {player_name} ({team_name}, {sport} {position}): {impact:.1f} impact")
                context_data.append("")
            
            # V6 NEW: Get team aggregate impact comparisons
            cursor.execute("""
                SELECT t.team_name, t.sport, COUNT(p.player_id) as player_count,
                       AVG(p.impact_rating) as avg_impact, MAX(p.impact_rating) as top_impact
                FROM teams t
                LEFT JOIN players p ON t.team_id = p.team_id AND p.impact_rating > 0
                WHERE t.sport IN ('NBA', 'NFL')
                GROUP BY t.team_id, t.team_name, t.sport
                HAVING player_count > 0
                ORDER BY avg_impact DESC
                LIMIT 8
            """)
            team_impacts = cursor.fetchall()
            if team_impacts:
                context_data.append("🏆 V6 Team Aggregate Impact Rankings (Lineup Intelligence):")
                for team, sport, count, avg_impact, top_impact in team_impacts:
                    lineup_strength = "🔥" if avg_impact > 50 else "⚡" if avg_impact > 40 else "📊"
                    context_data.append(f"   {lineup_strength} {team} ({sport}): {avg_impact:.1f} avg ({count} players, {top_impact:.1f} top)")
                context_data.append("")
            
            # Get recent predictions with model insights (now including Player Impact models)
            cursor.execute("""
                SELECT matchup, predicted_pick, confidence_score, calculated_edge, model_version,
                       projected_score, created_at, sport
                FROM predictions 
                ORDER BY created_at DESC 
                LIMIT 8
            """)
            recent_predictions = cursor.fetchall()
            if recent_predictions:
                context_data.append("🤖 Recent AI Predictions (V6 Player-Impact-Aware Ensemble Models):")
                for pred in recent_predictions:
                    edge_indicator = "🔥" if pred[3] and pred[3] > 4 else "📊"
                    model_type = "⭐" if "player-impact" in pred[4].lower() else "🏆" if "elo" in pred[4].lower() else "⚡"
                    context_data.append(f"{edge_indicator} {pred[0]}: {pred[1]} ({pred[2]:.1f}% confidence, {pred[3]:.1f}% edge) {model_type}")
                    if pred[5]:  # projected_score
                        context_data.append(f"   Projected: {pred[5]} | Model: {pred[4]}")
                context_data.append("")
            
            # Get Elo ratings context for top teams
            cursor.execute("""
                SELECT t.team_name, t.sport, er.elo_rating
                FROM elo_ratings er
                JOIN teams t ON er.team_id = t.team_id
                WHERE er.date = date('now')
                ORDER BY er.elo_rating DESC
                LIMIT 6
            """)
            top_elo_teams = cursor.fetchall()
            if top_elo_teams:
                context_data.append("🏆 Current Top Elo Ratings (Dynamic Rating Engine):")
                for team_name, sport, elo_rating in top_elo_teams:
                    elo_strength = "🔥" if elo_rating > 1600 else "⚡" if elo_rating > 1500 else "📊"
                    context_data.append(f"   {elo_strength} {team_name} ({sport}): {elo_rating:.1f} Elo")
                context_data.append("")
            
            # Get CLV performance data
            cursor.execute("""
                SELECT 
                    odds_at_tracking, 
                    closing_line_odds,
                    sport,
                    matchup,
                    bet_date
                FROM bets 
                WHERE odds_at_tracking IS NOT NULL 
                  AND closing_line_odds IS NOT NULL
                ORDER BY bet_date DESC
                LIMIT 5
            """)
            clv_data = cursor.fetchall()
            if clv_data:
                context_data.append("💎 Closing Line Value (CLV) Performance:")
                total_clv = []
                for odds_tracking, odds_closing, sport, matchup, bet_date in clv_data:
                    clv = calculate_clv(odds_tracking, odds_closing)
                    if clv is not None:
                        total_clv.append(clv)
                        clv_indicator = "💎" if clv > 0 else "⚠️"
                        context_data.append(f"   {clv_indicator} {matchup[:30]}... ({sport}): {clv:+.1f}% CLV")
                
                if total_clv:
                    avg_clv = sum(total_clv) / len(total_clv)
                    clv_quality = "🌟 Exceptional" if avg_clv > 2 else "✅ Positive" if avg_clv > 0 else "🔍 Needs Work"
                    context_data.append(f"   Average CLV: {avg_clv:+.1f}% ({clv_quality})")
                context_data.append("")
            
            # Get comprehensive betting performance analytics
            cursor.execute("""
                SELECT 
                    COALESCE(SUM(profit_loss), 0) as total_pl,
                    COUNT(*) as total_bets,
                    COUNT(CASE WHEN status = 'Won' THEN 1 END) as wins,
                    COUNT(CASE WHEN status = 'Lost' THEN 1 END) as losses,
                    COUNT(CASE WHEN status = 'Pending' THEN 1 END) as pending,
                    COALESCE(AVG(stake), 0) as avg_stake,
                    COALESCE(MAX(profit_loss), 0) as best_win,
                    COALESCE(MIN(profit_loss), 0) as worst_loss
                FROM bets 
            """)
            performance = cursor.fetchone()
            if performance and performance[1] > 0:
                win_rate = (performance[2] / (performance[2] + performance[3]) * 100) if (performance[2] + performance[3]) > 0 else 0
                total_risked = performance[1] * performance[5] if performance[5] > 0 else 0
                roi = (performance[0] / total_risked * 100) if total_risked > 0 else 0
                
                context_data.append("📊 Quantitative Performance Analytics:")
                context_data.append(f"   Overall P&L: ${performance[0]:.2f} (ROI: {roi:.1f}%)")
                context_data.append(f"   Record: {performance[2]}-{performance[3]}-{performance[4]} ({win_rate:.1f}% win rate)")
                context_data.append(f"   Best Win: ${performance[6]:.2f} | Worst Loss: ${performance[7]:.2f}")
                context_data.append(f"   Average Stake: ${performance[5]:.2f}")
                context_data.append("")
            
            # Get recent betting activity with outcomes
            cursor.execute("""
                SELECT matchup, bet_type, stake, odds, status, profit_loss, created_at, sport
                FROM bets 
                ORDER BY created_at DESC 
                LIMIT 6
            """)
            recent_bets = cursor.fetchall()
            if recent_bets:
                context_data.append("💰 Recent Betting Activity:")
                for bet in recent_bets:
                    status_emoji = "✅" if bet[4] == "Won" else "❌" if bet[4] == "Lost" else "⏳"
                    pl_text = f" ({bet[5]:+.2f})" if bet[5] != 0 else ""
                    sport_emoji = "🏀" if bet[7] == "NBA" else "🏈" if bet[7] == "NFL" else "⚾"
                    context_data.append(f"{status_emoji} {sport_emoji} {bet[0][:25]}... ({bet[1]}): ${bet[2]} at {bet[3]:+d}{pl_text}")
                context_data.append("")
            
            # Get current bankroll status
            cursor.execute("SELECT running_balance FROM ledger ORDER BY entry_id DESC LIMIT 1")
            balance = cursor.fetchone()
            if balance:
                context_data.append(f"💳 Current Bankroll: ${balance[0]:.2f}")
                context_data.append("")
        
        # Build comprehensive context
        context = "\n".join(context_data) if context_data else "No recent data available."
        
        # V6 Enhanced system prompt with Player Impact Intelligence, Elo and CLV context
        system_prompt = f"""You are BetAI V6, an elite quantitative sports analyst powered by advanced Player Impact Intelligence, Dynamic Elo Rating Engine, and Closing Line Value validation systems. You have access to state-of-the-art lineup-aware ensemble models with real-time player impact ratings, Elo integration, and professional-grade CLV tracking.

V6 PLAYER-LEVEL INTELLIGENCE SYSTEMS:
{context}

CORE V6 CAPABILITIES:
- Player Impact Model: Advanced PER-style calculation quantifying individual player value (0-50+ scale)
- Lineup-Aware Predictions: Ensemble models using aggregated team player impact ratings as key features
- Dynamic Elo Rating Engine: Real-time team strength calculations with sport-specific parameters
- Closing Line Value (CLV) Validation: Gold-standard metric for sustainable betting edge
- Enhanced Ensemble Models: LightGBM + XGBoost with Player Impact + Elo feature integration
- Professional Performance Analytics: ROI, Sharpe ratios, and advanced risk metrics
- Strategic bankroll management with quantitative position sizing

ANALYSIS FRAMEWORK (V6 PROTOCOLS):
1. Player-Level Insights: Reference key player impact ratings and lineup strength differentials
2. Elo-Driven Context: Reference current team Elo ratings and strength differentials  
3. CLV Validation: Analyze closing line value performance as edge proof
4. Lineup Intelligence: Assess team aggregate impact ratings and star player advantages
5. Model Integration: Explain how player impact + Elo features enhance prediction accuracy
6. Performance Context: Assess user's quantitative metrics and betting patterns
7. Risk Management: Emphasize disciplined position sizing (1-3% of bankroll)

RESPONSE GUIDELINES:
- Lead with player-level insights when relevant matchups involve star players or depth advantages
- Reference specific player impact ratings and their predictive significance for lineup mismatches
- Explain team aggregate impact differentials and their importance for game outcomes
- Reference team Elo differentials and their predictive significance alongside player data
- Explain CLV performance and its importance for long-term profitability
- Provide strategic context based on V6 enhanced model outputs combining player + team intelligence
- Emphasize professional-grade risk management and disciplined execution
- Be concise but comprehensive, focusing on actionable quantitative intelligence

Remember: You're operating at a professional quantitative analyst level with granular player intelligence - provide strategic insights based on advanced player impact dynamics, Elo ratings, CLV validation, and lineup-aware ensemble model predictions."""

        # Prepare enhanced request to LM Studio
        payload = {
            "model": "local-model",
            "messages": [
                {
                    "role": "system", 
                    "content": system_prompt
                },
                {
                    "role": "user", 
                    "content": query.message
                }
            ],
            "temperature": 0.7,
            "max_tokens": 700,
            "stop": None
        }
        
        # Make request to LM Studio with enhanced error handling
        response = requests.post(
            LM_STUDIO_API_URL,
            json=payload,
            timeout=45,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            result = response.json()
            ai_response = result.get("choices", [{}])[0].get("message", {}).get("content", "No response from AI")
            return {"response": ai_response}
        else:
            # V6 Enhanced fallback response with Player Impact context
            fallback_response = f"""🤖 BetAI V6 (Player Impact + Elo + CLV Intelligence - Offline Mode)

I'm currently running on fallback intelligence while the LM Studio connection is restored. Here's what I can tell you based on your V6 player-level quantitative systems:

{context}

QUESTION: "{query.message}"

V6 ANALYSIS (Based on Available Data):
• Player Impact Model quantifies individual player value with PER-style calculations (0-50+ scale)
• Lineup-aware ensemble models aggregate team player impact ratings for game predictions
• Dynamic Elo Engine shows current team strength rankings with real-time updates
• CLV validation system tracking closing line performance for sustainable edge proof
• Enhanced ensemble models integrate Player Impact + Elo ratings as key predictive features  
• Look for predictions with player impact advantages > 10 points AND positive CLV for optimal opportunities
• Current performance metrics demonstrate the importance of disciplined position sizing

V6 QUANTITATIVE GUIDANCE:
• Maintain 1-3% position sizing relative to current bankroll for optimal Kelly growth
• Prioritize predictions with strong player impact differentials (>10 points) AND high model confidence
• Monitor team aggregate impact ratings - teams with 45+ average impact have significant advantages
• Focus on V6 Player-Impact-Aware model outputs for superior lineup-specific accuracy
• Star player advantages (impact rating >70) often create profitable betting opportunities
• Monitor CLV performance - positive CLV indicates sustainable predictive edge

For detailed V6 model explanations, player impact analysis, and lineup-specific predictions, please ensure LM Studio is running on localhost:1234."""
            return {"response": fallback_response}
            
    except requests.exceptions.Timeout:
        return {"response": "⏱️ BetAI V6 response timeout. The quantitative analysis engine is processing complex Player Impact, Elo and CLV calculations - please try again. Ensure LM Studio has sufficient resources allocated."}
    except requests.exceptions.ConnectionError:
        # Comprehensive offline analysis with V6 features
        try:
            offline_context = "Unable to retrieve recent V6 quantitative data"
            with get_db() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT running_balance FROM ledger ORDER BY entry_id DESC LIMIT 1")
                balance = cursor.fetchone()
                cursor.execute("SELECT COUNT(*) FROM elo_ratings WHERE date = date('now')")
                current_elos = cursor.fetchone()
                cursor.execute("SELECT COUNT(*) FROM bets WHERE odds_at_tracking IS NOT NULL")
                clv_bets = cursor.fetchone()
                cursor.execute("SELECT COUNT(*) FROM players WHERE impact_rating > 0")
                player_count = cursor.fetchone()
                
                if balance and current_elos and clv_bets and player_count:
                    offline_context = f"V6 Systems: Balance ${balance[0]:.2f} | Current Elo Ratings: {current_elos[0]} teams | CLV Tracked Bets: {clv_bets[0]} | Player Impact Models: {player_count[0]} players"
        except:
            offline_context = "Limited V6 quantitative data access in offline mode"
            
        return {"response": f"🔌 BetAI V6 is offline. LM Studio connection failed. {offline_context}\n\nPlease start LM Studio on localhost:1234 for full V6 quantitative analysis capabilities with Player Impact Intelligence, Elo dynamics and CLV validation."}
    except Exception as e:
        return {"response": f"⚠️ BetAI V6 encountered a quantitative analysis error: {str(e)}\n\nThis may indicate a V6 processing issue. Please verify system resources and try again."}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)