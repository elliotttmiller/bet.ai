#!/usr/bin/env python3
"""
Bet Copilot FastAPI Backend
Enterprise-grade betting analytics API with AI integration
"""

import os
import sqlite3
import requests
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any
from contextlib import contextmanager

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = FastAPI(
    title="Bet Copilot API",
    description="Enterprise-grade betting analytics with AI integration",
    version="1.0.0"
)

# CORS middleware for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Database path
DB_PATH = Path(__file__).parent.parent / "database" / "bet_copilot.db"
LM_STUDIO_API_URL = os.getenv("LM_STUDIO_API_URL", "http://localhost:1234/v1/chat/completions")

# Pydantic models
class BetCreate(BaseModel):
    matchup: str = Field(..., min_length=1, max_length=200)
    bet_type: str = Field(..., min_length=1, max_length=100)
    stake: float = Field(..., gt=0)
    odds: int = Field(..., ge=-1000, le=1000)

class BetSettle(BaseModel):
    result: str = Field(..., pattern="^(Won|Lost)$")

class BetResponse(BaseModel):
    bet_id: int
    matchup: str
    bet_type: str
    stake: float
    odds: int
    status: str
    profit_loss: float
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
        
        # Insert bet
        cursor.execute("""
            INSERT INTO bets (matchup, bet_type, stake, odds, bet_date, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (bet.matchup, bet.bet_type, bet.stake, bet.odds, now, now, now))
        
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

@app.get("/api/predictions", response_model=List[PredictionResponse])
async def get_predictions(limit: int = 10):
    """Get AI-generated ML predictions."""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT * FROM predictions 
            ORDER BY created_at DESC 
            LIMIT ?
        """, (limit,))
        
        rows = cursor.fetchall()
        if not rows:
            # If no predictions exist, generate some
            print("No predictions found, running model trainer...")
            try:
                from pathlib import Path
                import subprocess
                import sys
                
                model_trainer_path = Path(__file__).parent / "model_trainer.py"
                subprocess.run([sys.executable, str(model_trainer_path)], check=True)
                
                # Try fetching again
                cursor.execute("""
                    SELECT * FROM predictions 
                    ORDER BY created_at DESC 
                    LIMIT ?
                """, (limit,))
                rows = cursor.fetchall()
            except Exception as e:
                print(f"Error running model trainer: {e}")
        
        return [PredictionResponse(**dict(row)) for row in rows]

@app.post("/api/betai/query")
async def query_betai(query: BetAIQuery):
    """Proxy query to LM Studio AI."""
    try:
        # Prepare request to LM Studio
        payload = {
            "model": "local-model",
            "messages": [
                {
                    "role": "system", 
                    "content": "You are BetAI, an expert sports betting analyst. Provide insightful, data-driven responses about betting strategies, odds analysis, and sports predictions. Be concise but informative."
                },
                {
                    "role": "user", 
                    "content": query.message
                }
            ],
            "temperature": 0.7,
            "max_tokens": 500
        }
        
        # Make request to LM Studio
        response = requests.post(
            LM_STUDIO_API_URL,
            json=payload,
            timeout=30,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            result = response.json()
            ai_response = result.get("choices", [{}])[0].get("message", {}).get("content", "No response from AI")
            return {"response": ai_response}
        else:
            return {"response": "BetAI is currently unavailable. Please ensure LM Studio is running on localhost:1234"}
            
    except requests.exceptions.RequestException:
        return {"response": "BetAI is currently unavailable. Please ensure LM Studio is running on localhost:1234"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)