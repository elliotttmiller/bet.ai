#!/usr/bin/env python3
"""
Bet Copilot FastAPI Backend
Enterprise-grade betting analytics API with AI integration
"""

import os
import sqlite3
import requests
import joblib
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

# Database and model paths
DB_PATH = Path(__file__).parent.parent / "database" / "bet_copilot.db"
MODEL_PATH = Path(__file__).parent / "model.joblib"
SCALER_PATH = Path(__file__).parent / "scaler.joblib"
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
    """Get AI-generated ML predictions from trained model."""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT * FROM predictions 
            ORDER BY created_at DESC 
            LIMIT ?
        """, (limit,))
        
        rows = cursor.fetchall()
        
        # If no predictions exist, generate some using the trained model
        if not rows:
            print("No predictions found, checking for trained model...")
            try:
                if MODEL_PATH.exists() and SCALER_PATH.exists():
                    print("Found trained model, generating predictions...")
                    # Import and run model trainer to generate predictions
                    from pathlib import Path
                    import subprocess
                    import sys
                    
                    model_trainer_path = Path(__file__).parent / "model_trainer.py"
                    result = subprocess.run([sys.executable, str(model_trainer_path)], 
                                          capture_output=True, text=True, timeout=60)
                    
                    if result.returncode == 0:
                        print("Model trainer completed successfully")
                        # Try fetching again
                        cursor.execute("""
                            SELECT * FROM predictions 
                            ORDER BY created_at DESC 
                            LIMIT ?
                        """, (limit,))
                        rows = cursor.fetchall()
                    else:
                        print(f"Model trainer failed: {result.stderr}")
                else:
                    print("No trained model found, training new model...")
                    # Run model trainer to create and train model
                    from pathlib import Path
                    import subprocess
                    import sys
                    
                    model_trainer_path = Path(__file__).parent / "model_trainer.py"
                    result = subprocess.run([sys.executable, str(model_trainer_path)], 
                                          capture_output=True, text=True, timeout=120)
                    
                    if result.returncode == 0:
                        print("Model training completed successfully")
                        # Try fetching again
                        cursor.execute("""
                            SELECT * FROM predictions 
                            ORDER BY created_at DESC 
                            LIMIT ?
                        """, (limit,))
                        rows = cursor.fetchall()
                    else:
                        print(f"Model training failed: {result.stderr}")
                        
            except Exception as e:
                print(f"Error with model operations: {e}")
        
        return [PredictionResponse(**dict(row)) for row in rows]

@app.post("/api/betai/query")
async def query_betai(query: BetAIQuery):
    """RAG-enhanced AI query with retrieval from local database."""
    try:
        # Step 1: Analyze query to determine what data to retrieve
        query_text = query.message.lower()
        
        # Step 2: Retrieve relevant data from database based on query content
        relevant_data = {}
        
        with get_db() as conn:
            cursor = conn.cursor()
            
            # Check for betting/performance related queries
            if any(keyword in query_text for keyword in ['bet', 'profit', 'loss', 'roi', 'performance', 'win rate', 'balance']):
                # Get betting statistics
                cursor.execute("""
                    SELECT 
                        COUNT(*) as total_bets,
                        COUNT(CASE WHEN status = 'Won' THEN 1 END) as won_bets,
                        COUNT(CASE WHEN status = 'Lost' THEN 1 END) as lost_bets,
                        COUNT(CASE WHEN status = 'Pending' THEN 1 END) as pending_bets,
                        COALESCE(SUM(stake), 0) as total_staked,
                        COALESCE(SUM(profit_loss), 0) as total_profit_loss
                    FROM bets
                """)
                bet_stats = cursor.fetchone()
                
                if bet_stats and bet_stats[0] > 0:  # Check if there are any bets
                    total_bets = bet_stats[0] or 0
                    won_bets = bet_stats[1] or 0
                    lost_bets = bet_stats[2] or 0
                    pending_bets = bet_stats[3] or 0
                    total_staked = bet_stats[4] or 0
                    total_profit_loss = bet_stats[5] or 0
                    
                    win_rate = (won_bets / total_bets * 100) if total_bets > 0 else 0
                    roi = (total_profit_loss / total_staked * 100) if total_staked > 0 else 0
                    
                    relevant_data['betting_performance'] = {
                        'total_bets': total_bets,
                        'won_bets': won_bets,
                        'lost_bets': lost_bets,
                        'pending_bets': pending_bets,
                        'total_staked': total_staked,
                        'total_profit_loss': total_profit_loss,
                        'win_rate': win_rate,
                        'roi': roi
                    }
                
                # Get current balance
                cursor.execute("SELECT running_balance FROM ledger ORDER BY entry_id DESC LIMIT 1")
                balance_result = cursor.fetchone()
                relevant_data['current_balance'] = balance_result[0] if balance_result else 0
            
            # Check for recent bet queries
            if any(keyword in query_text for keyword in ['recent', 'latest', 'last', 'today', 'this week']):
                cursor.execute("""
                    SELECT matchup, bet_type, stake, odds, status, profit_loss, bet_date
                    FROM bets 
                    ORDER BY created_at DESC 
                    LIMIT 5
                """)
                recent_bets = cursor.fetchall()
                relevant_data['recent_bets'] = [dict(bet) for bet in recent_bets]
            
            # Check for prediction queries
            if any(keyword in query_text for keyword in ['prediction', 'pick', 'forecast', 'recommend']):
                cursor.execute("""
                    SELECT matchup, predicted_pick, confidence_score, predicted_odds, sport, league
                    FROM predictions 
                    ORDER BY created_at DESC 
                    LIMIT 5
                """)
                predictions = cursor.fetchall()
                relevant_data['recent_predictions'] = [dict(pred) for pred in predictions]
            
            # Check for sport-specific queries (NBA, NFL, etc.)
            sports = ['nba', 'nfl', 'basketball', 'football']
            mentioned_sport = next((sport for sport in sports if sport in query_text), None)
            if mentioned_sport:
                # Get sport-specific data
                sport_filter = 'NBA' if mentioned_sport in ['nba', 'basketball'] else 'NFL' if mentioned_sport in ['nfl', 'football'] else mentioned_sport.upper()
                
                cursor.execute("""
                    SELECT matchup, predicted_pick, confidence_score, sport
                    FROM predictions 
                    WHERE sport = ? 
                    ORDER BY created_at DESC 
                    LIMIT 3
                """, (sport_filter,))
                sport_predictions = cursor.fetchall()
                relevant_data[f'{sport_filter.lower()}_predictions'] = [dict(pred) for pred in sport_predictions]
        
        # Step 3: Build augmented prompt with retrieved data
        system_prompt = """You are BetAI, an expert sports betting analyst with access to the user's personal betting database. 
Provide insightful, data-driven responses based on the retrieved data. Be specific with numbers and percentages when available.
Always be helpful, professional, and focus on actionable insights."""
        
        # Create context from retrieved data
        context_parts = []
        if 'betting_performance' in relevant_data:
            perf = relevant_data['betting_performance']
            context_parts.append(f"""
BETTING PERFORMANCE DATA:
- Total Bets: {perf['total_bets']}
- Won: {perf['won_bets']}, Lost: {perf['lost_bets']}, Pending: {perf['pending_bets']}
- Win Rate: {perf['win_rate']:.1f}%
- Total Staked: ${perf['total_staked']:.2f}
- Total Profit/Loss: ${perf['total_profit_loss']:.2f}
- ROI: {perf['roi']:.1f}%
- Current Balance: ${relevant_data.get('current_balance', 0):.2f}""")
        
        if 'recent_bets' in relevant_data and relevant_data['recent_bets']:
            context_parts.append("RECENT BETS:")
            for bet in relevant_data['recent_bets'][:3]:
                context_parts.append(f"- {bet['matchup']}: {bet['bet_type']} ${bet['stake']} at {bet['odds']} odds - {bet['status']}")
        
        if 'recent_predictions' in relevant_data and relevant_data['recent_predictions']:
            context_parts.append("RECENT AI PREDICTIONS:")
            for pred in relevant_data['recent_predictions'][:3]:
                context_parts.append(f"- {pred['matchup']}: {pred['predicted_pick']} ({pred['confidence_score']:.1f}% confidence)")
        
        # Add sport-specific predictions if available
        for key, value in relevant_data.items():
            if key.endswith('_predictions') and value:
                sport_name = key.replace('_predictions', '').upper()
                context_parts.append(f"\n{sport_name} PREDICTIONS:")
                for pred in value[:2]:
                    context_parts.append(f"- {pred['matchup']}: {pred['predicted_pick']} ({pred['confidence_score']:.1f}% confidence)")
        
        # Combine context
        context = '\n'.join(context_parts) if context_parts else "No relevant betting data found in database."
        
        # Step 4: Create augmented prompt
        augmented_prompt = f"""Based on the following data from the user's personal betting database, please answer their question:

{context}

User Question: {query.message}

Please provide a helpful, data-driven response based on the available information."""
        
        # Step 5: Send to LM Studio
        payload = {
            "model": "local-model",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": augmented_prompt}
            ],
            "temperature": 0.7,
            "max_tokens": 800
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
            # Fallback response with retrieved data
            fallback_response = "BetAI is currently unavailable, but here's what I found in your data:\n\n"
            if context != "No relevant betting data found in database.":
                fallback_response += context.replace('\n', '\n\n')
            else:
                # Still try to show predictions even if no specific data retrieved
                with get_db() as conn:
                    cursor = conn.cursor()
                    cursor.execute("SELECT matchup, predicted_pick, confidence_score FROM predictions ORDER BY created_at DESC LIMIT 3")
                    predictions = cursor.fetchall()
                    if predictions:
                        fallback_response += "🤖 RECENT AI PREDICTIONS:\n"
                        for pred in predictions:
                            fallback_response += f"• {pred[0]}: {pred[1]} ({pred[2]:.1f}% confidence)\n"
                        fallback_response += f"\n💰 Current balance: ${relevant_data.get('current_balance', 1000):.2f}\n"
                    else:
                        fallback_response += "No relevant data found for your query. Try asking about your betting performance, recent bets, or predictions."
            return {"response": fallback_response}
            
    except requests.exceptions.RequestException:
        # Enhanced fallback with retrieved data
        fallback_response = "BetAI is currently unavailable (LM Studio connection failed), but I can provide some insights from your data:\n\n"
        
        # Include retrieved data in fallback
        if context != "No relevant betting data found in database.":
            fallback_response += context.replace('\n', '\n\n')
        else:
            # Try to provide basic analysis even without LLM
            try:
                with get_db() as conn:
                    cursor = conn.cursor()
                    cursor.execute("SELECT COUNT(*) as total_bets, COUNT(CASE WHEN status = 'Won' THEN 1 END) as won_bets FROM bets")
                    basic_stats = cursor.fetchone()
                    if basic_stats and basic_stats[0] > 0:
                        win_rate = (basic_stats[1] / basic_stats[0] * 100)
                        fallback_response += f"📊 You have {basic_stats[0]} total bets with a {win_rate:.1f}% win rate.\n"
                    
                    cursor.execute("SELECT running_balance FROM ledger ORDER BY entry_id DESC LIMIT 1")
                    balance = cursor.fetchone()
                    if balance:
                        fallback_response += f"💰 Current balance: ${balance[0]:.2f}\n"
                    
                    # Always include recent predictions if available
                    cursor.execute("SELECT matchup, predicted_pick, confidence_score FROM predictions ORDER BY created_at DESC LIMIT 3")
                    predictions = cursor.fetchall()
                    if predictions:
                        fallback_response += "\n🤖 RECENT AI PREDICTIONS:\n"
                        for pred in predictions:
                            fallback_response += f"• {pred[0]}: {pred[1]} ({pred[2]:.1f}% confidence)\n"
                        
                    fallback_response += "\nPlease ensure LM Studio is running on localhost:1234 for AI analysis."
            except Exception:
                fallback_response = "BetAI is currently unavailable. Please ensure LM Studio is running on localhost:1234"
        
        return {"response": fallback_response}
    except Exception as e:
        return {"response": f"Sorry, I encountered an error while processing your request: {str(e)}"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)