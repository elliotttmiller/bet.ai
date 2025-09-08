"""
Bet Copilot FastAPI Application

A high-performance betting analytics API built with enterprise-grade architecture.
Implements Core Protocols V6 with emphasis on performance, reliability, and maintainability.

Key Features:
- RESTful API design with automatic OpenAPI documentation
- Pydantic models for type safety and validation
- SQLite database with transaction integrity
- Comprehensive error handling and logging
- Resource-optimized for target hardware constraints

Author: Bet Copilot Development Team
Version: Alpha 1.0
"""

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional, Union
from datetime import datetime, timezone
import sqlite3
import os
from pathlib import Path
from contextlib import contextmanager
import logging
from enum import Enum

# Configure logging for production readiness
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Initialize FastAPI application
app = FastAPI(
    title="Bet Copilot API",
    description="Advanced betting analytics and performance tracking system",
    version="1.0.0-alpha",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],  # React dev servers
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Database configuration
DATABASE_PATH = Path(__file__).parent.parent / "database" / "bet_copilot.db"


class BetStatus(str, Enum):
    """Enumeration for bet status values."""
    PENDING = "Pending"
    WON = "Won"
    LOST = "Lost"


class TransactionType(str, Enum):
    """Enumeration for transaction types in bankroll ledger."""
    INITIAL = "Initial"
    BET_PLACED = "Bet Placed"
    BET_SETTLED = "Bet Settled"


# Pydantic Models for API Contract Definition
class BetCreate(BaseModel):
    """Model for creating a new bet."""
    matchup: str = Field(..., min_length=1, max_length=200, description="The matchup being bet on")
    bet_type: str = Field(..., min_length=1, max_length=100, description="Type of bet (e.g., Moneyline, Spread)")
    stake: float = Field(..., gt=0, description="Amount wagered (must be positive)")
    odds: int = Field(..., description="American odds format (e.g., +150, -110)")
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "matchup": "Lakers vs Warriors",
                "bet_type": "Moneyline",
                "stake": 100.0,
                "odds": 150
            }
        }
    )


class Bet(BaseModel):
    """Model representing a complete bet record."""
    bet_id: int
    matchup: str
    bet_type: str
    stake: float
    odds: int
    status: BetStatus
    profit_loss: float
    bet_date: datetime
    created_at: datetime
    updated_at: datetime
    
    model_config = ConfigDict(
        from_attributes=True,
        json_encoders={
            datetime: lambda v: v.isoformat()
        }
    )


class BetSettle(BaseModel):
    """Model for settling a bet."""
    result: BetStatus = Field(..., description="Outcome of the bet")
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "result": "Won"
            }
        }
    )


class DashboardStats(BaseModel):
    """Model for dashboard statistics."""
    total_profit_loss: float = Field(..., description="Total profit/loss across all bets")
    roi: float = Field(..., description="Return on investment as percentage")
    win_rate: float = Field(..., description="Win rate as percentage")
    total_bets: int = Field(..., description="Total number of bets placed")
    pending_bets: int = Field(..., description="Number of pending bets")
    current_balance: float = Field(..., description="Current bankroll balance")
    recent_bets: List[Bet] = Field(..., description="Last 20 bets")


class AIQueryRequest(BaseModel):
    """Model for AI query requests."""
    query: str = Field(..., min_length=1, max_length=1000, description="User query for the AI system")


class AIQueryResponse(BaseModel):
    """Model for AI query responses."""
    response: str = Field(..., description="AI system response")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


# Database Connection Management
@contextmanager
def get_db_connection():
    """
    Context manager for database connections with proper transaction handling.
    
    Ensures database connections are properly closed and transactions are handled.
    """
    conn = None
    try:
        conn = sqlite3.connect(DATABASE_PATH)
        conn.row_factory = sqlite3.Row  # Enable dict-like access to rows
        conn.execute("PRAGMA foreign_keys = ON")  # Enable foreign key constraints
        yield conn
    except Exception as e:
        if conn:
            conn.rollback()
        logger.error(f"Database error: {e}")
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
    finally:
        if conn:
            conn.close()


# Core Business Logic Functions
def calculate_profit_loss(stake: float, odds: int, won: bool) -> float:
    """
    Calculate profit/loss for a bet based on American odds.
    
    Args:
        stake: Amount wagered
        odds: American odds (positive or negative)
        won: Whether the bet was won
        
    Returns:
        float: Profit if won (positive), loss if lost (negative)
    """
    if not won:
        return -stake  # Full stake is lost
    
    if odds > 0:
        # Positive odds: profit = stake * (odds/100)
        return stake * (odds / 100)
    else:
        # Negative odds: profit = stake / (abs(odds)/100)
        return stake / (abs(odds) / 100)


def get_current_balance() -> float:
    """Get the current bankroll balance from the ledger."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT running_balance 
            FROM bankroll_ledger 
            ORDER BY entry_id DESC 
            LIMIT 1
        """)
        result = cursor.fetchone()
        return result['running_balance'] if result else 0.0


def add_bet(bet_data: BetCreate) -> int:
    """
    Add a new bet to the system with complete transaction integrity.
    
    Args:
        bet_data: Bet creation data
        
    Returns:
        int: ID of the created bet
        
    Raises:
        HTTPException: If insufficient funds or database error
    """
    current_balance = get_current_balance()
    
    if current_balance < bet_data.stake:
        raise HTTPException(
            status_code=400, 
            detail=f"Insufficient funds. Current balance: ${current_balance:.2f}, Required: ${bet_data.stake:.2f}"
        )
    
    with get_db_connection() as conn:
        cursor = conn.cursor()
        
        try:
            # Insert bet record
            cursor.execute("""
                INSERT INTO bets (matchup, bet_type, stake, odds, status, profit_loss, bet_date)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                bet_data.matchup,
                bet_data.bet_type,
                bet_data.stake,
                bet_data.odds,
                BetStatus.PENDING.value,
                0.0,
                datetime.now(timezone.utc)
            ))
            
            bet_id = cursor.lastrowid
            
            # Update bankroll ledger
            new_balance = current_balance - bet_data.stake
            cursor.execute("""
                INSERT INTO bankroll_ledger 
                (transaction_type, amount, running_balance, related_bet_id, description)
                VALUES (?, ?, ?, ?, ?)
            """, (
                TransactionType.BET_PLACED.value,
                -bet_data.stake,  # Negative amount for outgoing
                new_balance,
                bet_id,
                f"Stake for bet #{bet_id}: {bet_data.matchup}"
            ))
            
            conn.commit()
            logger.info(f"Bet #{bet_id} created successfully. Stake: ${bet_data.stake:.2f}")
            return bet_id
            
        except Exception as e:
            conn.rollback()
            logger.error(f"Error creating bet: {e}")
            raise HTTPException(status_code=500, detail=f"Error creating bet: {str(e)}")


def settle_bet(bet_id: int, result: BetStatus) -> None:
    """
    Settle a pending bet and update bankroll accordingly.
    
    Args:
        bet_id: ID of the bet to settle
        result: Outcome of the bet (Won/Lost)
        
    Raises:
        HTTPException: If bet not found, already settled, or database error
    """
    with get_db_connection() as conn:
        cursor = conn.cursor()
        
        # Get bet details
        cursor.execute("SELECT * FROM bets WHERE bet_id = ?", (bet_id,))
        bet_row = cursor.fetchone()
        
        if not bet_row:
            raise HTTPException(status_code=404, detail=f"Bet #{bet_id} not found")
        
        if bet_row['status'] != BetStatus.PENDING.value:
            raise HTTPException(
                status_code=400, 
                detail=f"Bet #{bet_id} is already settled with status: {bet_row['status']}"
            )
        
        try:
            # Calculate profit/loss
            won = (result == BetStatus.WON)
            profit_loss = calculate_profit_loss(bet_row['stake'], bet_row['odds'], won)
            
            # Update bet record
            cursor.execute("""
                UPDATE bets 
                SET status = ?, profit_loss = ?, updated_at = CURRENT_TIMESTAMP
                WHERE bet_id = ?
            """, (result.value, profit_loss, bet_id))
            
            # Update bankroll ledger if bet was won
            if won and profit_loss > 0:
                current_balance = get_current_balance()
                new_balance = current_balance + profit_loss
                
                cursor.execute("""
                    INSERT INTO bankroll_ledger 
                    (transaction_type, amount, running_balance, related_bet_id, description)
                    VALUES (?, ?, ?, ?, ?)
                """, (
                    TransactionType.BET_SETTLED.value,
                    profit_loss,
                    new_balance,
                    bet_id,
                    f"Winnings for bet #{bet_id}: {bet_row['matchup']}"
                ))
            
            conn.commit()
            logger.info(f"Bet #{bet_id} settled as {result.value}. P/L: ${profit_loss:.2f}")
            
        except Exception as e:
            conn.rollback()
            logger.error(f"Error settling bet #{bet_id}: {e}")
            raise HTTPException(status_code=500, detail=f"Error settling bet: {str(e)}")


def get_dashboard_statistics() -> DashboardStats:
    """
    Calculate comprehensive dashboard statistics.
    
    Returns:
        DashboardStats: Complete dashboard metrics
    """
    with get_db_connection() as conn:
        cursor = conn.cursor()
        
        # Get total profit/loss
        cursor.execute("SELECT COALESCE(SUM(profit_loss), 0) as total_pl FROM bets")
        total_profit_loss = cursor.fetchone()['total_pl']
        
        # Get total stakes for ROI calculation
        cursor.execute("SELECT COALESCE(SUM(stake), 0) as total_stakes FROM bets")
        total_stakes = cursor.fetchone()['total_stakes']
        
        # Calculate ROI
        roi = (total_profit_loss / total_stakes * 100) if total_stakes > 0 else 0.0
        
        # Get win rate
        cursor.execute("""
            SELECT 
                COUNT(CASE WHEN status = 'Won' THEN 1 END) as wins,
                COUNT(CASE WHEN status IN ('Won', 'Lost') THEN 1 END) as settled_bets
            FROM bets
        """)
        win_stats = cursor.fetchone()
        win_rate = (win_stats['wins'] / win_stats['settled_bets'] * 100) if win_stats['settled_bets'] > 0 else 0.0
        
        # Get bet counts
        cursor.execute("""
            SELECT 
                COUNT(*) as total_bets,
                COUNT(CASE WHEN status = 'Pending' THEN 1 END) as pending_bets
            FROM bets
        """)
        bet_counts = cursor.fetchone()
        
        # Get recent bets
        cursor.execute("""
            SELECT * FROM bets 
            ORDER BY bet_date DESC 
            LIMIT 20
        """)
        recent_bets_rows = cursor.fetchall()
        
        # Convert to Bet objects
        recent_bets = []
        for row in recent_bets_rows:
            bet = Bet(
                bet_id=row['bet_id'],
                matchup=row['matchup'],
                bet_type=row['bet_type'],
                stake=row['stake'],
                odds=row['odds'],
                status=BetStatus(row['status']),
                profit_loss=row['profit_loss'],
                bet_date=datetime.fromisoformat(row['bet_date'].replace('Z', '+00:00')),
                created_at=datetime.fromisoformat(row['created_at'].replace('Z', '+00:00')),
                updated_at=datetime.fromisoformat(row['updated_at'].replace('Z', '+00:00'))
            )
            recent_bets.append(bet)
        
        current_balance = get_current_balance()
        
        return DashboardStats(
            total_profit_loss=total_profit_loss,
            roi=roi,
            win_rate=win_rate,
            total_bets=bet_counts['total_bets'],
            pending_bets=bet_counts['pending_bets'],
            current_balance=current_balance,
            recent_bets=recent_bets
        )


# API Endpoints
@app.get("/", summary="Health Check")
async def root():
    """Health check endpoint."""
    return {
        "message": "Bet Copilot API is running",
        "version": "1.0.0-alpha",
        "status": "operational"
    }


@app.post("/api/bets", response_model=dict, summary="Create New Bet")
async def create_bet(bet: BetCreate):
    """
    Create a new bet and deduct stake from bankroll.
    
    Args:
        bet: Bet creation data
        
    Returns:
        dict: Success message with bet ID
    """
    bet_id = add_bet(bet)
    return {
        "message": "Bet created successfully",
        "bet_id": bet_id,
        "stake": bet.stake,
        "matchup": bet.matchup
    }


@app.put("/api/bets/{bet_id}/settle", response_model=dict, summary="Settle Bet")
async def settle_bet_endpoint(bet_id: int, settlement: BetSettle):
    """
    Settle a pending bet as Won or Lost.
    
    Args:
        bet_id: ID of the bet to settle
        settlement: Settlement data containing result
        
    Returns:
        dict: Success message with settlement details
    """
    if settlement.result not in [BetStatus.WON, BetStatus.LOST]:
        raise HTTPException(
            status_code=400, 
            detail="Result must be either 'Won' or 'Lost'"
        )
    
    settle_bet(bet_id, settlement.result)
    return {
        "message": f"Bet #{bet_id} settled successfully",
        "result": settlement.result.value,
        "bet_id": bet_id
    }


@app.get("/api/dashboard/stats", response_model=DashboardStats, summary="Get Dashboard Statistics")
async def get_dashboard_stats():
    """
    Get comprehensive dashboard statistics including P/L, ROI, win rate, and recent bets.
    
    Returns:
        DashboardStats: Complete dashboard metrics
    """
    return get_dashboard_statistics()


@app.post("/api/betai/query", response_model=AIQueryResponse, summary="AI Query (Placeholder)")
async def query_ai(query: AIQueryRequest):
    """
    Placeholder endpoint for AI integration.
    
    Args:
        query: User query for the AI system
        
    Returns:
        AIQueryResponse: Mock AI response
    """
    return AIQueryResponse(
        response=f"AI integration is pending. Your query was: {query.query}",
        timestamp=datetime.now(timezone.utc)
    )


# Error Handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Custom HTTP exception handler for consistent error responses."""
    return {
        "error": True,
        "message": exc.detail,
        "status_code": exc.status_code
    }


if __name__ == "__main__":
    import uvicorn
    
    # Ensure database exists before starting server
    if not DATABASE_PATH.exists():
        logger.warning("Database not found. Please run database/create_db.py first.")
        exit(1)
    
    logger.info("Starting Bet Copilot API server...")
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )