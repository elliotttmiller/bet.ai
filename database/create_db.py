#!/usr/bin/env python3
"""
Database Schema Creation Script
Creates SQLite database with optimized schema for betting analytics
"""

import sqlite3
from pathlib import Path

def create_database():
    """Create the SQLite database with complete schema."""
    db_path = Path(__file__).parent / "bet_copilot.db"
    
    # Remove existing database if it exists
    if db_path.exists():
        db_path.unlink()
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Create bets table
    cursor.execute("""
        CREATE TABLE bets (
            bet_id INTEGER PRIMARY KEY AUTOINCREMENT,
            matchup TEXT NOT NULL,
            bet_type TEXT NOT NULL,
            stake REAL NOT NULL,
            odds INTEGER NOT NULL,
            status TEXT NOT NULL DEFAULT 'Pending',
            profit_loss REAL NOT NULL DEFAULT 0.0,
            brier_score REAL,
            bet_date TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            CONSTRAINT valid_status CHECK (status IN ('Pending', 'Won', 'Lost')),
            CONSTRAINT positive_stake CHECK (stake > 0),
            CONSTRAINT valid_brier_score CHECK (brier_score IS NULL OR (brier_score >= 0 AND brier_score <= 1))
        )
    """)
    
    # Create ledger table for transaction tracking
    cursor.execute("""
        CREATE TABLE ledger (
            entry_id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            transaction_type TEXT NOT NULL,
            amount REAL NOT NULL,
            running_balance REAL NOT NULL,
            related_bet_id INTEGER,
            description TEXT NOT NULL,
            FOREIGN KEY (related_bet_id) REFERENCES bets (bet_id)
        )
    """)
    
    # Create teams table for ML model data
    cursor.execute("""
        CREATE TABLE teams (
            team_id INTEGER PRIMARY KEY AUTOINCREMENT,
            team_name TEXT NOT NULL UNIQUE,
            sport TEXT NOT NULL,
            league TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
    """)
    
    # Create games table for historical data
    cursor.execute("""
        CREATE TABLE games (
            game_id INTEGER PRIMARY KEY AUTOINCREMENT,
            external_id TEXT UNIQUE,
            sport TEXT NOT NULL,
            league TEXT NOT NULL,
            game_date TEXT NOT NULL,
            home_team_id INTEGER NOT NULL,
            away_team_id INTEGER NOT NULL,
            home_score INTEGER,
            away_score INTEGER,
            status TEXT NOT NULL DEFAULT 'scheduled',
            created_at TEXT NOT NULL,
            FOREIGN KEY (home_team_id) REFERENCES teams (team_id),
            FOREIGN KEY (away_team_id) REFERENCES teams (team_id),
            CONSTRAINT valid_status CHECK (status IN ('scheduled', 'completed', 'postponed', 'cancelled'))
        )
    """)
    
    # Create historical_stats table for team performance metrics
    cursor.execute("""
        CREATE TABLE historical_stats (
            stat_id INTEGER PRIMARY KEY AUTOINCREMENT,
            team_id INTEGER NOT NULL,
            games_played INTEGER NOT NULL DEFAULT 0,
            wins INTEGER NOT NULL DEFAULT 0,
            losses INTEGER NOT NULL DEFAULT 0,
            points_scored INTEGER NOT NULL DEFAULT 0,
            points_allowed INTEGER NOT NULL DEFAULT 0,
            last_updated TEXT NOT NULL,
            FOREIGN KEY (team_id) REFERENCES teams (team_id)
        )
    """)

    # Create predictions table (for ML-generated AI predictions)
    cursor.execute("""
        CREATE TABLE predictions (
            prediction_id INTEGER PRIMARY KEY AUTOINCREMENT,
            matchup TEXT NOT NULL,
            sport TEXT NOT NULL,
            league TEXT NOT NULL,
            game_date TEXT NOT NULL,
            team_a TEXT NOT NULL,
            team_b TEXT NOT NULL,
            predicted_pick TEXT NOT NULL,
            predicted_odds INTEGER NOT NULL,
            confidence_score REAL NOT NULL,
            projected_score TEXT,
            calculated_edge REAL,
            created_at TEXT NOT NULL,
            model_version TEXT DEFAULT 'v1.0',
            CONSTRAINT valid_confidence CHECK (confidence_score >= 0 AND confidence_score <= 100)
        )
    """)
    
    # Insert initial bankroll
    cursor.execute("""
        INSERT INTO ledger (timestamp, transaction_type, amount, running_balance, description)
        VALUES (datetime('now'), 'Initial', 1000.0, 1000.0, 'Initial bankroll deposit')
    """)
    
    conn.commit()
    conn.close()
    
    print(f"✅ Database created successfully at {db_path}")
    print(f"📊 Initialized with sample predictions and $1000 starting bankroll")

if __name__ == "__main__":
    create_database()