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
            bet_date TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            CONSTRAINT valid_status CHECK (status IN ('Pending', 'Won', 'Lost')),
            CONSTRAINT positive_stake CHECK (stake > 0)
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
    
    # Create predictions table (for future AI predictions)
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
            CONSTRAINT valid_confidence CHECK (confidence_score >= 0 AND confidence_score <= 100)
        )
    """)
    
    # Insert initial bankroll
    cursor.execute("""
        INSERT INTO ledger (timestamp, transaction_type, amount, running_balance, description)
        VALUES (datetime('now'), 'Initial', 1000.0, 1000.0, 'Initial bankroll deposit')
    """)
    
    # Insert sample predictions for demo
    sample_predictions = [
        ('Lakers vs Warriors', 'NBA', 'NBA', '2024-01-15 20:00:00', 'Los Angeles Lakers', 'Golden State Warriors', 'Lakers -5.5', -110, 78.5, 'Lakers 112, Warriors 105', 3.2),
        ('Chiefs vs Bills', 'NFL', 'NFL', '2024-01-14 16:00:00', 'Kansas City Chiefs', 'Buffalo Bills', 'Over 48.5', -105, 82.3, 'Chiefs 28, Bills 24', 4.1),
        ('Celtics vs Heat', 'NBA', 'NBA', '2024-01-16 19:30:00', 'Boston Celtics', 'Miami Heat', 'Celtics ML', 150, 65.7, 'Celtics 108, Heat 102', 2.8),
        ('Cowboys vs Eagles', 'NFL', 'NFL', '2024-01-14 13:00:00', 'Dallas Cowboys', 'Philadelphia Eagles', 'Eagles -3', -115, 71.2, 'Eagles 21, Cowboys 17', 3.5),
        ('Nuggets vs Suns', 'NBA', 'NBA', '2024-01-17 22:00:00', 'Denver Nuggets', 'Phoenix Suns', 'Under 225.5', -110, 74.9, 'Nuggets 110, Suns 108', 2.9)
    ]
    
    cursor.executemany("""
        INSERT INTO predictions (matchup, sport, league, game_date, team_a, team_b, predicted_pick, predicted_odds, confidence_score, projected_score, calculated_edge, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))
    """, sample_predictions)
    
    conn.commit()
    conn.close()
    
    print(f"✅ Database created successfully at {db_path}")
    print(f"📊 Initialized with sample predictions and $1000 starting bankroll")

if __name__ == "__main__":
    create_database()