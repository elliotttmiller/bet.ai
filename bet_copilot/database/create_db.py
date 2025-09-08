#!/usr/bin/env python3
"""
Bet Copilot Database Initialization Script

This script creates the bet_copilot.db SQLite database with the required schema.
It is idempotent and safe to run multiple times.

Schema:
- bets: Core betting records with odds, stakes, and outcomes
- bankroll_ledger: Immutable transaction log for complete financial audit trail

Adheres to Core Protocol #3 (Verifiable Logic) and #4 (Impeccable Craftsmanship)
"""

import sqlite3
import os
from pathlib import Path
from datetime import datetime
from typing import Optional


class DatabaseInitializer:
    """
    Handles the creation and initialization of the Bet Copilot database.
    
    Implements idempotent database creation following enterprise patterns.
    """
    
    def __init__(self, db_path: str = "bet_copilot.db"):
        """
        Initialize the database initializer.
        
        Args:
            db_path: Path to the SQLite database file
        """
        self.db_path = Path(db_path)
        self.db_dir = self.db_path.parent
        
    def ensure_directory_exists(self) -> None:
        """Ensure the database directory exists."""
        self.db_dir.mkdir(parents=True, exist_ok=True)
        
    def create_database(self) -> None:
        """
        Create the database with all required tables if they don't exist.
        
        This method is idempotent - safe to run multiple times.
        """
        self.ensure_directory_exists()
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Enable foreign key support
            cursor.execute("PRAGMA foreign_keys = ON")
            
            # Create bets table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS bets (
                    bet_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    matchup TEXT NOT NULL,
                    bet_type TEXT NOT NULL,
                    stake REAL NOT NULL CHECK (stake > 0),
                    odds INTEGER NOT NULL,  -- American odds format
                    status TEXT NOT NULL CHECK (status IN ('Pending', 'Won', 'Lost')) DEFAULT 'Pending',
                    profit_loss REAL DEFAULT 0.0,
                    bet_date DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create bankroll_ledger table for immutable transaction history
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS bankroll_ledger (
                    entry_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    transaction_type TEXT NOT NULL CHECK (
                        transaction_type IN ('Initial', 'Bet Placed', 'Bet Settled')
                    ),
                    amount REAL NOT NULL,  -- Negative for outgoing (stakes), positive for incoming (wins)
                    running_balance REAL NOT NULL,
                    related_bet_id INTEGER,
                    description TEXT,
                    FOREIGN KEY (related_bet_id) REFERENCES bets(bet_id)
                )
            """)
            
            # Create indexes for performance optimization
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_bets_status ON bets(status)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_bets_date ON bets(bet_date)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_ledger_timestamp ON bankroll_ledger(timestamp)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_ledger_bet_id ON bankroll_ledger(related_bet_id)")
            
            # Create trigger to update updated_at timestamp
            cursor.execute("""
                CREATE TRIGGER IF NOT EXISTS update_bet_timestamp 
                AFTER UPDATE ON bets
                BEGIN
                    UPDATE bets SET updated_at = CURRENT_TIMESTAMP WHERE bet_id = NEW.bet_id;
                END
            """)
            
            conn.commit()
            
    def initialize_with_seed_data(self, initial_bankroll: float = 1000.0) -> None:
        """
        Initialize the database with seed data if it's empty.
        
        Args:
            initial_bankroll: Starting bankroll amount
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Check if ledger is empty
            cursor.execute("SELECT COUNT(*) FROM bankroll_ledger")
            ledger_count = cursor.fetchone()[0]
            
            if ledger_count == 0:
                # Insert initial bankroll entry
                cursor.execute("""
                    INSERT INTO bankroll_ledger 
                    (transaction_type, amount, running_balance, description)
                    VALUES (?, ?, ?, ?)
                """, ('Initial', initial_bankroll, initial_bankroll, 'Initial bankroll deposit'))
                
                conn.commit()
                print(f"✅ Database seeded with initial bankroll: ${initial_bankroll:.2f}")
            else:
                print("✅ Database already contains data, skipping seed initialization")
                
    def verify_schema(self) -> bool:
        """
        Verify the database schema is correctly created.
        
        Returns:
            bool: True if schema is valid, False otherwise
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Check tables exist
                cursor.execute("""
                    SELECT name FROM sqlite_master 
                    WHERE type='table' AND name IN ('bets', 'bankroll_ledger')
                """)
                tables = [row[0] for row in cursor.fetchall()]
                
                required_tables = {'bets', 'bankroll_ledger'}
                if not required_tables.issubset(set(tables)):
                    print(f"❌ Missing tables: {required_tables - set(tables)}")
                    return False
                
                # Check bets table columns
                cursor.execute("PRAGMA table_info(bets)")
                bet_columns = {row[1] for row in cursor.fetchall()}
                required_bet_columns = {
                    'bet_id', 'matchup', 'bet_type', 'stake', 'odds', 
                    'status', 'profit_loss', 'bet_date', 'created_at', 'updated_at'
                }
                
                if not required_bet_columns.issubset(bet_columns):
                    print(f"❌ Missing bet columns: {required_bet_columns - bet_columns}")
                    return False
                
                # Check ledger table columns
                cursor.execute("PRAGMA table_info(bankroll_ledger)")
                ledger_columns = {row[1] for row in cursor.fetchall()}
                required_ledger_columns = {
                    'entry_id', 'timestamp', 'transaction_type', 'amount', 
                    'running_balance', 'related_bet_id', 'description'
                }
                
                if not required_ledger_columns.issubset(ledger_columns):
                    print(f"❌ Missing ledger columns: {required_ledger_columns - ledger_columns}")
                    return False
                
                print("✅ Database schema verification passed")
                return True
                
        except Exception as e:
            print(f"❌ Schema verification failed: {e}")
            return False
            
    def get_database_stats(self) -> dict:
        """Get basic statistics about the database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("SELECT COUNT(*) FROM bets")
                bet_count = cursor.fetchone()[0]
                
                cursor.execute("SELECT COUNT(*) FROM bankroll_ledger")
                ledger_count = cursor.fetchone()[0]
                
                cursor.execute("SELECT running_balance FROM bankroll_ledger ORDER BY entry_id DESC LIMIT 1")
                current_balance_result = cursor.fetchone()
                current_balance = current_balance_result[0] if current_balance_result else 0.0
                
                return {
                    'bet_count': bet_count,
                    'ledger_entries': ledger_count,
                    'current_balance': current_balance,
                    'database_size_mb': self.db_path.stat().st_size / 1024 / 1024 if self.db_path.exists() else 0
                }
        except Exception as e:
            print(f"❌ Error getting database stats: {e}")
            return {}


def main():
    """Main execution function - idempotent database initialization."""
    print("🚀 Bet Copilot Database Initializer")
    print("=" * 50)
    
    # Initialize database
    db_init = DatabaseInitializer()
    
    try:
        # Create database and schema
        print("📊 Creating database schema...")
        db_init.create_database()
        print("✅ Database schema created successfully")
        
        # Verify schema
        print("\n🔍 Verifying database schema...")
        if not db_init.verify_schema():
            print("❌ Schema verification failed")
            return False
            
        # Initialize with seed data
        print("\n💰 Initializing bankroll...")
        db_init.initialize_with_seed_data(initial_bankroll=1000.0)
        
        # Display stats
        print("\n📈 Database Statistics:")
        stats = db_init.get_database_stats()
        for key, value in stats.items():
            if key == 'current_balance':
                print(f"  {key.replace('_', ' ').title()}: ${value:.2f}")
            elif key == 'database_size_mb':
                print(f"  {key.replace('_', ' ').title()}: {value:.2f} MB")
            else:
                print(f"  {key.replace('_', ' ').title()}: {value}")
                
        print("\n✅ Database initialization completed successfully!")
        print(f"📁 Database location: {db_init.db_path.resolve()}")
        
        return True
        
    except Exception as e:
        print(f"❌ Database initialization failed: {e}")
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)