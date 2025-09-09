#!/usr/bin/env python3
"""
Dynamic Elo Rating Engine
Processes all historical game results in chronological order to calculate and store daily Elo ratings.
Implements advanced Elo rating system for sports betting edge calculation.
"""

import os
import sqlite3
import math
from datetime import datetime, timedelta
from pathlib import Path
from contextlib import contextmanager
from typing import Dict, List, Tuple, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Database path
DB_PATH = Path(__file__).parent.parent / "database" / "bet_copilot.db"

@contextmanager
def get_db():
    """Database connection context manager."""
    conn = sqlite3.connect(DB_PATH, timeout=30.0)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()

class DynamicEloEngine:
    """
    Advanced Dynamic Elo Rating System for Multi-Sport Analytics
    
    Features:
    - Sport-specific K-factors for optimal rating volatility
    - Home field advantage adjustments 
    - Margin of victory scaling for more accurate updates
    - Chronological processing for historical accuracy
    - Daily rating snapshots for time-series analysis
    """
    
    def __init__(self):
        # Sport-specific K-factors (rating volatility)
        self.sport_k_factors = {
            'NBA': 32.0,    # Higher volatility - more games per season
            'NFL': 40.0,    # Medium volatility - fewer games, higher impact  
            'MLB': 24.0     # Lower volatility - many games per season
        }
        
        # Home field advantage by sport (Elo points)
        self.home_advantage = {
            'NBA': 65.0,    # ~3 point spread equivalent
            'NFL': 85.0,    # ~3.5 point spread equivalent
            'MLB': 45.0     # ~2 point spread equivalent  
        }
        
        # Base Elo rating for new teams
        self.base_elo = 1500.0
        
        # Margin of victory scaling factors
        self.mov_multipliers = {
            'NBA': 2.5,     # Points difference scaling
            'NFL': 4.0,     # Points difference scaling  
            'MLB': 6.0      # Runs difference scaling
        }
    
    def calculate_expected_score(self, team_elo: float, opponent_elo: float, 
                               is_home: bool, sport: str) -> float:
        """
        Calculate expected score using Elo rating formula with home advantage.
        
        Formula: E = 1 / (1 + 10^((opponent_elo - team_elo - home_bonus) / 400))
        """
        home_bonus = self.home_advantage.get(sport, 65.0) if is_home else 0
        rating_diff = opponent_elo - team_elo - home_bonus
        expected_score = 1 / (1 + 10 ** (rating_diff / 400))
        return expected_score
    
    def calculate_margin_multiplier(self, margin: int, sport: str) -> float:
        """
        Calculate margin of victory multiplier for Elo updates.
        Larger margins of victory result in bigger rating changes.
        """
        if margin <= 0:
            return 1.0
        
        base_multiplier = self.mov_multipliers.get(sport, 3.0)
        # Logarithmic scaling: log(margin + 1) to prevent extreme multipliers
        multiplier = 1.0 + (math.log(margin + 1) / base_multiplier)
        return min(multiplier, 2.5)  # Cap at 2.5x to prevent extreme swings
    
    def update_elo_ratings(self, team_elo: float, opponent_elo: float,
                          actual_score: float, margin: int, is_home: bool, 
                          sport: str) -> float:
        """
        Update team Elo rating based on game result.
        
        Formula: New_Elo = Old_Elo + K * MOV_Multiplier * (Actual - Expected)
        """
        k_factor = self.sport_k_factors.get(sport, 32.0)
        expected_score = self.calculate_expected_score(team_elo, opponent_elo, is_home, sport)
        margin_multiplier = self.calculate_margin_multiplier(margin, sport)
        
        rating_change = k_factor * margin_multiplier * (actual_score - expected_score)
        new_elo = team_elo + rating_change
        
        # Ensure ratings stay within reasonable bounds
        new_elo = max(800, min(2400, new_elo))
        
        return new_elo
    
    def initialize_team_elos(self) -> Dict[Tuple[int, str], float]:
        """Initialize all teams with base Elo ratings."""
        team_elos = {}
        
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT team_id, sport FROM teams")
            teams = cursor.fetchall()
            
            for team in teams:
                team_id, sport = team[0], team[1]
                team_elos[(team_id, sport)] = self.base_elo
                
        logger.info(f"Initialized {len(team_elos)} team-sport Elo ratings at {self.base_elo}")
        return team_elos
    
    def process_historical_games(self) -> None:
        """
        Process all historical games in chronological order to calculate Elo ratings.
        This is the core method that implements the dynamic rating engine.
        """
        logger.info("🚀 Starting Dynamic Elo Rating Engine...")
        
        with get_db() as conn:
            cursor = conn.cursor()
            
            # Clear existing Elo ratings to rebuild from scratch
            cursor.execute("DELETE FROM elo_ratings")
            logger.info("   Cleared existing Elo ratings for fresh calculation")
            
            # First, ensure we have teams for demo
            cursor.execute("SELECT COUNT(*) FROM teams")
            team_count = cursor.fetchone()[0]
            
            if team_count == 0:
                logger.info("   Creating sample teams for demonstration...")
                teams = [
                    ('Lakers', 'NBA', 'NBA'), ('Warriors', 'NBA', 'NBA'),
                    ('Celtics', 'NBA', 'NBA'), ('Heat', 'NBA', 'NBA'),
                    ('Patriots', 'NFL', 'NFL'), ('Bills', 'NFL', 'NFL'),
                    ('Chiefs', 'NFL', 'NFL'), ('Cowboys', 'NFL', 'NFL'),
                    ('Yankees', 'MLB', 'MLB'), ('Red Sox', 'MLB', 'MLB'),
                    ('Dodgers', 'MLB', 'MLB'), ('Giants', 'MLB', 'MLB')
                ]
                
                for team_name, sport, league in teams:
                    cursor.execute("""
                        INSERT INTO teams (team_name, sport, league, created_at)
                        VALUES (?, ?, ?, ?)
                    """, (team_name, sport, league, datetime.now().isoformat()))
                
                conn.commit()
                logger.info(f"   Created {len(teams)} sample teams")
            
            # Initialize team Elos
            cursor.execute("SELECT team_id, sport FROM teams")
            teams_data = cursor.fetchall()
            
            current_elos = {}
            for team_id, sport in teams_data:
                current_elos[(team_id, sport)] = self.base_elo
                
            logger.info(f"Initialized {len(current_elos)} team-sport Elo ratings at {self.base_elo}")
            
            # Get all completed games in chronological order
            cursor.execute("""
                SELECT game_id, sport, game_date, home_team_id, away_team_id, 
                       home_score, away_score, status
                FROM games 
                WHERE status = 'completed'
                  AND home_score IS NOT NULL 
                  AND away_score IS NOT NULL
                ORDER BY game_date ASC, game_id ASC
            """)
            
            completed_games = cursor.fetchall()
            
            if not completed_games:
                # Generate synthetic historical data for demonstration
                logger.info("   No historical games found. Generating synthetic data...")
                self.generate_synthetic_games_inline(cursor, conn, teams_data)
                
                # Re-query for the generated games
                cursor.execute("""
                    SELECT game_id, sport, game_date, home_team_id, away_team_id, 
                           home_score, away_score, status
                    FROM games 
                    WHERE status = 'completed'
                      AND home_score IS NOT NULL 
                      AND away_score IS NOT NULL
                    ORDER BY game_date ASC, game_id ASC
                """)
                
                completed_games = cursor.fetchall()
            
            logger.info(f"   Processing {len(completed_games)} historical games...")
            
            # Track daily snapshots
            daily_snapshots = {}
            processed_games = 0
            
            for game in completed_games:
                game_id, sport, game_date, home_team_id, away_team_id = game[:5]
                home_score, away_score, status = game[5:]
                
                # Get current Elo ratings
                home_elo = current_elos.get((home_team_id, sport), self.base_elo)
                away_elo = current_elos.get((away_team_id, sport), self.base_elo)
                
                # Determine game outcome (1 = win, 0 = loss)
                home_won = home_score > away_score
                margin = abs(home_score - away_score)
                
                # Update Elo ratings
                new_home_elo = self.update_elo_ratings(
                    home_elo, away_elo, 1.0 if home_won else 0.0, 
                    margin, True, sport
                )
                
                new_away_elo = self.update_elo_ratings(
                    away_elo, home_elo, 0.0 if home_won else 1.0, 
                    margin, False, sport  
                )
                
                # Store updated ratings
                current_elos[(home_team_id, sport)] = new_home_elo
                current_elos[(away_team_id, sport)] = new_away_elo
                
                # Track daily snapshots
                date_key = game_date[:10]  # Extract YYYY-MM-DD
                if date_key not in daily_snapshots:
                    daily_snapshots[date_key] = {}
                
                daily_snapshots[date_key][(home_team_id, sport)] = new_home_elo
                daily_snapshots[date_key][(away_team_id, sport)] = new_away_elo
                
                processed_games += 1
                
                if processed_games % 100 == 0:
                    logger.info(f"   Processed {processed_games} games...")
            
            # Insert daily snapshots into database
            logger.info("   Storing daily Elo rating snapshots...")
            
            snapshot_count = 0
            for date_str, team_ratings in daily_snapshots.items():
                for (team_id, sport), elo_rating in team_ratings.items():
                    cursor.execute("""
                        INSERT OR REPLACE INTO elo_ratings 
                        (team_id, sport, elo_rating, date, created_at)
                        VALUES (?, ?, ?, ?, ?)
                    """, (
                        team_id, sport, elo_rating, date_str, 
                        datetime.now().isoformat()
                    ))
                    snapshot_count += 1
            
            # Insert current ratings for today
            today = datetime.now().strftime('%Y-%m-%d')
            for (team_id, sport), elo_rating in current_elos.items():
                cursor.execute("""
                    INSERT OR REPLACE INTO elo_ratings 
                    (team_id, sport, elo_rating, date, created_at)
                    VALUES (?, ?, ?, ?, ?)
                """, (
                    team_id, sport, elo_rating, today, 
                    datetime.now().isoformat()
                ))
                snapshot_count += 1
            
            conn.commit()
            
            logger.info(f"✅ Dynamic Elo Engine Processing Complete!")
            logger.info(f"   📊 Processed {processed_games} games across {len(current_elos)} teams")
            logger.info(f"   💾 Stored {snapshot_count} daily Elo rating snapshots")
            
            # Display sample current ratings
            cursor.execute("""
                SELECT t.team_name, t.sport, er.elo_rating
                FROM elo_ratings er
                JOIN teams t ON er.team_id = t.team_id
                WHERE er.date = ?
                ORDER BY er.elo_rating DESC
                LIMIT 10
            """, (today,))
            
            top_teams = cursor.fetchall()
            if top_teams:
                logger.info("   🏆 Current Top 10 Elo Ratings:")
                for i, (team_name, sport, elo_rating) in enumerate(top_teams, 1):
                    logger.info(f"      {i}. {team_name} ({sport}): {elo_rating:.1f}")
    
    def generate_synthetic_games_inline(self, cursor, conn, teams_data) -> None:
        """Generate synthetic games within the same database connection."""
        # Group teams by sport
        sport_teams = {}
        for team_id, sport in teams_data:
            if sport not in sport_teams:
                sport_teams[sport] = []
            sport_teams[sport].append(team_id)
        
        # Generate games for the last 6 months
        import random
        random.seed(42)
        
        base_date = datetime.now() - timedelta(days=180)
        
        games_to_insert = []
        game_id_counter = 1
        
        for days_offset in range(180):
            current_date = base_date + timedelta(days=days_offset)
            
            # Generate 2-4 games per day
            for _ in range(random.randint(2, 4)):
                # Pick a random sport
                sport = random.choice(list(sport_teams.keys()))
                sport_team_list = sport_teams[sport]
                
                if len(sport_team_list) < 2:
                    continue
                
                # Pick two different teams
                home_team_id = random.choice(sport_team_list)
                away_team_id = random.choice([t for t in sport_team_list if t != home_team_id])
                
                # Generate realistic scores based on sport
                if sport == 'NBA':
                    home_score = random.randint(95, 130)
                    away_score = random.randint(95, 130)
                elif sport == 'NFL':
                    home_score = random.randint(10, 45)
                    away_score = random.randint(10, 45)
                else:  # MLB
                    home_score = random.randint(2, 15)
                    away_score = random.randint(2, 15)
                
                # Slight home advantage in scoring
                if random.random() < 0.55:  # 55% chance home team wins
                    if home_score <= away_score:
                        home_score = away_score + random.randint(1, 5)
                
                games_to_insert.append((
                    f"synthetic_{game_id_counter}", sport, sport, 
                    current_date.isoformat(), home_team_id, away_team_id,
                    home_score, away_score, 'completed', datetime.now().isoformat()
                ))
                
                game_id_counter += 1
        
        # Batch insert games
        cursor.executemany("""
            INSERT INTO games (
                external_id, sport, league, game_date, home_team_id, away_team_id,
                home_score, away_score, status, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, games_to_insert)
        
        conn.commit()
        logger.info(f"   Generated {len(games_to_insert)} synthetic games")
    
    def generate_synthetic_games(self) -> None:
        """Generate synthetic historical games for demonstration."""
        logger.info("   Generating synthetic historical game data...")
        
        with get_db() as conn:
            cursor = conn.cursor()
            
            try:
                # First, ensure we have teams
                cursor.execute("SELECT COUNT(*) FROM teams")
                team_count = cursor.fetchone()[0]
                
                if team_count == 0:
                    # Create sample teams
                    teams = [
                        ('Lakers', 'NBA', 'NBA'), ('Warriors', 'NBA', 'NBA'),
                        ('Celtics', 'NBA', 'NBA'), ('Heat', 'NBA', 'NBA'),
                        ('Patriots', 'NFL', 'NFL'), ('Bills', 'NFL', 'NFL'),
                        ('Chiefs', 'NFL', 'NFL'), ('Cowboys', 'NFL', 'NFL'),
                        ('Yankees', 'MLB', 'MLB'), ('Red Sox', 'MLB', 'MLB'),
                        ('Dodgers', 'MLB', 'MLB'), ('Giants', 'MLB', 'MLB')
                    ]
                    
                    for team_name, sport, league in teams:
                        cursor.execute("""
                            INSERT INTO teams (team_name, sport, league, created_at)
                            VALUES (?, ?, ?, ?)
                        """, (team_name, sport, league, datetime.now().isoformat()))
                    
                    conn.commit()  # Commit team creation
                
                # Generate games for the last 6 months
                import random
                random.seed(42)
                
                cursor.execute("SELECT team_id, sport FROM teams")
                teams_data = cursor.fetchall()
                
                # Group teams by sport
                sport_teams = {}
                for team_id, sport in teams_data:
                    if sport not in sport_teams:
                        sport_teams[sport] = []
                    sport_teams[sport].append(team_id)
                
                base_date = datetime.now() - timedelta(days=180)
                
                game_id_counter = 1
                games_to_insert = []
                
                for days_offset in range(180):
                    current_date = base_date + timedelta(days=days_offset)
                    
                    # Generate 2-4 games per day
                    for _ in range(random.randint(2, 4)):
                        # Pick a random sport
                        sport = random.choice(list(sport_teams.keys()))
                        sport_team_list = sport_teams[sport]
                        
                        if len(sport_team_list) < 2:
                            continue
                        
                        # Pick two different teams
                        home_team_id = random.choice(sport_team_list)
                        away_team_id = random.choice([t for t in sport_team_list if t != home_team_id])
                        
                        # Generate realistic scores based on sport
                        if sport == 'NBA':
                            home_score = random.randint(95, 130)
                            away_score = random.randint(95, 130)
                        elif sport == 'NFL':
                            home_score = random.randint(10, 45)
                            away_score = random.randint(10, 45)
                        else:  # MLB
                            home_score = random.randint(2, 15)
                            away_score = random.randint(2, 15)
                        
                        # Slight home advantage in scoring
                        if random.random() < 0.55:  # 55% chance home team wins
                            if home_score <= away_score:
                                home_score = away_score + random.randint(1, 5)
                        
                        games_to_insert.append((
                            f"synthetic_{game_id_counter}", sport, sport, 
                            current_date.isoformat(), home_team_id, away_team_id,
                            home_score, away_score, 'completed', datetime.now().isoformat()
                        ))
                        
                        game_id_counter += 1
                
                # Batch insert games
                cursor.executemany("""
                    INSERT INTO games (
                        external_id, sport, league, game_date, home_team_id, away_team_id,
                        home_score, away_score, status, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, games_to_insert)
                
                conn.commit()
                logger.info(f"   Generated {len(games_to_insert)} synthetic games")
                
            except Exception as e:
                conn.rollback()
                logger.error(f"Error generating synthetic games: {e}")
                raise
    
    def get_current_elo(self, team_id: int, sport: str) -> Optional[float]:
        """Get the current Elo rating for a team in a specific sport."""
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT elo_rating FROM elo_ratings
                WHERE team_id = ? AND sport = ?
                ORDER BY date DESC
                LIMIT 1
            """, (team_id, sport))
            
            result = cursor.fetchone()
            return result[0] if result else self.base_elo
    
    def get_team_elo_history(self, team_id: int, sport: str, 
                           days: int = 90) -> List[Dict]:
        """Get historical Elo ratings for a team over specified days."""
        with get_db() as conn:
            cursor = conn.cursor()
            
            since_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
            
            cursor.execute("""
                SELECT date, elo_rating FROM elo_ratings
                WHERE team_id = ? AND sport = ? AND date >= ?
                ORDER BY date ASC
            """, (team_id, sport, since_date))
            
            results = cursor.fetchall()
            return [{'date': row[0], 'elo_rating': row[1]} for row in results]

def run_elo_calculator():
    """Main entry point to run the Dynamic Elo Rating Engine."""
    try:
        engine = DynamicEloEngine()
        engine.process_historical_games()
        return True
    except Exception as e:
        logger.error(f"❌ Elo Calculator failed: {e}")
        return False

if __name__ == "__main__":
    success = run_elo_calculator()
    exit(0 if success else 1)