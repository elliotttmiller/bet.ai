#!/usr/bin/env python3
"""
Mock Data Seeder for Testing
Populates database with sample teams, games and stats for development
"""

import sqlite3
import random
from datetime import datetime, timedelta
from pathlib import Path
from contextlib import contextmanager

DB_PATH = Path(__file__).parent.parent / "database" / "bet_copilot.db"

@contextmanager
def get_db():
    """Database connection context manager."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()

def seed_mock_data():
    """Seed the database with mock NBA and NFL data."""
    print("🎯 Seeding Mock Sports Data...")
    
    # NBA Teams
    nba_teams = [
        'Los Angeles Lakers', 'Golden State Warriors', 'Boston Celtics', 'Miami Heat',
        'Denver Nuggets', 'Phoenix Suns', 'Dallas Mavericks', 'Philadelphia 76ers',
        'Milwaukee Bucks', 'Brooklyn Nets', 'Toronto Raptors', 'Chicago Bulls',
        'New York Knicks', 'Memphis Grizzlies', 'Atlanta Hawks', 'Portland Trail Blazers'
    ]
    
    # NFL Teams
    nfl_teams = [
        'Kansas City Chiefs', 'Buffalo Bills', 'Dallas Cowboys', 'Philadelphia Eagles',
        'San Francisco 49ers', 'New England Patriots', 'Green Bay Packers', 'Pittsburgh Steelers',
        'Tampa Bay Buccaneers', 'Baltimore Ravens', 'Denver Broncos', 'Seattle Seahawks',
        'Los Angeles Rams', 'Minnesota Vikings', 'Indianapolis Colts', 'Tennessee Titans'
    ]
    
    with get_db() as conn:
        cursor = conn.cursor()
        
        # Clear existing data
        cursor.execute("DELETE FROM predictions")
        cursor.execute("DELETE FROM historical_stats")
        cursor.execute("DELETE FROM games")
        cursor.execute("DELETE FROM teams")
        
        team_ids = {}
        
        # Insert NBA teams
        print("   Adding NBA teams...")
        for team_name in nba_teams:
            cursor.execute("""
                INSERT INTO teams (team_name, sport, league, created_at)
                VALUES (?, ?, ?, ?)
            """, (team_name, 'NBA', 'NBA', datetime.now().isoformat()))
            team_ids[team_name] = cursor.lastrowid
        
        # Insert NFL teams
        print("   Adding NFL teams...")
        for team_name in nfl_teams:
            cursor.execute("""
                INSERT INTO teams (team_name, sport, league, created_at)
                VALUES (?, ?, ?, ?)
            """, (team_name, 'NFL', 'NFL', datetime.now().isoformat()))
            team_ids[team_name] = cursor.lastrowid
        
        # Generate NBA games (completed and upcoming)
        print("   Generating NBA games...")
        game_id_counter = 1
        
        # Generate past NBA games
        for i in range(50):  # 50 completed games
            home_team = random.choice(nba_teams)
            away_team = random.choice([t for t in nba_teams if t != home_team])
            
            game_date = datetime.now() - timedelta(days=random.randint(1, 90))
            home_score = random.randint(85, 130)
            away_score = random.randint(85, 130)
            
            cursor.execute("""
                INSERT INTO games (external_id, sport, league, game_date, home_team_id, away_team_id, home_score, away_score, status, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                f"nba_{game_id_counter}", 'NBA', 'NBA', game_date.isoformat(),
                team_ids[home_team], team_ids[away_team], home_score, away_score,
                'completed', datetime.now().isoformat()
            ))
            game_id_counter += 1
        
        # Generate upcoming NBA games
        for i in range(15):  # 15 upcoming games
            home_team = random.choice(nba_teams)
            away_team = random.choice([t for t in nba_teams if t != home_team])
            
            game_date = datetime.now() + timedelta(days=random.randint(1, 30))
            
            cursor.execute("""
                INSERT INTO games (external_id, sport, league, game_date, home_team_id, away_team_id, status, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                f"nba_{game_id_counter}", 'NBA', 'NBA', game_date.isoformat(),
                team_ids[home_team], team_ids[away_team], 'scheduled', datetime.now().isoformat()
            ))
            game_id_counter += 1
        
        # Generate NFL games
        print("   Generating NFL games...")
        
        # Generate past NFL games
        for i in range(30):  # 30 completed games
            home_team = random.choice(nfl_teams)
            away_team = random.choice([t for t in nfl_teams if t != home_team])
            
            game_date = datetime.now() - timedelta(days=random.randint(1, 120))
            home_score = random.randint(7, 45)
            away_score = random.randint(7, 45)
            
            cursor.execute("""
                INSERT INTO games (external_id, sport, league, game_date, home_team_id, away_team_id, home_score, away_score, status, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                f"nfl_{game_id_counter}", 'NFL', 'NFL', game_date.isoformat(),
                team_ids[home_team], team_ids[away_team], home_score, away_score,
                'completed', datetime.now().isoformat()
            ))
            game_id_counter += 1
        
        # Generate upcoming NFL games
        for i in range(10):  # 10 upcoming games
            home_team = random.choice(nfl_teams)
            away_team = random.choice([t for t in nfl_teams if t != home_team])
            
            game_date = datetime.now() + timedelta(days=random.randint(1, 21))
            
            cursor.execute("""
                INSERT INTO games (external_id, sport, league, game_date, home_team_id, away_team_id, status, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                f"nfl_{game_id_counter}", 'NFL', 'NFL', game_date.isoformat(),
                team_ids[home_team], team_ids[away_team], 'scheduled', datetime.now().isoformat()
            ))
            game_id_counter += 1
        
        # Generate team stats
        print("   Calculating team statistics...")
        all_teams = nba_teams + nfl_teams
        for team_name in all_teams:
            team_id = team_ids[team_name]
            
            # Calculate stats from games
            cursor.execute("""
                SELECT home_team_id, away_team_id, home_score, away_score
                FROM games 
                WHERE (home_team_id = ? OR away_team_id = ?) AND status = 'completed'
                AND home_score IS NOT NULL AND away_score IS NOT NULL
            """, (team_id, team_id))
            
            games = cursor.fetchall()
            
            games_played = len(games)
            wins = 0
            losses = 0
            points_scored = 0
            points_allowed = 0
            
            for game in games:
                home_team_id, away_team_id, home_score, away_score = game
                
                if home_team_id == team_id:
                    # Team was home
                    points_scored += home_score
                    points_allowed += away_score
                    if home_score > away_score:
                        wins += 1
                    else:
                        losses += 1
                else:
                    # Team was away
                    points_scored += away_score
                    points_allowed += home_score
                    if away_score > home_score:
                        wins += 1
                    else:
                        losses += 1
            
            if games_played > 0:
                cursor.execute("""
                    INSERT INTO historical_stats (team_id, games_played, wins, losses, points_scored, points_allowed, last_updated)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (team_id, games_played, wins, losses, points_scored, points_allowed, datetime.now().isoformat()))
        
        conn.commit()
        
        # Summary
        cursor.execute("SELECT COUNT(*) FROM teams")
        team_count = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM games")
        game_count = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM historical_stats")
        stats_count = cursor.fetchone()[0]
        
        print(f"✅ Mock Data Seeded Successfully!")
        print(f"   📊 Teams: {team_count}")
        print(f"   🏟️  Games: {game_count}")  
        print(f"   📈 Team Stats: {stats_count}")

if __name__ == "__main__":
    seed_mock_data()