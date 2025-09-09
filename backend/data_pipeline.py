#!/usr/bin/env python3
"""
Data Pipeline for Sports Data Ingestion
Fetches data from TheSportsDB API and populates database tables
"""

import os
import sqlite3
import requests
from datetime import datetime, timedelta
from pathlib import Path
from contextlib import contextmanager
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
DB_PATH = Path(__file__).parent.parent / "database" / "bet_copilot.db"
SPORTS_API_URL = os.getenv("SPORTS_API_URL", "https://www.thesportsdb.com/api/v1/json/3")

@contextmanager
def get_db():
    """Database connection context manager."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()

def fetch_nba_teams():
    """Fetch NBA teams from TheSportsDB API."""
    try:
        url = f"{SPORTS_API_URL}/search_all_teams.php?l=NBA"
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        data = response.json()
        return data.get('teams', []) if data else []
    except Exception as e:
        print(f"Error fetching NBA teams: {e}")
        return []

def fetch_nfl_teams():
    """Fetch NFL teams from TheSportsDB API."""
    try:
        url = f"{SPORTS_API_URL}/search_all_teams.php?l=NFL"
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        data = response.json()
        return data.get('teams', []) if data else []
    except Exception as e:
        print(f"Error fetching NFL teams: {e}")
        return []

def fetch_recent_games(league_id):
    """Fetch recent games for a league."""
    try:
        # Get last 15 rounds of games
        url = f"{SPORTS_API_URL}/eventspastleague.php?id={league_id}"
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        data = response.json()
        return data.get('events', []) if data else []
    except Exception as e:
        print(f"Error fetching recent games for league {league_id}: {e}")
        return []

def fetch_upcoming_games(league_id):
    """Fetch upcoming games for a league."""
    try:
        url = f"{SPORTS_API_URL}/eventsnextleague.php?id={league_id}"
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        data = response.json()
        return data.get('events', []) if data else []
    except Exception as e:
        print(f"Error fetching upcoming games for league {league_id}: {e}")
        return []

def upsert_team(cursor, team_data, sport, league):
    """Insert or update team in database."""
    team_name = team_data.get('strTeam')
    if not team_name:
        return None
        
    # Check if team exists
    cursor.execute(
        "SELECT team_id FROM teams WHERE team_name = ? AND sport = ? AND league = ?", 
        (team_name, sport, league)
    )
    existing = cursor.fetchone()
    
    if existing:
        return existing[0]
    
    # Insert new team
    cursor.execute("""
        INSERT INTO teams (team_name, sport, league, created_at)
        VALUES (?, ?, ?, ?)
    """, (team_name, sport, league, datetime.now().isoformat()))
    
    return cursor.lastrowid

def upsert_game(cursor, game_data, sport, league):
    """Insert or update game in database."""
    external_id = game_data.get('idEvent')
    if not external_id:
        return None
        
    home_team = game_data.get('strHomeTeam')
    away_team = game_data.get('strAwayTeam')
    game_date = game_data.get('dateEvent')
    game_time = game_data.get('strTime', '00:00:00')
    home_score = game_data.get('intHomeScore')
    away_score = game_data.get('intAwayScore')
    
    if not all([home_team, away_team, game_date]):
        return None
    
    # Get team IDs
    cursor.execute(
        "SELECT team_id FROM teams WHERE team_name = ? AND sport = ? AND league = ?", 
        (home_team, sport, league)
    )
    home_team_row = cursor.fetchone()
    if not home_team_row:
        return None
        
    cursor.execute(
        "SELECT team_id FROM teams WHERE team_name = ? AND sport = ? AND league = ?", 
        (away_team, sport, league)
    )
    away_team_row = cursor.fetchone()
    if not away_team_row:
        return None
    
    home_team_id = home_team_row[0]
    away_team_id = away_team_row[0]
    
    # Create datetime string
    game_datetime = f"{game_date} {game_time}"
    
    # Determine status
    status = 'scheduled'
    if home_score is not None and away_score is not None:
        status = 'completed'
    
    # Check if game exists
    cursor.execute("SELECT game_id FROM games WHERE external_id = ?", (external_id,))
    existing = cursor.fetchone()
    
    if existing:
        # Update existing game
        cursor.execute("""
            UPDATE games 
            SET home_score = ?, away_score = ?, status = ?
            WHERE external_id = ?
        """, (home_score, away_score, status, external_id))
        return existing[0]
    else:
        # Insert new game
        cursor.execute("""
            INSERT INTO games (external_id, sport, league, game_date, home_team_id, away_team_id, home_score, away_score, status, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (external_id, sport, league, game_datetime, home_team_id, away_team_id, home_score, away_score, status, datetime.now().isoformat()))
        return cursor.lastrowid

def update_team_stats(cursor, team_id):
    """Update historical stats for a team based on completed games."""
    # Get completed games for this team
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
    
    # Upsert stats
    cursor.execute("SELECT stat_id FROM historical_stats WHERE team_id = ?", (team_id,))
    existing = cursor.fetchone()
    
    if existing:
        cursor.execute("""
            UPDATE historical_stats 
            SET games_played = ?, wins = ?, losses = ?, points_scored = ?, points_allowed = ?, last_updated = ?
            WHERE team_id = ?
        """, (games_played, wins, losses, points_scored, points_allowed, datetime.now().isoformat(), team_id))
    else:
        cursor.execute("""
            INSERT INTO historical_stats (team_id, games_played, wins, losses, points_scored, points_allowed, last_updated)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (team_id, games_played, wins, losses, points_scored, points_allowed, datetime.now().isoformat()))

def run_data_pipeline():
    """Execute the complete data pipeline with player-level support (V6)."""
    print("🏈 Starting Enhanced V6 Data Pipeline (Player-Level Intelligence)...")
    
    with get_db() as conn:
        cursor = conn.cursor()
        
        # NBA Pipeline
        print("\n🏀 Processing NBA data...")
        nba_teams = fetch_nba_teams()
        print(f"   Fetched {len(nba_teams)} NBA teams")
        
        # Insert NBA teams
        nba_team_ids = []
        for team_data in nba_teams:
            team_id = upsert_team(cursor, team_data, "NBA", "NBA")
            if team_id:
                nba_team_ids.append(team_id)
        
        # Fetch NBA games (League ID: 4387)
        nba_recent_games = fetch_recent_games("4387")
        nba_upcoming_games = fetch_upcoming_games("4387")
        all_nba_games = nba_recent_games + nba_upcoming_games
        print(f"   Fetched {len(all_nba_games)} NBA games")
        
        # Insert NBA games
        for game_data in all_nba_games:
            upsert_game(cursor, game_data, "NBA", "NBA")
        
        # Update NBA team stats
        for team_id in nba_team_ids:
            update_team_stats(cursor, team_id)
        
        # NFL Pipeline
        print("\n🏈 Processing NFL data...")
        nfl_teams = fetch_nfl_teams()
        print(f"   Fetched {len(nfl_teams)} NFL teams")
        
        # Insert NFL teams
        nfl_team_ids = []
        for team_data in nfl_teams:
            team_id = upsert_team(cursor, team_data, "NFL", "NFL")
            if team_id:
                nfl_team_ids.append(team_id)
        
        # Fetch NFL games (League ID: 4391)
        nfl_recent_games = fetch_recent_games("4391")
        nfl_upcoming_games = fetch_upcoming_games("4391")
        all_nfl_games = nfl_recent_games + nfl_upcoming_games
        print(f"   Fetched {len(all_nfl_games)} NFL games")
        
        # Insert NFL games
        for game_data in all_nfl_games:
            upsert_game(cursor, game_data, "NFL", "NFL")
        
        # Update NFL team stats
        for team_id in nfl_team_ids:
            update_team_stats(cursor, team_id)
        
        # V6 Enhancement: Player-Level Data Processing
        print("\n⭐ Processing V6 Player-Level Data...")
        
        # Initialize players if they don't exist (V6 feature)
        cursor.execute("SELECT COUNT(*) FROM players")
        player_count = cursor.fetchone()[0]
        
        if player_count == 0:
            print("   Initializing player database...")
            # Import and run player initialization
            try:
                import sys
                from pathlib import Path
                
                # Add the backend directory to path
                backend_path = Path(__file__).parent
                if str(backend_path) not in sys.path:
                    sys.path.append(str(backend_path))
                
                from player_impact_model import PlayerImpactCalculator
                
                calculator = PlayerImpactCalculator()
                calculator.initialize_sample_players()
                print("   ✅ Player initialization complete")
                
            except Exception as e:
                print(f"   ⚠️ Player initialization failed: {e}")
        else:
            print(f"   Found {player_count} existing players")
        
        # Update player game stats based on recent games (V6 feature)
        print("   Updating player game statistics...")
        updated_stats = update_player_game_stats(cursor, conn)
        print(f"   Updated {updated_stats} player stat records")
        
        conn.commit()
        
        # Summary statistics
        cursor.execute("SELECT COUNT(*) FROM teams")
        team_count = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM games")
        game_count = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM historical_stats")
        stats_count = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM players")
        player_count = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM player_game_stats")
        player_stats_count = cursor.fetchone()[0]
        
        print(f"\n✅ Enhanced V6 Data Pipeline Complete!")
        print(f"   📊 Teams: {team_count}")
        print(f"   🏟️  Games: {game_count}")
        print(f"   📈 Team Stats: {stats_count}")
        print(f"   ⭐ Players: {player_count} (V6)")
        print(f"   📊 Player Stats: {player_stats_count} (V6)")

def update_player_game_stats(cursor, conn):
    """Update player game statistics based on recent completed games (V6 Feature)."""
    try:
        # Get recent completed games
        cursor.execute("""
            SELECT game_id, sport, home_team_id, away_team_id, game_date
            FROM games 
            WHERE status = 'completed' AND game_date >= datetime('now', '-7 days')
        """)
        
        recent_games = cursor.fetchall()
        
        if not recent_games:
            return 0
        
        updated_count = 0
        
        # For each recent game, check if we need to generate/update player stats
        for game in recent_games:
            game_id, sport, home_team_id, away_team_id, game_date = game
            
            # Check if we already have player stats for this game
            cursor.execute("""
                SELECT COUNT(*) FROM player_game_stats 
                WHERE game_id = ?
            """, (game_id,))
            
            existing_stats = cursor.fetchone()[0]
            
            if existing_stats > 0:
                continue  # Already have stats for this game
            
            # Get players from both teams
            cursor.execute("""
                SELECT player_id, player_name, position 
                FROM players 
                WHERE team_id IN (?, ?) AND sport = ?
            """, (home_team_id, away_team_id, sport))
            
            players = cursor.fetchall()
            
            # Generate stats for a subset of players (simulating partial participation)
            import random
            random.seed(game_id)  # Deterministic but varied
            
            participating_players = random.sample(players, min(len(players), random.randint(5, 12)))
            
            for player_id, player_name, position in participating_players:
                # Generate game stats
                from player_impact_model import PlayerImpactCalculator
                calculator = PlayerImpactCalculator()
                stats = calculator.generate_game_stats_by_sport(sport, position, player_name)
                
                # Insert stats
                cursor.execute("""
                    INSERT OR IGNORE INTO player_game_stats (
                        player_id, game_id, minutes_played, points, rebounds, assists,
                        steals, blocks, turnovers, field_goals_made, field_goals_attempted,
                        three_pointers_made, three_pointers_attempted, free_throws_made,
                        free_throws_attempted, fouls, plus_minus, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    player_id, game_id, stats['minutes'], stats['points'], stats['rebounds'],
                    stats['assists'], stats['steals'], stats['blocks'], stats['turnovers'],
                    stats['fgm'], stats['fga'], stats['3pm'], stats['3pa'],
                    stats['ftm'], stats['fta'], stats['fouls'], stats['plus_minus'],
                    datetime.now().isoformat()
                ))
                
                updated_count += 1
        
        conn.commit()
        return updated_count
        
    except Exception as e:
        print(f"   ⚠️ Player stats update failed: {e}")
        return 0

if __name__ == "__main__":
    run_data_pipeline()