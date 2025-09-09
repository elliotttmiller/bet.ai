#!/usr/bin/env python3
"""
Player Impact Model (V6 Core Intelligence)
Implements Player Efficiency Rating (PER)-style calculation for granular player value quantification.
Synthesizes per-minute box score contributions into a single impact_rating stored in players table.
"""

import os
import sqlite3
import math
import numpy as np
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

class PlayerImpactCalculator:
    """
    Advanced Player Impact Model implementing PER-style calculation.
    
    Features:
    - Sport-specific impact metrics and scaling factors
    - Per-minute efficiency normalization 
    - Advanced statistical aggregation (usage rate, true shooting %)
    - Position-based adjustments for different roles
    - Season-long impact rating calculation and storage
    """
    
    def __init__(self):
        # Sport-specific scaling factors for impact calculation
        self.sport_scales = {
            'NBA': {
                'pace_factor': 100.0,  # Possessions per game normalization
                'league_average_per': 15.0,  # NBA baseline PER
                'minutes_weight': 0.3,  # Importance of playing time
                'efficiency_weight': 0.7  # Importance of per-minute production
            },
            'NFL': {
                'pace_factor': 70.0,  # Plays per game normalization  
                'league_average_per': 12.0,  # NFL baseline rating
                'minutes_weight': 0.4,  # Playing time importance
                'efficiency_weight': 0.6  # Per-play efficiency
            },
            'MLB': {
                'pace_factor': 25.0,  # At-bats per game normalization
                'league_average_per': 10.0,  # MLB baseline rating
                'minutes_weight': 0.2,  # Playing time less critical
                'efficiency_weight': 0.8  # Per-opportunity efficiency
            }
        }
        
        # Position adjustment factors (NBA example)
        self.position_adjustments = {
            'PG': {'passing_bonus': 1.2, 'rebounding_penalty': 0.8},
            'SG': {'scoring_bonus': 1.1, 'passing_bonus': 1.0},
            'SF': {'versatility_bonus': 1.05, 'all_around': 1.0},
            'PF': {'rebounding_bonus': 1.15, 'scoring_bonus': 1.0},
            'C': {'rebounding_bonus': 1.25, 'blocks_bonus': 1.3, 'assists_penalty': 0.9}
        }
    
    def calculate_per_style_rating(self, player_stats: List[sqlite3.Row], sport: str) -> float:
        """
        Calculate Player Efficiency Rating using advanced box score synthesis.
        
        PER Formula (adapted):
        PER = (Points + Rebounds + Assists + Steals + Blocks - Missed_FG - Missed_FT - Turnovers) 
              / Minutes * Pace_Adjustment * League_Adjustment
        """
        if not player_stats:
            return 0.0
        
        sport_scale = self.sport_scales.get(sport, self.sport_scales['NBA'])
        total_minutes = 0.0
        total_production = 0.0
        game_count = len(player_stats)
        
        for game in player_stats:
            minutes = game['minutes_played'] or 0
            if minutes <= 0:
                continue
                
            total_minutes += minutes
            
            # Core production metrics
            points = game['points'] or 0
            rebounds = game['rebounds'] or 0
            assists = game['assists'] or 0
            steals = game['steals'] or 0
            blocks = game['blocks'] or 0
            turnovers = game['turnovers'] or 0
            
            # Shooting efficiency
            fgm = game['field_goals_made'] or 0
            fga = game['field_goals_attempted'] or 0
            ftm = game['free_throws_made'] or 0
            fta = game['free_throws_attempted'] or 0
            
            # Calculate game production
            positive_actions = points + rebounds + assists + steals + blocks
            negative_actions = (fga - fgm) + (fta - ftm) + turnovers
            
            # Plus/minus consideration (if available)
            plus_minus = game['plus_minus'] or 0
            plus_minus_factor = 1.0 + (plus_minus / 100.0)  # Convert to multiplicative factor
            
            game_production = (positive_actions - negative_actions) * plus_minus_factor
            total_production += game_production
        
        if total_minutes <= 0:
            return 0.0
        
        # Calculate per-minute efficiency
        per_minute_production = total_production / total_minutes
        
        # Apply sport-specific pace adjustment
        pace_adjusted = per_minute_production * sport_scale['pace_factor']
        
        # League adjustment to center around average
        league_adjusted = pace_adjusted + sport_scale['league_average_per']
        
        # Apply playing time factor (players who play more get bonus)
        avg_minutes_per_game = total_minutes / game_count if game_count > 0 else 0
        minutes_factor = 1.0 + (avg_minutes_per_game / 40.0) * sport_scale['minutes_weight']
        
        # Final impact rating
        impact_rating = league_adjusted * minutes_factor
        
        # Ensure reasonable bounds (0-50 scale)
        impact_rating = max(0.0, min(50.0, impact_rating))
        
        return round(impact_rating, 2)
    
    def calculate_usage_rate(self, player_stats: List[sqlite3.Row], team_stats: Dict) -> float:
        """Calculate player usage rate (percentage of team possessions used)."""
        if not player_stats:
            return 0.0
        
        total_fga = sum(game['field_goals_attempted'] or 0 for game in player_stats)
        total_fta = sum(game['free_throws_attempted'] or 0 for game in player_stats)
        total_turnovers = sum(game['turnovers'] or 0 for game in player_stats)
        total_minutes = sum(game['minutes_played'] or 0 for game in player_stats)
        
        if total_minutes <= 0:
            return 0.0
        
        # Simplified usage rate calculation
        player_possessions = total_fga + (total_fta * 0.44) + total_turnovers
        
        # Estimate team possessions based on player's share
        estimated_team_possessions = player_possessions * (48.0 / total_minutes * len(player_stats))
        
        usage_rate = (player_possessions / estimated_team_possessions * 100) if estimated_team_possessions > 0 else 0
        
        return min(100.0, max(0.0, usage_rate))
    
    def apply_position_adjustments(self, base_rating: float, position: str, player_stats: List[sqlite3.Row]) -> float:
        """Apply position-specific adjustments to the base impact rating."""
        if not position or position not in self.position_adjustments:
            return base_rating
        
        adjustments = self.position_adjustments[position]
        adjusted_rating = base_rating
        
        if not player_stats:
            return adjusted_rating
        
        # Calculate player's statistical profile
        avg_assists = np.mean([game['assists'] or 0 for game in player_stats])
        avg_rebounds = np.mean([game['rebounds'] or 0 for game in player_stats])
        avg_blocks = np.mean([game['blocks'] or 0 for game in player_stats])
        
        # Apply position-specific bonuses/penalties
        if avg_assists > 5 and 'passing_bonus' in adjustments:
            adjusted_rating *= adjustments['passing_bonus']
        
        if avg_rebounds > 8 and 'rebounding_bonus' in adjustments:
            adjusted_rating *= adjustments['rebounding_bonus']
        elif avg_rebounds < 3 and 'rebounding_penalty' in adjustments:
            adjusted_rating *= adjustments['rebounding_penalty']
        
        if avg_blocks > 1.5 and 'blocks_bonus' in adjustments:
            adjusted_rating *= adjustments['blocks_bonus']
        
        return round(adjusted_rating, 2)
    
    def initialize_sample_players(self) -> None:
        """Initialize sample players and generate synthetic stats for demonstration."""
        logger.info("🏀 Initializing sample players and generating synthetic performance data...")
        
        with get_db() as conn:
            cursor = conn.cursor()
            
            # First ensure we have teams
            cursor.execute("SELECT team_id, team_name, sport FROM teams")
            teams = cursor.fetchall()
            
            if not teams:
                logger.info("   No teams found, skipping player initialization")
                return
            
            # Sample players for demonstration
            sample_players = [
                # NBA Players
                ('LeBron James', 'SF', 'NBA'),
                ('Stephen Curry', 'PG', 'NBA'), 
                ('Kevin Durant', 'SF', 'NBA'),
                ('Giannis Antetokounmpo', 'PF', 'NBA'),
                ('Luka Dončić', 'PG', 'NBA'),
                ('Jayson Tatum', 'SF', 'NBA'),
                ('Nikola Jokić', 'C', 'NBA'),
                ('Joel Embiid', 'C', 'NBA'),
                
                # NFL Players  
                ('Tom Brady', 'QB', 'NFL'),
                ('Aaron Rodgers', 'QB', 'NFL'),
                ('Derrick Henry', 'RB', 'NFL'),
                ('Travis Kelce', 'TE', 'NFL'),
                ('Cooper Kupp', 'WR', 'NFL'),
                ('Aaron Donald', 'DT', 'NFL'),
                
                # MLB Players
                ('Mike Trout', 'OF', 'MLB'),
                ('Mookie Betts', 'OF', 'MLB'),
                ('Ronald Acuña Jr.', 'OF', 'MLB'),
                ('Manny Machado', 'SS', 'MLB'),
                ('Jacob deGrom', 'P', 'MLB'),
                ('Juan Soto', 'OF', 'MLB')
            ]
            
            created_players = 0
            
            for player_name, position, sport in sample_players:
                # Find a team of matching sport
                sport_teams = [team for team in teams if team['sport'] == sport]
                if not sport_teams:
                    continue
                
                # Assign to random team for demonstration
                import random
                random.seed(hash(player_name) % 1000)  # Deterministic but varied
                team = random.choice(sport_teams)
                team_id = team['team_id']
                
                # Check if player already exists
                cursor.execute("""
                    SELECT player_id FROM players 
                    WHERE player_name = ? AND team_id = ? AND sport = ?
                """, (player_name, team_id, sport))
                
                if cursor.fetchone():
                    continue  # Player already exists
                
                # Insert player
                cursor.execute("""
                    INSERT INTO players (player_name, team_id, sport, position, impact_rating, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    player_name, team_id, sport, position, 0.0,
                    datetime.now().isoformat(), datetime.now().isoformat()
                ))
                
                created_players += 1
            
            conn.commit()
            logger.info(f"   Created {created_players} sample players")
            
            # Generate synthetic game stats for these players
            self.generate_synthetic_player_stats(cursor, conn)
    
    def generate_synthetic_player_stats(self, cursor, conn) -> None:
        """Generate realistic synthetic player game statistics."""
        logger.info("   Generating synthetic player game statistics...")
        
        import random
        random.seed(42)
        
        # Get all players
        cursor.execute("SELECT player_id, player_name, sport, position FROM players")
        players = cursor.fetchall()
        
        # Get recent games (last 30 days)
        recent_date = (datetime.now() - timedelta(days=30)).isoformat()
        cursor.execute("""
            SELECT game_id, sport, home_team_id, away_team_id
            FROM games 
            WHERE status = 'completed' AND game_date >= ?
        """, (recent_date,))
        
        games = cursor.fetchall()
        
        stats_created = 0
        
        for player in players:
            player_id, player_name, sport, position = player
            
            # Find games this player's team participated in
            cursor.execute("""
                SELECT team_id FROM players WHERE player_id = ?
            """, (player_id,))
            team_result = cursor.fetchone()
            if not team_result:
                continue
            
            team_id = team_result[0]
            
            # Get games where this team played
            team_games = [
                game for game in games 
                if game['sport'] == sport and (
                    game['home_team_id'] == team_id or game['away_team_id'] == team_id
                )
            ]
            
            # Generate stats for random selection of these games
            num_games = min(len(team_games), random.randint(5, 15))
            selected_games = random.sample(team_games, num_games) if team_games else []
            
            for game in selected_games:
                game_id = game['game_id']
                
                # Check if stats already exist
                cursor.execute("""
                    SELECT stat_id FROM player_game_stats 
                    WHERE player_id = ? AND game_id = ?
                """, (player_id, game_id))
                
                if cursor.fetchone():
                    continue  # Stats already exist
                
                # Generate sport-specific stats
                stats = self.generate_game_stats_by_sport(sport, position, player_name)
                
                # Insert stats
                cursor.execute("""
                    INSERT INTO player_game_stats (
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
                
                stats_created += 1
        
        conn.commit()
        logger.info(f"   Generated {stats_created} player game stat records")
    
    def generate_game_stats_by_sport(self, sport: str, position: str, player_name: str) -> Dict:
        """Generate realistic game statistics based on sport and position."""
        import random
        
        # Use player name for deterministic but varied stats
        random.seed(hash(player_name) % 10000)
        
        if sport == 'NBA':
            return self.generate_nba_stats(position, player_name)
        elif sport == 'NFL':
            return self.generate_nfl_stats(position, player_name)
        elif sport == 'MLB':
            return self.generate_mlb_stats(position, player_name)
        else:
            return self.generate_nba_stats(position, player_name)  # Default to NBA
    
    def generate_nba_stats(self, position: str, player_name: str) -> Dict:
        """Generate realistic NBA game statistics."""
        import random
        
        # Star player boost based on name recognition
        star_boost = 1.0
        if any(star in player_name for star in ['LeBron', 'Curry', 'Durant', 'Giannis', 'Luka', 'Jokić']):
            star_boost = 1.4
        elif any(good in player_name for good in ['Tatum', 'Embiid']):
            star_boost = 1.2
        
        base_minutes = random.uniform(25, 40) * star_boost
        minutes = min(48, max(10, base_minutes))
        
        # Position-based stat generation
        if position == 'PG':
            points = random.uniform(8, 30) * star_boost
            rebounds = random.uniform(3, 8)
            assists = random.uniform(5, 12) * star_boost
        elif position == 'SG':
            points = random.uniform(12, 35) * star_boost
            rebounds = random.uniform(3, 7)
            assists = random.uniform(2, 8)
        elif position == 'SF':
            points = random.uniform(10, 32) * star_boost
            rebounds = random.uniform(5, 10)
            assists = random.uniform(3, 9)
        elif position == 'PF':
            points = random.uniform(8, 28) * star_boost
            rebounds = random.uniform(7, 14)
            assists = random.uniform(2, 6)
        elif position == 'C':
            points = random.uniform(10, 30) * star_boost
            rebounds = random.uniform(8, 16) * star_boost
            assists = random.uniform(1, 8)
        else:
            # Default
            points = random.uniform(8, 25)
            rebounds = random.uniform(3, 10)
            assists = random.uniform(2, 8)
        
        # Other stats
        steals = random.uniform(0, 3)
        blocks = random.uniform(0, 3) if position in ['PF', 'C'] else random.uniform(0, 1.5)
        turnovers = random.uniform(1, 5)
        fouls = random.randint(0, 6)
        
        # Shooting stats
        fga = max(1, int(points * random.uniform(0.8, 1.3)))
        fgm = max(0, int(fga * random.uniform(0.35, 0.65)))
        fta = max(0, int(points * random.uniform(0.15, 0.4)))
        ftm = max(0, int(fta * random.uniform(0.65, 0.95)))
        
        three_pa = max(0, int(fga * random.uniform(0.2, 0.6)))
        three_pm = max(0, int(three_pa * random.uniform(0.25, 0.45)))
        
        plus_minus = random.uniform(-20, 20) * star_boost
        
        return {
            'minutes': round(minutes, 1),
            'points': int(points),
            'rebounds': int(rebounds),
            'assists': int(assists),
            'steals': int(steals),
            'blocks': int(blocks),
            'turnovers': int(turnovers),
            'fgm': fgm,
            'fga': fga,
            '3pm': three_pm,
            '3pa': three_pa,
            'ftm': ftm,
            'fta': fta,
            'fouls': fouls,
            'plus_minus': round(plus_minus, 1)
        }
    
    def generate_nfl_stats(self, position: str, player_name: str) -> Dict:
        """Generate simplified NFL statistics mapped to NBA schema."""
        import random
        
        # NFL stats are quite different, so we'll map them creatively
        star_boost = 1.0
        if any(star in player_name for star in ['Brady', 'Rodgers', 'Henry', 'Kelce']):
            star_boost = 1.3
        
        if position == 'QB':
            points = random.uniform(15, 35) * star_boost  # Passing yards / 10
            rebounds = random.uniform(1, 3)               # Rushing attempts
            assists = random.uniform(1, 4) * star_boost   # Passing TDs
            turnovers = random.uniform(0, 3)              # Interceptions
            steals = random.uniform(0, 1)                 # Sacks avoided
            blocks = 0
        elif position == 'RB':
            points = random.uniform(5, 25) * star_boost   # Rushing yards / 5
            rebounds = random.uniform(2, 6)               # Carries / 3
            assists = random.uniform(0, 2)                # TDs
            turnovers = random.uniform(0, 2)              # Fumbles
            steals = random.uniform(0, 1)                 # Broken tackles
            blocks = 0
        elif position in ['WR', 'TE']:
            points = random.uniform(3, 20) * star_boost   # Receiving yards / 5
            rebounds = random.uniform(2, 8)               # Receptions
            assists = random.uniform(0, 2)                # TDs
            turnovers = random.uniform(0, 1)              # Fumbles
            steals = random.uniform(0, 2)                 # YAC
            blocks = random.uniform(0, 1) if position == 'TE' else 0
        else:
            points = random.uniform(2, 15)                # Tackles
            rebounds = random.uniform(1, 5)               # Assists
            assists = random.uniform(0, 1)                # Sacks/INTs
            turnovers = 0                                 # Minimal
            steals = random.uniform(0, 2)                 # Forced fumbles
            blocks = random.uniform(0, 1)                 # Pass breakups
        
        return {
            'minutes': random.uniform(20, 60),  # Snaps played
            'points': int(points),
            'rebounds': int(rebounds),
            'assists': int(assists),
            'steals': random.randint(0, 2),
            'blocks': random.randint(0, 2),
            'turnovers': int(turnovers) if position == 'QB' else random.randint(0, 1),
            'fgm': int(points * 0.6),
            'fga': int(points * 1.0),
            '3pm': 0,
            '3pa': 0,
            'ftm': 0,
            'fta': 0,
            'fouls': random.randint(0, 5),
            'plus_minus': random.uniform(-10, 15) * star_boost
        }
    
    def generate_mlb_stats(self, position: str, player_name: str) -> Dict:
        """Generate simplified MLB statistics mapped to NBA schema."""
        import random
        
        star_boost = 1.0
        if any(star in player_name for star in ['Trout', 'Betts', 'Acuña', 'Soto']):
            star_boost = 1.4
        
        if position == 'P':  # Pitcher
            points = random.uniform(3, 15) * star_boost   # Strikeouts
            rebounds = random.uniform(0, 1)               # RBIs (minimal for pitchers)
            assists = random.uniform(5, 9)                # Innings pitched
            turnovers = random.uniform(0, 5)              # Earned runs
            steals = 0
        else:  # Position player
            points = random.uniform(0, 4) * star_boost    # Hits
            rebounds = random.uniform(0, 3)               # RBIs
            assists = random.uniform(0, 2)                # Runs scored
            steals = random.uniform(0, 2)                 # Stolen bases
        
        return {
            'minutes': random.uniform(1, 9),  # Innings played
            'points': int(points),
            'rebounds': int(rebounds),
            'assists': int(assists),
            'steals': int(steals) if position != 'P' else 0,
            'blocks': 0,
            'turnovers': int(turnovers) if position == 'P' else random.randint(0, 1),
            'fgm': int(points),
            'fga': random.randint(int(points), int(points) + 3),
            '3pm': 0,
            '3pa': 0,
            'ftm': 0,
            'fta': 0,
            'fouls': 0,
            'plus_minus': random.uniform(-3, 3) * star_boost
        }
    
    def calculate_all_player_impacts(self) -> None:
        """Calculate and store impact ratings for all players based on their game statistics."""
        logger.info("🧮 Calculating Player Impact Ratings using advanced PER methodology...")
        
        with get_db() as conn:
            cursor = conn.cursor()
            
            # Get all players
            cursor.execute("SELECT player_id, player_name, sport, position FROM players")
            players = cursor.fetchall()
            
            if not players:
                logger.info("   No players found for impact calculation")
                return
            
            updated_players = 0
            total_ratings = []
            
            for player in players:
                player_id, player_name, sport, position = player
                
                # Get player's game statistics
                cursor.execute("""
                    SELECT * FROM player_game_stats 
                    WHERE player_id = ?
                    ORDER BY created_at DESC
                """, (player_id,))
                
                player_stats = cursor.fetchall()
                
                if not player_stats:
                    continue  # No stats to calculate from
                
                # Calculate base impact rating using PER methodology
                base_rating = self.calculate_per_style_rating(player_stats, sport)
                
                # Apply position adjustments
                final_rating = self.apply_position_adjustments(base_rating, position, player_stats)
                
                # Calculate additional metrics for validation
                usage_rate = self.calculate_usage_rate(player_stats, {})
                
                # Update player's impact rating
                cursor.execute("""
                    UPDATE players 
                    SET impact_rating = ?, updated_at = ?
                    WHERE player_id = ?
                """, (final_rating, datetime.now().isoformat(), player_id))
                
                updated_players += 1
                total_ratings.append(final_rating)
                
                logger.info(f"   {player_name} ({sport}, {position}): {final_rating:.2f} impact (usage: {usage_rate:.1f}%)")
            
            conn.commit()
            
            # Calculate summary statistics
            if total_ratings:
                avg_rating = np.mean(total_ratings)
                std_rating = np.std(total_ratings)
                max_rating = max(total_ratings)
                min_rating = min(total_ratings)
                
                logger.info(f"✅ Player Impact Calculation Complete!")
                logger.info(f"   📊 Updated {updated_players} players")
                logger.info(f"   📈 Average Impact: {avg_rating:.2f} (±{std_rating:.2f})")
                logger.info(f"   🏆 Range: {min_rating:.2f} - {max_rating:.2f}")
                
                # Show top performers
                cursor.execute("""
                    SELECT player_name, sport, position, impact_rating
                    FROM players 
                    WHERE impact_rating > 0
                    ORDER BY impact_rating DESC
                    LIMIT 10
                """)
                
                top_players = cursor.fetchall()
                if top_players:
                    logger.info("   🌟 Top 10 Impact Players:")
                    for i, (name, sport, pos, rating) in enumerate(top_players, 1):
                        logger.info(f"      {i}. {name} ({sport}, {pos}): {rating:.2f}")
            
    def get_player_impact(self, player_id: int) -> Optional[float]:
        """Get the current impact rating for a specific player."""
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT impact_rating FROM players WHERE player_id = ?", (player_id,))
            result = cursor.fetchone()
            return result[0] if result else None
    
    def get_team_aggregate_impact(self, team_id: int, sport: str) -> Dict[str, float]:
        """Calculate team's aggregate impact rating from all players."""
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT impact_rating, position FROM players 
                WHERE team_id = ? AND sport = ? AND impact_rating > 0
                ORDER BY impact_rating DESC
            """, (team_id, sport))
            
            players = cursor.fetchall()
            
            if not players:
                return {'total_impact': 0.0, 'avg_impact': 0.0, 'top_5_impact': 0.0}
            
            impact_ratings = [player[0] for player in players]
            
            total_impact = sum(impact_ratings)
            avg_impact = total_impact / len(impact_ratings)
            top_5_impact = sum(impact_ratings[:5])  # Top 5 players
            
            return {
                'total_impact': round(total_impact, 2),
                'avg_impact': round(avg_impact, 2),
                'top_5_impact': round(top_5_impact, 2),
                'player_count': len(impact_ratings)
            }

def run_player_impact_model():
    """Main entry point to run the Player Impact Model calculation."""
    try:
        calculator = PlayerImpactCalculator()
        
        # Initialize sample players if none exist
        calculator.initialize_sample_players()
        
        # Calculate impact ratings for all players
        calculator.calculate_all_player_impacts()
        
        return True
    except Exception as e:
        logger.error(f"❌ Player Impact Model failed: {e}")
        return False

if __name__ == "__main__":
    success = run_player_impact_model()
    exit(0 if success else 1)