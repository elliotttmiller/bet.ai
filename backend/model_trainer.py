#!/usr/bin/env python3
"""
Model Trainer for Sports Prediction Engine
Implements a basic statistical model for game outcome prediction
"""

import os
import sqlite3
import json
import random
from datetime import datetime, timedelta
from pathlib import Path
from contextlib import contextmanager
from typing import Dict, List, Tuple, Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
DB_PATH = Path(__file__).parent.parent / "database" / "bet_copilot.db"
MODEL_PATH = Path(__file__).parent / "model.json"

@contextmanager
def get_db():
    """Database connection context manager."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()

class SimpleStatsModel:
    """
    A basic statistical model for sports predictions
    Uses win rates, point differentials, and basic features
    """
    
    def __init__(self):
        self.model_version = "v1.0-stats"
        self.weights = {
            'win_rate_home': 0.35,
            'win_rate_away': 0.35,
            'home_advantage': 0.15,
            'point_differential_home': 0.075,
            'point_differential_away': 0.075
        }
        self.home_advantage_boost = 0.05  # 5% boost for home teams
    
    def calculate_team_metrics(self, team_stats: Dict) -> Dict:
        """Calculate key metrics for a team."""
        games_played = team_stats.get('games_played', 0)
        if games_played == 0:
            return {
                'win_rate': 0.5,
                'point_differential': 0.0,
                'avg_points_scored': 0.0,
                'avg_points_allowed': 0.0
            }
        
        wins = team_stats.get('wins', 0)
        points_scored = team_stats.get('points_scored', 0)
        points_allowed = team_stats.get('points_allowed', 0)
        
        win_rate = wins / games_played
        point_differential = (points_scored - points_allowed) / games_played
        avg_points_scored = points_scored / games_played
        avg_points_allowed = points_allowed / games_played
        
        return {
            'win_rate': win_rate,
            'point_differential': point_differential,
            'avg_points_scored': avg_points_scored,
            'avg_points_allowed': avg_points_allowed
        }
    
    def predict_game_outcome(self, home_team_metrics: Dict, away_team_metrics: Dict) -> Dict:
        """Predict the outcome of a game between two teams."""
        
        # Calculate base probability for home team winning
        home_win_prob = (
            self.weights['win_rate_home'] * home_team_metrics['win_rate'] +
            self.weights['win_rate_away'] * (1 - away_team_metrics['win_rate']) +
            self.weights['home_advantage'] * self.home_advantage_boost +
            self.weights['point_differential_home'] * (home_team_metrics['point_differential'] / 50.0) +  # Normalize
            self.weights['point_differential_away'] * (-away_team_metrics['point_differential'] / 50.0)
        )
        
        # Ensure probability is between 0.1 and 0.9
        home_win_prob = max(0.1, min(0.9, home_win_prob))
        away_win_prob = 1 - home_win_prob
        
        # Determine predicted winner and confidence
        if home_win_prob > away_win_prob:
            predicted_winner = 'home'
            confidence = home_win_prob * 100
        else:
            predicted_winner = 'away'
            confidence = away_win_prob * 100
        
        # Predict score based on team averages
        home_predicted_score = int(home_team_metrics['avg_points_scored'] * 0.6 + 
                                   (120 - away_team_metrics['avg_points_allowed']) * 0.4)
        away_predicted_score = int(away_team_metrics['avg_points_scored'] * 0.6 + 
                                   (120 - home_team_metrics['avg_points_allowed']) * 0.4)
        
        # Apply some randomness and home advantage
        home_predicted_score += random.randint(-3, 5)  # Home advantage
        away_predicted_score += random.randint(-5, 3)
        
        # Ensure reasonable scores
        home_predicted_score = max(70, min(150, home_predicted_score))
        away_predicted_score = max(70, min(150, away_predicted_score))
        
        return {
            'predicted_winner': predicted_winner,
            'confidence': confidence,
            'home_win_prob': home_win_prob,
            'away_win_prob': away_win_prob,
            'home_predicted_score': home_predicted_score,
            'away_predicted_score': away_predicted_score
        }
    
    def save_model(self):
        """Save model parameters to file."""
        model_data = {
            'model_version': self.model_version,
            'weights': self.weights,
            'home_advantage_boost': self.home_advantage_boost,
            'trained_at': datetime.now().isoformat()
        }
        
        with open(MODEL_PATH, 'w') as f:
            json.dump(model_data, f, indent=2)
    
    @classmethod
    def load_model(cls):
        """Load model parameters from file."""
        if not MODEL_PATH.exists():
            return cls()  # Return default model
        
        try:
            with open(MODEL_PATH, 'r') as f:
                model_data = json.load(f)
            
            model = cls()
            model.model_version = model_data.get('model_version', 'v1.0-stats')
            model.weights = model_data.get('weights', model.weights)
            model.home_advantage_boost = model_data.get('home_advantage_boost', 0.05)
            
            return model
        except Exception as e:
            print(f"Error loading model: {e}")
            return cls()  # Return default model

def get_team_stats_from_db(team_id: int) -> Dict:
    """Get team statistics from database."""
    with get_db() as conn:
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT * FROM historical_stats WHERE team_id = ?
        """, (team_id,))
        
        stats = cursor.fetchone()
        if stats:
            return dict(stats)
        else:
            return {
                'games_played': 0,
                'wins': 0,
                'losses': 0,
                'points_scored': 0,
                'points_allowed': 0
            }

def get_upcoming_games() -> List[Dict]:
    """Get upcoming games that need predictions."""
    with get_db() as conn:
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT g.*, ht.team_name as home_team_name, at.team_name as away_team_name
            FROM games g
            JOIN teams ht ON g.home_team_id = ht.team_id
            JOIN teams at ON g.away_team_id = at.team_id
            WHERE g.status = 'scheduled' 
            AND g.game_date >= datetime('now')
            ORDER BY g.game_date ASC
            LIMIT 20
        """)
        
        games = cursor.fetchall()
        return [dict(game) for game in games]

def generate_predictions():
    """Generate predictions for upcoming games."""
    print("🤖 Starting Model Training and Prediction Generation...")
    
    # Load or create model
    model = SimpleStatsModel.load_model()
    
    # Get upcoming games
    upcoming_games = get_upcoming_games()
    print(f"   Found {len(upcoming_games)} upcoming games")
    
    if not upcoming_games:
        print("   No upcoming games to predict")
        return
    
    predictions_generated = 0
    
    with get_db() as conn:
        cursor = conn.cursor()
        
        for game in upcoming_games:
            # Get team statistics
            home_team_stats = get_team_stats_from_db(game['home_team_id'])
            away_team_stats = get_team_stats_from_db(game['away_team_id'])
            
            # Calculate team metrics
            home_metrics = model.calculate_team_metrics(home_team_stats)
            away_metrics = model.calculate_team_metrics(away_team_stats)
            
            # Generate prediction
            prediction = model.predict_game_outcome(home_metrics, away_metrics)
            
            # Format prediction data
            matchup = f"{game['home_team_name']} vs {game['away_team_name']}"
            
            if prediction['predicted_winner'] == 'home':
                predicted_pick = f"{game['home_team_name']} ML"
                predicted_odds = -150 if prediction['confidence'] > 60 else -110
            else:
                predicted_pick = f"{game['away_team_name']} ML"
                predicted_odds = 130 if prediction['confidence'] > 60 else -110
            
            # Calculate edge (simplified)
            calculated_edge = max(0, (prediction['confidence'] - 55) / 10)
            
            projected_score = f"{game['home_team_name']} {prediction['home_predicted_score']}, {game['away_team_name']} {prediction['away_predicted_score']}"
            
            # Check if prediction already exists
            cursor.execute("""
                SELECT prediction_id FROM predictions 
                WHERE matchup = ? AND game_date = ?
            """, (matchup, game['game_date']))
            
            existing = cursor.fetchone()
            
            if not existing:
                # Insert new prediction
                cursor.execute("""
                    INSERT INTO predictions (
                        matchup, sport, league, game_date, team_a, team_b,
                        predicted_pick, predicted_odds, confidence_score,
                        projected_score, calculated_edge, created_at, model_version
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    matchup, game['sport'], game['league'], game['game_date'],
                    game['home_team_name'], game['away_team_name'], predicted_pick,
                    predicted_odds, prediction['confidence'], projected_score,
                    calculated_edge, datetime.now().isoformat(), model.model_version
                ))
                
                predictions_generated += 1
        
        conn.commit()
    
    # Save model
    model.save_model()
    
    print(f"✅ Model Training Complete!")
    print(f"   📊 Generated {predictions_generated} new predictions")
    print(f"   💾 Model saved to: {MODEL_PATH}")

def train_model():
    """Train the model and generate predictions."""
    generate_predictions()

if __name__ == "__main__":
    train_model()