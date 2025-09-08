#!/usr/bin/env python3
"""
Model Trainer for Sports Prediction Engine  
Implements a scikit-learn based machine learning model for game outcome prediction
"""

import os
import sqlite3
import json
import random
import numpy as np
import joblib
from datetime import datetime, timedelta
from pathlib import Path
from contextlib import contextmanager
from typing import Dict, List, Tuple, Optional
from dotenv import load_dotenv
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Load environment variables
load_dotenv()

# Configuration
DB_PATH = Path(__file__).parent.parent / "database" / "bet_copilot.db"
MODEL_PATH = Path(__file__).parent / "model.joblib"
SCALER_PATH = Path(__file__).parent / "scaler.joblib"

@contextmanager
def get_db():
    """Database connection context manager."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()

class MLPredictionModel:
    """
    Machine Learning model for sports predictions using scikit-learn
    Uses LogisticRegression for win probability prediction
    """
    
    def __init__(self):
        self.model = LogisticRegression(random_state=42, max_iter=1000)
        self.scaler = StandardScaler()
        self.model_version = "v1.0-sklearn"
        self.is_trained = False
        self.feature_names = [
            'home_win_rate', 'away_win_rate', 'home_avg_points_scored', 
            'away_avg_points_scored', 'home_avg_points_allowed', 'away_avg_points_allowed',
            'home_point_differential', 'away_point_differential', 'home_games_played', 'away_games_played'
        ]
    
    def extract_features(self, home_stats: Dict, away_stats: Dict) -> np.ndarray:
        """Extract features for ML model from team statistics."""
        home_games = max(home_stats.get('games_played', 1), 1)  # Avoid division by zero
        away_games = max(away_stats.get('games_played', 1), 1)
        
        home_wins = home_stats.get('wins', 0)
        away_wins = away_stats.get('wins', 0)
        home_points_scored = home_stats.get('points_scored', 0)
        away_points_scored = away_stats.get('points_scored', 0)
        home_points_allowed = home_stats.get('points_allowed', 0) 
        away_points_allowed = away_stats.get('points_allowed', 0)
        
        features = np.array([
            home_wins / home_games,  # home_win_rate
            away_wins / away_games,  # away_win_rate
            home_points_scored / home_games,  # home_avg_points_scored
            away_points_scored / away_games,  # away_avg_points_scored
            home_points_allowed / home_games,  # home_avg_points_allowed
            away_points_allowed / away_games,  # away_avg_points_allowed
            (home_points_scored - home_points_allowed) / home_games,  # home_point_differential
            (away_points_scored - away_points_allowed) / away_games,  # away_point_differential
            min(home_games, 100),  # home_games_played (capped)
            min(away_games, 100)   # away_games_played (capped)
        ])
        
        return features.reshape(1, -1)
    
    def train_on_historical_data(self):
        """Train the model on historical game data."""
        print("🤖 Training ML model on historical data...")
        
        with get_db() as conn:
            cursor = conn.cursor()
            
            # Get completed games with team stats
            cursor.execute("""
                SELECT g.*, 
                       ht.team_name as home_team_name, 
                       at.team_name as away_team_name,
                       hs.games_played as home_games, hs.wins as home_wins, 
                       hs.points_scored as home_points_scored, hs.points_allowed as home_points_allowed,
                       as_table.games_played as away_games, as_table.wins as away_wins,
                       as_table.points_scored as away_points_scored, as_table.points_allowed as away_points_allowed
                FROM games g
                JOIN teams ht ON g.home_team_id = ht.team_id
                JOIN teams at ON g.away_team_id = at.team_id
                LEFT JOIN historical_stats hs ON g.home_team_id = hs.team_id
                LEFT JOIN historical_stats as_table ON g.away_team_id = as_table.team_id
                WHERE g.status = 'completed' 
                AND g.home_score IS NOT NULL 
                AND g.away_score IS NOT NULL
                AND hs.games_played IS NOT NULL
                AND as_table.games_played IS NOT NULL
                LIMIT 1000
            """)
            
            games = cursor.fetchall()
        
        if len(games) < 10:  # Not enough data for training
            print("   ⚠️  Not enough historical data for ML training, creating synthetic training data...")
            self._create_synthetic_training_data()
            return
        
        features = []
        labels = []
        
        for game in games:
            game_dict = dict(game)
            home_stats = {
                'games_played': game_dict.get('home_games', 10),
                'wins': game_dict.get('home_wins', 5),
                'points_scored': game_dict.get('home_points_scored', 1000),
                'points_allowed': game_dict.get('home_points_allowed', 1000)
            }
            away_stats = {
                'games_played': game_dict.get('away_games', 10),
                'wins': game_dict.get('away_wins', 5),
                'points_scored': game_dict.get('away_points_scored', 1000),
                'points_allowed': game_dict.get('away_points_allowed', 1000)
            }
            
            feature_vec = self.extract_features(home_stats, away_stats)
            features.append(feature_vec[0])
            
            # Label: 1 if home team won, 0 if away team won
            home_won = game_dict['home_score'] > game_dict['away_score']
            labels.append(1 if home_won else 0)
        
        X = np.array(features)
        y = np.array(labels)
        
        # Split data for validation
        if len(X) > 20:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        else:
            X_train, X_test, y_train, y_test = X, X, y, y
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model
        self.model.fit(X_train_scaled, y_train)
        self.is_trained = True
        
        # Evaluate
        train_pred = self.model.predict(X_train_scaled)
        test_pred = self.model.predict(X_test_scaled)
        
        train_acc = accuracy_score(y_train, train_pred)
        test_acc = accuracy_score(y_test, test_pred)
        
        print(f"   📊 Training accuracy: {train_acc:.3f}")
        print(f"   📊 Test accuracy: {test_acc:.3f}")
        print(f"   📊 Trained on {len(X)} games")
    
    def _create_synthetic_training_data(self):
        """Create synthetic training data when historical data is insufficient."""
        print("   🎲 Creating synthetic training data...")
        
        # Generate synthetic game data
        np.random.seed(42)
        n_games = 200
        
        features = []
        labels = []
        
        for _ in range(n_games):
            # Random team stats
            home_win_rate = np.random.beta(2, 2)  # Bias toward 0.5
            away_win_rate = np.random.beta(2, 2)
            
            home_avg_scored = np.random.normal(110, 10)  # NBA-like scoring
            away_avg_scored = np.random.normal(110, 10)
            home_avg_allowed = np.random.normal(110, 10)
            away_avg_allowed = np.random.normal(110, 10)
            
            home_diff = home_avg_scored - home_avg_allowed
            away_diff = away_avg_scored - away_avg_allowed
            
            games_played = np.random.randint(10, 82)
            
            feature_vec = np.array([
                home_win_rate, away_win_rate, 
                home_avg_scored, away_avg_scored,
                home_avg_allowed, away_avg_allowed,
                home_diff, away_diff,
                games_played, games_played
            ])
            
            # Home team wins if they have better overall stats + some randomness
            home_advantage = 0.1  # 10% home advantage
            win_prob = 0.5 + (home_win_rate - away_win_rate) * 0.3 + (home_diff - away_diff) * 0.01 + home_advantage
            home_won = np.random.random() < win_prob
            
            features.append(feature_vec)
            labels.append(1 if home_won else 0)
        
        X = np.array(features)
        y = np.array(labels)
        
        # Train the model
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        self.is_trained = True
        
        print(f"   📊 Synthetic training completed on {n_games} games")
    
    def predict_game_outcome(self, home_stats: Dict, away_stats: Dict) -> Dict:
        """Predict the outcome of a game using the trained ML model."""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        # Extract features
        features = self.extract_features(home_stats, away_stats)
        features_scaled = self.scaler.transform(features)
        
        # Get prediction probabilities
        proba = self.model.predict_proba(features_scaled)[0]
        home_win_prob = proba[1]  # Probability of home team winning
        away_win_prob = proba[0]  # Probability of away team winning
        
        # Determine predicted winner and confidence
        if home_win_prob > away_win_prob:
            predicted_winner = 'home'
            confidence = home_win_prob * 100
        else:
            predicted_winner = 'away'
            confidence = away_win_prob * 100
        
        # Generate predicted scores based on team averages with some variance
        home_games = max(home_stats.get('games_played', 1), 1)
        away_games = max(away_stats.get('games_played', 1), 1)
        
        home_avg_scored = home_stats.get('points_scored', 1100) / home_games
        away_avg_scored = away_stats.get('points_scored', 1100) / away_games
        
        # Add some randomness and home advantage
        home_predicted_score = int(home_avg_scored + np.random.normal(2, 5))  # Home advantage
        away_predicted_score = int(away_avg_scored + np.random.normal(-1, 5))
        
        # Ensure reasonable scores (NBA range)
        home_predicted_score = max(85, min(140, home_predicted_score))
        away_predicted_score = max(85, min(140, away_predicted_score))
        
        return {
            'predicted_winner': predicted_winner,
            'confidence': confidence,
            'home_win_prob': home_win_prob,
            'away_win_prob': away_win_prob,
            'home_predicted_score': home_predicted_score,
            'away_predicted_score': away_predicted_score
        }
    
    def save_model(self):
        """Save the trained model and scaler to disk."""
        if not self.is_trained:
            raise ValueError("Cannot save untrained model")
        
        # Save model
        joblib.dump({
            'model': self.model,
            'model_version': self.model_version,
            'feature_names': self.feature_names,
            'trained_at': datetime.now().isoformat()
        }, MODEL_PATH)
        
        # Save scaler
        joblib.dump(self.scaler, SCALER_PATH)
        
        print(f"   💾 Model saved to: {MODEL_PATH}")
        print(f"   💾 Scaler saved to: {SCALER_PATH}")
    
    @classmethod
    def load_model(cls):
        """Load a trained model from disk."""
        if not MODEL_PATH.exists() or not SCALER_PATH.exists():
            return cls()  # Return new untrained model
        
        try:
            # Load model
            model_data = joblib.load(MODEL_PATH)
            scaler = joblib.load(SCALER_PATH)
            
            instance = cls()
            instance.model = model_data['model']
            instance.scaler = scaler
            instance.model_version = model_data.get('model_version', 'v1.0-sklearn')
            instance.feature_names = model_data.get('feature_names', instance.feature_names)
            instance.is_trained = True
            
            print(f"   ✅ Model loaded from: {MODEL_PATH}")
            return instance
        except Exception as e:
            print(f"   ⚠️  Error loading model: {e}")
            return cls()  # Return new untrained model

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
    """Generate predictions for upcoming games using ML model."""
    print("🤖 Starting Model Training and Prediction Generation...")
    
    # Load or create model
    model = MLPredictionModel.load_model()
    
    # Train the model if not already trained
    if not model.is_trained:
        model.train_on_historical_data()
        model.save_model()
    
    # Get upcoming games
    upcoming_games = get_upcoming_games()
    print(f"   Found {len(upcoming_games)} upcoming games")
    
    # If no upcoming games, create some synthetic ones for demonstration
    if not upcoming_games:
        print("   Creating synthetic upcoming games for demonstration...")
        upcoming_games = create_synthetic_upcoming_games()
    
    if not upcoming_games:
        print("   No games available for prediction")
        return
    
    predictions_generated = 0
    
    with get_db() as conn:
        cursor = conn.cursor()
        
        for game in upcoming_games:
            # Get team statistics
            home_team_stats = get_team_stats_from_db(game['home_team_id'])
            away_team_stats = get_team_stats_from_db(game['away_team_id'])
            
            # Generate prediction using ML model
            try:
                prediction = model.predict_game_outcome(home_team_stats, away_team_stats)
            except ValueError as e:
                print(f"   ⚠️  Skipping prediction: {e}")
                continue
            
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
    
    print(f"✅ Model Training Complete!")
    print(f"   📊 Generated {predictions_generated} new predictions")
    print(f"   💾 Model version: {model.model_version}")


def create_synthetic_upcoming_games() -> List[Dict]:
    """Create synthetic upcoming games for demonstration when no real games exist."""
    with get_db() as conn:
        cursor = conn.cursor()
        
        # Get some random teams
        cursor.execute("SELECT * FROM teams ORDER BY RANDOM() LIMIT 10")
        teams = cursor.fetchall()
        
        if len(teams) < 4:
            return []
        
        synthetic_games = []
        current_date = datetime.now()
        
        # Create 5 synthetic matchups
        for i in range(min(5, len(teams) // 2)):
            home_team = teams[i * 2]
            away_team = teams[i * 2 + 1]
            
            game_date = (current_date + timedelta(days=i + 1)).strftime('%Y-%m-%d %H:%M:%S')
            
            synthetic_games.append({
                'game_id': f'synthetic_{i}',
                'home_team_id': home_team['team_id'],
                'away_team_id': away_team['team_id'],
                'home_team_name': home_team['team_name'],
                'away_team_name': away_team['team_name'],
                'sport': home_team['sport'],
                'league': home_team['league'],
                'game_date': game_date,
                'status': 'scheduled'
            })
        
        return synthetic_games

def train_model():
    """Train the model and generate predictions."""
    generate_predictions()

if __name__ == "__main__":
    train_model()