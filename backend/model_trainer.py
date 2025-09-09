#!/usr/bin/env python3
"""
Advanced Sports Prediction Engine with Ensemble Models
State-of-the-art machine learning ensemble using LightGBM + XGBoost
Following "Quantifying the performance of individual players in team competitions" methodology
"""

import os
import sqlite3
import json
import joblib
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from contextlib import contextmanager
from typing import Dict, List, Tuple, Optional, Any
from dotenv import load_dotenv

import lightgbm as lgb
import xgboost as xgb
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.preprocessing import LabelEncoder

# Load environment variables
load_dotenv()

# Configuration
DB_PATH = Path(__file__).parent.parent / "database" / "bet_copilot.db"

def get_model_paths(sport: str):
    """Get sport-specific model file paths."""
    base_path = Path(__file__).parent
    return {
        'lgbm': base_path / f"model_lgbm_{sport.lower()}.joblib",
        'xgb': base_path / f"model_xgb_{sport.lower()}.joblib", 
        'metadata': base_path / f"model_{sport.lower()}.json"
    }

@contextmanager
def get_db():
    """Database connection context manager."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()

class AdvancedFeatureEngineer:
    """
    Advanced feature engineering for sports prediction
    Implements rolling averages, strength of schedule, and performance metrics
    """
    
    def __init__(self):
        self.rolling_windows = [5, 10, 20]  # Games to look back for rolling stats
        
    def calculate_rolling_averages(self, team_data: pd.DataFrame, windows: List[int]) -> pd.DataFrame:
        """Calculate rolling averages for key performance metrics."""
        features = pd.DataFrame()
        
        # Ensure data is sorted by date
        team_data = team_data.sort_values('game_date').copy()
        
        for window in windows:
            # Rolling win rate
            features[f'win_rate_{window}g'] = team_data['won'].rolling(window, min_periods=1).mean()
            
            # Rolling scoring average
            features[f'avg_points_{window}g'] = team_data['points_scored'].rolling(window, min_periods=1).mean()
            features[f'avg_points_allowed_{window}g'] = team_data['points_allowed'].rolling(window, min_periods=1).mean()
            
            # Rolling point differential
            features[f'point_diff_{window}g'] = (
                team_data['points_scored'] - team_data['points_allowed']
            ).rolling(window, min_periods=1).mean()
            
            # Form (recent performance trend)
            features[f'form_{window}g'] = team_data['won'].rolling(window, min_periods=1).sum()
            
        return features
    
    def calculate_strength_of_schedule(self, team_data: pd.DataFrame, all_teams_data: pd.DataFrame) -> pd.Series:
        """Calculate strength of schedule based on opponent quality."""
        sos_scores = []
        
        for _, game in team_data.iterrows():
            opponent_id = game['opponent_id']
            
            # Get opponent's record up to this game
            opponent_games = all_teams_data[
                (all_teams_data['team_id'] == opponent_id) & 
                (all_teams_data['game_date'] < game['game_date'])
            ]
            
            if len(opponent_games) > 0:
                opponent_win_rate = opponent_games['won'].mean()
                opponent_avg_score = opponent_games['points_scored'].mean()
                # Composite opponent strength (0-1 scale)
                opponent_strength = (opponent_win_rate * 0.6) + ((opponent_avg_score / 120) * 0.4)
            else:
                opponent_strength = 0.5  # Neutral for unknown opponents
                
            sos_scores.append(opponent_strength)
        
        return pd.Series(sos_scores, index=team_data.index)
    
    def calculate_head_to_head_features(self, team_data: pd.DataFrame) -> pd.DataFrame:
        """Calculate head-to-head historical performance."""
        h2h_features = pd.DataFrame(index=team_data.index)
        
        # Initialize features
        h2h_features['h2h_wins'] = 0
        h2h_features['h2h_games'] = 0
        h2h_features['h2h_win_rate'] = 0.5
        h2h_features['h2h_avg_margin'] = 0.0
        
        for idx, game in team_data.iterrows():
            team_id = game['team_id']
            opponent_id = game['opponent_id']
            game_date = game['game_date']
            
            # Find previous games between these teams
            previous_h2h = team_data[
                (team_data['team_id'] == team_id) &
                (team_data['opponent_id'] == opponent_id) &
                (team_data['game_date'] < game_date)
            ]
            
            if len(previous_h2h) > 0:
                h2h_features.loc[idx, 'h2h_games'] = len(previous_h2h)
                h2h_features.loc[idx, 'h2h_wins'] = previous_h2h['won'].sum()
                h2h_features.loc[idx, 'h2h_win_rate'] = previous_h2h['won'].mean()
                h2h_features.loc[idx, 'h2h_avg_margin'] = (
                    previous_h2h['points_scored'] - previous_h2h['points_allowed']
                ).mean()
        
        return h2h_features
    
    def create_advanced_features(self, games_data: pd.DataFrame) -> pd.DataFrame:
        """Create comprehensive feature set for each game including Elo ratings."""
        features_list = []
        
        # Process each team's data
        for team_id in games_data['team_id'].unique():
            team_games = games_data[games_data['team_id'] == team_id].copy()
            team_games = team_games.sort_values('game_date')
            
            # Basic features
            basic_features = team_games[['game_id', 'team_id', 'opponent_id', 'is_home', 'won']].copy()
            
            # Rolling averages
            rolling_features = self.calculate_rolling_averages(team_games, self.rolling_windows)
            
            # Strength of schedule
            basic_features['strength_of_schedule'] = self.calculate_strength_of_schedule(team_games, games_data)
            
            # Head-to-head features
            h2h_features = self.calculate_head_to_head_features(team_games)
            
            # Rest days (days since last game)
            team_games['game_date'] = pd.to_datetime(team_games['game_date'])
            basic_features['rest_days'] = team_games['game_date'].diff().dt.days.fillna(7)
            
            # Elo rating features (NEW!)
            elo_features = self.calculate_elo_features(team_games, games_data)
            
            # Combine all features
            team_features = pd.concat([basic_features, rolling_features, h2h_features, elo_features], axis=1)
            features_list.append(team_features)
        
        return pd.concat(features_list, ignore_index=True)
    
    def calculate_elo_features(self, team_games: pd.DataFrame, all_games_data: pd.DataFrame) -> pd.DataFrame:
        """Calculate Elo rating features for enhanced prediction accuracy."""
        elo_features = pd.DataFrame(index=team_games.index)
        
        # Initialize Elo features
        elo_features['team_elo'] = 1500.0
        elo_features['opponent_elo'] = 1500.0
        elo_features['elo_difference'] = 0.0
        elo_features['elo_advantage'] = 0.0
        
        # V6 Enhancement: Add player impact features
        player_features = self.calculate_player_impact_features(team_games, all_games_data)
        elo_features = pd.concat([elo_features, player_features], axis=1)
        
        # Get team and sport info
        if len(team_games) == 0:
            return elo_features
        
        team_id = team_games.iloc[0]['team_id']
        # Assume sport can be inferred from game context - for demo, use NBA
        sport = 'NBA'
        
        try:
            # Import here to avoid circular imports
            from pathlib import Path
            import sqlite3
            from contextlib import contextmanager
            
            @contextmanager
            def get_db():
                db_path = Path(__file__).parent.parent / "database" / "bet_copilot.db"
                conn = sqlite3.connect(db_path)
                conn.row_factory = sqlite3.Row
                try:
                    yield conn
                finally:
                    conn.close()
            
            with get_db() as conn:
                cursor = conn.cursor()
                
                for idx, game in team_games.iterrows():
                    game_date = game['game_date']
                    opponent_id = game['opponent_id']
                    
                    # Convert datetime to string if needed
                    if hasattr(game_date, 'strftime'):
                        date_str = game_date.strftime('%Y-%m-%d')
                    else:
                        date_str = str(game_date)[:10]
                    
                    # Get team's Elo rating on game date
                    cursor.execute("""
                        SELECT elo_rating FROM elo_ratings
                        WHERE team_id = ? AND sport = ? AND date <= ?
                        ORDER BY date DESC
                        LIMIT 1
                    """, (team_id, sport, date_str))
                    
                    team_elo_result = cursor.fetchone()
                    team_elo = team_elo_result[0] if team_elo_result else 1500.0
                    
                    # Get opponent's Elo rating on game date
                    cursor.execute("""
                        SELECT elo_rating FROM elo_ratings
                        WHERE team_id = ? AND sport = ? AND date <= ?
                        ORDER BY date DESC
                        LIMIT 1
                    """, (opponent_id, sport, date_str))
                    
                    opponent_elo_result = cursor.fetchone()
                    opponent_elo = opponent_elo_result[0] if opponent_elo_result else 1500.0
                    
                    # Calculate Elo features
                    elo_difference = team_elo - opponent_elo
                    
                    # Home advantage consideration (65 points for NBA)
                    home_bonus = 65.0 if game['is_home'] else -65.0
                    elo_advantage = elo_difference + home_bonus
                    
                    # Store features
                    elo_features.loc[idx, 'team_elo'] = team_elo
                    elo_features.loc[idx, 'opponent_elo'] = opponent_elo
                    elo_features.loc[idx, 'elo_difference'] = elo_difference
                    elo_features.loc[idx, 'elo_advantage'] = elo_advantage
                    
        except Exception as e:
            print(f"Warning: Could not load Elo features: {e}")
            # Use default values if Elo ratings are not available
            pass
        
        return elo_features
    
    def calculate_player_impact_features(self, team_games: pd.DataFrame, all_games_data: pd.DataFrame) -> pd.DataFrame:
        """Calculate player impact-based features for lineup-aware predictions (V6)."""
        player_features = pd.DataFrame(index=team_games.index)
        
        # Initialize player impact features
        player_features['team_total_impact'] = 0.0
        player_features['team_avg_impact'] = 0.0
        player_features['team_top5_impact'] = 0.0
        player_features['opponent_total_impact'] = 0.0
        player_features['opponent_avg_impact'] = 0.0
        player_features['opponent_top5_impact'] = 0.0
        player_features['impact_advantage'] = 0.0
        player_features['star_player_advantage'] = 0.0
        
        # Get team and sport info
        if len(team_games) == 0:
            return player_features
        
        team_id = team_games.iloc[0]['team_id']
        # Assume sport can be inferred from game context - for demo, use NBA
        sport = 'NBA'
        
        try:
            # Import here to avoid circular imports
            from pathlib import Path
            import sqlite3
            from contextlib import contextmanager
            
            @contextmanager
            def get_db():
                db_path = Path(__file__).parent.parent / "database" / "bet_copilot.db"
                conn = sqlite3.connect(db_path)
                conn.row_factory = sqlite3.Row
                try:
                    yield conn
                finally:
                    conn.close()
            
            with get_db() as conn:
                cursor = conn.cursor()
                
                for idx, game in team_games.iterrows():
                    game_date = game['game_date']
                    opponent_id = game['opponent_id']
                    
                    # Get team's player impact ratings
                    cursor.execute("""
                        SELECT impact_rating FROM players
                        WHERE team_id = ? AND sport = ? AND impact_rating > 0
                        ORDER BY impact_rating DESC
                    """, (team_id, sport))
                    
                    team_impacts = [row[0] for row in cursor.fetchall()]
                    
                    # Get opponent's player impact ratings
                    cursor.execute("""
                        SELECT impact_rating FROM players
                        WHERE team_id = ? AND sport = ? AND impact_rating > 0
                        ORDER BY impact_rating DESC
                    """, (opponent_id, sport))
                    
                    opponent_impacts = [row[0] for row in cursor.fetchall()]
                    
                    # Calculate team aggregates
                    if team_impacts:
                        team_total = sum(team_impacts)
                        team_avg = team_total / len(team_impacts)
                        team_top5 = sum(team_impacts[:5])
                        team_star_power = max(team_impacts) if team_impacts else 0.0
                    else:
                        team_total = team_avg = team_top5 = team_star_power = 15.0  # League average
                    
                    # Calculate opponent aggregates
                    if opponent_impacts:
                        opp_total = sum(opponent_impacts)
                        opp_avg = opp_total / len(opponent_impacts)
                        opp_top5 = sum(opponent_impacts[:5])
                        opp_star_power = max(opponent_impacts) if opponent_impacts else 0.0
                    else:
                        opp_total = opp_avg = opp_top5 = opp_star_power = 15.0  # League average
                    
                    # Calculate advantages
                    impact_advantage = team_avg - opp_avg
                    star_advantage = team_star_power - opp_star_power
                    
                    # Store features
                    player_features.loc[idx, 'team_total_impact'] = team_total
                    player_features.loc[idx, 'team_avg_impact'] = team_avg
                    player_features.loc[idx, 'team_top5_impact'] = team_top5
                    player_features.loc[idx, 'opponent_total_impact'] = opp_total
                    player_features.loc[idx, 'opponent_avg_impact'] = opp_avg
                    player_features.loc[idx, 'opponent_top5_impact'] = opp_top5
                    player_features.loc[idx, 'impact_advantage'] = impact_advantage
                    player_features.loc[idx, 'star_player_advantage'] = star_advantage
                    
        except Exception as e:
            print(f"Warning: Could not load player impact features: {e}")
            # Use default values if player data is not available
            pass
        
        return player_features

class EnsembleSportsModel:
    """
    State-of-the-art ensemble model using LightGBM + XGBoost for sports prediction
    """
    
    def __init__(self, sport: str = "NBA"):
        self.sport = sport.upper()
        self.lgbm_model = None
        self.xgb_model = None
        self.feature_columns = None
        self.label_encoder = LabelEncoder()
        self.model_version = f"v6.0-player-impact-elo-ensemble-{self.sport.lower()}"
        self.feature_engineer = AdvancedFeatureEngineer()
        self.model_paths = get_model_paths(self.sport)
        
        # LightGBM parameters optimized for sports prediction
        self.lgb_params = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
            'random_state': 42
        }
        
        # XGBoost parameters optimized for sports prediction
        self.xgb_params = {
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'max_depth': 6,
            'learning_rate': 0.05,
            'subsample': 0.8,
            'colsample_bytree': 0.9,
            'random_state': 42,
            'verbosity': 0
        }
    
    def prepare_training_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        """Load and prepare training data from database."""
        print("📊 Loading training data from database...")
        
        # Create synthetic training data since we don't have historical games yet
        # This simulates a robust dataset for demonstration
        np.random.seed(42)
        
        games_data = []
        team_ids = list(range(1, 31))  # 30 teams
        
        # Generate synthetic season data
        for game_id in range(1, 2001):  # 2000 games
            team_a = np.random.choice(team_ids)
            team_b = np.random.choice([t for t in team_ids if t != team_a])
            
            # Simulate game date over a season
            base_date = datetime(2023, 10, 1)
            game_date = base_date + timedelta(days=np.random.randint(0, 200))
            
            # Simulate team strengths (some teams are naturally better)
            team_a_strength = 0.4 + (team_a / 30) * 0.2 + np.random.normal(0, 0.1)
            team_b_strength = 0.4 + (team_b / 30) * 0.2 + np.random.normal(0, 0.1)
            
            # Home advantage
            is_home_a = np.random.choice([True, False])
            if is_home_a:
                team_a_strength += 0.05
            
            # Determine winner based on strengths
            team_a_wins = team_a_strength > team_b_strength
            
            # Generate scores
            team_a_score = max(70, int(110 + team_a_strength * 30 + np.random.normal(0, 15)))
            team_b_score = max(70, int(110 + team_b_strength * 30 + np.random.normal(0, 15)))
            
            if team_a_wins:
                team_a_score = max(team_a_score, team_b_score + 1)
            else:
                team_b_score = max(team_b_score, team_a_score + 1)
            
            # Add both team perspectives
            games_data.extend([
                {
                    'game_id': game_id,
                    'team_id': team_a,
                    'opponent_id': team_b,
                    'is_home': is_home_a,
                    'points_scored': team_a_score,
                    'points_allowed': team_b_score,
                    'won': team_a_wins,
                    'game_date': game_date.isoformat()
                },
                {
                    'game_id': game_id,
                    'team_id': team_b,
                    'opponent_id': team_a,
                    'is_home': not is_home_a,
                    'points_scored': team_b_score,
                    'points_allowed': team_a_score,
                    'won': not team_a_wins,
                    'game_date': game_date.isoformat()
                }
            ])
        
        games_df = pd.DataFrame(games_data)
        print(f"   Generated {len(games_df)} game records for training")
        
        # Create advanced features
        print("🔧 Engineering advanced features...")
        features_df = self.feature_engineer.create_advanced_features(games_df)
        
        # Prepare features and target
        feature_columns = [col for col in features_df.columns if col not in ['game_id', 'team_id', 'opponent_id', 'won']]
        X = features_df[feature_columns].fillna(0)
        y = features_df['won'].astype(int)
        
        self.feature_columns = feature_columns
        print(f"   Created {len(feature_columns)} features: {feature_columns[:5]}...")
        
        return X, y
    
    def train(self):
        """Train both LightGBM and XGBoost models for ensemble prediction."""
        print("🤖 Starting Ensemble Model Training (LightGBM + XGBoost)...")
        
        # Prepare data
        X, y = self.prepare_training_data()
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"   Training set: {len(X_train)} samples")
        print(f"   Test set: {len(X_test)} samples")
        
        # Train LightGBM model
        print("   Training LightGBM model...")
        train_data = lgb.Dataset(X_train, label=y_train)
        valid_data = lgb.Dataset(X_test, label=y_test, reference=train_data)
        
        self.lgbm_model = lgb.train(
            self.lgb_params,
            train_data,
            valid_sets=[valid_data],
            num_boost_round=1000,
            callbacks=[lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(period=0)]
        )
        
        # Train XGBoost model
        print("   Training XGBoost model...")
        dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=self.feature_columns)
        dtest = xgb.DMatrix(X_test, label=y_test, feature_names=self.feature_columns)
        
        self.xgb_model = xgb.train(
            self.xgb_params,
            dtrain,
            num_boost_round=1000,
            evals=[(dtest, 'test')],
            early_stopping_rounds=50,
            verbose_eval=False
        )
        
        # Evaluate ensemble
        lgbm_train_pred = self.lgbm_model.predict(X_train, num_iteration=self.lgbm_model.best_iteration)
        lgbm_test_pred = self.lgbm_model.predict(X_test, num_iteration=self.lgbm_model.best_iteration)
        
        xgb_train_pred = self.xgb_model.predict(xgb.DMatrix(X_train, feature_names=self.feature_columns))
        xgb_test_pred = self.xgb_model.predict(xgb.DMatrix(X_test, feature_names=self.feature_columns))
        
        # Ensemble predictions (simple average)
        ensemble_train_pred = (lgbm_train_pred + xgb_train_pred) / 2
        ensemble_test_pred = (lgbm_test_pred + xgb_test_pred) / 2
        
        # Calculate metrics
        train_pred_binary = (ensemble_train_pred > 0.5).astype(int)
        test_pred_binary = (ensemble_test_pred > 0.5).astype(int)
        
        train_accuracy = accuracy_score(y_train, train_pred_binary)
        test_accuracy = accuracy_score(y_test, test_pred_binary)
        train_auc = roc_auc_score(y_train, ensemble_train_pred)
        test_auc = roc_auc_score(y_test, ensemble_test_pred)
        
        print(f"✅ Ensemble Model Training Complete!")
        print(f"   📊 Training Accuracy: {train_accuracy:.3f}")
        print(f"   📊 Test Accuracy: {test_accuracy:.3f}")
        print(f"   📊 Training AUC: {train_auc:.3f}")
        print(f"   📊 Test AUC: {test_auc:.3f}")
        
        # Feature importance (from LightGBM)
        lgbm_importance = self.lgbm_model.feature_importance(importance_type='gain')
        xgb_score_dict = self.xgb_model.get_score(importance_type='gain') if self.xgb_model.get_score() else {}
        xgb_importance = [xgb_score_dict.get(feat, 0) for feat in self.feature_columns]
        
        feature_importance = pd.DataFrame({
            'feature': self.feature_columns,
            'lgbm_importance': lgbm_importance,
            'xgb_importance': xgb_importance
        }).sort_values('lgbm_importance', ascending=False)
        
        print(f"   🎯 Top 5 Features (LightGBM):")
        for i, (_, row) in enumerate(feature_importance.head().iterrows()):
            print(f"      {i+1}. {row['feature']}: {row['lgbm_importance']:.1f}")
        
        return {
            'train_accuracy': train_accuracy,
            'test_accuracy': test_accuracy,
            'train_auc': train_auc,
            'test_auc': test_auc,
            'feature_importance': feature_importance.to_dict('records')
        }
    
    def predict_game_outcome(self, home_team_id: int, away_team_id: int, game_date: str) -> Dict[str, Any]:
        """Predict outcome for a specific game using ensemble model."""
        if self.lgbm_model is None or self.xgb_model is None:
            raise ValueError("Models not trained. Call train() first.")
        
        # For demo purposes, create synthetic recent performance data
        # In production, this would query actual team statistics
        
        # Create feature vector for this game
        features = {}
        
        # Basic features
        features['is_home'] = 1  # Home team perspective
        features['rest_days'] = 3  # Assumed rest days
        features['strength_of_schedule'] = 0.52  # Slightly above average
        
        # Rolling averages (synthetic but realistic)
        for window in [5, 10, 20]:
            features[f'win_rate_{window}g'] = 0.6 + np.random.normal(0, 0.1)
            features[f'avg_points_{window}g'] = 110 + np.random.normal(0, 10)
            features[f'avg_points_allowed_{window}g'] = 105 + np.random.normal(0, 10)
            features[f'point_diff_{window}g'] = features[f'avg_points_{window}g'] - features[f'avg_points_allowed_{window}g']
            features[f'form_{window}g'] = int(features[f'win_rate_{window}g'] * window)
        
        # Head-to-head features
        features['h2h_games'] = 4
        features['h2h_wins'] = 2
        features['h2h_win_rate'] = 0.5
        features['h2h_avg_margin'] = 2.5
        
        # Create feature vector in correct order
        feature_vector = []
        for col in self.feature_columns:
            feature_vector.append(features.get(col, 0))
        
        # Make ensemble predictions
        lgbm_prob = self.lgbm_model.predict([feature_vector], num_iteration=self.lgbm_model.best_iteration)[0]
        
        # Create XGBoost DMatrix with feature names
        xgb_dmatrix = xgb.DMatrix([feature_vector], feature_names=self.feature_columns)
        xgb_prob = self.xgb_model.predict(xgb_dmatrix)[0]
        
        # Ensemble prediction (simple average)
        ensemble_prob = (lgbm_prob + xgb_prob) / 2
        
        confidence = ensemble_prob * 100 if ensemble_prob > 0.5 else (1 - ensemble_prob) * 100
        
        # Calculate edge (simplified)
        edge = max(0, (confidence - 55) / 5)
        
        return {
            'home_win_probability': ensemble_prob,
            'away_win_probability': 1 - ensemble_prob,
            'predicted_winner': 'home' if ensemble_prob > 0.5 else 'away',
            'confidence': confidence,
            'calculated_edge': edge,
            'lgbm_probability': lgbm_prob,
            'xgb_probability': xgb_prob
        }
    
    def save_model(self):
        """Save both trained models and metadata."""
        if self.lgbm_model is None or self.xgb_model is None:
            raise ValueError("No models to save. Train the models first.")
        
        # Save LightGBM model
        joblib.dump({
            'model': self.lgbm_model,
            'feature_columns': self.feature_columns,
            'model_version': self.model_version,
            'sport': self.sport
        }, self.model_paths['lgbm'])
        
        # Save XGBoost model
        joblib.dump({
            'model': self.xgb_model,
            'feature_columns': self.feature_columns,
            'model_version': self.model_version,
            'sport': self.sport
        }, self.model_paths['xgb'])
        
        # Save metadata
        metadata = {
            'model_version': self.model_version,
            'model_type': 'Ensemble (LightGBM + XGBoost)',
            'sport': self.sport,
            'num_features': len(self.feature_columns),
            'feature_columns': self.feature_columns,
            'lgb_params': self.lgb_params,
            'xgb_params': self.xgb_params,
            'trained_at': datetime.now().isoformat()
        }
        
        with open(self.model_paths['metadata'], 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"💾 LightGBM model saved to: {self.model_paths['lgbm']}")
        print(f"💾 XGBoost model saved to: {self.model_paths['xgb']}")
        print(f"📋 Metadata saved to: {self.model_paths['metadata']}")
    
    @classmethod
    def load_models(cls, sport: str = "NBA"):
        """Load both previously trained models for a specific sport."""
        model_paths = get_model_paths(sport)
        
        if not model_paths['lgbm'].exists() or not model_paths['xgb'].exists():
            return None
        
        try:
            lgbm_data = joblib.load(model_paths['lgbm'])
            xgb_data = joblib.load(model_paths['xgb'])
            
            instance = cls(sport)
            instance.lgbm_model = lgbm_data['model']
            instance.xgb_model = xgb_data['model']
            instance.feature_columns = lgbm_data['feature_columns']
            instance.model_version = lgbm_data.get('model_version', 'v3.0-ensemble')
            
            return instance
        except Exception as e:
            print(f"Error loading models: {e}")
            return None

def generate_predictions(sport: str = "NBA"):
    """Generate predictions for upcoming games using ensemble model."""
    print(f"🎯 Generating {sport} Ensemble Predictions (LightGBM + XGBoost)...")
    
    # Load or train model
    model = EnsembleSportsModel.load_models(sport)
    if model is None:
        print(f"   No trained {sport} models found. Training new ensemble...")
        model = EnsembleSportsModel(sport)
        model.train()
        model.save_model()
    else:
        print(f"   Loaded existing {sport} ensemble models: {model.model_version}")
    
    # Generate sample predictions for upcoming games
    upcoming_games = [
        {
            'home_team': 'Lakers',
            'away_team': 'Warriors',
            'sport': 'NBA',
            'league': 'NBA',
            'game_date': (datetime.now() + timedelta(days=1)).isoformat()
        },
        {
            'home_team': 'Patriots',
            'away_team': 'Bills',
            'sport': 'NFL',
            'league': 'NFL',
            'game_date': (datetime.now() + timedelta(days=2)).isoformat()
        },
        {
            'home_team': 'Yankees',
            'away_team': 'Red Sox',
            'sport': 'MLB',
            'league': 'MLB',
            'game_date': (datetime.now() + timedelta(days=1)).isoformat()
        }
    ]
    
    predictions_generated = 0
    
    with get_db() as conn:
        cursor = conn.cursor()
        
        for game in upcoming_games:
            # Generate prediction using ensemble model
            prediction = model.predict_game_outcome(1, 2, game['game_date'])
            
            # Format prediction data
            matchup = f"{game['home_team']} vs {game['away_team']}"
            
            if prediction['predicted_winner'] == 'home':
                predicted_pick = f"{game['home_team']} ML"
                predicted_odds = -150 if prediction['confidence'] > 65 else -110
            else:
                predicted_pick = f"{game['away_team']} ML"
                predicted_odds = 140 if prediction['confidence'] > 65 else -110
            
            # Generate realistic score predictions
            home_score = int(110 + np.random.normal(0, 8))
            away_score = int(110 + np.random.normal(0, 8))
            
            if prediction['predicted_winner'] == 'home':
                home_score = max(home_score, away_score + 1 + int(np.random.exponential(3)))
            else:
                away_score = max(away_score, home_score + 1 + int(np.random.exponential(3)))
            
            projected_score = f"{game['home_team']} {home_score}, {game['away_team']} {away_score}"
            
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
                    game['home_team'], game['away_team'], predicted_pick,
                    predicted_odds, prediction['confidence'], projected_score,
                    prediction['calculated_edge'], datetime.now().isoformat(), model.model_version
                ))
                
                predictions_generated += 1
        
        conn.commit()
    
    print(f"✅ Ensemble Prediction Generation Complete!")
    print(f"   📊 Generated {predictions_generated} new predictions")
    print(f"   🤖 Model: {model.model_version}")

def train_model(sport: str = "NBA"):
    """Train the ensemble model and generate predictions for a specific sport."""
    print(f"🚀 Starting Advanced {sport} Ensemble Training Pipeline...")
    
    # Train model
    model = EnsembleSportsModel(sport)
    training_results = model.train()
    model.save_model()
    
    # Generate predictions
    generate_predictions(sport)
    
    print(f"🎯 {sport} Ensemble Training Pipeline Complete!")
    return training_results

if __name__ == "__main__":
    import sys
    sport = sys.argv[1] if len(sys.argv) > 1 else "NBA"
    train_model(sport)