"""
AI Models Configuration
Settings and hyperparameters for all AI models
"""

import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).parent
MODEL_DIR = BASE_DIR / 'ai_models'
DATA_DIR = BASE_DIR / 'training_data'
LOGS_DIR = BASE_DIR / 'logs'

# Create directories if they don't exist
MODEL_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)

# Model file paths
# Model file paths (convert to strings for compatibility)
LSTM_MODEL_PATH = str(MODEL_DIR / 'feeding_pattern_lstm.h5')
ANOMALY_MODEL_PATH = str(MODEL_DIR / 'anomaly_detector.pkl')
RL_MODEL_PATH = str(MODEL_DIR / 'schedule_optimizer.pkl')
SCALER_PATH = str(MODEL_DIR / 'data_scaler.pkl')

# Data requirements
MIN_TRAINING_SAMPLES = 50  # Minimum feeding events needed for training
MIN_DAYS_DATA = 14  # Minimum days of data collection

# LSTM Configuration
LSTM_CONFIG = {
    'sequence_length': 7,  # Use last 7 feeding events for prediction
    'features': [
        'hour_of_day',
        'day_of_week', 
        'amount_consumed',
        'eating_duration',
        'time_since_last_meal'
    ],
    'lstm_units': [128, 64],  # Two LSTM layers
    'dropout_rate': 0.2,
    'epochs': 100,
    'batch_size': 32,
    'validation_split': 0.2,
    'early_stopping_patience': 10,
    'learning_rate': 0.001
}

# Isolation Forest Configuration
ANOMALY_CONFIG = {
    'contamination': 0.1,  # Expected proportion of anomalies (10%)
    'n_estimators': 100,
    'max_samples': 'auto',
    'random_state': 42,
    'features': [
        'amount_consumed',
        'eating_duration',
        'completion_rate',
        'time_since_last_meal',
        'hour_of_day',
        'day_of_week'
    ],
    'anomaly_threshold': -0.5  # Threshold for anomaly score
}

# Q-Learning Configuration
RL_CONFIG = {
    'learning_rate': 0.1,
    'discount_factor': 0.95,  # Gamma
    'exploration_rate': 1.0,  # Epsilon (initial)
    'exploration_decay': 0.995,
    'min_exploration_rate': 0.01,
    'episodes': 1000,
    'max_steps_per_episode': 100,
    
    # State space
    'time_slots': 24,  # Hours in a day
    'hunger_levels': 5,  # Discretized hunger levels
    
    # Actions
    'actions': ['dispense_small', 'dispense_medium', 'dispense_large', 'wait'],
    'action_amounts': {
        'dispense_small': 50,   # grams
        'dispense_medium': 100,
        'dispense_large': 150,
        'wait': 0
    },
    
    # Rewards
    'reward_completion_rate': 1.0,      # Reward for high completion rate
    'reward_healthy_duration': 0.5,     # Reward for normal eating speed
    'reward_daily_target': 2.0,         # Reward for meeting daily target
    'penalty_too_frequent': -0.5,       # Penalty for feeding too often
    'penalty_too_infrequent': -1.0,     # Penalty for long gaps
    'penalty_waste': -1.0               # Penalty for low completion rate
}

# Training Configuration
TRAINING_CONFIG = {
    'test_size': 0.2,  # 20% for testing
    'random_state': 42,
    'cross_validation_folds': 5,
    'retrain_interval_days': 7,  # Retrain weekly
    'auto_retrain': True
}

# Prediction Thresholds
PREDICTION_CONFIG = {
    'confidence_threshold': 0.7,  # Minimum confidence to use prediction
    'max_prediction_hours': 24,   # Don't predict more than 24 hours ahead
    'anomaly_alert_threshold': 0.8  # Confidence needed to send alert
}

# Feature Engineering
FEATURE_CONFIG = {
    'cyclical_encoding': True,  # Use sin/cos encoding for time features
    'normalize_features': True,  # Normalize numerical features
    'handle_missing': 'mean'     # Strategy for missing values
}

# Logging Configuration
LOGGING_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'file': str(LOGS_DIR / 'ai_models.log')
}
