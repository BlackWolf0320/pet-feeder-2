"""
LSTM Pattern Prediction Model
Predict next feeding time and optimal food amount
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
import logging
import pickle

from ai_config import LSTM_CONFIG, LSTM_MODEL_PATH, PREDICTION_CONFIG
from data_preparer import DataPreparer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeedingPatternPredictor:
    """LSTM model for predicting feeding patterns"""
    
    def __init__(self):
        self.model = None
        self.history = None
        self.data_preparer = DataPreparer()
        
    def build_model(self, input_shape):
        """
        Build LSTM neural network architecture
        
        Args:
            input_shape: (sequence_length, num_features)
        """
        model = Sequential([
            # First LSTM layer
            LSTM(
                LSTM_CONFIG['lstm_units'][0],
                return_sequences=True,
                input_shape=input_shape
            ),
            Dropout(LSTM_CONFIG['dropout_rate']),
            BatchNormalization(),
            
            # Second LSTM layer
            LSTM(
                LSTM_CONFIG['lstm_units'][1],
                return_sequences=False
            ),
            Dropout(LSTM_CONFIG['dropout_rate']),
            BatchNormalization(),
            
            # Dense layers
            Dense(32, activation='relu'),
            Dropout(LSTM_CONFIG['dropout_rate']),
            
            Dense(16, activation='relu'),
            
            # Output layer - predicting amount consumed
            Dense(1, activation='linear')
        ])
        
        # Compile model
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=LSTM_CONFIG['learning_rate']),
            loss='mse',
            metrics=['mae', 'mse']
        )
        
        self.model = model
        logger.info("Built LSTM model")
        logger.info(f"Input shape: {input_shape}")
        
        return model
    
    def train(self, pet_id=None, save_model=True):
        """
        Train LSTM model on feeding data
        
        Args:
            pet_id: Train for specific pet or None for all pets
            save_model: Whether to save trained model
            
        Returns:
            Training history
        """
        logger.info(f"Starting LSTM training for pet_id={pet_id}")
        
        # Prepare data
        X, y = self.data_preparer.prepare_for_lstm(
            pet_id=pet_id,
            sequence_length=LSTM_CONFIG['sequence_length']
        )
        
        if X is None or len(X) == 0:
            logger.error("No training data available")
            return None
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=0.2, 
            random_state=42
        )
        
        logger.info(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")
        
        # Build model
        self.build_model(input_shape=(X.shape[1], X.shape[2]))
        
        # Callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=LSTM_CONFIG['early_stopping_patience'],
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=0.00001,
                verbose=1
            )
        ]
        
        if save_model:
            callbacks.append(
                ModelCheckpoint(
                    str(LSTM_MODEL_PATH),  # Convert Path to string
                    monitor='val_loss',
                    save_best_only=True,
                    verbose=1
                )
            )
        
        # Train model
        logger.info("Training LSTM model...")
        self.history = self.model.fit(
            X_train, y_train,
            epochs=LSTM_CONFIG['epochs'],
            batch_size=LSTM_CONFIG['batch_size'],
            validation_data=(X_test, y_test),
            callbacks=callbacks,
            verbose=1
        )
        
        # Evaluate
        test_loss, test_mae, test_mse = self.model.evaluate(X_test, y_test, verbose=0)
        
        logger.info("=" * 60)
        logger.info("TRAINING COMPLETE")
        logger.info("=" * 60)
        logger.info(f"Test Loss (MSE): {test_loss:.4f}")
        logger.info(f"Test MAE: {test_mae:.4f}")
        logger.info(f"Test RMSE: {np.sqrt(test_mse):.4f}")
        logger.info("=" * 60)
        
        return self.history
    
    def load_model(self):
        """Load trained model from disk"""
        try:
            self.model = load_model(str(LSTM_MODEL_PATH))  # Convert Path to string
            logger.info(f"Loaded model from {LSTM_MODEL_PATH}")
            return True
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False
    
    def predict_next_feeding(self, pet_id, return_confidence=True):
        """
        Predict next feeding time and amount
        
        Args:
            pet_id: Pet to predict for
            return_confidence: Whether to return confidence score
            
        Returns:
            Dictionary with predictions
        """
        if self.model is None:
            if not self.load_model():
                logger.error("No model available for prediction")
                return None
        
        # Get recent feeding history
        df = self.data_preparer.extract_feeding_data(pet_id=pet_id, min_samples=LSTM_CONFIG['sequence_length'])
        
        if df is None or len(df) < LSTM_CONFIG['sequence_length']:
            logger.warning(f"Not enough data for prediction (need {LSTM_CONFIG['sequence_length']} samples)")
            return None
        
        # Engineer features
        df = self.data_preparer.engineer_features(df)
        df = self.data_preparer.normalize_data(df, fit=False)
        
        # Get last sequence
        feature_cols = [
            'hour_of_day', 'day_of_week', 'amount_consumed',
            'eating_duration', 'time_since_last_meal', 'completion_rate',
            'hour_sin', 'hour_cos', 'day_sin', 'day_cos',
            'eating_speed', 'hours_since_last_meal'
        ]
        
        feature_cols = [col for col in feature_cols if col in df.columns]
        
        last_sequence = df[feature_cols].tail(LSTM_CONFIG['sequence_length']).values
        X_pred = np.array([last_sequence])
        
        # Predict
        predicted_amount = self.model.predict(X_pred, verbose=0)[0][0]
        
        # Estimate next feeding time based on patterns
        avg_interval = df['time_since_last_meal'].mean()
        last_feeding_time = df['timestamp'].iloc[-1]
        predicted_time = last_feeding_time + timedelta(minutes=avg_interval)
        
        # Calculate confidence based on prediction variance
        # Make multiple predictions with slight noise to estimate uncertainty
        predictions = []
        for _ in range(10):
            noise = np.random.normal(0, 0.01, X_pred.shape)
            noisy_input = X_pred + noise
            pred = self.model.predict(noisy_input, verbose=0)[0][0]
            predictions.append(pred)
        
        prediction_std = np.std(predictions)
        confidence = max(0.0, min(1.0, 1.0 - (prediction_std / np.mean(predictions))))
        
        result = {
            'predicted_amount': float(predicted_amount),
            'predicted_time': predicted_time,
            'confidence': float(confidence),
            'avg_interval_minutes': float(avg_interval),
            'last_feeding_time': last_feeding_time,
            'model_version': 'lstm_v1'
        }
        
        logger.info(f"Prediction for pet {pet_id}: {predicted_amount:.1f}g at {predicted_time} (confidence: {confidence:.2f})")
        
        return result
    
    def predict_daily_pattern(self, pet_id, target_date=None):
        """
        Predict feeding pattern for entire day
        
        Args:
            pet_id: Pet to predict for
            target_date: Date to predict (default: tomorrow)
            
        Returns:
            List of predicted feeding times and amounts
        """
        if target_date is None:
            target_date = datetime.now() + timedelta(days=1)
        
        predictions = []
        
        # Get recent patterns to understand typical feeding times
        df = self.data_preparer.extract_feeding_data(pet_id=pet_id, days=30)
        
        if df is None or len(df) == 0:
            logger.error("No historical data for prediction")
            return None
        
        # Find typical feeding times
        df['hour'] = df['timestamp'].dt.hour
        feeding_hours = df.groupby('hour').size().sort_values(ascending=False).head(5).index.tolist()
        
        logger.info(f"Typical feeding hours: {feeding_hours}")
        
        # Predict for each typical feeding time
        for hour in sorted(feeding_hours):
            feeding_datetime = target_date.replace(hour=hour, minute=0, second=0)
            
            # Make prediction
            pred = self.predict_next_feeding(pet_id, return_confidence=True)
            
            if pred:
                predictions.append({
                    'time': feeding_datetime,
                    'amount': pred['predicted_amount'],
                    'confidence': pred['confidence']
                })
        
        return predictions
    
    def evaluate_model(self, pet_id=None):
        """
        Evaluate model performance on test data
        
        Args:
            pet_id: Specific pet or None for all
            
        Returns:
            Dictionary with evaluation metrics
        """
        if self.model is None:
            if not self.load_model():
                return None
        
        # Prepare test data
        X, y = self.data_preparer.prepare_for_lstm(pet_id=pet_id)
        
        if X is None:
            return None
        
        _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Predictions
        y_pred = self.model.predict(X_test)
        
        # Calculate metrics
        mae = np.mean(np.abs(y_test - y_pred.flatten()))
        mse = np.mean((y_test - y_pred.flatten()) ** 2)
        rmse = np.sqrt(mse)
        mape = np.mean(np.abs((y_test - y_pred.flatten()) / y_test)) * 100
        
        # R² score
        ss_res = np.sum((y_test - y_pred.flatten()) ** 2)
        ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
        r2 = 1 - (ss_res / ss_tot)
        
        metrics = {
            'mae': float(mae),
            'mse': float(mse),
            'rmse': float(rmse),
            'mape': float(mape),
            'r2': float(r2),
            'test_samples': len(X_test)
        }
        
        logger.info("Model Evaluation:")
        logger.info(f"  MAE: {mae:.2f}g")
        logger.info(f"  RMSE: {rmse:.2f}g")
        logger.info(f"  MAPE: {mape:.2f}%")
        logger.info(f"  R²: {r2:.4f}")
        
        return metrics


# CLI functions
def train_model_cli(pet_id=None):
    """Train model from command line"""
    predictor = FeedingPatternPredictor()
    history = predictor.train(pet_id=pet_id)
    
    if history:
        print("\n✅ Model training complete!")
        print(f"Model saved to: {LSTM_MODEL_PATH}")
    else:
        print("\n❌ Training failed - check if you have enough data")


def test_prediction_cli(pet_id):
    """Test prediction from command line"""
    predictor = FeedingPatternPredictor()
    
    prediction = predictor.predict_next_feeding(pet_id)
    
    if prediction:
        print("\n" + "=" * 60)
        print("FEEDING PREDICTION")
        print("=" * 60)
        print(f"Predicted Amount: {prediction['predicted_amount']:.1f}g")
        print(f"Predicted Time: {prediction['predicted_time']}")
        print(f"Confidence: {prediction['confidence']:.2%}")
        print("=" * 60)
    else:
        print("\n❌ Prediction failed")


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == 'train':
            pet_id = int(sys.argv[2]) if len(sys.argv) > 2 else None
            train_model_cli(pet_id)
        elif sys.argv[1] == 'predict':
            if len(sys.argv) < 3:
                print("Usage: python lstm_predictor.py predict <pet_id>")
            else:
                test_prediction_cli(int(sys.argv[2]))
    else:
        print("Usage:")
        print("  python lstm_predictor.py train [pet_id]")
        print("  python lstm_predictor.py predict <pet_id>")
