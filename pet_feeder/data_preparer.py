"""
Data Preparation Module
Prepare data from database for AI model training
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from database import get_session, init_db, Pet, FeedingEvent, HealthMetric
import pickle
from ai_config import FEATURE_CONFIG, DATA_DIR, SCALER_PATH
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataPreparer:
    """Prepare feeding data for AI models"""
    
    def __init__(self, db_session=None):
        if db_session is None:
            engine = init_db()
            self.session = get_session(engine)
        else:
            self.session = db_session
        
        self.scaler = None
    
    def extract_feeding_data(self, pet_id=None, days=None, min_samples=50):
        """
        Extract feeding events from database
        
        Args:
            pet_id: Specific pet ID or None for all pets
            days: Number of days to look back or None for all data
            min_samples: Minimum number of samples required
            
        Returns:
            DataFrame with feeding data
        """
        query = self.session.query(FeedingEvent)
        
        if pet_id is not None:
            query = query.filter(FeedingEvent.pet_id == pet_id)
        
        if days is not None:
            start_date = datetime.utcnow() - timedelta(days=days)
            query = query.filter(FeedingEvent.timestamp >= start_date)
        
        # Only get complete feeding events
        query = query.filter(
            FeedingEvent.amount_consumed.isnot(None),
            FeedingEvent.eating_duration.isnot(None)
        )
        
        feedings = query.order_by(FeedingEvent.timestamp).all()
        
        if len(feedings) < min_samples:
            logger.warning(f"Only {len(feedings)} samples found, need at least {min_samples}")
            return None
        
        logger.info(f"Extracted {len(feedings)} feeding events")
        
        # Convert to DataFrame
        data = []
        for f in feedings:
            data.append({
                'id': f.id,
                'pet_id': f.pet_id,
                'timestamp': f.timestamp,
                'amount_dispensed': f.amount_dispensed,
                'amount_consumed': f.amount_consumed,
                'eating_duration': f.eating_duration,
                'time_since_last_meal': f.time_since_last_meal or 0,
                'completion_rate': f.completion_rate or 0,
                'is_manual': f.is_manual_dispense,
                'anomaly_detected': f.anomaly_detected
            })
        
        df = pd.DataFrame(data)
        return df
    
    def engineer_features(self, df):
        """
        Create features from raw data
        
        Args:
            df: DataFrame with feeding events
            
        Returns:
            DataFrame with engineered features
        """
        df = df.copy()
        
        # Time-based features
        df['hour_of_day'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['day_of_month'] = df['timestamp'].dt.day
        df['month'] = df['timestamp'].dt.month
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        
        # Cyclical encoding for time features (better for neural networks)
        if FEATURE_CONFIG['cyclical_encoding']:
            df['hour_sin'] = np.sin(2 * np.pi * df['hour_of_day'] / 24)
            df['hour_cos'] = np.cos(2 * np.pi * df['hour_of_day'] / 24)
            df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
            df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        
        # Rolling statistics (per pet)
        for pet_id in df['pet_id'].unique():
            pet_mask = df['pet_id'] == pet_id
            
            # Rolling average of last 3 feedings
            df.loc[pet_mask, 'avg_amount_3'] = df.loc[pet_mask, 'amount_consumed'].rolling(
                window=3, min_periods=1
            ).mean()
            
            df.loc[pet_mask, 'avg_duration_3'] = df.loc[pet_mask, 'eating_duration'].rolling(
                window=3, min_periods=1
            ).mean()
            
            # Rolling average of last 7 feedings
            df.loc[pet_mask, 'avg_amount_7'] = df.loc[pet_mask, 'amount_consumed'].rolling(
                window=7, min_periods=1
            ).mean()
            
            # Trend (difference from moving average)
            df.loc[pet_mask, 'amount_trend'] = (
                df.loc[pet_mask, 'amount_consumed'] - df.loc[pet_mask, 'avg_amount_3']
            )
        
        # Eating speed (grams per second)
        df['eating_speed'] = df['amount_consumed'] / df['eating_duration'].replace(0, 1)
        
        # Time features
        df['minutes_since_last_meal'] = df['time_since_last_meal']
        df['hours_since_last_meal'] = df['time_since_last_meal'] / 60.0
        
        # Binary features
        df['is_fast_eater'] = (df['eating_duration'] < df['eating_duration'].quantile(0.25)).astype(int)
        df['is_slow_eater'] = (df['eating_duration'] > df['eating_duration'].quantile(0.75)).astype(int)
        
        logger.info(f"Engineered {len(df.columns)} features")
        return df
    
    def create_sequences(self, df, sequence_length=7, target_col='amount_consumed'):
        """
        Create sequences for LSTM training
        
        Args:
            df: DataFrame with features
            sequence_length: Number of past events to use
            target_col: Column to predict
            
        Returns:
            X (sequences), y (targets)
        """
        sequences = []
        targets = []
        
        # Group by pet
        for pet_id in df['pet_id'].unique():
            pet_data = df[df['pet_id'] == pet_id].sort_values('timestamp')
            
            if len(pet_data) < sequence_length + 1:
                continue
            
            # Select feature columns
            feature_cols = [
                'hour_of_day', 'day_of_week', 'amount_consumed',
                'eating_duration', 'time_since_last_meal', 'completion_rate',
                'hour_sin', 'hour_cos', 'day_sin', 'day_cos',
                'eating_speed', 'hours_since_last_meal'
            ]
            
            # Filter to only existing columns
            feature_cols = [col for col in feature_cols if col in pet_data.columns]
            
            values = pet_data[feature_cols].values
            targets_values = pet_data[target_col].values
            
            # Create sequences
            for i in range(len(values) - sequence_length):
                sequences.append(values[i:i+sequence_length])
                targets.append(targets_values[i+sequence_length])
        
        X = np.array(sequences)
        y = np.array(targets)
        
        logger.info(f"Created {len(X)} sequences of length {sequence_length}")
        logger.info(f"Sequence shape: {X.shape}, Target shape: {y.shape}")
        
        return X, y
    
    def normalize_data(self, df, fit=True):
        """
        Normalize numerical features
        
        Args:
            df: DataFrame to normalize
            fit: Whether to fit scaler (True for training, False for inference)
            
        Returns:
            Normalized DataFrame
        """
        df = df.copy()
        
        numerical_cols = [
            'amount_consumed', 'eating_duration', 'time_since_last_meal',
            'completion_rate', 'eating_speed', 'avg_amount_3', 'avg_duration_3'
        ]
        
        # Filter to existing columns
        numerical_cols = [col for col in numerical_cols if col in df.columns]
        
        if fit:
            self.scaler = StandardScaler()
            df[numerical_cols] = self.scaler.fit_transform(df[numerical_cols])
            
            # Save scaler
            with open(SCALER_PATH, 'wb') as f:
                pickle.dump(self.scaler, f)
            logger.info("Fitted and saved scaler")
        else:
            if self.scaler is None:
                # Load existing scaler
                try:
                    with open(SCALER_PATH, 'rb') as f:
                        self.scaler = pickle.load(f)
                    logger.info("Loaded existing scaler")
                except FileNotFoundError:
                    logger.warning("No scaler found, returning unnormalized data")
                    return df
            
            df[numerical_cols] = self.scaler.transform(df[numerical_cols])
        
        return df
    
    def prepare_for_lstm(self, pet_id=None, sequence_length=7, normalize=True):
        """
        Complete pipeline for LSTM training data
        
        Args:
            pet_id: Specific pet or None for all
            sequence_length: Length of input sequences
            normalize: Whether to normalize features
            
        Returns:
            X, y for training
        """
        # Extract data
        df = self.extract_feeding_data(pet_id=pet_id)
        
        if df is None or len(df) == 0:
            logger.error("No data available")
            return None, None
        
        # Engineer features
        df = self.engineer_features(df)
        
        # Normalize
        if normalize:
            df = self.normalize_data(df, fit=True)
        
        # Create sequences
        X, y = self.create_sequences(df, sequence_length=sequence_length)
        
        return X, y
    
    def prepare_for_anomaly_detection(self, pet_id=None, normalize=True):
        """
        Prepare data for anomaly detection model
        
        Args:
            pet_id: Specific pet or None for all
            normalize: Whether to normalize features
            
        Returns:
            DataFrame with features for anomaly detection
        """
        # Extract data
        df = self.extract_feeding_data(pet_id=pet_id)
        
        if df is None or len(df) == 0:
            logger.error("No data available")
            return None
        
        # Engineer features
        df = self.engineer_features(df)
        
        # Select features for anomaly detection
        feature_cols = [
            'amount_consumed', 'eating_duration', 'completion_rate',
            'time_since_last_meal', 'hour_of_day', 'day_of_week',
            'eating_speed', 'hours_since_last_meal'
        ]
        
        feature_cols = [col for col in feature_cols if col in df.columns]
        
        X = df[feature_cols].copy()
        
        # Normalize
        if normalize:
            scaler = StandardScaler()
            X[feature_cols] = scaler.fit_transform(X[feature_cols])
            
            # Save scaler
            with open(SCALER_PATH, 'wb') as f:
                pickle.dump(scaler, f)
        
        logger.info(f"Prepared {len(X)} samples for anomaly detection")
        
        return X, df
    
    def export_training_data(self, pet_id=None, filename='training_data.csv'):
        """
        Export data to CSV for external training/analysis
        
        Args:
            pet_id: Specific pet or None for all
            filename: Output filename
        """
        df = self.extract_feeding_data(pet_id=pet_id)
        
        if df is None:
            logger.error("No data to export")
            return
        
        df = self.engineer_features(df)
        
        output_path = DATA_DIR / filename
        df.to_csv(output_path, index=False)
        
        logger.info(f"Exported {len(df)} records to {output_path}")
        return output_path
    
    def get_data_summary(self, pet_id=None):
        """
        Get summary statistics about available data
        
        Args:
            pet_id: Specific pet or None for all
            
        Returns:
            Dictionary with summary statistics
        """
        df = self.extract_feeding_data(pet_id=pet_id, min_samples=1)
        
        if df is None or len(df) == 0:
            return {
                'total_samples': 0,
                'pets': 0,
                'date_range': None,
                'ready_for_training': False
            }
        
        summary = {
            'total_samples': len(df),
            'pets': df['pet_id'].nunique(),
            'date_range': {
                'start': df['timestamp'].min(),
                'end': df['timestamp'].max(),
                'days': (df['timestamp'].max() - df['timestamp'].min()).days
            },
            'samples_per_pet': df.groupby('pet_id').size().to_dict(),
            'avg_amount_consumed': df['amount_consumed'].mean(),
            'avg_eating_duration': df['eating_duration'].mean(),
            'avg_completion_rate': df['completion_rate'].mean(),
            'anomalies_detected': df['anomaly_detected'].sum(),
            'ready_for_training': len(df) >= 50 and (df['timestamp'].max() - df['timestamp'].min()).days >= 14
        }
        
        return summary


# Utility functions
def check_data_readiness(pet_id=None):
    """Check if enough data is available for training"""
    preparer = DataPreparer()
    summary = preparer.get_data_summary(pet_id=pet_id)
    
    print("=" * 60)
    print("DATA READINESS CHECK")
    print("=" * 60)
    print(f"Total Samples: {summary['total_samples']}")
    print(f"Number of Pets: {summary['pets']}")
    
    if summary['date_range']:
        print(f"Date Range: {summary['date_range']['start'].date()} to {summary['date_range']['end'].date()}")
        print(f"Days of Data: {summary['date_range']['days']}")
    
    print(f"\nReady for Training: {'✅ YES' if summary['ready_for_training'] else '❌ NO'}")
    
    if not summary['ready_for_training']:
        if summary['total_samples'] < 50:
            print(f"  • Need {50 - summary['total_samples']} more samples")
        if summary['date_range'] and summary['date_range']['days'] < 14:
            print(f"  • Need {14 - summary['date_range']['days']} more days of data")
    
    print("=" * 60)
    
    return summary['ready_for_training']


if __name__ == '__main__':
    # Test data preparation
    check_data_readiness()
