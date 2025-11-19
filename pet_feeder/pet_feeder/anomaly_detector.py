"""
Anomaly Detection Model
Detect unusual eating behaviors using Isolation Forest
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import pickle
import logging
from datetime import datetime

from ai_config import ANOMALY_CONFIG, ANOMALY_MODEL_PATH, SCALER_PATH
from data_preparer import DataPreparer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BehaviorAnomalyDetector:
    """Detect anomalies in pet feeding behavior"""
    
    def __init__(self):
        self.model = None
        self.scaler = None
        self.feature_names = None
        self.data_preparer = DataPreparer()
        self.normal_ranges = {}  # Store normal ranges for interpretation
        
    def train(self, pet_id=None, save_model=True):
        """
        Train Isolation Forest on normal feeding patterns
        
        Args:
            pet_id: Train for specific pet or None for all pets
            save_model: Whether to save trained model
            
        Returns:
            Trained model
        """
        logger.info(f"Starting anomaly detection training for pet_id={pet_id}")
        
        # Prepare data
        X, df = self.data_preparer.prepare_for_anomaly_detection(pet_id=pet_id)
        
        if X is None or len(X) == 0:
            logger.error("No training data available")
            return None
        
        self.feature_names = X.columns.tolist()
        
        # Calculate normal ranges for each feature (for interpretation)
        for col in X.columns:
            self.normal_ranges[col] = {
                'mean': X[col].mean(),
                'std': X[col].std(),
                'q25': X[col].quantile(0.25),
                'q75': X[col].quantile(0.75),
                'min': X[col].min(),
                'max': X[col].max()
            }
        
        logger.info(f"Training on {len(X)} samples with {len(X.columns)} features")
        
        # Train Isolation Forest
        self.model = IsolationForest(
            contamination=ANOMALY_CONFIG['contamination'],
            n_estimators=ANOMALY_CONFIG['n_estimators'],
            max_samples=ANOMALY_CONFIG['max_samples'],
            random_state=ANOMALY_CONFIG['random_state'],
            n_jobs=-1,
            verbose=1
        )
        
        self.model.fit(X)
        
        # Evaluate on training data
        predictions = self.model.predict(X)
        scores = self.model.score_samples(X)
        
        n_anomalies = (predictions == -1).sum()
        anomaly_rate = n_anomalies / len(predictions) * 100
        
        logger.info("=" * 60)
        logger.info("TRAINING COMPLETE")
        logger.info("=" * 60)
        logger.info(f"Total samples: {len(X)}")
        logger.info(f"Detected anomalies: {n_anomalies} ({anomaly_rate:.1f}%)")
        logger.info(f"Average anomaly score: {scores.mean():.4f}")
        logger.info(f"Score range: [{scores.min():.4f}, {scores.max():.4f}]")
        logger.info("=" * 60)
        
        # Save model
        if save_model:
            self.save_model()
        
        return self.model
    
    def save_model(self):
        """Save trained model to disk"""
        model_data = {
            'model': self.model,
            'feature_names': self.feature_names,
            'normal_ranges': self.normal_ranges,
            'config': ANOMALY_CONFIG,
            'trained_at': datetime.utcnow()
        }
        
        with open(str(ANOMALY_MODEL_PATH), 'wb') as f:  # Convert Path to string
            pickle.dump(model_data, f)
        
        logger.info(f"Model saved to {ANOMALY_MODEL_PATH}")
    
    def load_model(self):
        """Load trained model from disk"""
        try:
            with open(str(ANOMALY_MODEL_PATH), 'rb') as f:  # Convert Path to string
                model_data = pickle.load(f)
            
            self.model = model_data['model']
            self.feature_names = model_data['feature_names']
            self.normal_ranges = model_data.get('normal_ranges', {})
            
            logger.info(f"Loaded model from {ANOMALY_MODEL_PATH}")
            logger.info(f"Model trained at: {model_data.get('trained_at')}")
            return True
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False
    
    def detect_anomaly(self, feeding_event_data, return_details=True):
        """
        Detect if a feeding event is anomalous
        
        Args:
            feeding_event_data: Dictionary with feeding event features
            return_details: Whether to return detailed explanation
            
        Returns:
            Dictionary with anomaly detection results
        """
        if self.model is None:
            if not self.load_model():
                logger.error("No model available for detection")
                return None
        
        # Prepare features
        features = self._prepare_features(feeding_event_data)
        
        if features is None:
            return None
        
        # Predict
        X = pd.DataFrame([features])
        
        # Ensure feature order matches training
        X = X[self.feature_names]
        
        prediction = self.model.predict(X)[0]
        score = self.model.score_samples(X)[0]
        
        is_anomaly = prediction == -1
        
        # Normalize score to 0-1 range (lower score = more anomalous)
        # Isolation forest scores are typically between -0.5 and 0.5
        anomaly_confidence = max(0.0, min(1.0, (0.5 - score)))
        
        result = {
            'is_anomaly': bool(is_anomaly),
            'anomaly_score': float(score),
            'anomaly_confidence': float(anomaly_confidence),
            'timestamp': datetime.utcnow()
        }
        
        # Add detailed explanation
        if return_details and is_anomaly:
            result['explanation'] = self._explain_anomaly(features, feeding_event_data)
            result['anomaly_type'] = self._classify_anomaly_type(features, feeding_event_data)
        
        logger.info(f"Anomaly detection: {'ANOMALY' if is_anomaly else 'NORMAL'} (score: {score:.4f}, confidence: {anomaly_confidence:.2f})")
        
        return result
    
    def _prepare_features(self, feeding_event_data):
        """Prepare features from feeding event data"""
        try:
            # Extract basic features
            features = {
                'amount_consumed': feeding_event_data.get('amount_consumed', 0),
                'eating_duration': feeding_event_data.get('eating_duration', 0),
                'completion_rate': feeding_event_data.get('completion_rate', 0),
                'time_since_last_meal': feeding_event_data.get('time_since_last_meal', 0),
            }
            
            # Time features
            timestamp = feeding_event_data.get('timestamp', datetime.utcnow())
            if isinstance(timestamp, str):
                timestamp = datetime.fromisoformat(timestamp)
            
            features['hour_of_day'] = timestamp.hour
            features['day_of_week'] = timestamp.weekday()
            
            # Calculate eating speed
            if features['eating_duration'] > 0:
                features['eating_speed'] = features['amount_consumed'] / features['eating_duration']
            else:
                features['eating_speed'] = 0
            
            features['hours_since_last_meal'] = features['time_since_last_meal'] / 60.0
            
            return features
            
        except Exception as e:
            logger.error(f"Error preparing features: {e}")
            return None
    
    def _explain_anomaly(self, features, raw_data):
        """
        Generate human-readable explanation of anomaly
        
        Args:
            features: Prepared features
            raw_data: Original feeding event data
            
        Returns:
            List of explanation strings
        """
        explanations = []
        
        # Check each feature against normal ranges
        for feature, value in features.items():
            if feature not in self.normal_ranges:
                continue
            
            ranges = self.normal_ranges[feature]
            
            # Check if significantly outside normal range
            if value < ranges['q25'] - 1.5 * (ranges['q75'] - ranges['q25']):
                explanations.append(f"Very low {feature.replace('_', ' ')}: {value:.1f} (normal: {ranges['mean']:.1f})")
            elif value > ranges['q75'] + 1.5 * (ranges['q75'] - ranges['q25']):
                explanations.append(f"Very high {feature.replace('_', ' ')}: {value:.1f} (normal: {ranges['mean']:.1f})")
        
        return explanations if explanations else ["Anomaly detected in overall pattern"]
    
    def _classify_anomaly_type(self, features, raw_data):
        """
        Classify the type of anomaly
        
        Returns:
            String describing anomaly type
        """
        # Check for specific patterns
        if features['completion_rate'] < 50:
            return 'reduced_appetite'
        elif features['eating_duration'] > self.normal_ranges.get('eating_duration', {}).get('mean', 200) * 1.5:
            return 'slow_eating'
        elif features['eating_duration'] < self.normal_ranges.get('eating_duration', {}).get('mean', 200) * 0.5:
            return 'fast_eating'
        elif features['amount_consumed'] < self.normal_ranges.get('amount_consumed', {}).get('mean', 100) * 0.5:
            return 'low_consumption'
        elif features['time_since_last_meal'] > 1440:  # More than 24 hours
            return 'missed_feeding'
        else:
            return 'unusual_pattern'
    
    def batch_detect(self, pet_id=None, days=7):
        """
        Run anomaly detection on recent feeding events
        
        Args:
            pet_id: Specific pet or None for all
            days: Number of days to analyze
            
        Returns:
            DataFrame with anomaly detection results
        """
        if self.model is None:
            if not self.load_model():
                return None
        
        # Get recent data
        df = self.data_preparer.extract_feeding_data(pet_id=pet_id, days=days, min_samples=1)
        
        if df is None or len(df) == 0:
            logger.warning("No data to analyze")
            return None
        
        # Engineer features
        df = self.data_preparer.engineer_features(df)
        
        # Select features
        X = df[self.feature_names].copy()
        
        # Detect anomalies
        predictions = self.model.predict(X)
        scores = self.model.score_samples(X)
        
        # Add results to dataframe
        df['is_anomaly'] = predictions == -1
        df['anomaly_score'] = scores
        df['anomaly_confidence'] = np.maximum(0, np.minimum(1, 0.5 - scores))
        
        # Classify anomaly types
        df['anomaly_type'] = df.apply(
            lambda row: self._classify_anomaly_type(
                row[self.feature_names].to_dict(),
                row.to_dict()
            ) if row['is_anomaly'] else None,
            axis=1
        )
        
        n_anomalies = df['is_anomaly'].sum()
        logger.info(f"Batch detection: {n_anomalies} anomalies out of {len(df)} events ({n_anomalies/len(df)*100:.1f}%)")
        
        return df
    
    def get_anomaly_summary(self, pet_id=None, days=30):
        """
        Get summary of detected anomalies
        
        Args:
            pet_id: Specific pet or None for all
            days: Number of days to analyze
            
        Returns:
            Dictionary with anomaly statistics
        """
        df = self.batch_detect(pet_id=pet_id, days=days)
        
        if df is None:
            return None
        
        anomalies = df[df['is_anomaly'] == True]
        
        summary = {
            'total_events': len(df),
            'anomalies_detected': len(anomalies),
            'anomaly_rate': len(anomalies) / len(df) * 100,
            'anomaly_types': anomalies['anomaly_type'].value_counts().to_dict(),
            'avg_anomaly_score': anomalies['anomaly_score'].mean(),
            'recent_anomalies': anomalies.tail(5)[['timestamp', 'anomaly_type', 'anomaly_confidence']].to_dict('records')
        }
        
        return summary


# CLI functions
def train_model_cli(pet_id=None):
    """Train anomaly detection model from command line"""
    detector = BehaviorAnomalyDetector()
    model = detector.train(pet_id=pet_id)
    
    if model:
        print("\n✅ Anomaly detection model training complete!")
        print(f"Model saved to: {ANOMALY_MODEL_PATH}")
    else:
        print("\n❌ Training failed - check if you have enough data")


def analyze_anomalies_cli(pet_id=None, days=7):
    """Analyze recent anomalies from command line"""
    detector = BehaviorAnomalyDetector()
    
    summary = detector.get_anomaly_summary(pet_id=pet_id, days=days)
    
    if summary:
        print("\n" + "=" * 60)
        print("ANOMALY DETECTION SUMMARY")
        print("=" * 60)
        print(f"Total Events: {summary['total_events']}")
        print(f"Anomalies Detected: {summary['anomalies_detected']}")
        print(f"Anomaly Rate: {summary['anomaly_rate']:.1f}%")
        print("\nAnomaly Types:")
        for atype, count in summary['anomaly_types'].items():
            print(f"  • {atype}: {count}")
        print("\nRecent Anomalies:")
        for i, anomaly in enumerate(summary['recent_anomalies'], 1):
            print(f"  {i}. {anomaly['timestamp']} - {anomaly['anomaly_type']} (confidence: {anomaly['anomaly_confidence']:.2f})")
        print("=" * 60)
    else:
        print("\n❌ Analysis failed")


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == 'train':
            pet_id = int(sys.argv[2]) if len(sys.argv) > 2 else None
            train_model_cli(pet_id)
        elif sys.argv[1] == 'analyze':
            pet_id = int(sys.argv[2]) if len(sys.argv) > 2 else None
            days = int(sys.argv[3]) if len(sys.argv) > 3 else 7
            analyze_anomalies_cli(pet_id, days)
    else:
        print("Usage:")
        print("  python anomaly_detector.py train [pet_id]")
        print("  python anomaly_detector.py analyze [pet_id] [days]")
