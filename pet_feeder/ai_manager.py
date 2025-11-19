"""
AI Integration Module
Integrates all AI models with the main pet feeder application
"""

import logging
from datetime import datetime
from database import get_session, init_db, FeedingEvent, HealthMetric, Pet, FeedingSchedule
from lstm_predictor import FeedingPatternPredictor
from anomaly_detector import BehaviorAnomalyDetector
from schedule_optimizer import ScheduleOptimizer
from data_preparer import DataPreparer
from ai_config import PREDICTION_CONFIG

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AIManager:
    """Manages all AI models and provides unified interface"""
    
    def __init__(self, db_session=None):
        if db_session is None:
            engine = init_db()
            self.session = get_session(engine)
        else:
            self.session = db_session
        
        # Initialize AI models
        self.lstm_predictor = FeedingPatternPredictor()
        self.anomaly_detector = BehaviorAnomalyDetector()
        self.schedule_optimizer = ScheduleOptimizer()
        self.data_preparer = DataPreparer(self.session)
        
        # Load models if available
        self._load_models()
    
    def _load_models(self):
        """Attempt to load all trained models"""
        logger.info("Loading AI models...")
        
        lstm_loaded = self.lstm_predictor.load_model()
        anomaly_loaded = self.anomaly_detector.load_model()
        rl_loaded = self.schedule_optimizer.load_model()
        
        logger.info(f"LSTM Model: {'✅ Loaded' if lstm_loaded else '❌ Not available'}")
        logger.info(f"Anomaly Detector: {'✅ Loaded' if anomaly_loaded else '❌ Not available'}")
        logger.info(f"Schedule Optimizer: {'✅ Loaded' if rl_loaded else '❌ Not available'}")
    
    def check_training_readiness(self, pet_id=None):
        """
        Check if enough data is available to train models
        
        Args:
            pet_id: Specific pet or None for all
            
        Returns:
            Dictionary with readiness status
        """
        summary = self.data_preparer.get_data_summary(pet_id=pet_id)
        
        return {
            'ready': summary['ready_for_training'],
            'total_samples': summary['total_samples'],
            'days_of_data': summary['date_range']['days'] if summary['date_range'] else 0,
            'needs_more_samples': max(0, 50 - summary['total_samples']),
            'needs_more_days': max(0, 14 - (summary['date_range']['days'] if summary['date_range'] else 0)),
            'pets': summary['pets']
        }
    
    def train_all_models(self, pet_id=None, force=False):
        """
        Train all AI models
        
        Args:
            pet_id: Train for specific pet or None for all
            force: Force training even if not enough data
            
        Returns:
            Dictionary with training results
        """
        results = {
            'lstm': None,
            'anomaly': None,
            'schedule': None,
            'success': False
        }
        
        # Check readiness
        if not force:
            readiness = self.check_training_readiness(pet_id)
            if not readiness['ready']:
                logger.warning("Not enough data for training")
                return results
        
        logger.info("=" * 60)
        logger.info("TRAINING ALL AI MODELS")
        logger.info("=" * 60)
        
        # Train LSTM predictor
        try:
            logger.info("\n1. Training LSTM Pattern Predictor...")
            history = self.lstm_predictor.train(pet_id=pet_id)
            results['lstm'] = {'success': history is not None}
            logger.info("✅ LSTM training complete")
        except Exception as e:
            logger.error(f"❌ LSTM training failed: {e}")
            results['lstm'] = {'success': False, 'error': str(e)}
        
        # Train anomaly detector
        try:
            logger.info("\n2. Training Anomaly Detector...")
            model = self.anomaly_detector.train(pet_id=pet_id)
            results['anomaly'] = {'success': model is not None}
            logger.info("✅ Anomaly detector training complete")
        except Exception as e:
            logger.error(f"❌ Anomaly detector training failed: {e}")
            results['anomaly'] = {'success': False, 'error': str(e)}
        
        # Train schedule optimizer
        if pet_id is not None:  # RL is pet-specific
            try:
                logger.info("\n3. Training Schedule Optimizer...")
                history = self.schedule_optimizer.train_from_historical_data(pet_id)
                results['schedule'] = {'success': history is not None}
                if history:
                    self.schedule_optimizer.save_model()
                logger.info("✅ Schedule optimizer training complete")
            except Exception as e:
                logger.error(f"❌ Schedule optimizer training failed: {e}")
                results['schedule'] = {'success': False, 'error': str(e)}
        
        results['success'] = all(
            r['success'] for r in [results['lstm'], results['anomaly']] if r is not None
        )
        
        logger.info("\n" + "=" * 60)
        logger.info("TRAINING SUMMARY")
        logger.info("=" * 60)
        logger.info(f"LSTM: {'✅ Success' if results['lstm']['success'] else '❌ Failed'}")
        logger.info(f"Anomaly Detection: {'✅ Success' if results['anomaly']['success'] else '❌ Failed'}")
        logger.info(f"Schedule Optimization: {'✅ Success' if results['schedule'] and results['schedule']['success'] else '⚠️ Skipped or Failed'}")
        logger.info("=" * 60)
        
        return results
    
    def analyze_feeding_event(self, feeding_event):
        """
        Analyze a feeding event using AI models
        
        Args:
            feeding_event: FeedingEvent object or dict with feeding data
            
        Returns:
            Dictionary with AI analysis results
        """
        # Convert to dict if necessary
        if hasattr(feeding_event, '__dict__'):
            event_data = {
                'pet_id': feeding_event.pet_id,
                'timestamp': feeding_event.timestamp,
                'amount_consumed': feeding_event.amount_consumed,
                'eating_duration': feeding_event.eating_duration,
                'completion_rate': feeding_event.completion_rate,
                'time_since_last_meal': feeding_event.time_since_last_meal
            }
        else:
            event_data = feeding_event
        
        results = {
            'anomaly_detection': None,
            'prediction': None,
            'recommendations': []
        }
        
        # Anomaly detection
        try:
            anomaly_result = self.anomaly_detector.detect_anomaly(event_data)
            results['anomaly_detection'] = anomaly_result
            
            if anomaly_result and anomaly_result['is_anomaly']:
                results['recommendations'].append({
                    'type': 'alert',
                    'priority': 'high',
                    'message': f"Anomaly detected: {anomaly_result.get('anomaly_type', 'unknown')}",
                    'details': anomaly_result.get('explanation', [])
                })
        except Exception as e:
            logger.error(f"Anomaly detection failed: {e}")
        
        # Pattern prediction for next feeding
        try:
            prediction = self.lstm_predictor.predict_next_feeding(event_data['pet_id'])
            results['prediction'] = prediction
            
            if prediction and prediction['confidence'] > PREDICTION_CONFIG['confidence_threshold']:
                results['recommendations'].append({
                    'type': 'schedule',
                    'priority': 'medium',
                    'message': f"Next feeding predicted at {prediction['predicted_time'].strftime('%H:%M')}",
                    'amount': prediction['predicted_amount']
                })
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
        
        return results
    
    def update_health_metrics(self, pet_id, date=None):
        """
        Update health metrics with AI predictions
        
        Args:
            pet_id: Pet to update metrics for
            date: Date to update (default: today)
            
        Returns:
            Updated HealthMetric object
        """
        if date is None:
            date = datetime.utcnow().date()
        
        # Get or create health metric
        from sqlalchemy import func
        metric = self.session.query(HealthMetric)\
            .filter(
                HealthMetric.pet_id == pet_id,
                func.date(HealthMetric.date) == date
            ).first()
        
        if not metric:
            # Calculate basic metrics first
            metric = self.data_preparer.calculate_daily_metrics(pet_id, date)
            if not metric:
                return None
        
        # Add AI predictions
        try:
            prediction = self.lstm_predictor.predict_next_feeding(pet_id)
            if prediction:
                metric.predicted_next_feeding_time = prediction['predicted_time']
                metric.predicted_food_amount = prediction['predicted_amount']
        except Exception as e:
            logger.error(f"Failed to update predictions: {e}")
        
        # Calculate behavior risk score using anomaly detector
        try:
            recent_anomalies = self.anomaly_detector.batch_detect(pet_id=pet_id, days=7)
            if recent_anomalies is not None:
                anomaly_rate = recent_anomalies['is_anomaly'].sum() / len(recent_anomalies)
                metric.behavior_risk_score = anomaly_rate * 100  # 0-100 scale
        except Exception as e:
            logger.error(f"Failed to calculate risk score: {e}")
        
        self.session.commit()
        logger.info(f"Updated health metrics for pet {pet_id}")
        
        return metric
    
    def optimize_and_update_schedule(self, pet_id):
        """
        Generate optimized schedule and update database
        
        Args:
            pet_id: Pet to optimize schedule for
            
        Returns:
            List of created FeedingSchedule objects
        """
        logger.info(f"Optimizing schedule for pet {pet_id}")
        
        # Generate optimal schedule
        schedule = self.schedule_optimizer.generate_optimal_schedule(pet_id)
        
        if not schedule:
            logger.error("Failed to generate schedule")
            return None
        
        # Deactivate old schedules
        old_schedules = self.session.query(FeedingSchedule)\
            .filter_by(pet_id=pet_id, is_active=True)\
            .all()
        
        for old in old_schedules:
            old.is_active = False
        
        # Create new schedules
        new_schedules = []
        for feeding in schedule:
            schedule_obj = FeedingSchedule(
                pet_id=pet_id,
                scheduled_time=feeding['time'],
                food_amount=feeding['amount'],
                is_active=True,
                is_ai_optimized=True,
                optimization_date=datetime.utcnow()
            )
            self.session.add(schedule_obj)
            new_schedules.append(schedule_obj)
        
        self.session.commit()
        
        logger.info(f"Created {len(new_schedules)} optimized feeding times")
        
        return new_schedules
    
    def get_ai_insights(self, pet_id, days=7):
        """
        Get comprehensive AI insights for a pet
        
        Args:
            pet_id: Pet to analyze
            days: Number of days to analyze
            
        Returns:
            Dictionary with all AI insights
        """
        insights = {
            'pet_id': pet_id,
            'period_days': days,
            'generated_at': datetime.utcnow(),
            'anomalies': None,
            'predictions': None,
            'schedule': None,
            'recommendations': []
        }
        
        # Get anomaly summary
        try:
            anomaly_summary = self.anomaly_detector.get_anomaly_summary(pet_id, days)
            insights['anomalies'] = anomaly_summary
            
            if anomaly_summary and anomaly_summary['anomaly_rate'] > 20:
                insights['recommendations'].append({
                    'type': 'health_check',
                    'priority': 'high',
                    'message': f"High anomaly rate ({anomaly_summary['anomaly_rate']:.1f}%). Consider vet consultation."
                })
        except Exception as e:
            logger.error(f"Failed to get anomaly summary: {e}")
        
        # Get predictions
        try:
            prediction = self.lstm_predictor.predict_next_feeding(pet_id)
            insights['predictions'] = prediction
        except Exception as e:
            logger.error(f"Failed to get predictions: {e}")
        
        # Get optimized schedule
        try:
            schedule = self.schedule_optimizer.generate_optimal_schedule(pet_id)
            insights['schedule'] = schedule
            
            if schedule:
                total_daily = sum(s['amount'] for s in schedule)
                pet = self.session.query(Pet).get(pet_id)
                if pet and pet.daily_food_target:
                    if total_daily < pet.daily_food_target * 0.8:
                        insights['recommendations'].append({
                            'type': 'nutrition',
                            'priority': 'medium',
                            'message': f"Scheduled amount ({total_daily}g) is below target ({pet.daily_food_target}g)"
                        })
        except Exception as e:
            logger.error(f"Failed to get schedule: {e}")
        
        return insights
    
    def auto_analyze_recent_feedings(self, hours=24):
        """
        Automatically analyze recent feedings and create alerts
        
        Args:
            hours: Number of hours to look back
            
        Returns:
            Number of alerts created
        """
        from datetime import timedelta
        from database import Alert
        
        cutoff = datetime.utcnow() - timedelta(hours=hours)
        
        # Get recent feedings that haven't been analyzed
        recent_feedings = self.session.query(FeedingEvent)\
            .filter(
                FeedingEvent.timestamp >= cutoff,
                FeedingEvent.amount_consumed.isnot(None)
            )\
            .all()
        
        alerts_created = 0
        
        for feeding in recent_feedings:
            # Analyze with AI
            analysis = self.analyze_feeding_event(feeding)
            
            # Create alerts based on analysis
            if analysis['anomaly_detection'] and analysis['anomaly_detection']['is_anomaly']:
                anomaly = analysis['anomaly_detection']
                
                # Check if alert already exists for this feeding
                existing_alert = self.session.query(Alert)\
                    .filter_by(pet_id=feeding.pet_id)\
                    .filter(Alert.message.contains(f"feeding event {feeding.id}"))\
                    .first()
                
                if not existing_alert:
                    alert = Alert(
                        pet_id=feeding.pet_id,
                        alert_type='behavior',
                        severity='high' if anomaly['anomaly_confidence'] > 0.8 else 'medium',
                        title=f"Unusual Behavior: {anomaly.get('anomaly_type', 'Unknown')}",
                        message=f"AI detected anomaly in feeding event {feeding.id}. " + 
                                " ".join(anomaly.get('explanation', [])),
                        is_ai_generated=True,
                        confidence_score=anomaly['anomaly_confidence']
                    )
                    self.session.add(alert)
                    alerts_created += 1
        
        self.session.commit()
        logger.info(f"Created {alerts_created} AI-generated alerts")
        
        return alerts_created


# Utility functions
def check_and_train_models(pet_id=None):
    """Check data readiness and train models if ready"""
    manager = AIManager()
    
    readiness = manager.check_training_readiness(pet_id)
    
    print("=" * 60)
    print("AI TRAINING READINESS CHECK")
    print("=" * 60)
    print(f"Total Samples: {readiness['total_samples']}")
    print(f"Days of Data: {readiness['days_of_data']}")
    print(f"Ready: {'✅ YES' if readiness['ready'] else '❌ NO'}")
    
    if not readiness['ready']:
        if readiness['needs_more_samples'] > 0:
            print(f"  • Need {readiness['needs_more_samples']} more samples")
        if readiness['needs_more_days'] > 0:
            print(f"  • Need {readiness['needs_more_days']} more days")
    else:
        print("\nStarting training...")
        results = manager.train_all_models(pet_id)
        
        if results['success']:
            print("\n✅ All models trained successfully!")
        else:
            print("\n⚠️ Some models failed to train")
    
    print("=" * 60)


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == 'train':
        pet_id = int(sys.argv[2]) if len(sys.argv) > 2 else None
        check_and_train_models(pet_id)
    else:
        print("Usage: python ai_manager.py train [pet_id]")
