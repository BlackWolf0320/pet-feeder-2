from datetime import datetime, timedelta
from database import (
    Pet, FeedingEvent, HealthMetric, DeviceStatus, 
    Alert, get_session
)
import logging
from sqlalchemy import func

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataProcessor:
    """Process and store incoming IoT data"""
    
    def __init__(self, db_session):
        self.session = db_session
        self.current_feeding_sessions = {}  # Track active feeding sessions
        
    def process_rfid_detection(self, payload):
        """Handle RFID tag detection"""
        rfid_tag = payload.get('rfid_tag')
        timestamp = datetime.fromisoformat(payload.get('timestamp'))
        
        # Find pet by RFID
        pet = self.session.query(Pet).filter_by(rfid_tag=rfid_tag).first()
        
        if not pet:
            logger.warning(f"Unknown RFID tag detected: {rfid_tag}")
            self._create_alert(
                None, 
                'device', 
                'medium', 
                'Unknown Pet Detected',
                f'RFID tag {rfid_tag} is not registered in the system'
            )
            return None
        
        logger.info(f"Pet detected: {pet.name} (ID: {pet.id})")
        
        # Check if there's an ongoing feeding session
        if pet.id in self.current_feeding_sessions:
            logger.info(f"Resuming feeding session for {pet.name}")
            return self.current_feeding_sessions[pet.id]
        
        # Calculate time since last meal
        last_feeding = self.session.query(FeedingEvent)\
            .filter_by(pet_id=pet.id)\
            .order_by(FeedingEvent.timestamp.desc())\
            .first()
        
        time_since_last_meal = None
        if last_feeding:
            time_diff = timestamp - last_feeding.timestamp
            time_since_last_meal = int(time_diff.total_seconds() / 60)
        
        # Create new feeding event
        feeding_event = FeedingEvent(
            pet_id=pet.id,
            timestamp=timestamp,
            time_since_last_meal=time_since_last_meal,
            eating_start_time=timestamp
        )
        
        self.session.add(feeding_event)
        self.session.commit()
        
        # Store in active sessions
        self.current_feeding_sessions[pet.id] = feeding_event
        
        logger.info(f"Created new feeding event (ID: {feeding_event.id}) for {pet.name}")
        return feeding_event
    
    def process_weight_sensor(self, payload):
        """Handle weight sensor readings"""
        weight = payload.get('weight')  # Current weight in bowl
        timestamp = datetime.fromisoformat(payload.get('timestamp'))
        is_stable = payload.get('stable', False)
        
        # Find active feeding session
        # In real scenario, you'd correlate this with recent RFID detection
        active_sessions = list(self.current_feeding_sessions.values())
        
        if not active_sessions:
            logger.debug("Weight reading received but no active feeding session")
            return
        
        # Get most recent session
        feeding_event = active_sessions[-1]
        
        # Update consumption data
        if feeding_event.amount_dispensed:
            amount_consumed = feeding_event.amount_dispensed - weight
            feeding_event.amount_consumed = max(0, amount_consumed)
            feeding_event.completion_rate = (amount_consumed / feeding_event.amount_dispensed) * 100
        
        # If weight is stable and low, feeding is complete
        if is_stable and weight < 10:  # Less than 10g remaining
            feeding_event.eating_end_time = timestamp
            
            if feeding_event.eating_start_time:
                duration = (timestamp - feeding_event.eating_start_time).total_seconds()
                feeding_event.eating_duration = int(duration)
            
            # Remove from active sessions
            if feeding_event.pet_id in self.current_feeding_sessions:
                del self.current_feeding_sessions[feeding_event.pet_id]
            
            logger.info(f"Feeding session completed for event ID: {feeding_event.id}")
            
            # Trigger AI analysis placeholder
            self._analyze_feeding_pattern(feeding_event)
        
        self.session.commit()
        logger.debug(f"Updated weight for feeding event {feeding_event.id}: {weight}g")
    
    def process_dispenser_status(self, payload):
        """Handle dispenser status updates"""
        status = payload.get('status')
        amount_dispensed = payload.get('amount_dispensed')
        start_time = payload.get('start_time')
        end_time = payload.get('end_time')
        
        if status == 'dispensing' and start_time:
            # Find the most recent feeding event
            active_sessions = list(self.current_feeding_sessions.values())
            if active_sessions:
                feeding_event = active_sessions[-1]
                feeding_event.dispense_start_time = datetime.fromisoformat(start_time)
                self.session.commit()
                logger.info(f"Dispensing started for event {feeding_event.id}")
        
        elif status == 'completed' and amount_dispensed:
            active_sessions = list(self.current_feeding_sessions.values())
            if active_sessions:
                feeding_event = active_sessions[-1]
                feeding_event.amount_dispensed = amount_dispensed
                feeding_event.dispense_end_time = datetime.fromisoformat(end_time)
                self.session.commit()
                logger.info(f"Dispensed {amount_dispensed}g for event {feeding_event.id}")
        
        elif status == 'jammed' or status == 'error':
            self._create_alert(
                None,
                'device',
                'high',
                'Dispenser Malfunction',
                f'Dispenser status: {status}. Please check the device.'
            )
    
    def process_device_status(self, payload):
        """Handle general device status updates"""
        device_status = DeviceStatus(
            timestamp=datetime.fromisoformat(payload.get('timestamp')),
            food_level=payload.get('food_level'),
            dispenser_status=payload.get('dispenser_status'),
            rfid_reader_status=payload.get('rfid_status'),
            weight_sensor_status=payload.get('weight_sensor_status'),
            temperature=payload.get('temperature'),
            humidity=payload.get('humidity'),
            wifi_signal_strength=payload.get('wifi_signal')
        )
        
        self.session.add(device_status)
        self.session.commit()
        
        # Check for low food level
        if payload.get('food_level', 100) < 20:
            self._create_alert(
                None,
                'food_low',
                'medium',
                'Food Level Low',
                f'Food container is at {payload.get("food_level")}%. Please refill soon.'
            )
        
        logger.info("Device status updated")
    
    def _analyze_feeding_pattern(self, feeding_event):
        """
        PLACEHOLDER: AI analysis will go here
        This will be implemented when we add ML models
        """
        pet = feeding_event.pet
        logger.info(f"[AI PLACEHOLDER] Analyzing feeding pattern for {pet.name}")
        
        # Get recent feeding history (last 7 days)
        week_ago = datetime.utcnow() - timedelta(days=7)
        recent_feedings = self.session.query(FeedingEvent)\
            .filter(
                FeedingEvent.pet_id == pet.id,
                FeedingEvent.timestamp >= week_ago,
                FeedingEvent.eating_duration != None
            ).all()
        
        if len(recent_feedings) < 5:
            logger.info("Not enough data for pattern analysis yet")
            return
        
        # Calculate basic statistics
        avg_duration = sum(f.eating_duration for f in recent_feedings) / len(recent_feedings)
        avg_amount = sum(f.amount_consumed for f in recent_feedings if f.amount_consumed) / len(recent_feedings)
        
        # Simple anomaly detection (to be replaced with AI)
        if feeding_event.eating_duration and feeding_event.eating_duration > avg_duration * 1.5:
            feeding_event.anomaly_detected = True
            feeding_event.anomaly_type = 'slow_eating'
            self._create_alert(
                pet.id,
                'behavior',
                'medium',
                'Unusual Eating Behavior',
                f'{pet.name} is eating slower than usual. Duration: {feeding_event.eating_duration}s vs avg: {avg_duration:.0f}s'
            )
        
        if feeding_event.amount_consumed and feeding_event.amount_consumed < avg_amount * 0.5:
            feeding_event.anomaly_detected = True
            feeding_event.anomaly_type = 'reduced_appetite'
            self._create_alert(
                pet.id,
                'behavior',
                'high',
                'Reduced Appetite Detected',
                f'{pet.name} ate only {feeding_event.amount_consumed:.0f}g vs usual {avg_amount:.0f}g'
            )
        
        self.session.commit()
        logger.info("[AI PLACEHOLDER] Basic pattern analysis completed")
    
    def _create_alert(self, pet_id, alert_type, severity, title, message):
        """Create a new alert"""
        alert = Alert(
            pet_id=pet_id,
            alert_type=alert_type,
            severity=severity,
            title=title,
            message=message,
            is_ai_generated=False  # Will be True when AI generates alerts
        )
        self.session.add(alert)
        self.session.commit()
        logger.warning(f"Alert created: {title}")
        return alert
    
    def calculate_daily_metrics(self, pet_id, date=None):
        """Calculate daily health metrics for a pet"""
        if date is None:
            date = datetime.utcnow().date()
        
        start_of_day = datetime.combine(date, datetime.min.time())
        end_of_day = datetime.combine(date, datetime.max.time())
        
        # Get all feeding events for the day
        daily_feedings = self.session.query(FeedingEvent)\
            .filter(
                FeedingEvent.pet_id == pet_id,
                FeedingEvent.timestamp >= start_of_day,
                FeedingEvent.timestamp <= end_of_day
            ).all()
        
        if not daily_feedings:
            logger.info(f"No feeding data for pet {pet_id} on {date}")
            return None
        
        # Calculate metrics
        total_consumed = sum(f.amount_consumed for f in daily_feedings if f.amount_consumed)
        feeding_frequency = len(daily_feedings)
        
        durations = [f.eating_duration for f in daily_feedings if f.eating_duration]
        avg_duration = sum(durations) / len(durations) if durations else 0
        
        times_between = [f.time_since_last_meal for f in daily_feedings if f.time_since_last_meal]
        avg_time_between = sum(times_between) / len(times_between) if times_between else 0
        
        # Create or update health metric
        health_metric = self.session.query(HealthMetric)\
            .filter(
                HealthMetric.pet_id == pet_id,
                func.date(HealthMetric.date) == date
            ).first()
        
        if not health_metric:
            health_metric = HealthMetric(pet_id=pet_id, date=start_of_day)
            self.session.add(health_metric)
        
        health_metric.total_food_consumed = total_consumed
        health_metric.feeding_frequency = feeding_frequency
        health_metric.average_eating_duration = avg_duration
        health_metric.average_time_between_meals = avg_time_between
        
        # PLACEHOLDER: AI will calculate these scores
        health_metric.eating_consistency_score = 85.0  # Placeholder
        health_metric.appetite_score = 90.0  # Placeholder
        
        self.session.commit()
        logger.info(f"Daily metrics calculated for pet {pet_id} on {date}")
        
        return health_metric
