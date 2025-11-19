"""
Pet Feeder App - NO MQTT VERSION
Run this if you don't have Mosquitto installed
This version works with API-only (no IoT communication)
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
from datetime import datetime, timedelta
from database import (
    init_db, get_session, Pet, FeedingEvent, 
    HealthMetric, DeviceStatus, Alert, FeedingSchedule
)
from data_processor import DataProcessor
import logging
from sqlalchemy import func, desc

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Initialize database
engine = init_db()
db_session = get_session(engine)

# Initialize data processor
data_processor = DataProcessor(db_session)

logger.warning("=" * 60)
logger.warning("RUNNING IN NO-MQTT MODE")
logger.warning("IoT features disabled - API-only mode")
logger.warning("Use manual feeding and simulator with direct API calls")
logger.warning("=" * 60)

# ==================== PET MANAGEMENT ENDPOINTS ====================

@app.route('/api/pets', methods=['GET'])
def get_pets():
    """Get all pets"""
    pets = db_session.query(Pet).all()
    return jsonify([{
        'id': p.id,
        'name': p.name,
        'rfid_tag': p.rfid_tag,
        'pet_type': p.pet_type,
        'breed': p.breed,
        'weight': p.weight,
        'age': p.age,
        'daily_food_target': p.daily_food_target
    } for p in pets])

@app.route('/api/pets', methods=['POST'])
def add_pet():
    """Register a new pet"""
    data = request.json
    
    pet = Pet(
        rfid_tag=data['rfid_tag'],
        name=data['name'],
        pet_type=data.get('pet_type'),
        breed=data.get('breed'),
        weight=data.get('weight'),
        age=data.get('age'),
        activity_level=data.get('activity_level', 'medium'),
        daily_food_target=data.get('daily_food_target', 200)
    )
    
    db_session.add(pet)
    db_session.commit()
    
    logger.info(f"New pet registered: {pet.name} (ID: {pet.id})")
    
    return jsonify({
        'message': 'Pet registered successfully',
        'pet_id': pet.id
    }), 201

@app.route('/api/pets/<int:pet_id>', methods=['GET'])
def get_pet(pet_id):
    """Get specific pet details"""
    pet = db_session.query(Pet).filter_by(id=pet_id).first()
    
    if not pet:
        return jsonify({'error': 'Pet not found'}), 404
    
    week_ago = datetime.utcnow() - timedelta(days=7)
    recent_feedings = db_session.query(FeedingEvent)\
        .filter(FeedingEvent.pet_id == pet_id, FeedingEvent.timestamp >= week_ago)\
        .count()
    
    return jsonify({
        'id': pet.id,
        'name': pet.name,
        'rfid_tag': pet.rfid_tag,
        'pet_type': pet.pet_type,
        'breed': pet.breed,
        'weight': pet.weight,
        'age': pet.age,
        'activity_level': pet.activity_level,
        'daily_food_target': pet.daily_food_target,
        'recent_feedings_count': recent_feedings
    })

@app.route('/api/pets/<int:pet_id>', methods=['PUT'])
def update_pet(pet_id):
    """Update pet information"""
    pet = db_session.query(Pet).filter_by(id=pet_id).first()
    
    if not pet:
        return jsonify({'error': 'Pet not found'}), 404
    
    data = request.json
    
    if 'name' in data:
        pet.name = data['name']
    if 'weight' in data:
        pet.weight = data['weight']
    if 'daily_food_target' in data:
        pet.daily_food_target = data['daily_food_target']
    if 'activity_level' in data:
        pet.activity_level = data['activity_level']
    
    pet.updated_at = datetime.utcnow()
    db_session.commit()
    
    return jsonify({'message': 'Pet updated successfully'})

# ==================== FEEDING ENDPOINTS ====================

@app.route('/api/feeding/manual', methods=['POST'])
def manual_dispense():
    """Manually trigger food dispensing (simulated without MQTT)"""
    data = request.json
    pet_id = data.get('pet_id')
    amount = data.get('amount', 100)
    
    pet = db_session.query(Pet).filter_by(id=pet_id).first()
    if not pet:
        return jsonify({'error': 'Pet not found'}), 404
    
    # Create feeding event (simulated)
    feeding_event = FeedingEvent(
        pet_id=pet_id,
        timestamp=datetime.utcnow(),
        amount_dispensed=amount,
        amount_consumed=amount * 0.9,  # Simulate 90% consumption
        eating_duration=180,  # Simulate 3 minutes
        completion_rate=90.0,
        is_manual_dispense=True,
        is_scheduled=False
    )
    
    db_session.add(feeding_event)
    db_session.commit()
    
    logger.info(f"Manual dispense (simulated): {amount}g for {pet.name}")
    
    return jsonify({
        'message': 'Feeding simulated (no MQTT available)',
        'amount': amount,
        'feeding_event_id': feeding_event.id,
        'note': 'MQTT disabled - this is a simulated feeding'
    })

@app.route('/api/feeding/simulate', methods=['POST'])
def simulate_feeding():
    """Simulate a complete feeding event (for testing without hardware)"""
    data = request.json
    
    pet_id = data.get('pet_id')
    amount_dispensed = data.get('amount_dispensed', 150)
    amount_consumed = data.get('amount_consumed', amount_dispensed * 0.9)
    eating_duration = data.get('eating_duration', 180)
    
    pet = db_session.query(Pet).filter_by(id=pet_id).first()
    if not pet:
        return jsonify({'error': 'Pet not found'}), 404
    
    # Calculate time since last meal
    last_feeding = db_session.query(FeedingEvent)\
        .filter_by(pet_id=pet_id)\
        .order_by(FeedingEvent.timestamp.desc())\
        .first()
    
    time_since_last_meal = None
    if last_feeding:
        time_diff = datetime.utcnow() - last_feeding.timestamp
        time_since_last_meal = int(time_diff.total_seconds() / 60)
    
    # Create feeding event
    feeding_event = FeedingEvent(
        pet_id=pet_id,
        timestamp=datetime.utcnow(),
        amount_dispensed=amount_dispensed,
        amount_consumed=amount_consumed,
        eating_duration=eating_duration,
        time_since_last_meal=time_since_last_meal,
        completion_rate=(amount_consumed / amount_dispensed) * 100 if amount_dispensed > 0 else 0,
        is_manual_dispense=True,
        is_scheduled=False
    )
    
    db_session.add(feeding_event)
    db_session.commit()
    
    logger.info(f"Simulated feeding for {pet.name}: {amount_consumed}g consumed")
    
    return jsonify({
        'message': 'Feeding event created',
        'feeding_event_id': feeding_event.id,
        'pet_name': pet.name,
        'completion_rate': feeding_event.completion_rate
    }), 201

@app.route('/api/feeding/history/<int:pet_id>', methods=['GET'])
def get_feeding_history(pet_id):
    """Get feeding history for a pet"""
    days = request.args.get('days', 7, type=int)
    start_date = datetime.utcnow() - timedelta(days=days)
    
    feedings = db_session.query(FeedingEvent)\
        .filter(
            FeedingEvent.pet_id == pet_id,
            FeedingEvent.timestamp >= start_date
        )\
        .order_by(desc(FeedingEvent.timestamp))\
        .all()
    
    return jsonify([{
        'id': f.id,
        'timestamp': f.timestamp.isoformat(),
        'amount_dispensed': f.amount_dispensed,
        'amount_consumed': f.amount_consumed,
        'eating_duration': f.eating_duration,
        'completion_rate': f.completion_rate,
        'is_manual': f.is_manual_dispense,
        'anomaly_detected': f.anomaly_detected,
        'anomaly_type': f.anomaly_type
    } for f in feedings])

@app.route('/api/feeding/schedule/<int:pet_id>', methods=['GET'])
def get_schedule(pet_id):
    """Get feeding schedule for a pet"""
    schedules = db_session.query(FeedingSchedule)\
        .filter_by(pet_id=pet_id, is_active=True)\
        .all()
    
    return jsonify([{
        'id': s.id,
        'scheduled_time': s.scheduled_time,
        'food_amount': s.food_amount,
        'is_ai_optimized': s.is_ai_optimized
    } for s in schedules])

@app.route('/api/feeding/schedule/<int:pet_id>', methods=['POST'])
def add_schedule(pet_id):
    """Add feeding schedule"""
    data = request.json
    
    schedule = FeedingSchedule(
        pet_id=pet_id,
        scheduled_time=data['scheduled_time'],
        food_amount=data['food_amount']
    )
    
    db_session.add(schedule)
    db_session.commit()
    
    return jsonify({
        'message': 'Schedule added successfully',
        'schedule_id': schedule.id,
        'note': 'MQTT disabled - schedules will not trigger automatically'
    }), 201

# ==================== ANALYTICS ENDPOINTS ====================

@app.route('/api/analytics/daily/<int:pet_id>', methods=['GET'])
def get_daily_analytics(pet_id):
    """Get daily analytics for a pet"""
    date_str = request.args.get('date')
    
    if date_str:
        target_date = datetime.fromisoformat(date_str).date()
    else:
        target_date = datetime.utcnow().date()
    
    metric = data_processor.calculate_daily_metrics(pet_id, target_date)
    
    if not metric:
        return jsonify({'error': 'No data available for this date'}), 404
    
    return jsonify({
        'date': metric.date.isoformat(),
        'total_food_consumed': metric.total_food_consumed,
        'feeding_frequency': metric.feeding_frequency,
        'average_eating_duration': metric.average_eating_duration,
        'average_time_between_meals': metric.average_time_between_meals,
        'eating_consistency_score': metric.eating_consistency_score,
        'appetite_score': metric.appetite_score
    })

@app.route('/api/analytics/trends/<int:pet_id>', methods=['GET'])
def get_trends(pet_id):
    """Get trend data over time"""
    days = request.args.get('days', 30, type=int)
    start_date = datetime.utcnow() - timedelta(days=days)
    
    metrics = db_session.query(HealthMetric)\
        .filter(
            HealthMetric.pet_id == pet_id,
            HealthMetric.date >= start_date
        )\
        .order_by(HealthMetric.date)\
        .all()
    
    return jsonify([{
        'date': m.date.isoformat(),
        'total_food_consumed': m.total_food_consumed,
        'feeding_frequency': m.feeding_frequency,
        'appetite_score': m.appetite_score,
        'weight': m.weight
    } for m in metrics])

# ==================== ALERT ENDPOINTS ====================

@app.route('/api/alerts', methods=['GET'])
def get_alerts():
    """Get all alerts"""
    unread_only = request.args.get('unread', 'false').lower() == 'true'
    
    query = db_session.query(Alert)
    
    if unread_only:
        query = query.filter_by(is_read=False)
    
    alerts = query.order_by(desc(Alert.timestamp)).limit(50).all()
    
    return jsonify([{
        'id': a.id,
        'pet_id': a.pet_id,
        'timestamp': a.timestamp.isoformat(),
        'alert_type': a.alert_type,
        'severity': a.severity,
        'title': a.title,
        'message': a.message,
        'is_read': a.is_read,
        'is_ai_generated': a.is_ai_generated
    } for a in alerts])

@app.route('/api/alerts/<int:alert_id>/read', methods=['PUT'])
def mark_alert_read(alert_id):
    """Mark alert as read"""
    alert = db_session.query(Alert).filter_by(id=alert_id).first()
    
    if not alert:
        return jsonify({'error': 'Alert not found'}), 404
    
    alert.is_read = True
    db_session.commit()
    
    return jsonify({'message': 'Alert marked as read'})

# ==================== STATISTICS ENDPOINTS ====================

@app.route('/api/stats/summary/<int:pet_id>', methods=['GET'])
def get_summary_stats(pet_id):
    """Get summary statistics for a pet"""
    pet = db_session.query(Pet).filter_by(id=pet_id).first()
    
    if not pet:
        return jsonify({'error': 'Pet not found'}), 404
    
    week_ago = datetime.utcnow() - timedelta(days=7)
    
    total_feedings = db_session.query(func.count(FeedingEvent.id))\
        .filter(FeedingEvent.pet_id == pet_id, FeedingEvent.timestamp >= week_ago)\
        .scalar()
    
    total_consumed = db_session.query(func.sum(FeedingEvent.amount_consumed))\
        .filter(FeedingEvent.pet_id == pet_id, FeedingEvent.timestamp >= week_ago)\
        .scalar() or 0
    
    anomalies = db_session.query(func.count(FeedingEvent.id))\
        .filter(
            FeedingEvent.pet_id == pet_id,
            FeedingEvent.timestamp >= week_ago,
            FeedingEvent.anomaly_detected == True
        )\
        .scalar()
    
    return jsonify({
        'pet_name': pet.name,
        'period': '7 days',
        'total_feedings': total_feedings,
        'total_food_consumed': float(total_consumed),
        'average_per_feeding': float(total_consumed / total_feedings) if total_feedings > 0 else 0,
        'anomalies_detected': anomalies,
        'daily_target': pet.daily_food_target
    })

# ==================== AI ENDPOINTS ====================

# Import AI manager only if available
try:
    from ai_manager import AIManager
    ai_manager = AIManager(db_session)
    AI_AVAILABLE = True
except ImportError:
    AI_AVAILABLE = False
    logger.warning("AI modules not available - AI endpoints disabled")

if AI_AVAILABLE:
    @app.route('/api/ai/readiness', methods=['GET'])
    def check_ai_readiness():
        """Check if enough data is available for AI training"""
        pet_id = request.args.get('pet_id', type=int)
        readiness = ai_manager.check_training_readiness(pet_id=pet_id)
        return jsonify(readiness)
    
    @app.route('/api/ai/train', methods=['POST'])
    def train_ai_models():
        """Train all AI models"""
        data = request.json or {}
        pet_id = data.get('pet_id')
        force = data.get('force', False)
        results = ai_manager.train_all_models(pet_id=pet_id, force=force)
        return jsonify(results)
    
    @app.route('/api/ai/predict/<int:pet_id>', methods=['GET'])
    def predict_next_feeding(pet_id):
        """AI prediction for next feeding"""
        try:
            prediction = ai_manager.lstm_predictor.predict_next_feeding(pet_id)
            if prediction:
                return jsonify({
                    'success': True,
                    'prediction': {
                        'predicted_time': prediction['predicted_time'].isoformat(),
                        'predicted_amount': prediction['predicted_amount'],
                        'confidence': prediction['confidence']
                    }
                })
            else:
                return jsonify({'success': False, 'message': 'Not enough data'}), 400
        except Exception as e:
            return jsonify({'success': False, 'error': str(e)}), 500

# ==================== HEALTH CHECK ====================

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'mqtt_enabled': False,
        'ai_available': AI_AVAILABLE,
        'mode': 'no-mqtt',
        'timestamp': datetime.utcnow().isoformat(),
        'note': 'Running without MQTT - use /api/feeding/simulate for testing'
    })

@app.route('/api/info', methods=['GET'])
def get_info():
    """Get system information"""
    return jsonify({
        'version': '1.0',
        'mode': 'no-mqtt',
        'mqtt_enabled': False,
        'ai_enabled': AI_AVAILABLE,
        'features': {
            'pet_management': True,
            'manual_feeding': True,
            'simulated_feeding': True,
            'analytics': True,
            'alerts': True,
            'ai_training': AI_AVAILABLE,
            'ai_prediction': AI_AVAILABLE,
            'iot_communication': False
        },
        'endpoints': {
            'simulate_feeding': '/api/feeding/simulate'
        }
    })

if __name__ == '__main__':
    logger.info("=" * 60)
    logger.info("Starting Pet Feeder System (NO-MQTT MODE)")
    logger.info("=" * 60)
    logger.info("IoT features: DISABLED")
    logger.info("AI features: " + ("ENABLED" if AI_AVAILABLE else "DISABLED"))
    logger.info("Use /api/feeding/simulate to create test data")
    logger.info("=" * 60)
    
    try:
        app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)
    except KeyboardInterrupt:
        logger.info("Shutting down...")
        db_session.close()
