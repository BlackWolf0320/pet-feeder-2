from flask import Flask, request, jsonify
from flask_cors import CORS
from datetime import datetime, timedelta
from database import (
    init_db, get_session, Pet, FeedingEvent, 
    HealthMetric, DeviceStatus, Alert, FeedingSchedule
)
from mqtt_client import PetFeederMQTTClient
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

# Initialize MQTT client
mqtt_client = PetFeederMQTTClient()
data_processor = DataProcessor(db_session)

# MQTT message handler
def handle_mqtt_message(topic, payload):
    """Route MQTT messages to appropriate processor"""
    try:
        if 'rfid/detected' in topic:
            data_processor.process_rfid_detection(payload)
        elif 'sensor/weight' in topic:
            data_processor.process_weight_sensor(payload)
        elif 'dispenser/status' in topic:
            data_processor.process_dispenser_status(payload)
        elif 'device/status' in topic:
            data_processor.process_device_status(payload)
    except Exception as e:
        logger.error(f"Error handling MQTT message: {e}")

mqtt_client.set_data_handler(handle_mqtt_message)

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
    
    # Get recent feeding stats
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
    """Manually trigger food dispensing"""
    data = request.json
    pet_id = data.get('pet_id')
    amount = data.get('amount', 100)
    
    pet = db_session.query(Pet).filter_by(id=pet_id).first()
    if not pet:
        return jsonify({'error': 'Pet not found'}), 404
    
    # Send MQTT command to dispenser
    command = mqtt_client.publish_dispense_command(pet_id, amount)
    
    # Create feeding event
    feeding_event = FeedingEvent(
        pet_id=pet_id,
        timestamp=datetime.utcnow(),
        amount_dispensed=amount,
        is_manual_dispense=True,
        is_scheduled=False
    )
    
    db_session.add(feeding_event)
    db_session.commit()
    
    logger.info(f"Manual dispense triggered: {amount}g for {pet.name}")
    
    return jsonify({
        'message': 'Dispensing food',
        'amount': amount,
        'feeding_event_id': feeding_event.id
    })

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
        scheduled_time=data['scheduled_time'],  # Format: "08:00"
        food_amount=data['food_amount']
    )
    
    db_session.add(schedule)
    db_session.commit()
    
    return jsonify({
        'message': 'Schedule added successfully',
        'schedule_id': schedule.id
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
    
    # Calculate or retrieve metrics
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

# ==================== DEVICE STATUS ENDPOINTS ====================

@app.route('/api/device/status', methods=['GET'])
def get_device_status():
    """Get current device status"""
    latest_status = db_session.query(DeviceStatus)\
        .order_by(desc(DeviceStatus.timestamp))\
        .first()
    
    if not latest_status:
        return jsonify({'error': 'No device status available'}), 404
    
    return jsonify({
        'timestamp': latest_status.timestamp.isoformat(),
        'food_level': latest_status.food_level,
        'dispenser_status': latest_status.dispenser_status,
        'rfid_reader_status': latest_status.rfid_reader_status,
        'weight_sensor_status': latest_status.weight_sensor_status,
        'temperature': latest_status.temperature,
        'humidity': latest_status.humidity,
        'wifi_signal_strength': latest_status.wifi_signal_strength
    })

# ==================== AI PLACEHOLDER ENDPOINTS ====================

@app.route('/api/ai/predict/<int:pet_id>', methods=['GET'])
def predict_next_feeding(pet_id):
    """
    PLACEHOLDER: AI prediction for next feeding
    This will be implemented with ML models
    """
    return jsonify({
        'message': 'AI prediction not yet implemented',
        'placeholder_data': {
            'predicted_time': (datetime.utcnow() + timedelta(hours=6)).isoformat(),
            'predicted_amount': 120.0,
            'confidence': 0.0
        }
    })

@app.route('/api/ai/optimize/<int:pet_id>', methods=['POST'])
def optimize_schedule(pet_id):
    """
    PLACEHOLDER: AI schedule optimization
    This will use ML to optimize feeding schedule
    """
    return jsonify({
        'message': 'AI optimization not yet implemented',
        'status': 'pending'
    })

# ==================== STATISTICS ENDPOINTS ====================

@app.route('/api/stats/summary/<int:pet_id>', methods=['GET'])
def get_summary_stats(pet_id):
    """Get summary statistics for a pet"""
    pet = db_session.query(Pet).filter_by(id=pet_id).first()
    
    if not pet:
        return jsonify({'error': 'Pet not found'}), 404
    
    # Last 7 days stats
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

# ==================== STARTUP ====================

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.utcnow().isoformat()
    })

if __name__ == '__main__':
    logger.info("Starting Pet Feeder System...")
    
    # Connect to MQTT broker
    mqtt_client.connect()
    
    try:
        # Start Flask server
        app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)
    except KeyboardInterrupt:
        logger.info("Shutting down...")
        mqtt_client.disconnect()
        db_session.close()
