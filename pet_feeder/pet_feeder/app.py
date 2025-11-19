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

# ==================== AI ENDPOINTS ====================

from ai_manager import AIManager

# Initialize AI Manager
ai_manager = AIManager(db_session)

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
    
    logger.info(f"Training AI models for pet_id={pet_id}")
    
    results = ai_manager.train_all_models(pet_id=pet_id, force=force)
    
    return jsonify(results)

@app.route('/api/ai/predict/<int:pet_id>', methods=['GET'])
def predict_next_feeding(pet_id):
    """
    AI prediction for next feeding time and amount
    """
    try:
        prediction = ai_manager.lstm_predictor.predict_next_feeding(pet_id)
        
        if prediction:
            return jsonify({
                'success': True,
                'prediction': {
                    'predicted_time': prediction['predicted_time'].isoformat(),
                    'predicted_amount': prediction['predicted_amount'],
                    'confidence': prediction['confidence'],
                    'avg_interval_minutes': prediction['avg_interval_minutes']
                }
            })
        else:
            return jsonify({
                'success': False,
                'message': 'Not enough data for prediction'
            }), 400
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/ai/optimize/<int:pet_id>', methods=['POST'])
def optimize_schedule(pet_id):
    """
    Generate and apply optimized feeding schedule
    """
    try:
        schedules = ai_manager.optimize_and_update_schedule(pet_id)
        
        if schedules:
            return jsonify({
                'success': True,
                'message': 'Schedule optimized',
                'schedule': [{
                    'id': s.id,
                    'time': s.scheduled_time,
                    'amount': s.food_amount,
                    'is_ai_optimized': s.is_ai_optimized
                } for s in schedules]
            })
        else:
            return jsonify({
                'success': False,
                'message': 'Schedule optimization failed'
            }), 400
    except Exception as e:
        logger.error(f"Optimization error: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/ai/analyze/<int:feeding_event_id>', methods=['GET'])
def analyze_feeding(feeding_event_id):
    """Analyze specific feeding event with AI"""
    feeding_event = db_session.query(FeedingEvent).get(feeding_event_id)
    
    if not feeding_event:
        return jsonify({'error': 'Feeding event not found'}), 404
    
    try:
        analysis = ai_manager.analyze_feeding_event(feeding_event)
        
        return jsonify({
            'success': True,
            'analysis': analysis
        })
    except Exception as e:
        logger.error(f"Analysis error: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/ai/insights/<int:pet_id>', methods=['GET'])
def get_ai_insights(pet_id):
    """Get comprehensive AI insights for a pet"""
    days = request.args.get('days', 7, type=int)
    
    try:
        insights = ai_manager.get_ai_insights(pet_id, days)
        
        # Convert datetime objects to strings for JSON serialization
        if insights['predictions'] and 'predicted_time' in insights['predictions']:
            insights['predictions']['predicted_time'] = insights['predictions']['predicted_time'].isoformat()
        
        insights['generated_at'] = insights['generated_at'].isoformat()
        
        return jsonify({
            'success': True,
            'insights': insights
        })
    except Exception as e:
        logger.error(f"Insights error: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/ai/auto-analyze', methods=['POST'])
def auto_analyze():
    """Automatically analyze recent feedings and create alerts"""
    data = request.json or {}
    hours = data.get('hours', 24)
    
    try:
        alerts_created = ai_manager.auto_analyze_recent_feedings(hours)
        
        return jsonify({
            'success': True,
            'alerts_created': alerts_created,
            'message': f'Analyzed feedings from last {hours} hours'
        })
    except Exception as e:
        logger.error(f"Auto-analysis error: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/ai/anomaly/detect', methods=['POST'])
def detect_anomaly():
    """Detect anomaly in feeding data"""
    data = request.json
    
    if not data:
        return jsonify({'error': 'No data provided'}), 400
    
    try:
        result = ai_manager.anomaly_detector.detect_anomaly(data)
        
        if result:
            return jsonify({
                'success': True,
                'result': result
            })
        else:
            return jsonify({
                'success': False,
                'message': 'Anomaly detection failed'
            }), 400
    except Exception as e:
        logger.error(f"Anomaly detection error: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/ai/models/status', methods=['GET'])
def get_models_status():
    """Get status of all AI models"""
    status = {
        'lstm': {
            'loaded': ai_manager.lstm_predictor.model is not None,
            'type': 'Pattern Prediction (LSTM)'
        },
        'anomaly': {
            'loaded': ai_manager.anomaly_detector.model is not None,
            'type': 'Anomaly Detection (Isolation Forest)'
        },
        'schedule': {
            'loaded': bool(ai_manager.schedule_optimizer.q_table),
            'type': 'Schedule Optimization (Q-Learning)'
        }
    }
    
    return jsonify(status)

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
