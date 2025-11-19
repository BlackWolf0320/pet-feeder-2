# AI Pet Food Feeder System - Base System

A smart pet feeding system with RFID identification, automated dispensing, eating pattern analysis, and AI-ready architecture.

## 🎯 Project Overview

This is the **base system** that handles all data collection, storage, and device communication. It's designed with clear placeholders for AI models to be added later.

### Current Features ✅
- RFID pet identification
- Real-time weight sensor monitoring
- Automated food dispensing control
- Data collection and storage
- REST API for all operations
- MQTT-based IoT communication
- Basic anomaly detection (statistical)
- Alert system
- Device status monitoring

### AI Features (Placeholders Ready) 🤖
- Eating pattern prediction (LSTM model placeholder)
- Anomaly detection (ML model placeholder)
- Schedule optimization (RL placeholder)
- Behavioral analysis (ready for AI)

---

## 📁 Project Structure

```
pet_feeder/
├── app.py                  # Main Flask API server
├── database.py             # Database models and setup
├── mqtt_client.py          # IoT device communication
├── data_processor.py       # Data processing and basic analysis
├── simulator.py            # IoT device simulator for testing
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

---

## 🚀 Quick Start Guide

### Step 1: Install Dependencies

```bash
cd pet_feeder
pip install -r requirements.txt
```

### Step 2: Start MQTT Broker (Required)

You need an MQTT broker. Install Mosquitto:

**Ubuntu/Debian:**
```bash
sudo apt-get install mosquitto mosquitto-clients
sudo systemctl start mosquitto
```

**macOS:**
```bash
brew install mosquitto
brew services start mosquitto
```

**Windows:**
Download from: https://mosquitto.org/download/

Or use Docker:
```bash
docker run -it -p 1883:1883 eclipse-mosquitto
```

### Step 3: Start the System

**Terminal 1 - Start API Server:**
```bash
python app.py
```

The server will start on `http://localhost:5000`

**Terminal 2 - Run Device Simulator (for testing):**
```bash
python simulator.py
```

---

## 📊 Database Schema

### Tables:
1. **pets** - Pet information and profiles
2. **feeding_events** - Individual feeding sessions
3. **health_metrics** - Daily aggregated health data
4. **device_status** - Hardware status logs
5. **alerts** - System notifications
6. **feeding_schedules** - Scheduled feeding times

All tables include AI-ready fields for future model integration.

---

## 🔌 API Endpoints

### Pet Management

**Get all pets:**
```http
GET /api/pets
```

**Register new pet:**
```http
POST /api/pets
Content-Type: application/json

{
  "rfid_tag": "RFID_12345",
  "name": "Max",
  "pet_type": "dog",
  "breed": "Golden Retriever",
  "weight": 25.5,
  "age": 36,
  "daily_food_target": 400
}
```

**Get specific pet:**
```http
GET /api/pets/{pet_id}
```

**Update pet:**
```http
PUT /api/pets/{pet_id}
Content-Type: application/json

{
  "weight": 26.0,
  "daily_food_target": 420
}
```

### Feeding Operations

**Manual dispense:**
```http
POST /api/feeding/manual
Content-Type: application/json

{
  "pet_id": 1,
  "amount": 150
}
```

**Get feeding history:**
```http
GET /api/feeding/history/{pet_id}?days=7
```

**Get feeding schedule:**
```http
GET /api/feeding/schedule/{pet_id}
```

**Add feeding schedule:**
```http
POST /api/feeding/schedule/{pet_id}
Content-Type: application/json

{
  "scheduled_time": "08:00",
  "food_amount": 150
}
```

### Analytics

**Get daily analytics:**
```http
GET /api/analytics/daily/{pet_id}?date=2025-11-20
```

**Get trends:**
```http
GET /api/analytics/trends/{pet_id}?days=30
```

**Get summary statistics:**
```http
GET /api/stats/summary/{pet_id}
```

### Alerts

**Get alerts:**
```http
GET /api/alerts?unread=true
```

**Mark alert as read:**
```http
PUT /api/alerts/{alert_id}/read
```

### Device Status

**Get device status:**
```http
GET /api/device/status
```

### AI Endpoints (Placeholders)

**Predict next feeding:**
```http
GET /api/ai/predict/{pet_id}
```

**Optimize schedule:**
```http
POST /api/ai/optimize/{pet_id}
```

---

## 📡 MQTT Topics

### Device → Server (Subscribe):
- `feeder/rfid/detected` - RFID tag detection
- `feeder/sensor/weight` - Weight sensor readings
- `feeder/dispenser/status` - Dispenser status updates
- `feeder/device/status` - General device status
- `feeder/food/level` - Food container level

### Server → Device (Publish):
- `feeder/dispenser/command` - Dispensing commands

### Example Payloads:

**RFID Detection:**
```json
{
  "rfid_tag": "RFID_12345",
  "timestamp": "2025-11-20T10:30:00",
  "signal_strength": -45
}
```

**Weight Sensor:**
```json
{
  "weight": 145.5,
  "stable": true,
  "timestamp": "2025-11-20T10:30:00"
}
```

**Dispense Command:**
```json
{
  "pet_id": 1,
  "amount": 150,
  "timestamp": "2025-11-20T10:30:00",
  "command": "dispense"
}
```

---

## 🧪 Testing with Simulator

The simulator mimics real hardware without physical devices.

### Available Test Scenarios:

1. **Normal Feeding** - Pet eats normally (90% completion)
2. **Reduced Appetite** - Pet eats only 40% (triggers alert)
3. **Slow Eating** - Pet takes 10 minutes (triggers alert)
4. **Multiple Pets** - Test with different pets
5. **Continuous Monitoring** - Periodic status updates
6. **Custom Mode** - Interactive testing

### Running Tests:

```bash
# First, register a pet using the API
curl -X POST http://localhost:5000/api/pets \
  -H "Content-Type: application/json" \
  -d '{
    "rfid_tag": "RFID_12345",
    "name": "Max",
    "pet_type": "dog",
    "weight": 25.5,
    "daily_food_target": 400
  }'

# Then run simulator and select a scenario
python simulator.py
```

---

## 🤖 AI Integration Points

The system is designed with clear placeholders for AI models. Here's where to add them:

### 1. Eating Pattern Prediction (LSTM)
**Location:** `data_processor.py` → `_analyze_feeding_pattern()`

**Input Data Available:**
- Historical feeding times
- Amount consumed per session
- Eating duration
- Time between meals

**What to Implement:**
```python
# Add LSTM model to predict next feeding time and amount
def predict_next_feeding(pet_id):
    # Load historical data
    # Preprocess for LSTM
    # Run prediction
    # Return predicted time and amount
    pass
```

### 2. Anomaly Detection (Isolation Forest / Autoencoder)
**Location:** `data_processor.py` → `_analyze_feeding_pattern()`

**Current:** Simple statistical thresholds
**Upgrade to:** ML-based anomaly detection

**What to Implement:**
```python
def detect_anomalies_ml(feeding_event):
    # Train Isolation Forest on normal patterns
    # Detect outliers in eating behavior
    # Calculate anomaly score
    # Generate appropriate alerts
    pass
```

### 3. Schedule Optimization (Reinforcement Learning)
**Location:** New file `ai_optimizer.py`

**What to Implement:**
```python
def optimize_feeding_schedule(pet_id):
    # Use Q-Learning or similar
    # Optimize based on:
    #   - Eating completion rates
    #   - Pet health metrics
    #   - Time preferences
    # Update FeedingSchedule table with optimized times
    pass
```

### 4. Behavioral Risk Scoring
**Location:** `database.py` → `HealthMetric.behavior_risk_score`

**What to Implement:**
```python
def calculate_risk_score(pet_id):
    # Analyze patterns over time
    # Score based on:
    #   - Appetite changes
    #   - Eating duration changes
    #   - Frequency changes
    # Return risk score 0-100
    pass
```

---

## 📈 Data Available for AI Training

### Feeding Events Data:
- Timestamp
- Amount dispensed
- Amount consumed
- Eating duration
- Time since last meal
- Approach count
- Hesitation time
- Completion rate

### Aggregated Metrics:
- Daily food consumption
- Feeding frequency
- Average eating duration
- Average time between meals
- Weight changes over time

### Features for ML:
- Time-based features (hour, day of week)
- Pet characteristics (age, weight, breed)
- Environmental factors (temperature, humidity)
- Historical patterns

---

## 🔧 Configuration

### Database
Default: SQLite (`pet_feeder.db`)

To use PostgreSQL:
```python
# In app.py
engine = init_db('postgresql://user:password@localhost/pet_feeder')
```

### MQTT Broker
Default: `localhost:1883`

To change:
```python
# In app.py
mqtt_client = PetFeederMQTTClient(
    broker_address="your-broker.com",
    port=1883
)
```

---

## 🛠️ Next Steps

### Phase 1: AI Model Development (Upcoming)
1. Collect data for 2-4 weeks using current system
2. Implement LSTM for pattern prediction
3. Add Isolation Forest for anomaly detection
4. Develop RL agent for schedule optimization

### Phase 2: Advanced Features
1. Computer vision for pet recognition
2. Multi-pet separation logic
3. Integration with vet systems
4. Mobile app development

---

## 📝 Notes

### Alert Types:
- `behavior` - Eating pattern anomalies
- `health` - Health-related concerns
- `device` - Hardware issues
- `food_low` - Container needs refilling

### Severity Levels:
- `low` - Informational
- `medium` - Attention needed
- `high` - Urgent attention
- `critical` - Immediate action required

### System Architecture:
```
IoT Device (RFID + Sensors)
        ↓ MQTT
   MQTT Broker
        ↓
  Data Processor → Database → REST API
                       ↓
                  AI Models (Future)
```

---

## 🐛 Troubleshooting

**MQTT Connection Failed:**
- Ensure Mosquitto is running: `systemctl status mosquitto`
- Check port 1883 is not blocked

**Database Errors:**
- Delete `pet_feeder.db` and restart to recreate
- Check file permissions

**Simulator Not Working:**
- Ensure API server is running first
- Check MQTT broker is accessible

---

## 📄 License

This is a base system for educational/development purposes.

---

## 🤝 Contributing

When adding AI models:
1. Keep data processing logic separate
2. Add model files in `ai_models/` directory
3. Update API endpoints for predictions
4. Document model architecture and training process

---

**Ready to add AI? The system is waiting for your ML models! 🚀**
