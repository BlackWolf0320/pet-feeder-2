# System Architecture Documentation

## Overview Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                        IoT Hardware Layer                        │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐         │
│  │ RFID Reader  │  │ Weight Sensor│  │   Dispenser  │         │
│  │  (Collar)    │  │   (Load Cell)│  │ (Servo Motor)│         │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘         │
│         │                  │                  │                  │
│         └──────────────────┴──────────────────┘                  │
│                            │                                      │
│                   ┌────────▼─────────┐                           │
│                   │  Microcontroller  │                           │
│                   │  (RPi/Arduino)    │                           │
│                   └────────┬──────────┘                           │
└────────────────────────────┼──────────────────────────────────────┘
                             │ WiFi/Ethernet
                             │
┌────────────────────────────▼──────────────────────────────────────┐
│                      Communication Layer                          │
│                   ┌─────────────────────┐                         │
│                   │   MQTT Broker       │                         │
│                   │  (Mosquitto)        │                         │
│                   │  Port: 1883         │                         │
│                   └─────────┬───────────┘                         │
└──────────────────────────────┼────────────────────────────────────┘
                               │
┌──────────────────────────────▼────────────────────────────────────┐
│                    Application Layer                              │
│                                                                    │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │              MQTT Client (mqtt_client.py)                   │  │
│  │  • Subscribes to device topics                              │  │
│  │  • Publishes commands to device                             │  │
│  └───────────────────────┬────────────────────────────────────┘  │
│                          │                                        │
│  ┌───────────────────────▼────────────────────────────────────┐  │
│  │         Data Processor (data_processor.py)                  │  │
│  │  • Processes RFID detections                                │  │
│  │  • Monitors weight changes                                  │  │
│  │  • Tracks feeding sessions                                  │  │
│  │  • Basic anomaly detection                                  │  │
│  │  • Generates alerts                                         │  │
│  └───────────────────────┬────────────────────────────────────┘  │
│                          │                                        │
│  ┌───────────────────────▼────────────────────────────────────┐  │
│  │              Database Layer (database.py)                   │  │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐      │  │
│  │  │   Pets   │ │ Feeding  │ │  Health  │ │  Device  │      │  │
│  │  │          │ │  Events  │ │ Metrics  │ │  Status  │      │  │
│  │  └──────────┘ └──────────┘ └──────────┘ └──────────┘      │  │
│  │  ┌──────────┐ ┌──────────┐                                 │  │
│  │  │  Alerts  │ │ Schedule │                                 │  │
│  │  └──────────┘ └──────────┘                                 │  │
│  └────────────────────────────────────────────────────────────┘  │
│                          │                                        │
│  ┌───────────────────────▼────────────────────────────────────┐  │
│  │               Flask API (app.py)                            │  │
│  │  • Pet management endpoints                                 │  │
│  │  • Feeding control endpoints                                │  │
│  │  • Analytics endpoints                                      │  │
│  │  • Alert management                                         │  │
│  │  • AI placeholder endpoints                                 │  │
│  └───────────────────────┬────────────────────────────────────┘  │
└────────────────────────────┼──────────────────────────────────────┘
                             │ HTTP/REST
┌────────────────────────────▼──────────────────────────────────────┐
│                      Client Layer                                 │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐           │
│  │  Web App     │  │  Mobile App  │  │  Dashboard   │           │
│  │  (Future)    │  │  (Future)    │  │  (Future)    │           │
│  └──────────────┘  └──────────────┘  └──────────────┘           │
└───────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────────┐
│                    AI Layer (To Be Implemented)                    │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐            │
│  │ LSTM Model   │  │   Isolation  │  │      RL      │            │
│  │  (Pattern    │  │    Forest    │  │  (Schedule   │            │
│  │ Prediction)  │  │  (Anomaly)   │  │ Optimizer)   │            │
│  └──────────────┘  └──────────────┘  └──────────────┘            │
└────────────────────────────────────────────────────────────────────┘
```

## Data Flow

### 1. Feeding Event Flow

```
Pet Approaches
      ↓
RFID Detected → MQTT: feeder/rfid/detected
      ↓
Data Processor: Create FeedingEvent
      ↓
System Decides: Manual or Scheduled Dispense
      ↓
MQTT Command → feeder/dispenser/command
      ↓
Dispenser Activates
      ↓
MQTT Status → feeder/dispenser/status
      ↓
Data Processor: Update FeedingEvent.amount_dispensed
      ↓
Pet Eats (Weight decreases)
      ↓
MQTT Weight → feeder/sensor/weight (continuous)
      ↓
Data Processor: Update FeedingEvent.amount_consumed
      ↓
Eating Complete (Weight stable & low)
      ↓
Data Processor: Calculate duration & completion_rate
      ↓
Basic Anomaly Detection
      ↓
Create Alert if anomaly detected
      ↓
Store in Database
```

### 2. Daily Metrics Calculation

```
End of Day or API Request
      ↓
Query all FeedingEvents for the day
      ↓
Calculate:
  • Total food consumed
  • Feeding frequency
  • Average eating duration
  • Average time between meals
      ↓
[AI PLACEHOLDER] Calculate:
  • Eating consistency score
  • Appetite score
  • Behavior risk score
      ↓
Store in HealthMetric table
```

### 3. Alert Generation

```
Anomaly Detected
      ↓
Check Type:
  • Reduced appetite
  • Increased appetite
  • Slow eating
  • Fast eating
  • No feeding in 24h
      ↓
Calculate Severity:
  • Low: Minor deviation
  • Medium: Moderate concern
  • High: Significant change
  • Critical: Emergency
      ↓
Create Alert Record
      ↓
[Future] Send Notification:
  • Push notification
  • Email
  • SMS
```

## Database Schema Details

### Core Tables

**1. pets**
```
id                  INTEGER PRIMARY KEY
rfid_tag            VARCHAR(50) UNIQUE
name                VARCHAR(100)
pet_type            VARCHAR(50)
breed               VARCHAR(100)
weight              FLOAT (kg)
age                 INTEGER (months)
activity_level      VARCHAR(20)
daily_food_target   FLOAT (grams)
created_at          DATETIME
updated_at          DATETIME
```

**2. feeding_events**
```
id                      INTEGER PRIMARY KEY
pet_id                  INTEGER FK → pets.id
timestamp               DATETIME
amount_dispensed        FLOAT (grams)
amount_consumed         FLOAT (grams)
eating_start_time       DATETIME
eating_end_time         DATETIME
eating_duration         INTEGER (seconds)
time_since_last_meal    INTEGER (minutes)
completion_rate         FLOAT (percentage)
is_manual_dispense      BOOLEAN
anomaly_detected        BOOLEAN
anomaly_type            VARCHAR(100)
```

**3. health_metrics**
```
id                              INTEGER PRIMARY KEY
pet_id                          INTEGER FK → pets.id
date                            DATETIME
total_food_consumed             FLOAT
feeding_frequency               INTEGER
average_eating_duration         FLOAT
eating_consistency_score        FLOAT [AI]
appetite_score                  FLOAT [AI]
predicted_next_feeding_time     DATETIME [AI]
predicted_food_amount           FLOAT [AI]
behavior_risk_score             FLOAT [AI]
```

## API Request Flow

### Example: Manual Feeding

```
Client Request
      ↓
POST /api/feeding/manual
{
  "pet_id": 1,
  "amount": 150
}
      ↓
Flask Handler (app.py)
      ↓
1. Validate pet exists
2. Create FeedingEvent record
3. Publish MQTT command
      ↓
MQTT Client sends to device
      ↓
Device dispenses food
      ↓
Device sends status updates
      ↓
MQTT Client receives updates
      ↓
Data Processor updates record
      ↓
Response to Client
{
  "message": "Dispensing food",
  "feeding_event_id": 123
}
```

## AI Integration Points

### 1. Pattern Prediction (LSTM)

**Input:**
- Historical feeding times (timestamps)
- Amount consumed (sequence)
- Day of week, time of day (features)
- Pet characteristics (static features)

**Output:**
- Next feeding time prediction
- Recommended food amount
- Confidence score

**Where to implement:**
```python
# New file: ai_models/pattern_predictor.py

class FeedingPatternPredictor:
    def __init__(self):
        self.model = load_lstm_model()
    
    def predict_next_feeding(self, pet_id):
        # Get historical data
        # Preprocess for LSTM
        # Generate prediction
        return predicted_time, predicted_amount, confidence
```

**Hook into:**
- `data_processor.py` → `_analyze_feeding_pattern()`
- `app.py` → `/api/ai/predict/<pet_id>`
- `database.py` → Update `HealthMetric.predicted_*`

### 2. Anomaly Detection (Isolation Forest)

**Input:**
- Current feeding features
- Rolling window of past behavior

**Output:**
- Anomaly score (-1 to 1)
- Anomaly classification
- Confidence

**Where to implement:**
```python
# New file: ai_models/anomaly_detector.py

class BehaviorAnomalyDetector:
    def __init__(self):
        self.model = load_isolation_forest()
    
    def detect_anomaly(self, feeding_event):
        # Extract features
        # Get anomaly score
        # Classify type
        return is_anomaly, anomaly_type, score
```

**Hook into:**
- `data_processor.py` → Replace statistical detection
- Real-time as feeding completes

### 3. Schedule Optimizer (RL)

**State:**
- Current time
- Pet hunger level (inferred)
- Last feeding time
- Historical success rates

**Actions:**
- Dispense now
- Wait (various durations)
- Adjust amount

**Reward:**
- High completion rate: +reward
- Healthy eating duration: +reward
- Meets daily target: +reward
- Too frequent/infrequent: -reward

**Where to implement:**
```python
# New file: ai_models/schedule_optimizer.py

class ScheduleOptimizer:
    def __init__(self):
        self.agent = load_rl_agent()
    
    def optimize_schedule(self, pet_id):
        # Analyze current schedule performance
        # Run RL optimization
        # Generate new schedule
        return optimized_schedule
```

**Hook into:**
- `app.py` → `/api/ai/optimize/<pet_id>`
- Periodic background task (weekly)
- Update `FeedingSchedule` table

## Performance Considerations

### Database Indexing
```sql
CREATE INDEX idx_feeding_events_pet_timestamp 
ON feeding_events(pet_id, timestamp);

CREATE INDEX idx_health_metrics_pet_date 
ON health_metrics(pet_id, date);

CREATE INDEX idx_alerts_timestamp 
ON alerts(timestamp);
```

### Data Retention
- Feeding events: 1 year
- Device status: 30 days
- Alerts: Keep all (archive after resolution)

### Caching Strategy
- Current device status: Redis cache (5 min TTL)
- Daily metrics: Cache after calculation
- Pet profiles: Memory cache

## Security Considerations

### MQTT Security
- Enable authentication
- Use TLS for production
- Implement topic-level ACLs

### API Security
- Add authentication (JWT)
- Rate limiting
- Input validation
- SQL injection prevention (using ORM)

### Data Privacy
- Encrypt sensitive data
- Anonymize data for AI training
- Implement data deletion on request

## Monitoring & Logging

### What to Monitor
- MQTT connection status
- Database connection health
- API response times
- Error rates
- Alert frequency

### Logging Levels
- DEBUG: MQTT messages, detailed flow
- INFO: Normal operations, feeding events
- WARNING: Anomalies detected, retry attempts
- ERROR: Failures, exceptions
- CRITICAL: System failures

## Testing Strategy

### Unit Tests
- Database models
- Data processor logic
- MQTT message handling
- API endpoints

### Integration Tests
- End-to-end feeding flow
- MQTT → Database → API
- Alert generation

### Load Tests
- Multiple pets feeding simultaneously
- High-frequency sensor data
- API concurrent requests

## Deployment

### Development
- SQLite database
- Local MQTT broker
- Flask debug mode

### Production
- PostgreSQL database
- Cloud MQTT broker (AWS IoT, Azure IoT)
- Gunicorn + Nginx
- Docker containers
- Environment-based config

---

**This architecture is designed to be modular, scalable, and AI-ready!**
