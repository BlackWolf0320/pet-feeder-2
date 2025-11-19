# 🤖 AI Models Guide - Complete Documentation

## Overview

Your pet feeder system now includes three powerful AI models:

1. **LSTM Pattern Predictor** - Predicts next feeding time and optimal amount
2. **Isolation Forest Anomaly Detector** - Detects unusual eating behaviors
3. **Q-Learning Schedule Optimizer** - Optimizes feeding schedules

## 📋 Prerequisites

### Minimum Data Requirements

- **50+ feeding events** (samples)
- **14+ days** of continuous data collection
- Complete feeding records (amount consumed, duration, timestamps)

### Installation

```bash
# Install AI dependencies
pip install -r requirements_ai.txt

# This includes: TensorFlow, scikit-learn, pandas, numpy, etc.
```

## 🎓 Training Workflow

### Step 1: Check Data Readiness

```bash
# Check if you have enough data
python data_preparer.py
```

Or via API:
```bash
curl http://localhost:5000/api/ai/readiness?pet_id=1
```

Expected output:
```json
{
  "ready": true,
  "total_samples": 75,
  "days_of_data": 21,
  "needs_more_samples": 0,
  "needs_more_days": 0
}
```

### Step 2: Train All Models at Once

**Option A: Command Line**
```bash
# Train for specific pet
python ai_manager.py train 1

# Train for all pets
python ai_manager.py train
```

**Option B: API**
```bash
curl -X POST http://localhost:5000/api/ai/train \
  -H "Content-Type: application/json" \
  -d '{"pet_id": 1}'
```

**Option C: Train Individual Models**
```bash
# Train LSTM predictor
python lstm_predictor.py train 1

# Train anomaly detector
python anomaly_detector.py train 1

# Train schedule optimizer
python schedule_optimizer.py train 1
```

### Step 3: Verify Training

```bash
# Check model status
curl http://localhost:5000/api/ai/models/status
```

Expected output:
```json
{
  "lstm": {
    "loaded": true,
    "type": "Pattern Prediction (LSTM)"
  },
  "anomaly": {
    "loaded": true,
    "type": "Anomaly Detection (Isolation Forest)"
  },
  "schedule": {
    "loaded": true,
    "type": "Schedule Optimization (Q-Learning)"
  }
}
```

## 🔮 Using the AI Models

### 1. Pattern Prediction (LSTM)

**Predict next feeding:**
```bash
curl http://localhost:5000/api/ai/predict/1
```

Response:
```json
{
  "success": true,
  "prediction": {
    "predicted_time": "2025-11-20T16:30:00",
    "predicted_amount": 145.5,
    "confidence": 0.87,
    "avg_interval_minutes": 360
  }
}
```

**Command line testing:**
```bash
python lstm_predictor.py predict 1
```

**What it does:**
- Analyzes last 7 feeding events
- Considers time patterns, amounts, durations
- Predicts optimal next feeding time
- Suggests food amount based on patterns
- Provides confidence score (0-1)

**Use cases:**
- Proactive scheduling
- Inventory planning
- Owner notifications
- Health monitoring

### 2. Anomaly Detection (Isolation Forest)

**Analyze recent feedings:**
```bash
curl http://localhost:5000/api/ai/insights/1?days=7
```

**Detect anomaly in specific feeding:**
```bash
curl -X POST http://localhost:5000/api/ai/anomaly/detect \
  -H "Content-Type: application/json" \
  -d '{
    "amount_consumed": 45,
    "eating_duration": 600,
    "completion_rate": 30,
    "time_since_last_meal": 240,
    "timestamp": "2025-11-20T14:00:00"
  }'
```

Response:
```json
{
  "is_anomaly": true,
  "anomaly_score": -0.45,
  "anomaly_confidence": 0.92,
  "anomaly_type": "reduced_appetite",
  "explanation": [
    "Very low amount consumed: 45.0 (normal: 135.0)",
    "Very low completion rate: 30.0 (normal: 90.0)"
  ]
}
```

**Command line testing:**
```bash
python anomaly_detector.py analyze 1 7
```

**Anomaly Types Detected:**
- `reduced_appetite` - Eating much less than usual
- `slow_eating` - Taking unusually long to eat
- `fast_eating` - Eating too quickly
- `low_consumption` - Small amounts consumed
- `missed_feeding` - Long gaps between meals
- `unusual_pattern` - Other behavioral changes

**Auto-analysis:**
```bash
curl -X POST http://localhost:5000/api/ai/auto-analyze \
  -H "Content-Type: application/json" \
  -d '{"hours": 24}'
```

This automatically analyzes recent feedings and creates alerts!

### 3. Schedule Optimization (Q-Learning)

**Generate optimal schedule:**
```bash
curl -X POST http://localhost:5000/api/ai/optimize/1
```

Response:
```json
{
  "success": true,
  "message": "Schedule optimized",
  "schedule": [
    {"time": "08:00", "amount": 100, "is_ai_optimized": true},
    {"time": "13:00", "amount": 150, "is_ai_optimized": true},
    {"time": "18:00", "amount": 100, "is_ai_optimized": true}
  ]
}
```

**Command line testing:**
```bash
python schedule_optimizer.py optimize 1
```

**What it optimizes:**
- Feeding times (hourly slots)
- Food amounts (small/medium/large)
- Total daily intake
- Completion rates
- Eating patterns

**Optimization goals:**
- Maximize food consumption efficiency
- Meet daily nutritional targets
- Maintain healthy eating speeds
- Reduce food waste
- Adapt to pet preferences

## 📊 Model Details

### LSTM Architecture

```
Input: [7 past feedings × 12 features]
  ↓
LSTM Layer (128 units) + Dropout + BatchNorm
  ↓
LSTM Layer (64 units) + Dropout + BatchNorm
  ↓
Dense Layer (32 units, ReLU)
  ↓
Dense Layer (16 units, ReLU)
  ↓
Output: Predicted amount (1 value)
```

**Training:**
- 100 epochs max
- Early stopping (patience: 10)
- Learning rate: 0.001
- Batch size: 32
- Validation split: 20%

**Features used:**
- Hour of day (cyclical encoded)
- Day of week (cyclical encoded)
- Amount consumed (historical)
- Eating duration
- Time since last meal
- Completion rate
- Eating speed
- Rolling averages

### Isolation Forest Configuration

- **Estimators:** 100 trees
- **Contamination:** 10% (expected anomaly rate)
- **Max samples:** Auto
- **Features:** 8 behavioral + temporal

**Detection process:**
1. Build ensemble of isolation trees
2. Calculate anomaly score for new sample
3. Compare to threshold (-0.5)
4. Classify anomaly type
5. Generate explanation

**Advantages:**
- No labeled data needed
- Detects novel anomalies
- Fast inference
- Interpretable results

### Q-Learning Configuration

**State space:**
- Time slot (24 hours)
- Hunger level (0-4)
- Time since last meal (5 categories)

**Actions:**
- Dispense small (50g)
- Dispense medium (100g)
- Dispense large (150g)
- Wait (no feeding)

**Rewards:**
- High completion rate: +1.0
- Healthy eating speed: +0.5
- Meeting daily target: +2.0
- Too frequent feeding: -0.5
- Too infrequent: -1.0
- Food waste: -1.0

**Training:**
- 1000 episodes
- Epsilon decay: 0.995
- Learning rate: 0.1
- Discount factor: 0.95

## 🎯 Integration Examples

### Auto-analyze after feeding

```python
from ai_manager import AIManager

ai_manager = AIManager()

# After a feeding event is completed
feeding_event = session.query(FeedingEvent).get(feeding_id)
analysis = ai_manager.analyze_feeding_event(feeding_event)

if analysis['anomaly_detection']['is_anomaly']:
    # Create alert
    send_notification(
        f"Unusual behavior detected: {analysis['anomaly_detection']['anomaly_type']}"
    )
```

### Daily health metrics update

```python
from ai_manager import AIManager

ai_manager = AIManager()

# Update daily metrics with AI predictions
for pet in pets:
    metric = ai_manager.update_health_metrics(pet.id)
    print(f"Predicted next feeding: {metric.predicted_next_feeding_time}")
    print(f"Behavior risk score: {metric.behavior_risk_score}")
```

### Scheduled optimization

```python
# Run weekly to optimize all pet schedules
from ai_manager import AIManager

ai_manager = AIManager()

for pet in pets:
    schedules = ai_manager.optimize_and_update_schedule(pet.id)
    print(f"Optimized {pet.name}'s schedule: {len(schedules)} feeding times")
```

## 📈 Performance Metrics

### LSTM Predictor

Good performance:
- MAE < 20g (amount prediction)
- MAPE < 15% (mean absolute percentage error)
- R² > 0.7 (variance explained)

Check with:
```python
from lstm_predictor import FeedingPatternPredictor

predictor = FeedingPatternPredictor()
metrics = predictor.evaluate_model(pet_id=1)
print(f"MAE: {metrics['mae']:.2f}g")
print(f"R²: {metrics['r2']:.4f}")
```

### Anomaly Detector

Good performance:
- Anomaly rate: 5-15%
- False positive rate < 10%
- Catches major behavioral changes

Check with:
```python
from anomaly_detector import BehaviorAnomalyDetector

detector = BehaviorAnomalyDetector()
summary = detector.get_anomaly_summary(pet_id=1, days=30)
print(f"Anomaly rate: {summary['anomaly_rate']:.1f}%")
```

### Schedule Optimizer

Good performance:
- Avg episode reward > 5.0
- Stable Q-values after 500 episodes
- Meets daily targets consistently

## 🔄 Retraining

Models should be retrained periodically to adapt to changing patterns.

**Automatic retraining (recommended):**

Set up a cron job:
```bash
# Retrain weekly (every Monday at 2 AM)
0 2 * * 1 cd /path/to/pet_feeder && python ai_manager.py train
```

**Manual retraining:**
```bash
# Retrain when you notice drift in predictions
python ai_manager.py train 1
```

**When to retrain:**
- After 1-2 weeks of new data
- When pet's routine changes
- After dietary changes
- Seasonally
- When predictions become inaccurate

## 🎨 Visualization

### Plot training history (LSTM):

```python
import matplotlib.pyplot as plt
from lstm_predictor import FeedingPatternPredictor

predictor = FeedingPatternPredictor()
history = predictor.train(pet_id=1)

plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.show()
```

### Plot anomaly scores:

```python
from anomaly_detector import BehaviorAnomalyDetector
import matplotlib.pyplot as plt

detector = BehaviorAnomalyDetector()
df = detector.batch_detect(pet_id=1, days=30)

plt.scatter(df.index, df['anomaly_score'], 
            c=df['is_anomaly'], cmap='RdYlGn')
plt.xlabel('Feeding Event')
plt.ylabel('Anomaly Score')
plt.show()
```

## ⚠️ Common Issues

### "Not enough data for training"
- Collect more feeding events (need 50+)
- Wait longer (need 14+ days)
- Use `force=True` to train anyway (not recommended)

### "Model not found"
- Train models first
- Check `ai_models/` directory exists
- Verify file permissions

### Low prediction accuracy
- Collect more data
- Check data quality (missing values?)
- Retrain with more epochs
- Adjust model hyperparameters in `ai_config.py`

### Too many false anomalies
- Adjust contamination parameter (default: 0.1)
- Collect more "normal" training data
- Retrain model

### Schedule not optimal
- Train for more episodes (default: 1000)
- Adjust reward function in `ai_config.py`
- Fine-tune Q-learning parameters

## 🔒 Best Practices

1. **Always check readiness before training**
2. **Validate predictions before acting on them**
3. **Review anomaly alerts manually initially**
4. **Start with conservative confidence thresholds**
5. **Retrain regularly (weekly recommended)**
6. **Keep track of model versions**
7. **Monitor prediction accuracy over time**
8. **Test on validation data before deployment**

## 📚 Further Reading

- **LSTM:** Time series prediction with recurrent neural networks
- **Isolation Forest:** Unsupervised anomaly detection
- **Q-Learning:** Model-free reinforcement learning
- **Feature Engineering:** Creating predictive features from raw data

## 🆘 Getting Help

**Check logs:**
```bash
tail -f logs/ai_models.log
```

**Test individual components:**
```bash
# Test data preparation
python data_preparer.py

# Test each model separately
python lstm_predictor.py train 1
python anomaly_detector.py train 1
python schedule_optimizer.py train 1
```

**Debug mode:**
Set logging level to DEBUG in code:
```python
logging.basicConfig(level=logging.DEBUG)
```

---

**You're now ready to build intelligent, adaptive pet feeding! 🚀🤖**
