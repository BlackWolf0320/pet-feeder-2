# 🚀 Quick Start Guide

Get your AI Pet Feeder system running in 5 minutes!

## Step 1: Install Dependencies (1 minute)

```bash
cd pet_feeder
pip install -r requirements.txt
```

## Step 2: Verify Setup (30 seconds)

```bash
python test_setup.py
```

This will check if everything is installed correctly.

## Step 3: Start MQTT Broker (30 seconds)

**Choose one option:**

### Option A: Use Mosquitto (Recommended)
```bash
# Ubuntu/Debian
sudo apt-get install mosquitto
sudo systemctl start mosquitto

# macOS
brew install mosquitto
brew services start mosquitto
```

### Option B: Use Docker
```bash
docker run -d -p 1883:1883 eclipse-mosquitto
```

### Option C: Skip for now
You can test without MQTT - the API will work but IoT features won't.

## Step 4: Start the System (1 minute)

**Terminal 1 - Start API Server:**
```bash
python app.py
```

You should see:
```
Starting Pet Feeder System...
* Running on http://0.0.0.0:5000
```

**Terminal 2 - Start Device Simulator:**
```bash
python simulator.py
```

Choose a test scenario (1-6)

## Step 5: Test the System (2 minutes)

### Test 1: Register a Pet

```bash
curl -X POST http://localhost:5000/api/pets \
  -H "Content-Type: application/json" \
  -d '{
    "rfid_tag": "RFID_12345",
    "name": "Max",
    "pet_type": "dog",
    "breed": "Golden Retriever",
    "weight": 25.5,
    "age": 36,
    "daily_food_target": 400
  }'
```

Expected response:
```json
{
  "message": "Pet registered successfully",
  "pet_id": 1
}
```

### Test 2: Check Pets

```bash
curl http://localhost:5000/api/pets
```

### Test 3: Run Feeding Simulation

In the simulator terminal, choose **Option 1** (Normal feeding)

This will:
1. Simulate RFID detection
2. Trigger food dispensing
3. Simulate eating behavior
4. Store all data in database

### Test 4: Check Feeding History

```bash
curl http://localhost:5000/api/feeding/history/1
```

### Test 5: Check for Alerts

```bash
curl http://localhost:5000/api/alerts
```

### Test 6: Get Analytics

```bash
curl http://localhost:5000/api/stats/summary/1
```

## 📊 Access the Database

The system creates a SQLite database file: `pet_feeder.db`

To explore it:
```bash
sqlite3 pet_feeder.db
```

Useful queries:
```sql
-- View all pets
SELECT * FROM pets;

-- View recent feeding events
SELECT * FROM feeding_events ORDER BY timestamp DESC LIMIT 10;

-- View alerts
SELECT * FROM alerts ORDER BY timestamp DESC;
```

## 🧪 Test Scenarios Explained

### Scenario 1: Normal Feeding ✅
- Pet eats 90% of food
- Takes normal time (~3 minutes)
- No alerts generated

### Scenario 2: Reduced Appetite ⚠️
- Pet eats only 40% of food
- **Should trigger "Reduced Appetite" alert**
- Tests anomaly detection

### Scenario 3: Slow Eating ⚠️
- Pet takes 10 minutes to eat
- **Should trigger "Unusual Eating Behavior" alert**
- Tests duration-based detection

### Scenario 4: Multiple Pets 🐕🐈
- Tests system with 2 different pets
- Verifies RFID identification
- Checks data separation

### Scenario 5: Continuous Monitoring 📡
- Publishes device status every minute
- Tests real-time updates
- Monitors food levels

### Scenario 6: Custom Mode 🎮
- Interactive testing
- Manual control of all events
- Good for debugging

## 🔧 Common Issues & Solutions

### "Module not found" Error
```bash
pip install -r requirements.txt
```

### "MQTT Connection Failed"
- Make sure Mosquitto is running
- Check if port 1883 is free: `netstat -an | grep 1883`
- Try Docker option if Mosquitto won't install

### "Database is locked"
- Close any other processes using the database
- Delete `pet_feeder.db` and restart

### Simulator doesn't work
1. Make sure API server is running first
2. Check MQTT broker is running
3. Look for error messages in Terminal 1

## 📱 API Testing with Postman/Insomnia

Import these endpoints:

**Base URL:** `http://localhost:5000`

**Endpoints to try:**
- GET `/api/health` - Health check
- GET `/api/pets` - List all pets
- POST `/api/pets` - Register pet
- GET `/api/pets/1` - Get pet details
- POST `/api/feeding/manual` - Manual feeding
- GET `/api/feeding/history/1` - Feeding history
- GET `/api/analytics/daily/1` - Daily stats
- GET `/api/alerts` - View alerts
- GET `/api/device/status` - Device status

## 🎯 What You Should See

### After Normal Feeding Scenario:

**In Database:**
- ✅ New feeding_event record
- ✅ amount_dispensed = 150g
- ✅ amount_consumed ≈ 135g (90%)
- ✅ eating_duration ≈ 180 seconds
- ✅ completion_rate ≈ 90%

**In API Response:**
```json
{
  "total_feedings": 1,
  "total_food_consumed": 135.0,
  "average_per_feeding": 135.0
}
```

### After Reduced Appetite Scenario:

**In Alerts Table:**
- ⚠️ New alert with type "behavior"
- ⚠️ Title: "Reduced Appetite Detected"
- ⚠️ Severity: "high"

### Terminal Output Should Show:
```
[INFO] Pet detected: Max (ID: 1)
[INFO] Created new feeding event (ID: 1)
[INFO] Dispensed 150g for event 1
[INFO] Feeding session completed
[INFO] [AI PLACEHOLDER] Analyzing feeding pattern
```

## 🧪 Next Steps After Testing

1. **Collect Real Data:**
   - Run for 1-2 weeks
   - Let it collect actual patterns
   - Export data for AI training

2. **Implement AI Models:**
   - Start with LSTM for pattern prediction
   - Add Isolation Forest for anomalies
   - Train on collected data

3. **Add Features:**
   - Mobile app
   - Push notifications
   - Multi-user support
   - Camera integration

4. **Connect Real Hardware:**
   - Replace simulator with actual devices
   - Use ESP32/Arduino with sensors
   - Deploy to Raspberry Pi

## 💡 Pro Tips

1. **Keep simulator running** between tests to see continuous data flow
2. **Try different scenarios** to see how system handles various behaviors
3. **Check the database** after each test to understand data structure
4. **Read the logs** - they show exactly what's happening
5. **Use the API** to build your own client/dashboard

## 📚 Documentation

- `README.md` - Full documentation
- `ARCHITECTURE.md` - System design details
- `.env.example` - Configuration options

## 🆘 Need Help?

Check the logs:
```bash
# API logs show in Terminal 1
# Simulator logs show in Terminal 2
```

Database exploration:
```bash
sqlite3 pet_feeder.db "SELECT * FROM feeding_events;"
```

Test specific component:
```bash
python test_setup.py
```

---

**You're all set! 🎉 Start experimenting with the system and when you're ready, we'll add the AI models!**
