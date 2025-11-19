# 🔧 MQTT Connection Issues - Complete Fix Guide

## Common MQTT Errors

### Error 1: "Connection refused" or "Connection failed"
**Cause:** MQTT broker (Mosquitto) is not running

### Error 2: "Connection timeout"
**Cause:** Firewall blocking port 1883 or broker not accessible

### Error 3: "Module not found: paho.mqtt"
**Cause:** MQTT library not installed

---

## ✅ Solution 1: Install & Start Mosquitto MQTT Broker

### For Windows:

**Option A: Download and Install**
```bash
# 1. Download Mosquitto from:
https://mosquitto.org/download/

# 2. Install it (use default settings)

# 3. Start Mosquitto service
# Open Services (Win + R, type "services.msc")
# Find "Mosquitto Broker"
# Click "Start"

# Or from Command Prompt (as Administrator):
net start mosquitto
```

**Option B: Use Docker (Recommended)**
```bash
# Install Docker Desktop first, then:
docker run -d -p 1883:1883 --name mosquitto eclipse-mosquitto

# Verify it's running:
docker ps
```

### For macOS:

```bash
# Install Homebrew if not installed:
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install Mosquitto:
brew install mosquitto

# Start Mosquitto:
brew services start mosquitto

# Or run in foreground:
mosquitto -v
```

### For Linux (Ubuntu/Debian):

```bash
# Install Mosquitto:
sudo apt-get update
sudo apt-get install mosquitto mosquitto-clients

# Start Mosquitto:
sudo systemctl start mosquitto

# Enable auto-start on boot:
sudo systemctl enable mosquitto

# Check status:
sudo systemctl status mosquitto
```

### For Linux (CentOS/RHEL):

```bash
sudo yum install mosquitto mosquitto-clients
sudo systemctl start mosquitto
sudo systemctl enable mosquitto
```

---

## ✅ Solution 2: Verify MQTT Installation

```bash
# Test if MQTT broker is running:
# On Linux/macOS:
netstat -an | grep 1883

# On Windows:
netstat -an | findstr 1883

# You should see: 0.0.0.0:1883 or 127.0.0.1:1883
```

**Using mosquitto_sub (if installed):**
```bash
# Subscribe to test topic:
mosquitto_sub -h localhost -t test/topic

# In another terminal, publish:
mosquitto_pub -h localhost -t test/topic -m "Hello MQTT"

# If you see "Hello MQTT" in first terminal, MQTT works!
```

---

## ✅ Solution 3: Use Docker (Easiest)

If you have Docker installed, this is the simplest solution:

```bash
# Pull and run Mosquitto:
docker run -d \
  --name mosquitto \
  -p 1883:1883 \
  -p 9001:9001 \
  eclipse-mosquitto

# Verify it's running:
docker ps

# Check logs:
docker logs mosquitto

# Stop it:
docker stop mosquitto

# Start it again:
docker start mosquitto

# Remove it (if needed):
docker stop mosquitto && docker rm mosquitto
```

---

## ✅ Solution 4: Run Without MQTT (Testing Mode)

If you can't install MQTT right now, you can still test the system!

**Option A: Skip MQTT in Code**

Create a file called `app_no_mqtt.py`:

```python
"""
Pet Feeder App WITHOUT MQTT
Use this if you can't install Mosquitto
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
data_processor = DataProcessor(db_session)

# NOTE: MQTT client is disabled - using manual feeding only
logger.warning("Running in NO-MQTT mode - only manual feeding available")

# Include all your API endpoints from app.py here
# (Copy from the original app.py, but skip MQTT initialization)

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy (no MQTT)',
        'timestamp': datetime.utcnow().isoformat(),
        'mqtt_enabled': False
    })

# ... (copy other endpoints from app.py)

if __name__ == '__main__':
    logger.info("Starting Pet Feeder System (NO-MQTT mode)...")
    logger.warning("MQTT is disabled - using API-only mode")
    
    try:
        app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)
    except KeyboardInterrupt:
        logger.info("Shutting down...")
        db_session.close()
```

Then run:
```bash
python app_no_mqtt.py
```

**Option B: Mock MQTT Client**

Add this to the top of `app.py`:

```python
# Add after imports
import os
USE_MQTT = os.environ.get('USE_MQTT', 'true').lower() == 'true'

if USE_MQTT:
    try:
        mqtt_client = PetFeederMQTTClient()
        mqtt_client.set_data_handler(handle_mqtt_message)
        mqtt_client.connect()
        logger.info("MQTT client connected")
    except Exception as e:
        logger.error(f"MQTT connection failed: {e}")
        logger.warning("Running without MQTT - manual feeding only")
        mqtt_client = None
else:
    logger.info("MQTT disabled by environment variable")
    mqtt_client = None
```

Then run with:
```bash
# Disable MQTT
export USE_MQTT=false  # Linux/macOS
set USE_MQTT=false     # Windows CMD
$env:USE_MQTT="false"  # Windows PowerShell

python app.py
```

---

## ✅ Solution 5: Use Online MQTT Broker (Testing)

For quick testing without installing Mosquitto:

**Use a free public MQTT broker:**

Update `mqtt_client.py`:

```python
# Change broker address
def __init__(self, broker_address="test.mosquitto.org", port=1883, client_id="pet_feeder_server"):
    # ... rest of code
```

**Public MQTT brokers for testing:**
- `test.mosquitto.org` (port 1883)
- `broker.hivemq.com` (port 1883)
- `mqtt.eclipse.org` (port 1883)

**⚠️ Warning:** Public brokers are for testing only! Not secure for production.

---

## ✅ Solution 6: Firewall Configuration

### Windows Firewall:
```bash
# Run as Administrator:
netsh advfirewall firewall add rule name="Mosquitto MQTT" dir=in action=allow protocol=TCP localport=1883
```

### Linux (UFW):
```bash
sudo ufw allow 1883/tcp
sudo ufw reload
```

### macOS:
```bash
# Usually no firewall issues on localhost
# If needed, add rule in System Preferences > Security & Privacy > Firewall
```

---

## 🧪 Quick Test Script

Create `test_mqtt.py`:

```python
"""
Quick MQTT Connection Test
"""

import paho.mqtt.client as mqtt
import time

def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print("✅ MQTT Connected successfully!")
        client.subscribe("test/topic")
        client.publish("test/topic", "Hello from Pet Feeder!")
    else:
        print(f"❌ Connection failed with code {rc}")
        print(f"   Error: {mqtt.connack_string(rc)}")

def on_message(client, userdata, msg):
    print(f"📨 Received: {msg.payload.decode()} on topic: {msg.topic}")
    print("✅ MQTT is working correctly!")
    client.disconnect()

client = mqtt.Client("test_client")
client.on_connect = on_connect
client.on_message = on_message

print("Attempting to connect to MQTT broker...")
print("Broker: localhost:1883")

try:
    client.connect("localhost", 1883, 60)
    client.loop_start()
    time.sleep(2)
    client.loop_stop()
except Exception as e:
    print(f"❌ Connection failed: {e}")
    print("\nTroubleshooting:")
    print("1. Is Mosquitto installed?")
    print("2. Is Mosquitto running?")
    print("3. Check firewall settings")
    print("4. Try: mosquitto -v")
```

Run it:
```bash
python test_mqtt.py
```

---

## 📝 Step-by-Step Fix (Most Common)

### Step 1: Install Python MQTT Library
```bash
pip install paho-mqtt
```

### Step 2: Install Mosquitto Broker

**Windows:**
```bash
# Download from https://mosquitto.org/download/
# Install and start service
```

**macOS:**
```bash
brew install mosquitto
brew services start mosquitto
```

**Linux:**
```bash
sudo apt-get install mosquitto
sudo systemctl start mosquitto
```

**Docker (All Platforms):**
```bash
docker run -d -p 1883:1883 --name mosquitto eclipse-mosquitto
```

### Step 3: Verify Connection
```bash
# Check if port 1883 is listening:
netstat -an | grep 1883  # Linux/macOS
netstat -an | findstr 1883  # Windows
```

### Step 4: Test System
```bash
python test_mqtt.py
```

### Step 5: Run Your App
```bash
python app.py
```

---

## 🐛 Debugging Tips

### Check MQTT Broker Logs:

**Linux:**
```bash
sudo tail -f /var/log/mosquitto/mosquitto.log
```

**macOS (Homebrew):**
```bash
tail -f /usr/local/var/log/mosquitto.log
```

**Docker:**
```bash
docker logs -f mosquitto
```

### Run Mosquitto in Verbose Mode:
```bash
mosquitto -v
```

### Check if Process is Running:

**Linux/macOS:**
```bash
ps aux | grep mosquitto
```

**Windows:**
```bash
tasklist | findstr mosquitto
```

---

## 💡 Alternative: Use Redis Instead (Advanced)

If MQTT continues to cause issues, you can swap to Redis:

```bash
# Install Redis
pip install redis

# Run Redis
docker run -d -p 6379:6379 redis
```

Then modify your code to use Redis pub/sub instead of MQTT.

---

## 🎯 Recommended Solution Order

1. **Try Docker first** (easiest, works on all platforms)
   ```bash
   docker run -d -p 1883:1883 eclipse-mosquitto
   ```

2. **If no Docker, install Mosquitto natively**
   - Windows: Download installer
   - macOS: `brew install mosquitto`
   - Linux: `sudo apt-get install mosquitto`

3. **If still issues, run without MQTT**
   ```bash
   export USE_MQTT=false
   python app.py
   ```

4. **For testing only, use public broker**
   - Change broker to `test.mosquitto.org`

---

## ✅ Verify Everything Works

After fixing MQTT, run:

```bash
# Terminal 1: Start app
python app.py

# Terminal 2: Run test
python test_mqtt.py

# Terminal 3: Run simulator
python simulator.py
```

If all three work without errors, you're good to go! 🎉

---

## 🆘 Still Having Issues?

**Common Issues:**

1. **"No module named 'paho'"**
   ```bash
   pip install paho-mqtt
   ```

2. **"Connection refused [Errno 111]"**
   - Mosquitto is not running
   - Start it: `sudo systemctl start mosquitto`

3. **"Timeout"**
   - Check firewall
   - Try: `telnet localhost 1883`

4. **Port already in use**
   ```bash
   # Find what's using port 1883
   sudo lsof -i :1883  # Linux/macOS
   netstat -ano | findstr 1883  # Windows
   ```

---

**Choose the solution that works best for your system!**

Most users: **Docker is the easiest!** 🐳
