"""
IoT Device Simulator
Simulates RFID readings, weight sensors, and dispenser actions
Use this to test the system without actual hardware
"""

import paho.mqtt.client as mqtt
import json
import time
import random
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DeviceSimulator:
    """Simulate IoT device behavior"""
    
    def __init__(self, broker_address="localhost", port=1883):
        self.client = mqtt.Client("device_simulator")
        self.broker_address = broker_address
        self.port = port
        self.is_connected = False
        
        # Simulated device state
        self.food_container_level = 100  # percentage
        self.bowl_weight = 0  # grams
        self.is_dispensing = False
        
        # Setup callbacks
        self.client.on_connect = self.on_connect
        self.client.on_message = self.on_message
        
    def on_connect(self, client, userdata, flags, rc):
        if rc == 0:
            logger.info("Simulator connected to MQTT broker")
            self.is_connected = True
            # Subscribe to command topic
            client.subscribe("feeder/dispenser/command")
        else:
            logger.error(f"Connection failed with code {rc}")
    
    def on_message(self, client, userdata, msg):
        """Handle incoming commands from server"""
        try:
            if msg.topic == "feeder/dispenser/command":
                command = json.loads(msg.payload.decode())
                if command.get('command') == 'dispense':
                    amount = command.get('amount', 100)
                    logger.info(f"Received dispense command: {amount}g")
                    self.simulate_dispensing(amount)
        except Exception as e:
            logger.error(f"Error handling command: {e}")
    
    def connect(self):
        """Connect to MQTT broker"""
        try:
            self.client.connect(self.broker_address, self.port, 60)
            self.client.loop_start()
            logger.info("Simulator MQTT client started")
        except Exception as e:
            logger.error(f"Failed to connect: {e}")
    
    def disconnect(self):
        """Disconnect from broker"""
        self.client.loop_stop()
        self.client.disconnect()
        logger.info("Simulator disconnected")
    
    def simulate_rfid_detection(self, rfid_tag):
        """Simulate RFID tag detection"""
        payload = {
            'rfid_tag': rfid_tag,
            'timestamp': datetime.utcnow().isoformat(),
            'signal_strength': random.randint(-60, -40)
        }
        
        self.client.publish('feeder/rfid/detected', json.dumps(payload))
        logger.info(f"Simulated RFID detection: {rfid_tag}")
        return payload
    
    def simulate_dispensing(self, amount):
        """Simulate food dispensing process"""
        logger.info(f"Starting to dispense {amount}g")
        self.is_dispensing = True
        
        # Publish dispensing start
        start_time = datetime.utcnow()
        payload = {
            'status': 'dispensing',
            'start_time': start_time.isoformat(),
            'timestamp': start_time.isoformat()
        }
        self.client.publish('feeder/dispenser/status', json.dumps(payload))
        
        # Simulate dispensing time (about 5 seconds)
        time.sleep(2)
        
        # Update bowl weight
        self.bowl_weight += amount
        
        # Publish dispensing complete
        end_time = datetime.utcnow()
        payload = {
            'status': 'completed',
            'amount_dispensed': amount,
            'start_time': start_time.isoformat(),
            'end_time': end_time.isoformat(),
            'timestamp': end_time.isoformat()
        }
        self.client.publish('feeder/dispenser/status', json.dumps(payload))
        
        # Update food level
        self.food_container_level -= (amount / 5000) * 100  # Assuming 5kg capacity
        
        # Publish weight sensor reading
        self.publish_weight_reading()
        
        self.is_dispensing = False
        logger.info(f"Dispensing complete. Bowl now has {self.bowl_weight}g")
    
    def simulate_eating(self, duration_seconds, completion_rate=0.9):
        """Simulate pet eating from bowl"""
        logger.info(f"Simulating eating for {duration_seconds}s")
        
        initial_weight = self.bowl_weight
        target_consumption = initial_weight * completion_rate
        
        # Simulate gradual eating
        steps = 10
        for i in range(steps):
            time.sleep(duration_seconds / steps)
            
            # Gradually reduce weight
            progress = (i + 1) / steps
            self.bowl_weight = initial_weight - (target_consumption * progress)
            
            # Publish weight readings
            self.publish_weight_reading(stable=False)
        
        # Final stable reading
        time.sleep(1)
        self.publish_weight_reading(stable=True)
        
        logger.info(f"Eating simulation complete. Remaining: {self.bowl_weight:.1f}g")
    
    def publish_weight_reading(self, stable=True):
        """Publish weight sensor reading"""
        payload = {
            'weight': round(self.bowl_weight, 1),
            'stable': stable,
            'timestamp': datetime.utcnow().isoformat()
        }
        
        self.client.publish('feeder/sensor/weight', json.dumps(payload))
    
    def publish_device_status(self):
        """Publish overall device status"""
        payload = {
            'food_level': round(self.food_container_level, 1),
            'dispenser_status': 'operational',
            'rfid_status': 'operational',
            'weight_sensor_status': 'operational',
            'temperature': round(random.uniform(20, 25), 1),
            'humidity': round(random.uniform(40, 60), 1),
            'wifi_signal': random.randint(-70, -50),
            'timestamp': datetime.utcnow().isoformat()
        }
        
        self.client.publish('feeder/device/status', json.dumps(payload))
        logger.info(f"Device status published. Food level: {self.food_container_level:.1f}%")
    
    def reset_bowl(self):
        """Reset bowl weight (simulate cleaning)"""
        self.bowl_weight = 0
        self.publish_weight_reading(stable=True)
        logger.info("Bowl reset to 0g")

# ==================== TEST SCENARIOS ====================

def scenario_normal_feeding(simulator, pet_rfid):
    """Simulate a normal feeding session"""
    logger.info("\n=== Starting Normal Feeding Scenario ===")
    
    # Pet approaches
    simulator.simulate_rfid_detection(pet_rfid)
    time.sleep(2)
    
    # System dispenses food (this would normally be triggered by server)
    # For testing, we manually trigger it
    simulator.simulate_dispensing(150)
    time.sleep(1)
    
    # Pet eats normally (takes about 3 minutes, consumes 90%)
    simulator.simulate_eating(duration_seconds=180, completion_rate=0.9)
    
    # Clean up
    time.sleep(5)
    simulator.reset_bowl()

def scenario_reduced_appetite(simulator, pet_rfid):
    """Simulate reduced appetite - should trigger alert"""
    logger.info("\n=== Starting Reduced Appetite Scenario ===")
    
    simulator.simulate_rfid_detection(pet_rfid)
    time.sleep(2)
    
    simulator.simulate_dispensing(150)
    time.sleep(1)
    
    # Pet only eats 40% - should trigger anomaly
    simulator.simulate_eating(duration_seconds=60, completion_rate=0.4)
    
    time.sleep(5)
    simulator.reset_bowl()

def scenario_slow_eating(simulator, pet_rfid):
    """Simulate slow eating - should trigger alert"""
    logger.info("\n=== Starting Slow Eating Scenario ===")
    
    simulator.simulate_rfid_detection(pet_rfid)
    time.sleep(2)
    
    simulator.simulate_dispensing(150)
    time.sleep(1)
    
    # Pet eats normally but very slowly (10 minutes)
    simulator.simulate_eating(duration_seconds=600, completion_rate=0.85)
    
    time.sleep(5)
    simulator.reset_bowl()

def scenario_multiple_pets(simulator):
    """Simulate multiple pets feeding at different times"""
    logger.info("\n=== Starting Multiple Pets Scenario ===")
    
    pets = [
        ('RFID_12345', 150, 180),  # Pet 1: 150g, 3 min
        ('RFID_67890', 80, 120),   # Pet 2: 80g, 2 min
    ]
    
    for rfid, amount, duration in pets:
        simulator.simulate_rfid_detection(rfid)
        time.sleep(2)
        simulator.simulate_dispensing(amount)
        time.sleep(1)
        simulator.simulate_eating(duration, 0.9)
        time.sleep(5)
        simulator.reset_bowl()
        
        # Wait before next pet
        time.sleep(30)

def continuous_monitoring(simulator):
    """Publish device status periodically"""
    logger.info("\n=== Starting Continuous Monitoring ===")
    
    while True:
        simulator.publish_device_status()
        time.sleep(60)  # Every minute

# ==================== MAIN ====================

if __name__ == '__main__':
    print("""
    ╔═══════════════════════════════════════════════════════╗
    ║        Pet Feeder IoT Device Simulator               ║
    ║                                                       ║
    ║  Use this to test the system without hardware        ║
    ╚═══════════════════════════════════════════════════════╝
    
    Available test scenarios:
    1. Normal feeding
    2. Reduced appetite (triggers alert)
    3. Slow eating (triggers alert)
    4. Multiple pets
    5. Continuous monitoring
    6. Custom scenario
    """)
    
    # Initialize simulator
    simulator = DeviceSimulator()
    simulator.connect()
    
    # Wait for connection
    time.sleep(2)
    
    if not simulator.is_connected:
        logger.error("Failed to connect to MQTT broker. Is it running?")
        exit(1)
    
    try:
        choice = input("\nSelect scenario (1-6): ").strip()
        
        if choice == '1':
            scenario_normal_feeding(simulator, 'RFID_12345')
        elif choice == '2':
            scenario_reduced_appetite(simulator, 'RFID_12345')
        elif choice == '3':
            scenario_slow_eating(simulator, 'RFID_12345')
        elif choice == '4':
            scenario_multiple_pets(simulator)
        elif choice == '5':
            continuous_monitoring(simulator)
        elif choice == '6':
            # Custom interactive mode
            logger.info("Custom mode - send commands manually")
            while True:
                cmd = input("\nCommand (rfid/dispense/eat/status/quit): ").strip().lower()
                
                if cmd == 'quit':
                    break
                elif cmd == 'rfid':
                    tag = input("RFID tag: ").strip()
                    simulator.simulate_rfid_detection(tag)
                elif cmd == 'dispense':
                    amount = float(input("Amount (g): "))
                    simulator.simulate_dispensing(amount)
                elif cmd == 'eat':
                    duration = int(input("Duration (seconds): "))
                    rate = float(input("Completion rate (0-1): "))
                    simulator.simulate_eating(duration, rate)
                elif cmd == 'status':
                    simulator.publish_device_status()
                else:
                    print("Unknown command")
        
        logger.info("\nSimulation complete!")
        
    except KeyboardInterrupt:
        logger.info("\nStopping simulator...")
    finally:
        simulator.disconnect()
