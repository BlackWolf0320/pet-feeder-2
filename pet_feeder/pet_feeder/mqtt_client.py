import paho.mqtt.client as mqtt
import json
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PetFeederMQTTClient:
    """MQTT Client for IoT device communication"""
    
    def __init__(self, broker_address="localhost", port=1883, client_id="pet_feeder_server"):
        self.broker_address = broker_address
        self.port = port
        self.client = mqtt.Client(client_id)
        self.data_handler = None
        
        # MQTT Topics
        self.TOPICS = {
            'rfid_detected': 'feeder/rfid/detected',
            'weight_sensor': 'feeder/sensor/weight',
            'dispenser_status': 'feeder/dispenser/status',
            'dispense_command': 'feeder/dispenser/command',
            'device_status': 'feeder/device/status',
            'food_level': 'feeder/food/level'
        }
        
        # Setup callbacks
        self.client.on_connect = self.on_connect
        self.client.on_message = self.on_message
        self.client.on_disconnect = self.on_disconnect
        
    def on_connect(self, client, userdata, flags, rc):
        """Callback for when client connects to broker"""
        if rc == 0:
            logger.info("Connected to MQTT Broker successfully")
            # Subscribe to all relevant topics
            for topic_name, topic in self.TOPICS.items():
                if topic_name != 'dispense_command':  # Don't subscribe to command topics
                    client.subscribe(topic)
                    logger.info(f"Subscribed to topic: {topic}")
        else:
            logger.error(f"Failed to connect, return code {rc}")
    
    def on_disconnect(self, client, userdata, rc):
        """Callback for when client disconnects"""
        if rc != 0:
            logger.warning("Unexpected disconnection from MQTT Broker")
    
    def on_message(self, client, userdata, msg):
        """Callback for when a message is received"""
        try:
            topic = msg.topic
            payload = json.loads(msg.payload.decode())
            logger.info(f"Received message on topic '{topic}': {payload}")
            
            # Route message to appropriate handler
            if self.data_handler:
                self.data_handler(topic, payload)
            
        except json.JSONDecodeError:
            logger.error(f"Failed to decode JSON from topic {msg.topic}")
        except Exception as e:
            logger.error(f"Error processing message: {e}")
    
    def set_data_handler(self, handler_function):
        """Set the function to handle incoming data"""
        self.data_handler = handler_function
    
    def connect(self):
        """Connect to MQTT broker"""
        try:
            self.client.connect(self.broker_address, self.port, 60)
            self.client.loop_start()
            logger.info("MQTT client started")
        except Exception as e:
            logger.error(f"Failed to connect to MQTT broker: {e}")
    
    def disconnect(self):
        """Disconnect from MQTT broker"""
        self.client.loop_stop()
        self.client.disconnect()
        logger.info("MQTT client disconnected")
    
    def publish_dispense_command(self, pet_id, amount):
        """Send command to dispense food"""
        command = {
            'pet_id': pet_id,
            'amount': amount,
            'timestamp': datetime.utcnow().isoformat(),
            'command': 'dispense'
        }
        self.client.publish(self.TOPICS['dispense_command'], json.dumps(command))
        logger.info(f"Sent dispense command: {amount}g for pet {pet_id}")
        return command
    
    def request_device_status(self):
        """Request current device status"""
        command = {
            'command': 'status_request',
            'timestamp': datetime.utcnow().isoformat()
        }
        self.client.publish(self.TOPICS['device_status'], json.dumps(command))
        logger.info("Requested device status")

# Example payload structures for reference
EXAMPLE_PAYLOADS = {
    'rfid_detected': {
        'rfid_tag': 'RFID_12345',
        'timestamp': '2025-11-20T10:30:00',
        'signal_strength': -45
    },
    'weight_sensor': {
        'weight': 145.5,  # grams in food bowl
        'stable': True,
        'timestamp': '2025-11-20T10:30:00'
    },
    'dispenser_status': {
        'status': 'dispensing',  # idle, dispensing, jammed, error
        'amount_dispensed': 150,
        'start_time': '2025-11-20T10:30:00',
        'end_time': '2025-11-20T10:30:05'
    },
    'device_status': {
        'food_level': 75.5,  # percentage
        'dispenser_status': 'operational',
        'rfid_status': 'operational',
        'weight_sensor_status': 'operational',
        'temperature': 22.5,
        'humidity': 45.0,
        'wifi_signal': -65,
        'timestamp': '2025-11-20T10:30:00'
    },
    'food_level': {
        'percentage': 75.5,
        'weight_grams': 2500,
        'timestamp': '2025-11-20T10:30:00'
    }
}
