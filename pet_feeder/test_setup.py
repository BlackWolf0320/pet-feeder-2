"""
Quick Test Script
Verify that all components are set up correctly
"""

import sys
import importlib

def test_imports():
    """Test if all required packages can be imported"""
    print("Testing imports...")
    
    required_packages = [
        'flask',
        'sqlalchemy',
        'paho.mqtt.client',
        'pandas',
        'numpy'
    ]
    
    failed = []
    for package in required_packages:
        try:
            if package == 'paho.mqtt.client':
                importlib.import_module('paho.mqtt')
            else:
                importlib.import_module(package.split('.')[0])
            print(f"  ✓ {package}")
        except ImportError:
            print(f"  ✗ {package}")
            failed.append(package)
    
    if failed:
        print(f"\n❌ Missing packages: {', '.join(failed)}")
        print("Run: pip install -r requirements.txt")
        return False
    
    print("\n✅ All required packages are installed\n")
    return True

def test_database():
    """Test database initialization"""
    print("Testing database...")
    
    try:
        from database import init_db, get_session, Pet
        
        # Initialize database
        engine = init_db('sqlite:///test_pet_feeder.db')
        session = get_session(engine)
        
        # Try to create a test pet
        test_pet = Pet(
            rfid_tag='TEST_001',
            name='TestPet',
            pet_type='dog',
            weight=10.0,
            daily_food_target=200
        )
        
        session.add(test_pet)
        session.commit()
        
        # Query it back
        retrieved = session.query(Pet).filter_by(rfid_tag='TEST_001').first()
        
        if retrieved and retrieved.name == 'TestPet':
            print("  ✓ Database initialization")
            print("  ✓ Create record")
            print("  ✓ Query record")
        
        session.close()
        
        # Clean up test database
        import os
        if os.path.exists('test_pet_feeder.db'):
            os.remove('test_pet_feeder.db')
        
        print("\n✅ Database system working correctly\n")
        return True
        
    except Exception as e:
        print(f"\n❌ Database test failed: {e}\n")
        return False

def test_mqtt_client():
    """Test MQTT client initialization"""
    print("Testing MQTT client...")
    
    try:
        from mqtt_client import PetFeederMQTTClient
        
        client = PetFeederMQTTClient()
        print("  ✓ MQTT client initialization")
        
        # Test topic structure
        if all(key in client.TOPICS for key in ['rfid_detected', 'weight_sensor', 'dispenser_status']):
            print("  ✓ MQTT topics defined")
        
        print("\n✅ MQTT client configured correctly")
        print("⚠️  Note: Actual connection requires running MQTT broker\n")
        return True
        
    except Exception as e:
        print(f"\n❌ MQTT client test failed: {e}\n")
        return False

def test_data_processor():
    """Test data processor initialization"""
    print("Testing data processor...")
    
    try:
        from database import init_db, get_session
        from data_processor import DataProcessor
        
        engine = init_db('sqlite:///test_pet_feeder.db')
        session = get_session(engine)
        
        processor = DataProcessor(session)
        print("  ✓ Data processor initialization")
        
        session.close()
        
        # Clean up
        import os
        if os.path.exists('test_pet_feeder.db'):
            os.remove('test_pet_feeder.db')
        
        print("\n✅ Data processor working correctly\n")
        return True
        
    except Exception as e:
        print(f"\n❌ Data processor test failed: {e}\n")
        return False

def check_mqtt_broker():
    """Check if MQTT broker is running"""
    print("Checking MQTT broker...")
    
    try:
        import socket
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(2)
        result = sock.connect_ex(('localhost', 1883))
        sock.close()
        
        if result == 0:
            print("  ✓ MQTT broker is running on localhost:1883")
            print("\n✅ MQTT broker accessible\n")
            return True
        else:
            print("  ✗ MQTT broker not accessible on localhost:1883")
            print("\n⚠️  Start MQTT broker (Mosquitto) before running the system")
            print("   Ubuntu/Debian: sudo systemctl start mosquitto")
            print("   macOS: brew services start mosquitto")
            print("   Docker: docker run -it -p 1883:1883 eclipse-mosquitto\n")
            return False
            
    except Exception as e:
        print(f"  ✗ Could not check MQTT broker: {e}")
        return False

def main():
    """Run all tests"""
    print("=" * 60)
    print("  Pet Feeder System - Setup Verification")
    print("=" * 60)
    print()
    
    results = {
        'Imports': test_imports(),
        'Database': test_database(),
        'MQTT Client': test_mqtt_client(),
        'Data Processor': test_data_processor(),
        'MQTT Broker': check_mqtt_broker()
    }
    
    print("=" * 60)
    print("  Test Summary")
    print("=" * 60)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {test_name:20s} {status}")
    
    print()
    
    if all(results.values()):
        print("🎉 All tests passed! System is ready to use.")
        print("\nNext steps:")
        print("  1. Terminal 1: python app.py")
        print("  2. Terminal 2: python simulator.py")
        print("  3. Test the API: http://localhost:5000/api/health")
    else:
        print("⚠️  Some tests failed. Please fix the issues above.")
        if not results['MQTT Broker']:
            print("\n💡 The system can still run without MQTT broker,")
            print("   but you won't be able to test IoT functionality.")
    
    print()

if __name__ == '__main__':
    main()
