from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, ForeignKey, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from datetime import datetime
import os

Base = declarative_base()

class Pet(Base):
    """Pet information table"""
    __tablename__ = 'pets'
    
    id = Column(Integer, primary_key=True)
    rfid_tag = Column(String(50), unique=True, nullable=False)
    name = Column(String(100), nullable=False)
    pet_type = Column(String(50))  # dog, cat, etc
    breed = Column(String(100))
    weight = Column(Float)  # in kg
    age = Column(Integer)  # in months
    activity_level = Column(String(20))  # low, medium, high
    health_conditions = Column(String(500))
    daily_food_target = Column(Float)  # in grams
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    feeding_events = relationship("FeedingEvent", back_populates="pet")
    health_metrics = relationship("HealthMetric", back_populates="pet")

class FeedingEvent(Base):
    """Individual feeding event records"""
    __tablename__ = 'feeding_events'
    
    id = Column(Integer, primary_key=True)
    pet_id = Column(Integer, ForeignKey('pets.id'), nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    # Dispensing data
    amount_dispensed = Column(Float)  # grams
    dispense_start_time = Column(DateTime)
    dispense_end_time = Column(DateTime)
    
    # Consumption data
    amount_consumed = Column(Float)  # grams
    eating_start_time = Column(DateTime)
    eating_end_time = Column(DateTime)
    eating_duration = Column(Integer)  # seconds
    
    # Behavioral data
    time_since_last_meal = Column(Integer)  # minutes
    approach_count = Column(Integer)  # how many times pet approached
    hesitation_time = Column(Integer)  # seconds before starting to eat
    
    # Status
    is_manual_dispense = Column(Boolean, default=False)
    is_scheduled = Column(Boolean, default=True)
    completion_rate = Column(Float)  # percentage of food consumed
    
    # AI will use this later
    anomaly_detected = Column(Boolean, default=False)
    anomaly_type = Column(String(100))
    
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    pet = relationship("Pet", back_populates="feeding_events")

class HealthMetric(Base):
    """Pet health and behavior metrics over time"""
    __tablename__ = 'health_metrics'
    
    id = Column(Integer, primary_key=True)
    pet_id = Column(Integer, ForeignKey('pets.id'), nullable=False)
    date = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    # Daily aggregated metrics
    total_food_consumed = Column(Float)  # grams
    feeding_frequency = Column(Integer)  # times per day
    average_eating_duration = Column(Float)  # seconds
    average_time_between_meals = Column(Float)  # minutes
    
    # Behavioral patterns (to be calculated)
    eating_consistency_score = Column(Float)  # 0-100
    appetite_score = Column(Float)  # 0-100
    
    # Weight tracking
    weight = Column(Float)  # kg
    weight_change = Column(Float)  # kg difference from last week
    
    # AI predictions placeholder
    predicted_next_feeding_time = Column(DateTime)
    predicted_food_amount = Column(Float)
    behavior_risk_score = Column(Float)  # 0-100, AI calculated
    
    notes = Column(String(500))
    
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    pet = relationship("Pet", back_populates="health_metrics")

class DeviceStatus(Base):
    """IoT device status and sensor readings"""
    __tablename__ = 'device_status'
    
    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    # Food container status
    food_level = Column(Float)  # percentage
    food_weight_remaining = Column(Float)  # grams
    
    # Device health
    dispenser_status = Column(String(20))  # operational, jammed, error
    rfid_reader_status = Column(String(20))
    weight_sensor_status = Column(String(20))
    
    # Environmental
    temperature = Column(Float)  # celsius
    humidity = Column(Float)  # percentage
    
    # System
    wifi_signal_strength = Column(Integer)  # dBm
    last_maintenance = Column(DateTime)
    
    error_message = Column(String(500))

class Alert(Base):
    """System alerts and notifications"""
    __tablename__ = 'alerts'
    
    id = Column(Integer, primary_key=True)
    pet_id = Column(Integer, ForeignKey('pets.id'))
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    alert_type = Column(String(50), nullable=False)  # behavior, health, device, food_low
    severity = Column(String(20))  # low, medium, high, critical
    title = Column(String(200))
    message = Column(String(1000))
    
    # Status
    is_read = Column(Boolean, default=False)
    is_resolved = Column(Boolean, default=False)
    resolved_at = Column(DateTime)
    
    # AI generated
    is_ai_generated = Column(Boolean, default=False)
    confidence_score = Column(Float)  # AI confidence 0-1

class FeedingSchedule(Base):
    """Scheduled feeding times (can be AI-optimized later)"""
    __tablename__ = 'feeding_schedules'
    
    id = Column(Integer, primary_key=True)
    pet_id = Column(Integer, ForeignKey('pets.id'), nullable=False)
    
    scheduled_time = Column(String(5))  # HH:MM format
    food_amount = Column(Float)  # grams
    is_active = Column(Boolean, default=True)
    
    # AI optimization flags
    is_ai_optimized = Column(Boolean, default=False)
    optimization_date = Column(DateTime)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

# Database setup
def init_db(database_url='sqlite:///pet_feeder.db'):
    """Initialize database and create tables"""
    engine = create_engine(database_url, echo=True)
    Base.metadata.create_all(engine)
    return engine

def get_session(engine):
    """Get database session"""
    Session = sessionmaker(bind=engine)
    return Session()
