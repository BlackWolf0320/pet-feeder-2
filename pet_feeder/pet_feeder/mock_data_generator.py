"""
Mock Data Generator for AI Training
Creates realistic feeding data without any hardware
"""

import random
import numpy as np
from datetime import datetime, timedelta
from database import init_db, get_session, Pet, FeedingEvent, HealthMetric
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MockDataGenerator:
    """Generate realistic pet feeding data for AI training"""
    
    def __init__(self, db_session=None):
        if db_session is None:
            engine = init_db()
            self.session = get_session(engine)
        else:
            self.session = db_session
    
    def create_sample_pets(self):
        """Create sample pets with different characteristics"""
        pets_data = [
            {
                'rfid_tag': 'MOCK_001',
                'name': 'Max',
                'pet_type': 'dog',
                'breed': 'Golden Retriever',
                'weight': 25.5,
                'age': 36,
                'activity_level': 'high',
                'daily_food_target': 400
            },
            {
                'rfid_tag': 'MOCK_002',
                'name': 'Luna',
                'pet_type': 'cat',
                'breed': 'Siamese',
                'weight': 4.5,
                'age': 24,
                'activity_level': 'medium',
                'daily_food_target': 150
            },
            {
                'rfid_tag': 'MOCK_003',
                'name': 'Buddy',
                'pet_type': 'dog',
                'breed': 'Labrador',
                'weight': 30.0,
                'age': 48,
                'activity_level': 'medium',
                'daily_food_target': 450
            }
        ]
        
        created_pets = []
        for pet_data in pets_data:
            # Check if pet already exists
            existing = self.session.query(Pet).filter_by(rfid_tag=pet_data['rfid_tag']).first()
            if existing:
                logger.info(f"Pet {pet_data['name']} already exists (ID: {existing.id})")
                created_pets.append(existing)
            else:
                pet = Pet(**pet_data)
                self.session.add(pet)
                self.session.commit()
                logger.info(f"Created pet: {pet.name} (ID: {pet.id})")
                created_pets.append(pet)
        
        return created_pets
    
    def generate_feeding_pattern(self, pet, days=30):
        """
        Generate realistic feeding pattern for a pet
        
        Args:
            pet: Pet object
            days: Number of days to generate data for
        """
        logger.info(f"Generating {days} days of feeding data for {pet.name}")
        
        # Pet-specific feeding patterns
        if pet.pet_type == 'dog':
            feedings_per_day = random.choice([2, 3])  # Dogs typically eat 2-3 times
            typical_times = [8, 13, 19] if feedings_per_day == 3 else [8, 18]
            portion_size = pet.daily_food_target / feedings_per_day
        else:  # cat
            feedings_per_day = random.choice([3, 4])  # Cats eat more frequently
            typical_times = [7, 12, 17, 21][:feedings_per_day]
            portion_size = pet.daily_food_target / feedings_per_day
        
        start_date = datetime.utcnow() - timedelta(days=days)
        events_created = 0
        
        for day in range(days):
            current_date = start_date + timedelta(days=day)
            
            # Skip occasional days (pet might be away, sick, etc.)
            if random.random() < 0.05:  # 5% chance to skip a day
                continue
            
            daily_feedings = feedings_per_day
            
            # Occasional variations (more/fewer feedings)
            if random.random() < 0.1:  # 10% chance
                daily_feedings += random.choice([-1, 1])
                daily_feedings = max(1, min(5, daily_feedings))
            
            last_feeding_time = None
            
            for feeding_idx in range(daily_feedings):
                # Generate feeding time
                if feeding_idx < len(typical_times):
                    base_hour = typical_times[feeding_idx]
                else:
                    base_hour = random.randint(6, 22)
                
                # Add some randomness (±30 minutes)
                hour_variation = random.uniform(-0.5, 0.5)
                actual_hour = base_hour + hour_variation
                
                feeding_time = current_date.replace(
                    hour=int(actual_hour),
                    minute=int((actual_hour % 1) * 60),
                    second=random.randint(0, 59)
                )
                
                # Calculate time since last meal
                time_since_last = None
                if last_feeding_time:
                    time_diff = feeding_time - last_feeding_time
                    time_since_last = int(time_diff.total_seconds() / 60)
                
                # Generate feeding amounts with realistic variation
                amount_dispensed = portion_size * random.uniform(0.9, 1.1)
                
                # Completion rate varies (usually high, sometimes low)
                if random.random() < 0.8:  # 80% normal eating
                    completion_rate = random.uniform(85, 100)
                elif random.random() < 0.9:  # 10% reduced appetite
                    completion_rate = random.uniform(40, 70)
                else:  # 10% very low appetite (anomaly)
                    completion_rate = random.uniform(10, 40)
                
                amount_consumed = amount_dispensed * (completion_rate / 100.0)
                
                # Eating duration (typically 2-5 minutes)
                if completion_rate > 80:
                    # Normal eating speed
                    eating_duration = random.uniform(120, 300)
                elif completion_rate > 50:
                    # Slow eating
                    eating_duration = random.uniform(300, 600)
                else:
                    # Very slow or stopped eating
                    eating_duration = random.uniform(60, 200)
                
                # Add day-of-week patterns
                if current_date.weekday() in [5, 6]:  # Weekend
                    # Slightly different timing on weekends
                    feeding_time += timedelta(hours=random.uniform(-1, 1))
                
                # Create feeding event
                feeding_event = FeedingEvent(
                    pet_id=pet.id,
                    timestamp=feeding_time,
                    amount_dispensed=amount_dispensed,
                    amount_consumed=amount_consumed,
                    eating_duration=int(eating_duration),
                    time_since_last_meal=time_since_last,
                    completion_rate=completion_rate,
                    is_manual_dispense=False,
                    is_scheduled=True
                )
                
                self.session.add(feeding_event)
                last_feeding_time = feeding_time
                events_created += 1
        
        self.session.commit()
        logger.info(f"Created {events_created} feeding events for {pet.name}")
        return events_created
    
    def add_anomalies(self, pet, anomaly_rate=0.15):
        """
        Add some anomalous feeding events to make data more realistic
        
        Args:
            pet: Pet object
            anomaly_rate: Percentage of events to make anomalous
        """
        logger.info(f"Adding anomalies for {pet.name}")
        
        # Get recent feeding events
        events = self.session.query(FeedingEvent)\
            .filter_by(pet_id=pet.id)\
            .order_by(FeedingEvent.timestamp.desc())\
            .limit(100)\
            .all()
        
        if not events:
            logger.warning("No events to add anomalies to")
            return
        
        num_anomalies = int(len(events) * anomaly_rate)
        anomaly_events = random.sample(events, min(num_anomalies, len(events)))
        
        anomaly_types = ['reduced_appetite', 'slow_eating', 'fast_eating', 'unusual_time']
        
        for event in anomaly_events:
            anomaly_type = random.choice(anomaly_types)
            
            if anomaly_type == 'reduced_appetite':
                # Very low consumption
                event.amount_consumed = event.amount_dispensed * random.uniform(0.2, 0.4)
                event.completion_rate = (event.amount_consumed / event.amount_dispensed) * 100
                
            elif anomaly_type == 'slow_eating':
                # Very slow eating
                event.eating_duration = random.randint(600, 1200)  # 10-20 minutes
                
            elif anomaly_type == 'fast_eating':
                # Very fast eating
                event.eating_duration = random.randint(30, 60)  # 30-60 seconds
                
            elif anomaly_type == 'unusual_time':
                # Feeding at unusual hour
                unusual_hour = random.choice([2, 3, 4, 23])
                event.timestamp = event.timestamp.replace(hour=unusual_hour)
            
            event.anomaly_detected = True
            event.anomaly_type = anomaly_type
        
        self.session.commit()
        logger.info(f"Added {len(anomaly_events)} anomalies")
    
    def add_weekly_pattern_variation(self, pet):
        """Add weekly patterns (weekday vs weekend differences)"""
        events = self.session.query(FeedingEvent)\
            .filter_by(pet_id=pet.id)\
            .all()
        
        for event in events:
            if event.timestamp.weekday() in [5, 6]:  # Weekend
                # Slightly later feeding times on weekends
                if random.random() < 0.7:
                    event.timestamp += timedelta(hours=random.uniform(0.5, 2))
        
        self.session.commit()
    
    def generate_complete_dataset(self, pet, days=45, include_anomalies=True):
        """
        Generate complete realistic dataset for a pet
        
        Args:
            pet: Pet object
            days: Number of days of data (45 days = ~6 weeks, good for training)
            include_anomalies: Whether to include anomalous events
        """
        logger.info("=" * 60)
        logger.info(f"GENERATING COMPLETE DATASET FOR {pet.name}")
        logger.info("=" * 60)
        
        # Generate base feeding pattern
        num_events = self.generate_feeding_pattern(pet, days=days)
        
        # Add weekly variations
        self.add_weekly_pattern_variation(pet)
        
        # Add anomalies
        if include_anomalies:
            self.add_anomalies(pet, anomaly_rate=0.12)
        
        logger.info("=" * 60)
        logger.info("DATASET GENERATION COMPLETE")
        logger.info(f"Total events: {num_events}")
        logger.info(f"Period: {days} days")
        logger.info(f"Ready for AI training: {'✅ YES' if num_events >= 50 else '❌ NO'}")
        logger.info("=" * 60)
        
        return num_events
    
    def quick_generate_for_training(self, num_pets=1, days=45):
        """
        Quick method to generate everything needed for AI training
        
        Args:
            num_pets: Number of pets to create and generate data for
            days: Days of data per pet
            
        Returns:
            List of created pets
        """
        logger.info("🚀 QUICK DATASET GENERATION FOR AI TRAINING")
        logger.info("=" * 60)
        
        # Create sample pets
        all_pets = self.create_sample_pets()
        pets_to_use = all_pets[:num_pets]
        
        # Generate data for each pet
        for pet in pets_to_use:
            self.generate_complete_dataset(pet, days=days, include_anomalies=True)
        
        logger.info("\n" + "=" * 60)
        logger.info("✅ ALL DATA GENERATED!")
        logger.info("=" * 60)
        logger.info(f"Pets created: {len(pets_to_use)}")
        logger.info(f"Days per pet: {days}")
        logger.info("\nNext steps:")
        logger.info("1. Check data: python data_preparer.py")
        logger.info("2. Train AI: python ai_manager.py train 1")
        logger.info("3. Test predictions: python lstm_predictor.py predict 1")
        logger.info("=" * 60)
        
        return pets_to_use


def quick_start_ai_training():
    """Quick function to generate data and check readiness"""
    generator = MockDataGenerator()
    
    print("\n" + "🎯" * 30)
    print("QUICK START: AI TRAINING DATA GENERATION")
    print("🎯" * 30 + "\n")
    
    choice = input("Generate mock data? (y/n): ").lower()
    
    if choice != 'y':
        print("Cancelled.")
        return
    
    # Ask for preferences
    try:
        num_pets = int(input("Number of pets (1-3, default 1): ") or "1")
        num_pets = max(1, min(3, num_pets))
    except:
        num_pets = 1
    
    try:
        days = int(input("Days of data per pet (30-90, default 45): ") or "45")
        days = max(30, min(90, days))
    except:
        days = 45
    
    print(f"\nGenerating data for {num_pets} pet(s) with {days} days each...")
    print("This will take about 5-10 seconds...\n")
    
    # Generate data
    pets = generator.quick_generate_for_training(num_pets=num_pets, days=days)
    
    print("\n✅ Data generation complete!")
    print(f"\nCreated pets:")
    for pet in pets:
        print(f"  • {pet.name} (ID: {pet.id}) - {pet.pet_type}")
    
    print("\n" + "=" * 60)
    print("NOW YOU CAN TRAIN AI MODELS!")
    print("=" * 60)
    print("\nCommands to use:")
    for pet in pets:
        print(f"\nFor {pet.name} (ID: {pet.id}):")
        print(f"  python ai_manager.py train {pet.id}")
        print(f"  python lstm_predictor.py predict {pet.id}")
        print(f"  python anomaly_detector.py analyze {pet.id}")
        print(f"  python schedule_optimizer.py train {pet.id}")


def generate_specific_scenario(pet_id, scenario_type='normal', days=30):
    """
    Generate specific test scenarios
    
    Scenarios:
    - normal: Regular, healthy feeding pattern
    - declining: Gradually declining appetite (health issue simulation)
    - irregular: Very irregular feeding times
    - picky: Low completion rates (picky eater)
    """
    generator = MockDataGenerator()
    session = generator.session
    
    pet = session.query(Pet).get(pet_id)
    if not pet:
        logger.error(f"Pet {pet_id} not found")
        return
    
    logger.info(f"Generating '{scenario_type}' scenario for {pet.name}")
    
    if scenario_type == 'normal':
        generator.generate_complete_dataset(pet, days=days, include_anomalies=False)
    
    elif scenario_type == 'declining':
        # Generate data with gradually declining appetite
        start_date = datetime.utcnow() - timedelta(days=days)
        
        for day in range(days):
            current_date = start_date + timedelta(days=day)
            
            # Appetite declines over time
            appetite_factor = 1.0 - (day / days) * 0.5  # Decline to 50%
            
            for hour in [8, 18]:
                amount_dispensed = 150
                completion_rate = appetite_factor * random.uniform(80, 100)
                
                event = FeedingEvent(
                    pet_id=pet.id,
                    timestamp=current_date.replace(hour=hour),
                    amount_dispensed=amount_dispensed,
                    amount_consumed=amount_dispensed * (completion_rate / 100),
                    eating_duration=random.randint(120, 300),
                    completion_rate=completion_rate,
                    is_scheduled=True
                )
                session.add(event)
        
        session.commit()
        logger.info(f"Created declining appetite scenario ({days} days)")
    
    elif scenario_type == 'irregular':
        # Very irregular feeding times
        start_date = datetime.utcnow() - timedelta(days=days)
        
        for day in range(days):
            current_date = start_date + timedelta(days=day)
            
            # Random number of feedings per day
            num_feedings = random.randint(1, 4)
            
            for _ in range(num_feedings):
                hour = random.randint(0, 23)
                
                event = FeedingEvent(
                    pet_id=pet.id,
                    timestamp=current_date.replace(hour=hour, minute=random.randint(0, 59)),
                    amount_dispensed=random.uniform(100, 200),
                    amount_consumed=random.uniform(80, 180),
                    eating_duration=random.randint(120, 300),
                    completion_rate=random.uniform(70, 100),
                    is_scheduled=True
                )
                session.add(event)
        
        session.commit()
        logger.info(f"Created irregular feeding scenario ({days} days)")
    
    elif scenario_type == 'picky':
        # Picky eater - low completion rates
        start_date = datetime.utcnow() - timedelta(days=days)
        
        for day in range(days):
            current_date = start_date + timedelta(days=day)
            
            for hour in [8, 18]:
                amount_dispensed = 150
                completion_rate = random.uniform(40, 70)  # Consistently low
                
                event = FeedingEvent(
                    pet_id=pet.id,
                    timestamp=current_date.replace(hour=hour),
                    amount_dispensed=amount_dispensed,
                    amount_consumed=amount_dispensed * (completion_rate / 100),
                    eating_duration=random.randint(180, 400),  # Slow eating
                    completion_rate=completion_rate,
                    is_scheduled=True
                )
                session.add(event)
        
        session.commit()
        logger.info(f"Created picky eater scenario ({days} days)")


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == 'quick':
            # Quick generation for training
            generator = MockDataGenerator()
            num_pets = int(sys.argv[2]) if len(sys.argv) > 2 else 1
            days = int(sys.argv[3]) if len(sys.argv) > 3 else 45
            generator.quick_generate_for_training(num_pets=num_pets, days=days)
        
        elif sys.argv[1] == 'scenario':
            # Generate specific scenario
            if len(sys.argv) < 4:
                print("Usage: python mock_data_generator.py scenario <pet_id> <scenario_type> [days]")
                print("Scenarios: normal, declining, irregular, picky")
            else:
                pet_id = int(sys.argv[2])
                scenario = sys.argv[3]
                days = int(sys.argv[4]) if len(sys.argv) > 4 else 30
                generate_specific_scenario(pet_id, scenario, days)
    
    else:
        # Interactive mode
        quick_start_ai_training()
