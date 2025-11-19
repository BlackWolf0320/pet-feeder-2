"""
Reinforcement Learning Schedule Optimizer
Uses Q-Learning to optimize feeding schedules
"""

import numpy as np
import pandas as pd
import pickle
import logging
from datetime import datetime, timedelta
from collections import defaultdict

from ai_config import RL_CONFIG, RL_MODEL_PATH
from data_preparer import DataPreparer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ScheduleOptimizer:
    """Q-Learning agent for optimizing feeding schedules"""
    
    def __init__(self):
        self.q_table = defaultdict(lambda: defaultdict(float))
        self.data_preparer = DataPreparer()
        self.epsilon = RL_CONFIG['exploration_rate']
        self.training_history = []
        
    def get_state(self, current_hour, hunger_level, last_feeding_hours_ago):
        """
        Get current state representation
        
        Args:
            current_hour: Current hour of day (0-23)
            hunger_level: Estimated hunger (0-4)
            last_feeding_hours_ago: Hours since last feeding
            
        Returns:
            Tuple representing state
        """
        # Discretize continuous values
        hour_slot = current_hour
        hunger = min(4, max(0, hunger_level))
        
        # Discretize time since last feeding
        if last_feeding_hours_ago < 2:
            time_category = 0  # Very recent
        elif last_feeding_hours_ago < 4:
            time_category = 1  # Recent
        elif last_feeding_hours_ago < 8:
            time_category = 2  # Normal
        elif last_feeding_hours_ago < 12:
            time_category = 3  # Long time
        else:
            time_category = 4  # Very long time
        
        return (hour_slot, hunger, time_category)
    
    def estimate_hunger(self, hours_since_last_meal, typical_interval):
        """
        Estimate hunger level based on time since last meal
        
        Args:
            hours_since_last_meal: Hours since last feeding
            typical_interval: Typical feeding interval for this pet
            
        Returns:
            Hunger level (0-4)
        """
        if hours_since_last_meal < typical_interval * 0.5:
            return 0  # Not hungry
        elif hours_since_last_meal < typical_interval * 0.8:
            return 1  # Slightly hungry
        elif hours_since_last_meal < typical_interval * 1.2:
            return 2  # Moderately hungry
        elif hours_since_last_meal < typical_interval * 1.5:
            return 3  # Very hungry
        else:
            return 4  # Extremely hungry
    
    def calculate_reward(self, action, completion_rate, eating_duration, 
                         hours_since_last, daily_target_met, amount_dispensed):
        """
        Calculate reward for taken action
        
        Args:
            action: Action taken
            completion_rate: Percentage of food consumed
            eating_duration: Time taken to eat (seconds)
            hours_since_last: Hours since previous feeding
            daily_target_met: Whether daily food target is met
            amount_dispensed: Amount of food dispensed
            
        Returns:
            Reward value
        """
        reward = 0
        
        # Reward for high completion rate
        if completion_rate > 80:
            reward += RL_CONFIG['reward_completion_rate']
        elif completion_rate < 50:
            reward += RL_CONFIG['penalty_waste']
        
        # Reward for healthy eating duration (not too fast, not too slow)
        # Assuming healthy is 2-5 minutes (120-300 seconds)
        if 120 <= eating_duration <= 300:
            reward += RL_CONFIG['reward_healthy_duration']
        
        # Reward for meeting daily target
        if daily_target_met:
            reward += RL_CONFIG['reward_daily_target']
        
        # Penalty for feeding too frequently
        if hours_since_last < 2 and action != 'wait':
            reward += RL_CONFIG['penalty_too_frequent']
        
        # Penalty for feeding too infrequently
        if hours_since_last > 12:
            reward += RL_CONFIG['penalty_too_infrequent']
        
        # Small reward for waiting when not very hungry
        if action == 'wait' and hours_since_last < 4:
            reward += 0.1
        
        return reward
    
    def choose_action(self, state, training=True):
        """
        Choose action using epsilon-greedy strategy
        
        Args:
            state: Current state
            training: Whether in training mode (use exploration)
            
        Returns:
            Selected action
        """
        actions = RL_CONFIG['actions']
        
        # Exploration vs exploitation
        if training and np.random.random() < self.epsilon:
            # Explore: random action
            action = np.random.choice(actions)
        else:
            # Exploit: best known action
            q_values = [self.q_table[state][action] for action in actions]
            max_q = max(q_values)
            
            # If multiple actions have same Q-value, choose randomly among them
            best_actions = [actions[i] for i, q in enumerate(q_values) if q == max_q]
            action = np.random.choice(best_actions)
        
        return action
    
    def update_q_value(self, state, action, reward, next_state):
        """
        Update Q-value using Q-learning formula
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
        """
        # Get current Q-value
        current_q = self.q_table[state][action]
        
        # Get max Q-value for next state
        next_max_q = max([self.q_table[next_state][a] for a in RL_CONFIG['actions']])
        
        # Q-learning update
        new_q = current_q + RL_CONFIG['learning_rate'] * (
            reward + RL_CONFIG['discount_factor'] * next_max_q - current_q
        )
        
        self.q_table[state][action] = new_q
    
    def train_from_historical_data(self, pet_id, episodes=None):
        """
        Train Q-learning agent on historical feeding data
        
        Args:
            pet_id: Pet to train for
            episodes: Number of training episodes (None = use config)
            
        Returns:
            Training history
        """
        if episodes is None:
            episodes = RL_CONFIG['episodes']
        
        logger.info(f"Training schedule optimizer for pet {pet_id}")
        
        # Get historical data
        df = self.data_preparer.extract_feeding_data(pet_id=pet_id, min_samples=30)
        
        if df is None or len(df) < 30:
            logger.error("Not enough historical data for training")
            return None
        
        # Calculate typical feeding interval
        typical_interval = df['time_since_last_meal'].median() / 60.0  # Convert to hours
        daily_target = df['amount_consumed'].sum() / (
            (df['timestamp'].max() - df['timestamp'].min()).days + 1
        )
        
        logger.info(f"Typical interval: {typical_interval:.1f} hours")
        logger.info(f"Daily target: {daily_target:.1f}g")
        
        # Training loop
        for episode in range(episodes):
            episode_reward = 0
            
            # Simulate a day
            current_hour = 0
            hours_since_last = typical_interval  # Start as if typical interval passed
            daily_consumption = 0
            
            for step in range(24):  # 24 hours in a day
                # Get current state
                hunger = self.estimate_hunger(hours_since_last, typical_interval)
                state = self.get_state(current_hour, hunger, hours_since_last)
                
                # Choose action
                action = self.choose_action(state, training=True)
                
                # Simulate action result (using historical data statistics)
                if action == 'wait':
                    amount = 0
                    completion_rate = 0
                    eating_duration = 0
                else:
                    amount = RL_CONFIG['action_amounts'][action]
                    # Estimate completion rate based on hunger
                    completion_rate = min(100, 60 + hunger * 10 + np.random.normal(0, 10))
                    eating_duration = np.random.normal(180, 30)  # ~3 minutes average
                
                consumed = amount * (completion_rate / 100.0)
                daily_consumption += consumed
                
                # Calculate reward
                daily_target_met = daily_consumption >= daily_target
                reward = self.calculate_reward(
                    action, completion_rate, eating_duration,
                    hours_since_last, daily_target_met, amount
                )
                
                episode_reward += reward
                
                # Next state
                if action != 'wait':
                    next_hours_since_last = 1  # Reset to 1 hour
                else:
                    next_hours_since_last = hours_since_last + 1
                
                next_hour = (current_hour + 1) % 24
                next_hunger = self.estimate_hunger(next_hours_since_last, typical_interval)
                next_state = self.get_state(next_hour, next_hunger, next_hours_since_last)
                
                # Update Q-table
                self.update_q_value(state, action, reward, next_state)
                
                # Move to next state
                current_hour = next_hour
                hours_since_last = next_hours_since_last
            
            # Decay epsilon
            self.epsilon = max(
                RL_CONFIG['min_exploration_rate'],
                self.epsilon * RL_CONFIG['exploration_decay']
            )
            
            self.training_history.append({
                'episode': episode,
                'reward': episode_reward,
                'epsilon': self.epsilon
            })
            
            # Log progress
            if (episode + 1) % 100 == 0:
                avg_reward = np.mean([h['reward'] for h in self.training_history[-100:]])
                logger.info(f"Episode {episode + 1}/{episodes} - Avg Reward: {avg_reward:.2f}, Epsilon: {self.epsilon:.3f}")
        
        logger.info("=" * 60)
        logger.info("TRAINING COMPLETE")
        logger.info("=" * 60)
        logger.info(f"Total Q-states learned: {len(self.q_table)}")
        logger.info(f"Final epsilon: {self.epsilon:.3f}")
        logger.info(f"Final avg reward: {np.mean([h['reward'] for h in self.training_history[-100:]]):.2f}")
        logger.info("=" * 60)
        
        return self.training_history
    
    def generate_optimal_schedule(self, pet_id):
        """
        Generate optimal feeding schedule using trained Q-table
        
        Args:
            pet_id: Pet to generate schedule for
            
        Returns:
            List of optimal feeding times and amounts
        """
        if not self.q_table:
            logger.error("Model not trained. Train first using train_from_historical_data()")
            return None
        
        # Get typical interval
        df = self.data_preparer.extract_feeding_data(pet_id=pet_id, min_samples=1)
        if df is None:
            return None
        
        typical_interval = df['time_since_last_meal'].median() / 60.0
        
        schedule = []
        hours_since_last = typical_interval
        
        # Go through each hour of the day
        for hour in range(24):
            hunger = self.estimate_hunger(hours_since_last, typical_interval)
            state = self.get_state(hour, hunger, hours_since_last)
            
            # Get best action (no exploration)
            action = self.choose_action(state, training=False)
            
            if action != 'wait':
                schedule.append({
                    'hour': hour,
                    'time': f"{hour:02d}:00",
                    'action': action,
                    'amount': RL_CONFIG['action_amounts'][action],
                    'estimated_hunger': hunger
                })
                hours_since_last = 1
            else:
                hours_since_last += 1
        
        logger.info(f"Generated schedule with {len(schedule)} feeding times")
        
        return schedule
    
    def save_model(self):
        """Save trained Q-table to disk"""
        model_data = {
            'q_table': dict(self.q_table),
            'epsilon': self.epsilon,
            'training_history': self.training_history,
            'config': RL_CONFIG,
            'trained_at': datetime.utcnow()
        }
        
        with open(RL_MODEL_PATH, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"Model saved to {RL_MODEL_PATH}")
    
    def load_model(self):
        """Load trained Q-table from disk"""
        try:
            with open(RL_MODEL_PATH, 'rb') as f:
                model_data = pickle.load(f)
            
            self.q_table = defaultdict(lambda: defaultdict(float), model_data['q_table'])
            self.epsilon = model_data['epsilon']
            self.training_history = model_data.get('training_history', [])
            
            logger.info(f"Loaded model from {RL_MODEL_PATH}")
            logger.info(f"Model trained at: {model_data.get('trained_at')}")
            return True
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False
    
    def evaluate_schedule(self, pet_id, schedule):
        """
        Evaluate a feeding schedule against historical data
        
        Args:
            pet_id: Pet to evaluate for
            schedule: Schedule to evaluate
            
        Returns:
            Dictionary with evaluation metrics
        """
        # Get historical data
        df = self.data_preparer.extract_feeding_data(pet_id=pet_id, days=30, min_samples=1)
        
        if df is None:
            return None
        
        # Calculate metrics
        historical_avg_completion = df['completion_rate'].mean()
        historical_feedings_per_day = len(df) / (
            (df['timestamp'].max() - df['timestamp'].min()).days + 1
        )
        
        metrics = {
            'scheduled_feedings_per_day': len(schedule),
            'historical_feedings_per_day': historical_feedings_per_day,
            'total_daily_amount': sum([s['amount'] for s in schedule]),
            'historical_daily_amount': df['amount_consumed'].mean() * historical_feedings_per_day,
            'feeding_times': [s['time'] for s in schedule],
            'efficiency_score': 0.0  # Placeholder
        }
        
        # Calculate efficiency score
        # Compare scheduled amount vs historical consumption
        amount_ratio = metrics['total_daily_amount'] / metrics['historical_daily_amount']
        frequency_ratio = metrics['scheduled_feedings_per_day'] / metrics['historical_feedings_per_day']
        
        # Good schedule should match historical patterns but be more efficient
        metrics['efficiency_score'] = (amount_ratio + 1/frequency_ratio) / 2
        
        return metrics


# CLI functions
def train_optimizer_cli(pet_id, episodes=None):
    """Train schedule optimizer from command line"""
    optimizer = ScheduleOptimizer()
    history = optimizer.train_from_historical_data(pet_id, episodes)
    
    if history:
        optimizer.save_model()
        print("\n✅ Schedule optimizer training complete!")
        print(f"Model saved to: {RL_MODEL_PATH}")
        
        # Generate and display schedule
        schedule = optimizer.generate_optimal_schedule(pet_id)
        if schedule:
            print("\nOptimal Feeding Schedule:")
            print("-" * 40)
            for feeding in schedule:
                print(f"  {feeding['time']} - {feeding['amount']}g ({feeding['action']})")
    else:
        print("\n❌ Training failed")


def optimize_schedule_cli(pet_id):
    """Generate optimized schedule from command line"""
    optimizer = ScheduleOptimizer()
    
    if not optimizer.load_model():
        print("❌ No trained model found. Train first using 'train' command.")
        return
    
    schedule = optimizer.generate_optimal_schedule(pet_id)
    
    if schedule:
        print("\n" + "=" * 60)
        print("OPTIMIZED FEEDING SCHEDULE")
        print("=" * 60)
        for feeding in schedule:
            print(f"{feeding['time']} - {feeding['amount']}g (hunger level: {feeding['estimated_hunger']}/4)")
        print("=" * 60)
        print(f"Total daily amount: {sum([s['amount'] for s in schedule])}g")
        print(f"Feedings per day: {len(schedule)}")
        print("=" * 60)
    else:
        print("\n❌ Schedule generation failed")


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == 'train':
            if len(sys.argv) < 3:
                print("Usage: python schedule_optimizer.py train <pet_id> [episodes]")
            else:
                pet_id = int(sys.argv[2])
                episodes = int(sys.argv[3]) if len(sys.argv) > 3 else None
                train_optimizer_cli(pet_id, episodes)
        elif sys.argv[1] == 'optimize':
            if len(sys.argv) < 3:
                print("Usage: python schedule_optimizer.py optimize <pet_id>")
            else:
                optimize_schedule_cli(int(sys.argv[2]))
    else:
        print("Usage:")
        print("  python schedule_optimizer.py train <pet_id> [episodes]")
        print("  python schedule_optimizer.py optimize <pet_id>")
