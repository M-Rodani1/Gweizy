"""
Data loader for RL training from historical gas prices.
"""
import numpy as np
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta
import sqlite3
import os


class GasDataLoader:
    """Loads historical gas data for RL training."""

    def __init__(self, db_path: str = None):
        self.db_path = db_path or os.path.join(
            os.path.dirname(os.path.dirname(__file__)), 
            'data', 'gas_data.db'
        )
        self._cache = None
        self._stats = None

    def load_data(self, hours: int = 168) -> List[Dict]:
        """Load historical gas prices."""
        if not os.path.exists(self.db_path):
            return self._generate_synthetic_data(hours)
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cutoff = datetime.utcnow() - timedelta(hours=hours)
            cursor.execute("""
                SELECT timestamp, gas_price, base_fee 
                FROM gas_prices 
                WHERE timestamp > ? 
                ORDER BY timestamp ASC
            """, (cutoff.isoformat(),))
            
            rows = cursor.fetchall()
            conn.close()
            
            if len(rows) < 100:
                return self._generate_synthetic_data(hours)
            
            data = []
            for row in rows:
                ts = datetime.fromisoformat(row[0]) if isinstance(row[0], str) else row[0]
                data.append({
                    'timestamp': ts,
                    'gas_price': float(row[1]),
                    'base_fee': float(row[2]) if row[2] else float(row[1]),
                    'hour': ts.hour,
                    'day_of_week': ts.weekday()
                })
            
            self._cache = data
            return data
        except Exception as e:
            print(f"DB error: {e}, using synthetic data")
            return self._generate_synthetic_data(hours)

    def _generate_synthetic_data(self, hours: int) -> List[Dict]:
        """Generate realistic synthetic gas data for training."""
        data = []
        base_price = 0.001  # Base gwei
        now = datetime.utcnow()
        
        for i in range(hours * 12):  # 5-min intervals
            ts = now - timedelta(minutes=(hours * 60) - (i * 5))
            hour = ts.hour
            dow = ts.weekday()
            
            # Time-based patterns
            hour_factor = 1.0
            if 2 <= hour <= 6:
                hour_factor = 0.7  # Low at night
            elif 10 <= hour <= 14:
                hour_factor = 1.4  # High midday
            elif 18 <= hour <= 21:
                hour_factor = 1.2  # Higher evening
            
            # Weekend discount
            if dow >= 5:
                hour_factor *= 0.85
            
            # Random noise + occasional spikes
            noise = np.random.normal(0, 0.15)
            spike = 0.5 if np.random.random() < 0.02 else 0
            
            price = base_price * hour_factor * (1 + noise) + spike * base_price
            price = max(0.0001, price)
            
            data.append({
                'timestamp': ts,
                'gas_price': price,
                'base_fee': price * 0.9,
                'hour': hour,
                'day_of_week': dow
            })
        
        self._cache = data
        return data

    def get_statistics(self) -> Dict:
        """Get price statistics for normalization."""
        if self._stats:
            return self._stats
        
        if not self._cache:
            self.load_data()
        
        prices = [d['gas_price'] for d in self._cache]
        self._stats = {
            'mean': np.mean(prices),
            'std': np.std(prices),
            'min': np.min(prices),
            'max': np.max(prices),
            'median': np.median(prices)
        }
        return self._stats

    def get_episodes(self, episode_length: int = 48, num_episodes: int = 100) -> List[List[Dict]]:
        """Split data into training episodes."""
        if not self._cache:
            self.load_data()
        
        episodes = []
        data_len = len(self._cache)
        
        for _ in range(num_episodes):
            start_idx = np.random.randint(0, max(1, data_len - episode_length))
            episode = self._cache[start_idx:start_idx + episode_length]
            if len(episode) == episode_length:
                episodes.append(episode)
        
        return episodes
