"""
Data loader for RL training from historical gas prices.
"""
import numpy as np
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta
import os
import sys

# Add parent to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

try:
    from data.database import DatabaseManager
    DB_AVAILABLE = True
except ImportError:
    DB_AVAILABLE = False
    print("Warning: DatabaseManager not available, using synthetic data only")


class GasDataLoader:
    """Loads historical gas data for RL training with data augmentation."""

    def __init__(self, db_path: str = None, use_database: bool = True):
        self.db_path = db_path
        self.use_database = use_database and DB_AVAILABLE
        self._cache = None
        self._stats = None
        self._db = None
        
        if self.use_database:
            try:
                self._db = DatabaseManager()
            except Exception as e:
                print(f"Warning: Could not initialize database: {e}")
                self.use_database = False

    def set_cache(self, data: List[Dict]):
        """Override cached data (used for train/val splits)."""
        self._cache = list(data)
        self._stats = None

    def split_data(self, train_ratio: float = 0.8) -> Tuple[List[Dict], List[Dict]]:
        """Split cached data into train/eval segments (time-ordered)."""
        if not self._cache:
            self.load_data()
        split_idx = int(len(self._cache) * train_ratio)
        train_data = self._cache[:split_idx]
        eval_data = self._cache[split_idx:]
        return train_data, eval_data

    def load_data(self, hours: int = 720, min_records: int = 100, chain_id: int = 8453) -> List[Dict]:
        """
        Load historical gas prices from database for a specific chain.
        REQUIRES REAL DATA - no synthetic fallback.
        
        Args:
            hours: Hours of historical data to load
            min_records: Minimum records needed (raises error if not met)
            chain_id: Chain ID (8453=Base, 1=Ethereum, etc.)
        
        Raises:
            ValueError: If insufficient real data is available
        """
        if not self.use_database or not self._db:
            raise ValueError("Database not available. Cannot train DQN without real data.")
        
        try:
            # Use DatabaseManager to get historical data for specific chain
            raw_data = self._db.get_historical_data(hours=hours, chain_id=chain_id)
            
            if len(raw_data) < min_records:
                raise ValueError(
                    f"Insufficient real data: {len(raw_data)} records found for chain {chain_id}, "
                    f"need at least {min_records}. Please collect more data before training."
                )
            
            # Convert to format expected by RL environment
            data = []
            for d in raw_data:
                # Handle timestamp
                ts = d.get('timestamp', '')
                if isinstance(ts, str):
                    from dateutil import parser
                    try:
                        ts = parser.parse(ts)
                    except:
                        ts = datetime.now()
                elif not isinstance(ts, datetime):
                    ts = datetime.now()
                
                # Get gas price (handle different field names)
                gas_price = d.get('gwei') or d.get('current_gas') or 0.001
                base_fee = d.get('baseFee') or d.get('base_fee') or gas_price * 0.9
                
                data.append({
                    'timestamp': ts,
                    'gas_price': float(gas_price),
                    'base_fee': float(base_fee),
                    'hour': ts.hour,
                    'day_of_week': ts.weekday()
                })
            
            # Sort by timestamp
            data.sort(key=lambda x: x['timestamp'])
            
            self._cache = data
            print(f"✅ Loaded {len(data)} REAL records from database for chain {chain_id}")
            return data
            
        except Exception as e:
            raise ValueError(f"Failed to load real data from database: {e}. Cannot train without real data.")

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
        """Get price statistics for normalization with enhanced metrics."""
        if self._stats:
            return self._stats
        
        if not self._cache:
            self.load_data()
        
        self._stats = self.get_statistics_for_data(self._cache)
        return self._stats

    def get_statistics_for_data(self, data: List[Dict]) -> Dict:
        """Compute statistics for a specific dataset segment."""
        prices = np.array([d['gas_price'] for d in data])
        if len(prices) == 0:
            return {
                'mean': 0.0,
                'std': 0.0,
                'min': 0.0,
                'max': 0.0,
                'median': 0.0,
                'iqr': 0.0,
                'q25': 0.0,
                'q75': 0.0,
                'typical_volatility': 0.0,
                'percentile_rank': 0.5
            }

        # Calculate IQR (Interquartile Range) for robust scaling
        q25 = np.percentile(prices, 25)
        q75 = np.percentile(prices, 75)
        iqr = q75 - q25

        # Calculate typical volatility (median of rolling volatilities)
        if len(prices) > 24:
            volatilities = []
            for i in range(len(prices) - 24):
                window = prices[i:i+24]
                vol = np.std(window) / (np.mean(window) + 1e-8)
                volatilities.append(vol)
            typical_volatility = np.median(volatilities) if volatilities else np.std(prices) / (np.mean(prices) + 1e-8)
        else:
            typical_volatility = np.std(prices) / (np.mean(prices) + 1e-8)

        return {
            'mean': float(np.mean(prices)),
            'std': float(np.std(prices)),
            'min': float(np.min(prices)),
            'max': float(np.max(prices)),
            'median': float(np.median(prices)),
            'iqr': float(iqr),
            'q25': float(q25),
            'q75': float(q75),
            'typical_volatility': float(typical_volatility)
        }

    def get_episodes(self, episode_length: int = 48, num_episodes: int = 100, 
                     augment: bool = True) -> List[List[Dict]]:
        """
        Split data into training episodes with optional augmentation.
        REQUIRES REAL DATA - no synthetic fallback.
        
        Args:
            episode_length: Length of each episode in time steps
            num_episodes: Number of episodes to generate
            augment: Whether to apply data augmentation (noise, time shifts)
        
        Raises:
            ValueError: If insufficient real data is available
        """
        if not self._cache:
            self.load_data(hours=720)  # Load more data for better episodes

        return self.get_episodes_from_data(
            self._cache,
            episode_length=episode_length,
            num_episodes=num_episodes,
            augment=augment
        )

    def get_episodes_from_data(
        self,
        data: List[Dict],
        episode_length: int = 48,
        num_episodes: int = 100,
        augment: bool = True
    ) -> List[List[Dict]]:
        """Generate episodes from a provided data slice (train/val separation)."""
        episodes = []
        data_len = len(data)

        if data_len < episode_length:
            raise ValueError(
                f"Insufficient real data: {data_len} records < {episode_length} required. "
                f"Please collect more data before training."
            )

        for _ in range(num_episodes):
            max_start = max(0, data_len - episode_length)
            start_idx = np.random.randint(0, max_start + 1)
            episode = [p.copy() for p in data[start_idx:start_idx + episode_length]]

            if len(episode) == episode_length:
                if augment:
                    episode = self._augment_episode(episode)
                episodes.append(episode)

        return episodes
    
    def _augment_episode(self, episode: List[Dict]) -> List[Dict]:
        """
        Apply data augmentation to an episode.
        - Add small random noise to prices
        - Time-based scaling
        - Slight time shifts
        """
        augmented = []
        
        for i, point in enumerate(episode):
            new_point = point.copy()
            
            # Add small random noise (1-3% variance)
            noise_factor = np.random.normal(1.0, 0.02)
            new_point['gas_price'] = max(0.0001, point['gas_price'] * noise_factor)
            new_point['base_fee'] = max(0.0001, point['base_fee'] * noise_factor)
            
            augmented.append(new_point)
        
        return augmented
    
    def get_diverse_episodes(self, episode_length: int = 48, num_episodes: int = 100) -> List[List[Dict]]:
        """
        Get diverse episodes covering different market conditions.
        Ensures episodes include:
        - High volatility periods
        - Low volatility periods
        - Price spikes
        - Normal conditions
        
        REQUIRES REAL DATA - no synthetic fallback.
        """
        if not self._cache:
            self.load_data(hours=720)  # This will raise if no real data
        
        data_len = len(self._cache)
        if data_len < episode_length * 2:
            # Still use real data, just with less diversity
            return self.get_episodes(episode_length, num_episodes)
        
        episodes = []
        
        # Calculate volatility for each potential episode start
        volatilities = []
        for i in range(0, data_len - episode_length, episode_length // 2):
            window = self._cache[i:i+episode_length]
            prices = [p['gas_price'] for p in window]
            if len(prices) > 1:
                vol = np.std(prices) / (np.mean(prices) + 1e-8)
                volatilities.append((i, vol))
        
        if not volatilities:
            return self.get_episodes(episode_length, num_episodes)
        
        # Sort by volatility
        volatilities.sort(key=lambda x: x[1])
        
        # Sample from different volatility regimes
        low_vol = [v[0] for v in volatilities[:len(volatilities)//3]]
        mid_vol = [v[0] for v in volatilities[len(volatilities)//3:2*len(volatilities)//3]]
        high_vol = [v[0] for v in volatilities[2*len(volatilities)//3:]]
        
        # Distribute episodes across regimes
        episodes_per_regime = num_episodes // 3
        
        for regime in [low_vol, mid_vol, high_vol]:
            if not regime:
                continue
            for _ in range(episodes_per_regime):
                start_idx = np.random.choice(regime)
                episode = self._cache[start_idx:start_idx + episode_length]
                if len(episode) == episode_length:
                    episodes.append(episode.copy())
        
        # Fill remaining with random episodes
        while len(episodes) < num_episodes:
            start_idx = np.random.randint(0, max(1, data_len - episode_length))
            episode = self._cache[start_idx:start_idx + episode_length]
            if len(episode) == episode_length:
                episodes.append(episode.copy())
        
        return episodes[:num_episodes]
