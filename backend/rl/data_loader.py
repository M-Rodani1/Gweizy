"""
Data Loader for RL Transaction Agent

Loads historical gas price data from the database for:
- Training environments
- Backtesting agents
- Generating replay buffers
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataclasses import dataclass
from typing import Dict, Generator, List, Optional, Tuple
import numpy as np
import pandas as pd
from datetime import datetime, timedelta


@dataclass
class Episode:
    """Represents a single episode of gas price data"""
    timestamps: List[datetime]
    gas_prices: np.ndarray
    predictions: Optional[Dict[str, np.ndarray]] = None
    congestion: Optional[np.ndarray] = None

    def __len__(self):
        return len(self.gas_prices)


@dataclass
class DataLoaderConfig:
    """Configuration for data loading"""
    # Episode settings
    episode_length: int = 60  # Steps per episode
    step_interval_minutes: int = 1  # Time between steps
    min_episode_length: int = 30  # Minimum valid episode

    # Data filtering
    min_gas_price: float = 0.0001  # Filter out invalid readings
    max_gas_price: float = 1.0     # Filter out extreme outliers

    # Splitting
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15

    # Augmentation
    add_noise: bool = False
    noise_std: float = 0.001


class RLDataLoader:
    """
    Loads and prepares data for RL training

    Handles:
    - Loading from database
    - Episode generation
    - Train/val/test splitting
    - Data augmentation
    """

    def __init__(self, config: Optional[DataLoaderConfig] = None):
        self.config = config or DataLoaderConfig()
        self._data = None
        self._episodes = None

    def load_from_database(
        self,
        hours: int = 720,
        include_predictions: bool = True
    ) -> pd.DataFrame:
        """
        Load data from the database

        Args:
            hours: Hours of historical data to load
            include_predictions: Whether to include model predictions

        Returns:
            DataFrame with gas price data
        """
        from data.database import DatabaseManager

        print(f"Loading {hours} hours of data from database...")

        db = DatabaseManager()
        data = db.get_historical_data(hours=hours)

        if not data:
            raise ValueError("No data available in database")

        df = pd.DataFrame(data)

        # Parse timestamps
        df['timestamp'] = pd.to_datetime(df['timestamp'], format='mixed', errors='coerce')
        df = df.dropna(subset=['timestamp']).sort_values('timestamp').reset_index(drop=True)

        # Handle gas price column
        if 'gas_price' not in df.columns:
            if 'gwei' in df.columns:
                df['gas_price'] = df['gwei']
            elif 'current_gas' in df.columns:
                df['gas_price'] = df['current_gas']
            else:
                raise ValueError("No gas price column found")

        # Filter invalid data
        df = df[
            (df['gas_price'] >= self.config.min_gas_price) &
            (df['gas_price'] <= self.config.max_gas_price)
        ]

        print(f"Loaded {len(df)} valid records")
        print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        print(f"Gas range: {df['gas_price'].min():.6f} to {df['gas_price'].max():.6f} gwei")

        self._data = df
        return df

    def load_from_csv(self, path: str) -> pd.DataFrame:
        """Load data from CSV file for testing"""
        df = pd.read_csv(path)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp').reset_index(drop=True)

        self._data = df
        return df

    def generate_episodes(
        self,
        df: Optional[pd.DataFrame] = None
    ) -> List[Episode]:
        """
        Generate episodes from time series data

        Each episode is a contiguous sequence of gas prices
        that the agent will use for one training/evaluation run.

        Args:
            df: DataFrame with data (uses loaded data if None)

        Returns:
            List of Episode objects
        """
        if df is None:
            df = self._data

        if df is None:
            raise ValueError("No data loaded. Call load_from_database first.")

        episodes = []
        episode_length = self.config.episode_length
        min_length = self.config.min_episode_length

        # Calculate step size in dataframe rows
        # Assuming data is approximately uniform, find average interval
        time_diff = df['timestamp'].diff().median()
        rows_per_step = max(1, int(self.config.step_interval_minutes / (time_diff.seconds / 60 + 1)))

        print(f"Generating episodes (length={episode_length}, interval={self.config.step_interval_minutes}min)...")

        # Slide through data creating episodes
        i = 0
        while i + episode_length * rows_per_step < len(df):
            # Extract episode data
            end_idx = i + episode_length * rows_per_step
            episode_df = df.iloc[i:end_idx:rows_per_step]

            if len(episode_df) >= min_length:
                episode = Episode(
                    timestamps=episode_df['timestamp'].tolist(),
                    gas_prices=episode_df['gas_price'].values.astype(np.float32),
                    congestion=episode_df['congestion'].values if 'congestion' in episode_df else None
                )
                episodes.append(episode)

            # Move forward by half episode length for overlapping episodes
            i += episode_length * rows_per_step // 2

        print(f"Generated {len(episodes)} episodes")

        self._episodes = episodes
        return episodes

    def split_episodes(
        self,
        episodes: Optional[List[Episode]] = None
    ) -> Tuple[List[Episode], List[Episode], List[Episode]]:
        """
        Split episodes into train/val/test sets

        Uses temporal split (not random) to avoid data leakage.

        Args:
            episodes: List of episodes (uses generated episodes if None)

        Returns:
            (train_episodes, val_episodes, test_episodes)
        """
        if episodes is None:
            episodes = self._episodes

        if episodes is None:
            raise ValueError("No episodes generated. Call generate_episodes first.")

        n = len(episodes)
        train_end = int(n * self.config.train_ratio)
        val_end = int(n * (self.config.train_ratio + self.config.val_ratio))

        train = episodes[:train_end]
        val = episodes[train_end:val_end]
        test = episodes[val_end:]

        print(f"Split: {len(train)} train, {len(val)} val, {len(test)} test")

        return train, val, test

    def episode_generator(
        self,
        episodes: List[Episode],
        shuffle: bool = True,
        infinite: bool = True
    ) -> Generator[Episode, None, None]:
        """
        Generator that yields episodes for training

        Args:
            episodes: List of episodes to sample from
            shuffle: Whether to shuffle episode order
            infinite: Whether to loop indefinitely

        Yields:
            Episode objects
        """
        indices = np.arange(len(episodes))

        while True:
            if shuffle:
                np.random.shuffle(indices)

            for idx in indices:
                episode = episodes[idx]

                # Optionally add noise for augmentation
                if self.config.add_noise:
                    noise = np.random.normal(0, self.config.noise_std, len(episode))
                    augmented_prices = episode.gas_prices + noise
                    augmented_prices = np.clip(
                        augmented_prices,
                        self.config.min_gas_price,
                        self.config.max_gas_price
                    )
                    episode = Episode(
                        timestamps=episode.timestamps,
                        gas_prices=augmented_prices.astype(np.float32),
                        predictions=episode.predictions,
                        congestion=episode.congestion
                    )

                yield episode

            if not infinite:
                break

    def get_statistics(self, df: Optional[pd.DataFrame] = None) -> Dict:
        """Get statistics about the loaded data"""
        if df is None:
            df = self._data

        if df is None:
            return {}

        return {
            'total_records': len(df),
            'date_range': {
                'start': df['timestamp'].min().isoformat(),
                'end': df['timestamp'].max().isoformat()
            },
            'gas_price': {
                'mean': float(df['gas_price'].mean()),
                'std': float(df['gas_price'].std()),
                'min': float(df['gas_price'].min()),
                'max': float(df['gas_price'].max()),
                'median': float(df['gas_price'].median())
            },
            'num_episodes': len(self._episodes) if self._episodes else 0
        }


class ReplayBuffer:
    """
    Experience replay buffer for RL training

    Stores (state, action, reward, next_state, done) tuples
    for off-policy algorithms like DQN.
    """

    def __init__(self, capacity: int = 100000, state_dim: int = 15):
        self.capacity = capacity
        self.state_dim = state_dim

        # Pre-allocate arrays for efficiency
        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)

        self.position = 0
        self.size = 0

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ):
        """Add a transition to the buffer"""
        self.states[self.position] = state
        self.actions[self.position] = action
        self.rewards[self.position] = reward
        self.next_states[self.position] = next_state
        self.dones[self.position] = float(done)

        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> Tuple[np.ndarray, ...]:
        """Sample a batch of transitions"""
        indices = np.random.randint(0, self.size, size=batch_size)

        return (
            self.states[indices],
            self.actions[indices],
            self.rewards[indices],
            self.next_states[indices],
            self.dones[indices]
        )

    def __len__(self):
        return self.size


def create_training_data(
    hours: int = 720,
    episode_length: int = 60
) -> Tuple[List[Episode], List[Episode], List[Episode], Dict]:
    """
    Convenience function to load and prepare all training data

    Args:
        hours: Hours of historical data
        episode_length: Steps per episode

    Returns:
        (train_episodes, val_episodes, test_episodes, statistics)
    """
    config = DataLoaderConfig(episode_length=episode_length)
    loader = RLDataLoader(config=config)

    # Load data
    df = loader.load_from_database(hours=hours)

    # Generate episodes
    episodes = loader.generate_episodes(df)

    # Split
    train, val, test = loader.split_episodes(episodes)

    # Get statistics
    stats = loader.get_statistics(df)

    return train, val, test, stats


if __name__ == "__main__":
    # Test data loading
    print("="*60)
    print("Testing RL Data Loader")
    print("="*60)

    try:
        train, val, test, stats = create_training_data(hours=168)  # 1 week

        print(f"\nData Statistics:")
        print(f"  Total records: {stats['total_records']}")
        print(f"  Gas mean: {stats['gas_price']['mean']:.6f} gwei")
        print(f"  Gas std: {stats['gas_price']['std']:.6f} gwei")

        print(f"\nEpisode counts:")
        print(f"  Train: {len(train)}")
        print(f"  Val: {len(val)}")
        print(f"  Test: {len(test)}")

        if train:
            print(f"\nSample episode:")
            print(f"  Length: {len(train[0])}")
            print(f"  Gas range: {train[0].gas_prices.min():.6f} - {train[0].gas_prices.max():.6f}")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
