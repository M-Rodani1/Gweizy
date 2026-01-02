"""
Adaptive Data Collection Service

Dynamically adjusts collection intervals based on network volatility.
Collects more frequently during high volatility/spikes, less during stable periods.

This optimizes:
- Data quality (captures important events)
- Resource usage (reduces unnecessary calls during stable periods)
- Prediction accuracy (better training data)
"""

import time
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict
from data.collector import BaseGasCollector
from data.database import DatabaseManager
from utils.logger import logger


class AdaptiveCollector:
    """
    Adaptive data collection with dynamic intervals

    Intervals:
    - High volatility (spike detected): 30 seconds
    - Medium volatility: 60 seconds (default)
    - Low volatility (stable): 120 seconds
    """

    def __init__(self):
        self.collector = BaseGasCollector()
        self.db = DatabaseManager()

        # Interval settings (seconds)
        self.interval_high = 30      # During spikes
        self.interval_normal = 60    # Normal conditions
        self.interval_low = 120      # Stable periods

        # Current state
        self.current_interval = self.interval_normal
        self.recent_prices: List[float] = []
        self.window_size = 20  # Number of recent prices to analyze

        # Volatility thresholds
        self.high_volatility_threshold = 0.15  # 15% coefficient of variation
        self.low_volatility_threshold = 0.05   # 5% coefficient of variation

        # Statistics
        self.collections_count = 0
        self.interval_changes = 0
        self.spike_detections = 0

    def get_volatility_state(self) -> str:
        """
        Analyze recent prices and determine volatility state

        Returns:
            'high', 'normal', or 'low'
        """
        if len(self.recent_prices) < 5:
            return 'normal'

        prices = np.array(self.recent_prices[-self.window_size:])

        # Calculate coefficient of variation (std / mean)
        if np.mean(prices) > 0:
            cv = np.std(prices) / np.mean(prices)
        else:
            cv = 0

        # Check for recent spike
        if len(prices) >= 3:
            recent_change = abs(prices[-1] - prices[-2]) / prices[-2] if prices[-2] > 0 else 0
            if recent_change > 0.20:  # 20% sudden change
                logger.info(f"🚨 SPIKE DETECTED: {recent_change:.1%} change")
                self.spike_detections += 1
                return 'high'

        # Classify based on CV
        if cv > self.high_volatility_threshold:
            return 'high'
        elif cv < self.low_volatility_threshold:
            return 'low'
        else:
            return 'normal'

    def update_interval(self, volatility_state: str) -> int:
        """
        Update collection interval based on volatility

        Args:
            volatility_state: 'high', 'normal', or 'low'

        Returns:
            New interval in seconds
        """
        old_interval = self.current_interval

        if volatility_state == 'high':
            self.current_interval = self.interval_high
        elif volatility_state == 'low':
            self.current_interval = self.interval_low
        else:
            self.current_interval = self.interval_normal

        if old_interval != self.current_interval:
            self.interval_changes += 1
            emoji = "🔴" if volatility_state == 'high' else "🟢" if volatility_state == 'low' else "🟡"
            logger.info(
                f"{emoji} Interval adjusted: {old_interval}s → {self.current_interval}s "
                f"(volatility: {volatility_state})"
            )

        return self.current_interval

    def collect_with_metadata(self) -> Dict:
        """
        Collect gas price and return with collection metadata

        Returns:
            Dict with gas data and collection metadata
        """
        data = self.collector.get_current_gas()

        if data:
            # Add to recent prices
            self.recent_prices.append(data['current_gas'])
            if len(self.recent_prices) > self.window_size:
                self.recent_prices.pop(0)

            # Add metadata
            volatility_state = self.get_volatility_state()
            data['collection_metadata'] = {
                'interval': self.current_interval,
                'volatility_state': volatility_state,
                'recent_prices_count': len(self.recent_prices)
            }

            # Save to database
            try:
                self.db.save_gas_price(data)
                self.collections_count += 1
            except Exception as e:
                logger.warning(f"Failed to save to database: {e}")

        return data

    def run_collection_cycle(self):
        """
        Run one collection cycle with adaptive interval
        """
        start_time = time.time()

        # Collect data
        data = self.collect_with_metadata()

        if data:
            volatility_state = data['collection_metadata']['volatility_state']

            # Log collection
            elapsed = time.time() - start_time
            logger.info(
                f"✓ #{self.collections_count}: "
                f"{data['current_gas']:.6f} Gwei "
                f"(base: {data['base_fee']:.6f}, priority: {data['priority_fee']:.6f}) "
                f"| Volatility: {volatility_state} | Interval: {self.current_interval}s "
                f"[{elapsed:.2f}s]"
            )

            # Update interval for next collection
            self.update_interval(volatility_state)
        else:
            logger.warning("Collection failed, keeping current interval")

    def get_stats(self) -> Dict:
        """Get collection statistics"""
        if len(self.recent_prices) > 1:
            volatility = np.std(self.recent_prices) / np.mean(self.recent_prices) if np.mean(self.recent_prices) > 0 else 0
        else:
            volatility = 0

        return {
            'collections': self.collections_count,
            'interval_changes': self.interval_changes,
            'spike_detections': self.spike_detections,
            'current_interval': self.current_interval,
            'recent_prices_count': len(self.recent_prices),
            'current_volatility': volatility,
            'volatility_state': self.get_volatility_state()
        }

    def log_stats(self):
        """Log detailed statistics"""
        stats = self.get_stats()

        if len(self.recent_prices) >= 5:
            prices = np.array(self.recent_prices)
            logger.info("="*60)
            logger.info("Adaptive Collection Statistics:")
            logger.info(f"  Total collections: {stats['collections']}")
            logger.info(f"  Interval changes: {stats['interval_changes']}")
            logger.info(f"  Spike detections: {stats['spike_detections']}")
            logger.info(f"  Current interval: {stats['current_interval']}s")
            logger.info(f"  Volatility state: {stats['volatility_state']}")
            logger.info(f"  Current volatility: {stats['current_volatility']:.3f}")
            logger.info(f"  Recent price range: {np.min(prices):.6f} - {np.max(prices):.6f} Gwei")
            logger.info("="*60)


def main():
    """
    Main entry point for adaptive collection service

    Can be used as a standalone service or integrated into existing collector
    """
    import signal
    import sys

    collector = AdaptiveCollector()
    running = True

    def signal_handler(signum, frame):
        nonlocal running
        logger.info(f"Received signal {signum}, shutting down...")
        collector.log_stats()
        running = False

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    logger.info("="*60)
    logger.info("Adaptive Gas Price Collection Service Starting")
    logger.info(f"Intervals: High={collector.interval_high}s, Normal={collector.interval_normal}s, Low={collector.interval_low}s")
    logger.info("="*60)

    while running:
        try:
            collector.run_collection_cycle()

            # Log stats every 20 collections
            if collector.collections_count % 20 == 0 and collector.collections_count > 0:
                collector.log_stats()

            # Sleep for current interval
            if running:
                time.sleep(collector.current_interval)

        except Exception as e:
            logger.error(f"Error in collection loop: {e}")
            import traceback
            logger.error(traceback.format_exc())
            time.sleep(collector.current_interval)

    logger.info("Adaptive collector stopped gracefully")
    sys.exit(0)


if __name__ == "__main__":
    main()
