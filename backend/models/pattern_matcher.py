"""
Historical Pattern Matching for Gas Price Prediction

Finds similar patterns in historical gas price data to improve predictions.
Uses correlation-based matching with time-of-day and day-of-week awareness.

Features:
- Pattern extraction from recent price windows
- Historical pattern search with configurable lookback
- Time-aware matching (similar hours/days get priority)
- Weighted outcome analysis from matched patterns
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class PatternMatch:
    """Represents a matched historical pattern."""
    timestamp: datetime
    correlation: float
    time_similarity: float  # How similar the time-of-day/day-of-week is
    combined_score: float
    pattern_prices: np.ndarray
    outcome_prices: np.ndarray  # What happened after this pattern
    outcome_change_1h: float
    outcome_change_4h: float
    outcome_change_24h: float


class PatternMatcher:
    """
    Finds and analyzes similar historical gas price patterns.

    Uses a sliding window approach to find patterns in historical data
    that are similar to the current price window. Analyzes what happened
    after those historical patterns to inform predictions.
    """

    def __init__(
        self,
        pattern_length: int = 12,  # Number of data points for pattern (1 hour at 5min intervals)
        min_correlation: float = 0.7,
        max_matches: int = 10,
        time_weight: float = 0.3  # Weight for time similarity in combined score
    ):
        """
        Initialize the pattern matcher.

        Args:
            pattern_length: Number of data points to use for pattern matching
            min_correlation: Minimum correlation threshold for a valid match
            max_matches: Maximum number of matches to return
            time_weight: Weight given to time-of-day/week similarity
        """
        self.pattern_length = pattern_length
        self.min_correlation = min_correlation
        self.max_matches = max_matches
        self.time_weight = time_weight

    def find_similar_patterns(
        self,
        recent_data: pd.DataFrame,
        historical_data: pd.DataFrame,
        current_time: Optional[datetime] = None
    ) -> List[PatternMatch]:
        """
        Find historical patterns similar to the current price pattern.

        Args:
            recent_data: Recent gas price data (current pattern)
            historical_data: Historical data to search for patterns
            current_time: Current timestamp for time-aware matching

        Returns:
            List of PatternMatch objects sorted by combined score
        """
        if len(recent_data) < self.pattern_length:
            logger.warning(f"Not enough recent data for pattern matching: {len(recent_data)} < {self.pattern_length}")
            return []

        if len(historical_data) < self.pattern_length + 12:  # Need outcome data too
            logger.warning("Not enough historical data for pattern matching")
            return []

        # Extract current pattern
        current_pattern = self._extract_pattern(recent_data)
        if current_pattern is None:
            return []

        # Normalize current pattern
        current_normalized = self._normalize_pattern(current_pattern)

        # Get current time for time-aware matching
        if current_time is None:
            if 'timestamp' in recent_data.columns:
                current_time = pd.to_datetime(recent_data['timestamp'].iloc[-1])
            else:
                current_time = datetime.now()

        current_hour = current_time.hour
        current_dow = current_time.weekday()

        # Search historical data for similar patterns
        matches = []
        historical_prices = self._get_prices(historical_data)

        # Slide through historical data
        # Leave room for outcome analysis (24 periods = ~2 hours at 5min intervals)
        for i in range(len(historical_prices) - self.pattern_length - 24):
            # Extract historical pattern
            hist_pattern = historical_prices[i:i + self.pattern_length]
            hist_normalized = self._normalize_pattern(hist_pattern)

            if hist_normalized is None:
                continue

            # Calculate correlation
            correlation = self._calculate_correlation(current_normalized, hist_normalized)

            if correlation < self.min_correlation:
                continue

            # Get historical timestamp for time similarity
            if 'timestamp' in historical_data.columns:
                hist_time = pd.to_datetime(historical_data['timestamp'].iloc[i + self.pattern_length - 1])
                time_similarity = self._calculate_time_similarity(
                    current_hour, current_dow,
                    hist_time.hour, hist_time.weekday()
                )
            else:
                time_similarity = 0.5  # Neutral if no timestamps

            # Combined score
            combined_score = (1 - self.time_weight) * correlation + self.time_weight * time_similarity

            # Extract outcome (what happened after this pattern)
            outcome_start = i + self.pattern_length
            outcome_prices = historical_prices[outcome_start:outcome_start + 24]

            if len(outcome_prices) < 24:
                continue

            # Calculate outcome changes
            base_price = historical_prices[i + self.pattern_length - 1]
            outcome_1h = (outcome_prices[11] - base_price) / base_price if base_price > 0 else 0  # ~1 hour
            outcome_4h = (outcome_prices[23] - base_price) / base_price if base_price > 0 else 0  # ~2 hours (limited by data)
            outcome_24h = outcome_4h  # Use same for now until we have more data points

            match = PatternMatch(
                timestamp=hist_time if 'timestamp' in historical_data.columns else datetime.now(),
                correlation=correlation,
                time_similarity=time_similarity,
                combined_score=combined_score,
                pattern_prices=hist_pattern,
                outcome_prices=outcome_prices,
                outcome_change_1h=outcome_1h,
                outcome_change_4h=outcome_4h,
                outcome_change_24h=outcome_24h
            )
            matches.append(match)

        # Sort by combined score and return top matches
        matches.sort(key=lambda m: m.combined_score, reverse=True)
        return matches[:self.max_matches]

    def predict_from_patterns(
        self,
        matches: List[PatternMatch],
        current_price: float
    ) -> Dict[str, Any]:
        """
        Generate predictions based on matched patterns.

        Args:
            matches: List of matched patterns
            current_price: Current gas price

        Returns:
            Dictionary with pattern-based predictions
        """
        if not matches:
            return {
                'available': False,
                'reason': 'No matching patterns found'
            }

        # Weight predictions by combined score
        total_weight = sum(m.combined_score for m in matches)

        if total_weight == 0:
            return {
                'available': False,
                'reason': 'Invalid pattern weights'
            }

        # Weighted average of outcomes
        pred_1h = sum(m.outcome_change_1h * m.combined_score for m in matches) / total_weight
        pred_4h = sum(m.outcome_change_4h * m.combined_score for m in matches) / total_weight
        pred_24h = sum(m.outcome_change_24h * m.combined_score for m in matches) / total_weight

        # Calculate standard deviation for confidence
        std_1h = np.std([m.outcome_change_1h for m in matches]) if len(matches) > 1 else 0.1
        std_4h = np.std([m.outcome_change_4h for m in matches]) if len(matches) > 1 else 0.1
        std_24h = np.std([m.outcome_change_24h for m in matches]) if len(matches) > 1 else 0.1

        # Calculate confidence (higher when patterns agree)
        avg_correlation = sum(m.correlation for m in matches) / len(matches)
        confidence = avg_correlation * (1 - min(std_1h, 0.5))  # Lower confidence if high variance

        # Generate price predictions
        predictions = {
            'available': True,
            'match_count': len(matches),
            'avg_correlation': round(avg_correlation, 4),
            'confidence': round(confidence, 4),
            '1h': {
                'predicted_change': round(pred_1h, 4),
                'predicted_price': round(current_price * (1 + pred_1h), 6),
                'std_dev': round(std_1h, 4)
            },
            '4h': {
                'predicted_change': round(pred_4h, 4),
                'predicted_price': round(current_price * (1 + pred_4h), 6),
                'std_dev': round(std_4h, 4)
            },
            '24h': {
                'predicted_change': round(pred_24h, 4),
                'predicted_price': round(current_price * (1 + pred_24h), 6),
                'std_dev': round(std_24h, 4)
            },
            'pattern_insight': self._generate_insight(matches, pred_1h, pred_4h)
        }

        return predictions

    def _extract_pattern(self, data: pd.DataFrame) -> Optional[np.ndarray]:
        """Extract price pattern from dataframe."""
        prices = self._get_prices(data)
        if len(prices) < self.pattern_length:
            return None
        return prices[-self.pattern_length:]

    def _get_prices(self, data: pd.DataFrame) -> np.ndarray:
        """Get price array from dataframe."""
        for col in ['gas_price', 'current_gas', 'gas', 'gwei']:
            if col in data.columns:
                return data[col].values.astype(float)
        raise ValueError("No price column found in data")

    def _normalize_pattern(self, pattern: np.ndarray) -> Optional[np.ndarray]:
        """Normalize pattern to zero mean and unit variance."""
        if len(pattern) == 0:
            return None
        std = np.std(pattern)
        if std == 0:
            return None  # Flat pattern, can't normalize
        return (pattern - np.mean(pattern)) / std

    def _calculate_correlation(self, pattern1: np.ndarray, pattern2: np.ndarray) -> float:
        """Calculate Pearson correlation between two patterns."""
        if len(pattern1) != len(pattern2):
            return 0.0
        try:
            correlation = np.corrcoef(pattern1, pattern2)[0, 1]
            return float(correlation) if not np.isnan(correlation) else 0.0
        except Exception:
            return 0.0

    def _calculate_time_similarity(
        self,
        hour1: int, dow1: int,
        hour2: int, dow2: int
    ) -> float:
        """
        Calculate similarity between two time points.

        Considers:
        - Hour of day (cyclical)
        - Day of week (weekend vs weekday)
        """
        # Hour similarity (cyclical distance)
        hour_diff = min(abs(hour1 - hour2), 24 - abs(hour1 - hour2))
        hour_sim = 1 - (hour_diff / 12)  # Max diff is 12 hours

        # Day of week similarity
        # Weekday (0-4) vs weekend (5-6)
        is_weekend1 = dow1 >= 5
        is_weekend2 = dow2 >= 5
        dow_sim = 1.0 if is_weekend1 == is_weekend2 else 0.5

        # Combined (hour matters more)
        return 0.7 * hour_sim + 0.3 * dow_sim

    def _generate_insight(
        self,
        matches: List[PatternMatch],
        pred_1h: float,
        pred_4h: float
    ) -> str:
        """Generate human-readable insight from pattern analysis."""
        if not matches:
            return "No patterns matched"

        avg_corr = sum(m.correlation for m in matches) / len(matches)

        # Determine trend
        if pred_1h > 0.02:
            trend = "rising"
            action = "Prices historically increased after similar patterns"
        elif pred_1h < -0.02:
            trend = "falling"
            action = "Prices historically decreased after similar patterns"
        else:
            trend = "stable"
            action = "Prices historically remained stable after similar patterns"

        # Confidence statement
        if avg_corr > 0.85:
            conf_text = "Strong pattern match"
        elif avg_corr > 0.75:
            conf_text = "Good pattern match"
        else:
            conf_text = "Moderate pattern match"

        return f"{conf_text} ({len(matches)} similar patterns found). {action}."


# Singleton instance
_pattern_matcher: Optional[PatternMatcher] = None


def get_pattern_matcher() -> PatternMatcher:
    """Get or create the global pattern matcher instance."""
    global _pattern_matcher
    if _pattern_matcher is None:
        _pattern_matcher = PatternMatcher()
    return _pattern_matcher
