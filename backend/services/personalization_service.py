"""
Personalized recommendations service.
Analyzes user transaction history to provide personalized gas optimization suggestions.
"""
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from collections import defaultdict
import numpy as np
from data.database import DatabaseManager
from utils.logger import logger


class PersonalizationService:
    """Service for generating personalized recommendations based on user behavior."""
    
    def __init__(self):
        self.db = DatabaseManager()
    
    def analyze_user_patterns(self, user_address: str, chain_id: int = 8453) -> Dict:
        """
        Analyze user transaction patterns to identify optimal times.
        
        Returns:
            Dict with patterns, recommendations, and statistics
        """
        transactions = self.db.get_user_transactions(user_address, chain_id, limit=500)
        
        if len(transactions) < 5:
            return self._get_default_recommendations()
        
        # Analyze patterns
        hour_distribution = defaultdict(int)
        day_distribution = defaultdict(int)
        gas_prices_by_hour = defaultdict(list)
        total_gas_paid = 0
        total_potential_savings = 0
        
        for tx in transactions:
            if tx.get('status') != 'success':
                continue
            
            try:
                ts = datetime.fromisoformat(tx['timestamp'].replace('Z', '+00:00'))
                hour = ts.hour
                day = ts.weekday()
                
                hour_distribution[hour] += 1
                day_distribution[day] += 1
                gas_prices_by_hour[hour].append(tx['gas_price_gwei'])
                
                total_gas_paid += tx.get('gas_cost_eth', 0)
                if tx.get('saved_by_waiting'):
                    total_potential_savings += tx['saved_by_waiting']
            except:
                continue
        
        # Find most common transaction times
        most_common_hour = max(hour_distribution.items(), key=lambda x: x[1])[0] if hour_distribution else None
        most_common_day = max(day_distribution.items(), key=lambda x: x[1])[0] if day_distribution else None
        
        # Find hours with lowest average gas prices
        avg_gas_by_hour = {
            hour: np.mean(prices) for hour, prices in gas_prices_by_hour.items() if prices
        }
        best_hour = min(avg_gas_by_hour.items(), key=lambda x: x[1])[0] if avg_gas_by_hour else None
        
        # Calculate savings opportunity
        savings_percentage = (total_potential_savings / total_gas_paid * 100) if total_gas_paid > 0 else 0
        
        return {
            'total_transactions': len(transactions),
            'analysis_period_days': 30,
            'patterns': {
                'most_common_hour': most_common_hour,
                'most_common_day': self._day_name(most_common_day) if most_common_day is not None else None,
                'best_hour_for_gas': best_hour,
                'avg_gas_by_hour': {str(k): float(v) for k, v in avg_gas_by_hour.items()}
            },
            'recommendations': {
                'usual_time': f"{most_common_hour}:00 UTC" if most_common_hour is not None else "Varies",
                'best_time': f"{best_hour}:00 UTC" if best_hour is not None else "2:00 UTC",
                'savings_opportunity': round(savings_percentage, 1),
                'suggestion': self._generate_suggestion(most_common_hour, best_hour, savings_percentage)
            },
            'statistics': {
                'total_gas_paid_eth': round(total_gas_paid, 6),
                'potential_savings_eth': round(total_potential_savings, 6),
                'savings_percentage': round(savings_percentage, 1),
                'avg_gas_price_gwei': round(np.mean([t['gas_price_gwei'] for t in transactions if t.get('gas_price_gwei')]), 4)
            }
        }
    
    def get_personalized_recommendation(self, user_address: str, chain_id: int = 8453) -> Dict:
        """
        Get personalized "best time to transact" recommendation.
        
        Returns:
            Dict with personalized recommendation
        """
        patterns = self.analyze_user_patterns(user_address, chain_id)
        
        # Get current time
        now = datetime.utcnow()
        current_hour = now.hour
        
        # Get best hour from patterns
        best_hour = patterns['patterns'].get('best_hour_for_gas')
        most_common_hour = patterns['patterns'].get('most_common_hour')
        
        # Calculate hours until best time
        if best_hour is not None:
            if current_hour < best_hour:
                hours_until_best = best_hour - current_hour
            else:
                hours_until_best = (24 - current_hour) + best_hour
            
            recommendation = {
                'recommended_time': f"{best_hour}:00 UTC",
                'hours_until_best': hours_until_best,
                'confidence': 'high' if patterns['total_transactions'] >= 20 else 'medium',
                'reason': f"Based on your {patterns['total_transactions']} transactions, {best_hour}:00 UTC typically has the lowest gas prices",
                'your_usual_time': f"{most_common_hour}:00 UTC" if most_common_hour is not None else "Varies",
                'potential_savings': f"{patterns['recommendations']['savings_opportunity']}%"
            }
        else:
            recommendation = {
                'recommended_time': "2:00 UTC",
                'hours_until_best': 0,
                'confidence': 'low',
                'reason': "Not enough transaction history. Using general recommendation.",
                'your_usual_time': "Unknown",
                'potential_savings': "15-25%"
            }
        
        return {
            **recommendation,
            'patterns': patterns
        }
    
    def track_transaction_savings(self, user_address: str, tx_hash: str, 
                                 gas_price_gwei: float, gas_used: int,
                                 chain_id: int = 8453) -> Optional[Dict]:
        """
        Track a transaction and calculate potential savings if user had waited.
        
        Returns:
            Dict with savings analysis or None if couldn't calculate
        """
        try:
            # Get historical gas prices around transaction time
            historical = self.db.get_historical_data(hours=24, chain_id=chain_id)
            
            if len(historical) < 10:
                return None
            
            # Find minimum gas price in the 24h window
            gas_prices = [h.get('gwei', 0) for h in historical if h.get('gwei', 0) > 0]
            if not gas_prices:
                return None
            
            min_gas = min(gas_prices)
            optimal_time = None
            
            # Find when minimum occurred
            for h in historical:
                if h.get('gwei') == min_gas:
                    try:
                        ts = h.get('timestamp', '')
                        if isinstance(ts, str):
                            from dateutil import parser
                            optimal_time = parser.parse(ts)
                        else:
                            optimal_time = ts
                        break
                    except:
                        pass
            
            # Calculate savings
            gas_cost_eth = (gas_price_gwei * gas_used) / 1e9
            optimal_cost_eth = (min_gas * gas_used) / 1e9
            saved_by_waiting = gas_cost_eth - optimal_cost_eth
            
            # Save transaction
            self.db.save_user_transaction({
                'user_address': user_address.lower(),
                'chain_id': chain_id,
                'tx_hash': tx_hash,
                'block_number': 0,  # Will be filled from blockchain
                'timestamp': datetime.now(),
                'gas_price_gwei': gas_price_gwei,
                'gas_used': gas_used,
                'gas_cost_eth': gas_cost_eth,
                'tx_type': 'unknown',
                'status': 'success',
                'saved_by_waiting': saved_by_waiting if saved_by_waiting > 0 else 0,
                'optimal_time': optimal_time
            })
            
            return {
                'saved_by_waiting_eth': saved_by_waiting if saved_by_waiting > 0 else 0,
                'savings_percentage': (saved_by_waiting / gas_cost_eth * 100) if gas_cost_eth > 0 else 0,
                'optimal_gas_price': min_gas,
                'optimal_time': optimal_time.isoformat() if optimal_time else None
            }
            
        except Exception as e:
            logger.error(f"Error tracking transaction savings: {e}")
            return None
    
    def _get_default_recommendations(self) -> Dict:
        """Get default recommendations when insufficient data"""
        current_hour = datetime.utcnow().hour
        return {
            'total_transactions': 0,
            'analysis_period_days': 0,
            'patterns': {
                'most_common_hour': current_hour,
                'most_common_day': None,
                'best_hour_for_gas': 2,  # Typically lowest at 2 AM UTC
                'avg_gas_by_hour': {}
            },
            'recommendations': {
                'usual_time': f"{current_hour}:00 UTC",
                'best_time': "2:00 UTC",
                'savings_opportunity': 20,
                'suggestion': "Gas prices are typically lowest between 2-6 AM UTC. Consider scheduling transactions during these hours."
            },
            'statistics': {
                'total_gas_paid_eth': 0,
                'potential_savings_eth': 0,
                'savings_percentage': 0,
                'avg_gas_price_gwei': 0
            }
        }
    
    def _generate_suggestion(self, usual_hour: Optional[int], best_hour: Optional[int], 
                           savings_pct: float) -> str:
        """Generate personalized suggestion text"""
        if usual_hour is None or best_hour is None:
            return "Gas prices are typically lowest between 2-6 AM UTC. Consider scheduling transactions during these hours."
        
        if usual_hour == best_hour:
            return f"Great! You're already transacting at the optimal time ({best_hour}:00 UTC). Keep it up!"
        
        if savings_pct > 30:
            return f"You could save {savings_pct:.0f}% by shifting your transactions from {usual_hour}:00 UTC to {best_hour}:00 UTC. That's significant savings!"
        elif savings_pct > 15:
            return f"Consider transacting at {best_hour}:00 UTC instead of {usual_hour}:00 UTC. You could save around {savings_pct:.0f}%."
        else:
            return f"Your current transaction time ({usual_hour}:00 UTC) is reasonable. For maximum savings, try {best_hour}:00 UTC."
    
    def _day_name(self, day_num: int) -> str:
        """Convert day number to name"""
        days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        return days[day_num] if 0 <= day_num < 7 else 'Unknown'


# Global instance
_personalization_service = None

def get_personalization_service() -> PersonalizationService:
    """Get or create personalization service instance."""
    global _personalization_service
    if _personalization_service is None:
        _personalization_service = PersonalizationService()
    return _personalization_service

