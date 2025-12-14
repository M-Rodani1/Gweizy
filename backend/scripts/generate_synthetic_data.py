#!/usr/bin/env python3
"""
Generate synthetic Base gas price data for ML training
Run: python scripts/generate_synthetic_data.py
"""

import sys
sys.path.append('.')

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from data.database import DatabaseManager


def generate_base_gas_data(days=90):
    """Generate realistic Base L2 gas price data"""
    
    samples_per_hour = 12  # Every 5 minutes
    total_samples = days * 24 * samples_per_hour
    
    # Timestamps
    end_time = datetime.now()
    start_time = end_time - timedelta(days=days)
    timestamps = pd.date_range(start=start_time, end=end_time, periods=total_samples)
    
    # Base gas is VERY cheap (L2)
    base_level = 0.002  # ~0.002 gwei average
    
    data = []
    
    for i, timestamp in enumerate(timestamps):
        hour = timestamp.hour
        day_of_week = timestamp.dayofweek
        
        # Daily pattern (peak ~2pm UTC = 9am EST)
        hour_factor = 1.0 + 0.3 * np.sin(2 * np.pi * (hour - 14) / 24)
        
        # Weekly pattern
        week_factor = 1.2 if day_of_week < 5 else 0.8
        
        # Trend
        trend = 1.0 + 0.1 * (i / total_samples)
        
        # Random spikes (1% chance)
        spike = np.random.uniform(2.0, 5.0) if np.random.random() < 0.01 else 1.0
        
        # Noise
        noise = np.random.normal(1.0, 0.1)
        
        # Combine
        gas_price = base_level * hour_factor * week_factor * trend * spike * noise
        gas_price = max(0.0001, min(gas_price, 0.05))
        
        # EIP-1559 split
        base_fee = gas_price * 0.9
        priority_fee = gas_price * 0.1
        
        # Block number
        block_number = 39_000_000 + int(i * 2.5)
        
        data.append({
            'timestamp': timestamp,
            'current_gas': round(gas_price, 6),
            'base_fee': round(base_fee, 6),
            'priority_fee': round(priority_fee, 6),
            'block_number': block_number
        })
    
    return data


def main():
    print("\n" + "="*60)
    print("🎲 Generating Synthetic Base Gas Data for ML Training")
    print("="*60 + "\n")
    
    print("📊 Generating 30 days of data (8,640 points)...")
    data = generate_base_gas_data(days=30)
    
    print(f"✅ Generated {len(data)} data points")
    print(f"   Range: {data[0]['timestamp']} to {data[-1]['timestamp']}")
    print(f"   Gas: {min(d['current_gas'] for d in data):.6f} - {max(d['current_gas'] for d in data):.6f} gwei\n")
    
    print("💾 Saving to database...")
    db = DatabaseManager()
    
    for i, record in enumerate(data):
        db.save_gas_price(record)
        if (i + 1) % 1000 == 0:
            print(f"   Saved {i + 1}/{len(data)}...")
    
    print(f"\n✅ Complete! Saved {len(data)} records\n")
    print("="*60)
    print("Next steps:")
    print("  1. Train models: python scripts/train_model.py")
    print("  2. Start backend: python app.py")
    print("  3. Test frontend: npm run dev")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()

