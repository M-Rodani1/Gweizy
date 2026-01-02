#!/usr/bin/env python3
"""
Check Data Collection Status

Quick script to check how much data has been collected
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.database import DatabaseManager
from datetime import datetime, timedelta

db = DatabaseManager()
conn = db.get_connection()
cursor = conn.cursor()

# Gas prices
cursor.execute("SELECT COUNT(*), MIN(timestamp), MAX(timestamp) FROM gas_prices")
gas_count, gas_min, gas_max = cursor.fetchone()
print("📊 Gas Prices:")
print(f"   Total: {gas_count:,} records")
if gas_min:
    print(f"   Range: {gas_min} to {gas_max}")

# OnChain features
cursor.execute("SELECT COUNT(*), MIN(timestamp), MAX(timestamp) FROM onchain_features")
onchain_count, onchain_min, onchain_max = cursor.fetchone()
print(f"\n📊 OnChain Features:")
print(f"   Total: {onchain_count:,} records")
if onchain_min:
    print(f"   Range: {onchain_min} to {onchain_max}")

# Recent data
cursor.execute("SELECT COUNT(*) FROM onchain_features WHERE timestamp > datetime('now', '-1 hour')")
recent_1h = cursor.fetchone()[0]

cursor.execute("SELECT COUNT(*) FROM onchain_features WHERE timestamp > datetime('now', '-24 hours')")
recent_24h = cursor.fetchone()[0]

print(f"\n📊 Recent Collection:")
print(f"   Last hour: {recent_1h} records")
print(f"   Last 24h: {recent_24h} records")

# Enhanced features check
cursor.execute("PRAGMA table_info(onchain_features)")
columns = [row[1] for row in cursor.fetchall()]
enhanced_cols = ['pending_tx_count', 'unique_addresses', 'tx_per_second', 'gas_utilization_ratio']
found_cols = [c for c in enhanced_cols if c in columns]
print(f"\n📊 Enhanced Features:")
print(f"   Columns present: {len(found_cols)}/{len(enhanced_cols)}")
if found_cols:
    print(f"   ✅ {', '.join(found_cols)}")

# Training readiness (Optimal: 1,000+ records)
needed_optimal = 1000
needed_minimum = 100
ready_optimal = recent_24h >= needed_optimal
ready_minimum = recent_24h >= needed_minimum

print(f"\n🎯 Training Readiness:")
print(f"   Minimum: {needed_minimum}+ records (can train now, limited quality)")
print(f"   Optimal: {needed_optimal}+ records (best results)")
print(f"   Have: {recent_24h} records")
print(f"   Status: {'✅ OPTIMAL - Ready!' if ready_optimal else '✅ MINIMUM - Can train' if ready_minimum else '⏳ COLLECTING'}")
if not ready_optimal:
    remaining = needed_optimal - recent_24h
    # Calculate based on 1 record per minute = 60 records per hour (standard rate)
    # Use standard rate for consistency
    records_per_hour = 60
    hours_needed = remaining / records_per_hour
    days_needed = hours_needed / 24
    print(f"   Time to optimal: ~{hours_needed:.1f} hours (~{days_needed:.2f} days) at 1 record/min rate")

conn.close()
