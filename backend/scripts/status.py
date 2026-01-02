#!/usr/bin/env python3
"""
Quick Status Check - Shows everything at a glance
Run this anytime: python3 scripts/status.py
"""

import sqlite3
import sys
from pathlib import Path
from datetime import datetime, timedelta

print("=" * 70)
print("  📊 BASE GAS OPTIMIZER - STATUS CHECK")
print("=" * 70)

# Check database
db_path = Path("gas_data.db")
if not db_path.exists():
    print("\n❌ Database not found!")
    sys.exit(1)

conn = sqlite3.connect(str(db_path))
cursor = conn.cursor()

# 1. Total records
cursor.execute("SELECT COUNT(*) FROM gas_prices")
total_records = cursor.fetchone()[0]

# 2. Records with onchain features
cursor.execute("SELECT COUNT(*) FROM gas_prices WHERE pending_tx_count IS NOT NULL")
onchain_records = cursor.fetchone()[0]

# 3. Date range
cursor.execute("SELECT MIN(timestamp), MAX(timestamp) FROM gas_prices")
earliest, latest = cursor.fetchone()

# 4. Recent data (last hour)
one_hour_ago = (datetime.now() - timedelta(hours=1)).isoformat()
cursor.execute("SELECT COUNT(*) FROM gas_prices WHERE timestamp >= ?", (one_hour_ago,))
recent_records = cursor.fetchone()[0]

# 5. Average gas price (last 100 records)
cursor.execute("SELECT AVG(current_gas) FROM (SELECT current_gas FROM gas_prices ORDER BY timestamp DESC LIMIT 100)")
avg_gas = cursor.fetchone()[0] or 0

conn.close()

# Calculate progress
optimal_threshold = 1000
progress_pct = (onchain_records / optimal_threshold) * 100
remaining = max(0, optimal_threshold - onchain_records)
eta_hours = (remaining / 60) if remaining > 0 else 0  # 1 record/min

# Display results
print(f"\n📈 DATA COLLECTION")
print(f"   Total Records:        {total_records:,}")
print(f"   With Onchain Features: {onchain_records:,} / {optimal_threshold:,} ({progress_pct:.1f}%)")
print(f"   Recent (last hour):    {recent_records} records")
print(f"   Average Gas Price:     {avg_gas:.6f} Gwei")

print(f"\n📅 DATE RANGE")
print(f"   Earliest: {earliest}")
print(f"   Latest:   {latest}")

print(f"\n⏳ PROGRESS TO OPTIMAL TRAINING")
progress_bar = "█" * int(progress_pct / 5) + "░" * (20 - int(progress_pct / 5))
print(f"   [{progress_bar}] {progress_pct:.1f}%")
if remaining > 0:
    print(f"   Remaining: {remaining:,} records (~{eta_hours:.1f} hours)")
else:
    print(f"   ✅ READY FOR TRAINING!")

# Check models
print(f"\n🤖 MODELS")
model_dir = Path("models/saved_models")
models_exist = []
for horizon in ['1h', '4h', '24h']:
    model_path = model_dir / f"model_{horizon}.pkl"
    if model_path.exists():
        models_exist.append(horizon)
        print(f"   ✅ {horizon} model exists")
    else:
        print(f"   ⚠️  {horizon} model not found")

# Recommendations
print(f"\n💡 RECOMMENDATIONS")
if onchain_records < optimal_threshold:
    print(f"   ⏳ Wait {eta_hours:.1f} hours for optimal data collection")
    print(f"   📊 Currently at {progress_pct:.1f}% - collection is running")
elif not models_exist:
    print(f"   🎯 READY TO TRAIN! Run: python3 scripts/train_with_current_data.py")
else:
    print(f"   ✅ Models trained! Optionally retrain with: python3 scripts/train_with_current_data.py")
    print(f"   🧪 Test models with: python3 testing/backtest_current_models.py")

print(f"\n" + "=" * 70)
