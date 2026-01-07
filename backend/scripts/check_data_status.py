#!/usr/bin/env python3
"""
Quick script to check data collection status
Shows how much data has been collected and if it's enough for training
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.database import DatabaseManager
from datetime import datetime, timedelta

def check_data_status():
    """Check data collection status"""
    print("="*70)
    print("📊 DATA COLLECTION STATUS")
    print("="*70)
    
    try:
        db = DatabaseManager()
        
        # Get all data
        all_data = db.get_historical_data(hours=720)  # 30 days
        
        if not all_data:
            print("\n❌ No data collected yet!")
            print("   Data collection should start automatically on deployment.")
            return
        
        total_records = len(all_data)
        
        # Get recent data
        recent_24h = db.get_historical_data(hours=24)
        recent_7d = db.get_historical_data(hours=168)
        
        recent_24h_count = len(recent_24h)
        recent_7d_count = len(recent_7d)
        
        # Calculate date range
        if all_data:
            from dateutil import parser
            timestamps = []
            for d in all_data:
                ts = d.get('timestamp', '')
                if isinstance(ts, str):
                    try:
                        timestamps.append(parser.parse(ts))
                    except:
                        try:
                            timestamps.append(datetime.fromisoformat(ts.replace('Z', '+00:00')))
                        except:
                            pass
                elif hasattr(ts, 'year'):
                    timestamps.append(ts)
            
            if timestamps:
                timestamps.sort()
                days_of_data = (timestamps[-1] - timestamps[0]).total_seconds() / 86400
                oldest = timestamps[0]
                newest = timestamps[-1]
            else:
                days_of_data = 0
                oldest = None
                newest = None
        else:
            days_of_data = 0
            oldest = None
            newest = None
        
        # Training requirements
        min_samples = 500
        optimal_samples = 1000
        min_days = 7
        optimal_days = 30
        
        # Check if sufficient
        sufficient = total_records >= min_samples and days_of_data >= min_days
        optimal = total_records >= optimal_samples and days_of_data >= optimal_days
        
        print(f"\n📈 DATA SUMMARY")
        print(f"   Total Records:        {total_records:,}")
        print(f"   Recent (24h):        {recent_24h_count:,} records")
        print(f"   Recent (7d):         {recent_7d_count:,} records")
        
        if oldest and newest:
            print(f"\n📅 DATE RANGE")
            print(f"   Oldest:              {oldest.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"   Newest:              {newest.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"   Days of Data:        {days_of_data:.1f} days")
        
        print(f"\n🎯 TRAINING REQUIREMENTS")
        print(f"   Minimum:             {min_samples:,} samples, {min_days} days")
        print(f"   Optimal:             {optimal_samples:,} samples, {optimal_days} days")
        
        print(f"\n✅ STATUS")
        if optimal:
            print(f"   ✅ OPTIMAL - Ready for high-quality training!")
            print(f"      You have {total_records:,} samples and {days_of_data:.1f} days of data")
        elif sufficient:
            print(f"   ⚠️  SUFFICIENT - Can train, but more data recommended")
            print(f"      You have {total_records:,} samples and {days_of_data:.1f} days of data")
            print(f"      Progress: {min(100, (total_records/optimal_samples)*100):.1f}% samples, {min(100, (days_of_data/optimal_days)*100):.1f}% days")
        else:
            print(f"   ❌ INSUFFICIENT - Need more data for training")
            if total_records < min_samples:
                needed_samples = min_samples - total_records
                print(f"      Need {needed_samples:,} more samples (have {total_records:,})")
            if days_of_data < min_days:
                needed_days = min_days - days_of_data
                print(f"      Need {needed_days:.1f} more days (have {days_of_data:.1f})")
            
            # Estimate time to sufficient
            if recent_24h_count > 0:
                samples_per_day = recent_24h_count
                if total_records < min_samples:
                    days_needed_samples = (min_samples - total_records) / samples_per_day if samples_per_day > 0 else 999
                else:
                    days_needed_samples = 0
                
                if days_of_data < min_days:
                    days_needed_time = min_days - days_of_data
                else:
                    days_needed_time = 0
                
                days_needed = max(days_needed_samples, days_needed_time)
                if days_needed > 0:
                    print(f"\n   ⏱️  ESTIMATED TIME TO SUFFICIENT DATA: {days_needed:.1f} days")
                    print(f"      (Based on current collection rate: ~{samples_per_day:.0f} samples/day)")
        
        # Collection rate
        if recent_24h_count > 0:
            expected_24h = 24 * 12  # 12 samples per hour (5 min intervals)
            collection_rate = (recent_24h_count / expected_24h) * 100 if expected_24h > 0 else 0
            print(f"\n📊 COLLECTION RATE")
            print(f"   Last 24h:            {recent_24h_count} records")
            print(f"   Expected:           {expected_24h} records (5-min intervals)")
            print(f"   Rate:               {collection_rate:.1f}%")
            
            if collection_rate < 50:
                print(f"   ⚠️  Collection rate is low - check if data collection is running")
        
        print(f"\n" + "="*70)
        
    except Exception as e:
        print(f"\n❌ Error checking data status: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    check_data_status()

